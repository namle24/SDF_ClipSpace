"""
sdf_calculator_gpu.py — Nâng cấp: Clip-Space Möller-Trumbore SDF trên GPU

Kiến trúc Hybrid:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  GIAI ĐOẠN 1 (Ý tưởng Giáo sư):                                  │
  │    look_at(eye, target) → V          Ma trận View  4×4            │
  │    perspective(fov, near, far) → P   Ma trận Projection 4×4       │
  │    PV = P @ V                                                     │
  │    vertices_clip = (PV @ verts_homo.T).T   ← Perspective Divide   │
  │    ⇒ Chùm tia nón (Cone) trong World Space                       │
  │       được NẮN THẲNG thành chùm tia SONG SONG dọc trục Z         │
  │       trong Clipping Space.                                       │
  ├─────────────────────────────────────────────────────────────────────┤
  │  GIAI ĐOẠN 2 (Lõi GPU của chúng ta):                              │
  │    Sinh lưới tia song song D = [0, 0, -1] trên mặt phẳng XY      │
  │    (vòng tròn đồng tâm — Ring Pattern của Giáo sư).               │
  │    Đưa tam giác ĐÃ BIẾN ĐỔI + tia song song vào                  │
  │    Möller-Trumbore Batched GPU → tìm (u, v, t).                   │
  │    Dùng (u, v) nội suy ngược hit point trong WORLD SPACE          │
  │    → Tính khoảng cách Euclid chính xác.                           │
  └─────────────────────────────────────────────────────────────────────┘
"""

import trimesh
import numpy as np
import pyvista as pv
import time
import torch
from tqdm import tqdm


# =====================================================================
# 1. CAMERA MATRICES — PyTorch GPU (dịch nguyên bản từ fast_sdf.py)
# =====================================================================

def look_at_torch(eye, target, device='cuda'):
    """
    Ma trận View 4×4:  World Space → Camera Space.
    Giữ nguyên logic Gram–Schmidt random-up của Giáo sư.
    """
    forward = target - eye
    forward = forward / (torch.norm(forward) + 1e-12)

    # Random up → chiếu vuông góc → right → true_up  (Gram-Schmidt)
    up = torch.randn(3, device=device, dtype=torch.float32)
    up = up - torch.dot(up, forward) * forward
    up = up / (torch.norm(up) + 1e-12)

    right = torch.linalg.cross(forward, up)
    right = right / (torch.norm(right) + 1e-12)
    true_up = torch.linalg.cross(right, forward)

    rotation = torch.tensor([
        [right[0],    right[1],    right[2],    0],
        [true_up[0],  true_up[1],  true_up[2],  0],
        [-forward[0], -forward[1], -forward[2], 0],
        [0,           0,           0,           1]
    ], dtype=torch.float32, device=device)

    translation = torch.eye(4, dtype=torch.float32, device=device)
    translation[0, 3] = -eye[0]
    translation[1, 3] = -eye[1]
    translation[2, 3] = -eye[2]

    return rotation @ translation


def perspective_torch(fov_rad, near, far, device='cuda'):
    """
    Ma trận Projection 4×4:  Camera Space → Clip Space.
    Công thức giống hệt perspective() trong fast_sdf.py.
    """
    cot = 1.0 / torch.tan(fov_rad / 2)
    P = torch.zeros((4, 4), dtype=torch.float32, device=device)
    P[0, 0] = cot
    P[1, 1] = cot
    P[2, 2] = -far / (far - near)
    P[2, 3] = -(near * far) / (far - near)
    P[3, 2] = -1.0
    return P


# =====================================================================
# 2. SINH LƯỚI TIA SONG SONG TRONG CLIP SPACE (Ring Pattern)
# =====================================================================

def generate_clip_space_rays(fov_rad, num_rings=5, num_rays_per_ring=10, device='cuda'):
    """
    Sinh chùm tia song song D=[0,0,-1] trên mặt phẳng XY của Clip Space.
    Vòng tròn đồng tâm (Ring) — giữ nguyên logic sinh tia của Giáo sư.
    
    Returns:
        ray_origins_xy: (R, 2) — tọa độ XY gốc phát tia trong Clip Space
    """
    tan_fov = torch.tan(fov_rad / 2)
    all_xy = []
    for ring in range(num_rings):
        r = (ring + 0.5) / num_rings
        num_points = int(num_rays_per_ring * (ring + 1))
        angles = torch.linspace(0, 2 * torch.pi, num_points + 1, device=device)[:-1]
        x = r * torch.cos(angles) * tan_fov
        y = r * torch.sin(angles) * tan_fov
        all_xy.append(torch.stack([x, y], dim=-1))
    return torch.cat(all_xy, dim=0)  # (R, 2)


# =====================================================================
# 3. MÖLLER-TRUMBORE TRÊN CLIP SPACE — Lõi GPU Batching
# =====================================================================

def moller_trumbore_clip_batch(ray_origins, ray_dir, V0c, V1c, V2c,
                               V0w, V1w, V2w,
                               umbrella_mask, ray_batch_size, device):
    """
    Möller-Trumbore GPU Batching trong Clip Space.
    Tia song song D=[0,0,-1], gốc tại (px, py, 0).
    
    Trả về (u_hit, v_hit, face_hit_idx, hit_valid) cho mỗi tia.
    Barycentric coords (u,v) để nội suy lại World Space.
    
    Args:
        ray_origins:    (R, 3) — gốc tia trong Clip Space
        ray_dir:        (3,)   — hướng tia [0, 0, -1]
        V0c, V1c, V2c:  (Fv, 3) — đỉnh tam giác trong Clip Space (đã lọc)
        V0w, V1w, V2w:  (Fv, 3) — đỉnh tam giác trong World Space (để nội suy)
        umbrella_mask:  (Fv,) bool — mặt chứa đỉnh gốc = True
        ray_batch_size: int
    
    Returns:
        hit_distances: (R,) — khoảng cách Euclid World Space (inf nếu miss)
    """
    R = len(ray_origins)
    Fv = len(V0c)
    
    E1c = V1c - V0c  # (Fv, 3)
    E2c = V2c - V0c  # (Fv, 3)
    
    best_t = torch.full((R,), float('inf'), device=device)
    best_u = torch.zeros(R, device=device)
    best_v = torch.zeros(R, device=device)
    best_face = torch.zeros(R, dtype=torch.long, device=device)
    
    D = ray_dir.unsqueeze(0)  # (1, 3)
    
    for i in range(0, R, ray_batch_size):
        end = min(i + ray_batch_size, R)
        B = end - i
        
        O_batch = ray_origins[i:end]  # (B, 3)
        
        # Möller-Trumbore: P = D × E2  (broadcast D over F faces)
        # D shape (1,3), E2c shape (Fv,3) → P shape (Fv, 3), shared across batch
        P = torch.linalg.cross(D.expand(Fv, 3), E2c)  # (Fv, 3)
        
        # a = E1 · P
        a = torch.sum(E1c * P, dim=-1)  # (Fv,)
        
        # Lọc mặt gần song song với tia
        face_valid = torch.abs(a) > 1e-8  # (Fv,)
        face_valid = face_valid & (~umbrella_mask)  # Loại Umbrella
        
        safe_a = torch.where(face_valid, a, torch.ones_like(a))
        f = 1.0 / safe_a  # (Fv,)
        
        # s = O - V0  →  (B, Fv, 3)
        s = O_batch.unsqueeze(1) - V0c.unsqueeze(0)  # (B, Fv, 3)
        
        # u_bary = f * (s · P)
        u_bary = f.unsqueeze(0) * torch.sum(s * P.unsqueeze(0), dim=-1)  # (B, Fv)
        
        bary_valid = face_valid.unsqueeze(0) & (u_bary >= 0.0) & (u_bary <= 1.0)
        
        # Q = s × E1
        Q = torch.linalg.cross(s, E1c.unsqueeze(0).expand(B, Fv, 3))  # (B, Fv, 3)
        
        # v_bary = f * (D · Q)
        v_bary = f.unsqueeze(0) * torch.sum(D.unsqueeze(0) * Q, dim=-1)  # (B, Fv)
        bary_valid = bary_valid & (v_bary >= 0.0) & (u_bary + v_bary <= 1.0)
        
        # t = f * (E2 · Q)
        t = f.unsqueeze(0) * torch.sum(E2c.unsqueeze(0) * Q, dim=-1)  # (B, Fv)
        bary_valid = bary_valid & (t > 1e-4)  # Phía trước tia
        
        # Đặt những miss = inf
        t = torch.where(bary_valid, t, torch.tensor(float('inf'), device=device))
        
        # Tìm mặt gần nhất cho mỗi tia trong batch
        min_t, min_idx = torch.min(t, dim=-1)  # (B,)
        
        # Cập nhật best nếu t nhỏ hơn
        update_mask = min_t < best_t[i:end]
        best_t[i:end] = torch.where(update_mask, min_t, best_t[i:end])
        
        # Lấy u, v tại face gần nhất
        batch_arange = torch.arange(B, device=device)
        u_at_min = u_bary[batch_arange, min_idx]
        v_at_min = v_bary[batch_arange, min_idx]
        
        best_u[i:end] = torch.where(update_mask, u_at_min, best_u[i:end])
        best_v[i:end] = torch.where(update_mask, v_at_min, best_v[i:end])
        best_face[i:end] = torch.where(update_mask, min_idx, best_face[i:end])
    
    # Nội suy hit point trong WORLD SPACE bằng Barycentric
    hit_valid = best_t < float('inf')
    
    w_bary = 1.0 - best_u - best_v  # (R,)
    hit_world = (w_bary.unsqueeze(-1) * V0w[best_face] +
                 best_u.unsqueeze(-1) * V1w[best_face] +
                 best_v.unsqueeze(-1) * V2w[best_face])  # (R, 3)
    
    return hit_world, hit_valid


# =====================================================================
# 4. ENGINE CHÍNH — Clip-Space + Möller-Trumbore GPU
# =====================================================================

@torch.no_grad()
def compute_sdf_clipspace_gpu(mesh, fov_deg=90, num_rings=5, num_rays_per_ring=10,
                              vertex_batch_size=64, ray_batch_size=4096):
    """
    SDF Calculation kết hợp:
      - View-Projection (Giáo sư) chuyển mesh sang Clip Space
      - Möller-Trumbore GPU (chúng ta) tìm Hitting Point
      - Nội suy Barycentric ngược World Space tính khoảng cách chính xác
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  CLIP-SPACE MÖLLER-TRUMBORE SDF ENGINE (GPU)")
    print(f"  Thiết bị: {device}")
    print(f"{'='*60}")

    start_time = time.perf_counter()

    # --- Tải dữ liệu lên VRAM ---
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)

    N = len(vertices)
    F = len(faces)

    # Homogeneous (tính 1 lần)
    ones = torch.ones((N, 1), dtype=torch.float32, device=device)
    verts_homo = torch.cat([vertices, ones], dim=-1)  # (N, 4)

    # World-Space triangle vertices (bất biến — dùng để nội suy cuối cùng)
    V0w = vertices[faces[:, 0]]  # (F, 3)
    V1w = vertices[faces[:, 1]]
    V2w = vertices[faces[:, 2]]

    # Camera parameters
    fov_rad = torch.tensor(np.radians(fov_deg), dtype=torch.float32, device=device)
    near = torch.tensor(0.001, dtype=torch.float32, device=device)
    bbox_diag = torch.norm(vertices.max(dim=0).values - vertices.min(dim=0).values)
    far = bbox_diag + 1.0

    # Ma trận Projection (chung cho mọi đỉnh)
    P = perspective_torch(fov_rad, near, far, device)

    # Sinh lưới tia song song 1 lần
    ray_xy = generate_clip_space_rays(fov_rad, num_rings, num_rays_per_ring, device)
    R = len(ray_xy)
    # Gốc tia trong NDC: (px, py, z_start)
    # Sau Perspective Divide của Giáo sư, NDC z nằm trong khoảng [0..1] (near→far).
    # Camera nhìn theo -Z trong Camera Space → sau phép chiếu, vật thể nằm ở Z dương trong NDC.
    # Đặt gốc tia ngay mặt phẳng Near (z ≈ 0) và bắn dọc Z+ hướng vào sâu.
    ray_origins_clip = torch.zeros((R, 3), dtype=torch.float32, device=device)
    ray_origins_clip[:, 0] = ray_xy[:, 0]
    ray_origins_clip[:, 1] = ray_xy[:, 1]
    ray_origins_clip[:, 2] = -0.01  # Hơi trước mặt phẳng Near
    # Hướng tia: D = [0, 0, +1] (dọc trục Z dương trong NDC)
    ray_dir = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)

    print(f"[*] Mesh: {N} đỉnh, {F} tam giác")
    print(f"[*] Tia/đỉnh: {R} | FOV: {fov_deg}° | Near/Far: {near.item():.3f}/{far.item():.1f}")

    sdf_values = torch.zeros(N, dtype=torch.float32, device=device)

    # --- Xử lý theo Batch đỉnh ---
    for b_start in tqdm(range(0, N, vertex_batch_size),
                        desc="Clip-Space MT Batches"):
        b_end = min(b_start + vertex_batch_size, N)

        for idx in range(b_start, b_end):
            # ========== GIAI ĐOẠN 1: View-Projection (Giáo sư) ==========
            inward = -normals[idx]
            norm_val = torch.norm(inward)
            if norm_val < 1e-8:
                continue
            inward = inward / norm_val

            eye = vertices[idx] + 0.001 * inward
            target = eye + inward

            V_mat = look_at_torch(eye, target, device)  # (4, 4)
            PV = P @ V_mat  # (4, 4)

            # Chuyển MỌI đỉnh sang Clip Space 1 phát
            verts_clip4 = (PV @ verts_homo.T).T  # (N, 4)
            w_clip = verts_clip4[:, 3:4]
            w_safe = torch.where(torch.abs(w_clip) > 1e-8, w_clip,
                                 torch.ones_like(w_clip) * 1e-8)
            # ★ Perspective Divide: Nón → Ống trụ song song
            verts_ndc = verts_clip4[:, :3] / w_safe  # (N, 3)

            # Clip-Space triangle vertices
            V0c = verts_ndc[faces[:, 0]]  # (F, 3)
            V1c = verts_ndc[faces[:, 1]]
            V2c = verts_ndc[faces[:, 2]]

            # Lọc sơ bộ: Camera Space Z < 0 (phía trước camera)
            verts_cam = (V_mat @ verts_homo.T).T[:, :3]
            face_z = (verts_cam[faces[:, 0], 2] +
                      verts_cam[faces[:, 1], 2] +
                      verts_cam[faces[:, 2], 2]) / 3.0
            in_front = face_z < -0.005

            # Umbrella: mặt chứa chính đỉnh gốc
            is_umbrella = ((faces[:, 0] == idx) |
                           (faces[:, 1] == idx) |
                           (faces[:, 2] == idx))

            valid_mask = in_front & (~is_umbrella)
            if valid_mask.sum() == 0:
                continue

            vf_idx = torch.where(valid_mask)[0]
            V0c_f = V0c[vf_idx]
            V1c_f = V1c[vf_idx]
            V2c_f = V2c[vf_idx]
            V0w_f = V0w[vf_idx]
            V1w_f = V1w[vf_idx]
            V2w_f = V2w[vf_idx]
            umbrella_f = torch.zeros(len(vf_idx), dtype=torch.bool, device=device)

            # ========== GIAI ĐOẠN 2: Möller-Trumbore GPU ==========
            hit_world, hit_valid = moller_trumbore_clip_batch(
                ray_origins_clip, ray_dir,
                V0c_f, V1c_f, V2c_f,
                V0w_f, V1w_f, V2w_f,
                umbrella_f, ray_batch_size, device
            )

            if hit_valid.sum() == 0:
                continue

            # Khoảng cách Euclid trong World Space
            dists = torch.norm(hit_world - vertices[idx].unsqueeze(0), dim=-1)
            dists = torch.where(hit_valid, dists, torch.zeros_like(dists))

            sdf_values[idx] = dists.sum() / hit_valid.sum()

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    avg = sdf_values.mean().item()

    print(f"\n{'='*60}")
    print(f"  THỜI GIAN: {elapsed:.4f} giây")
    print(f"  SDF Trung bình: {avg:.4f}")
    print(f"{'='*60}\n")

    return sdf_values.cpu().numpy()


# =====================================================================
# 5. MAIN
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Clip-Space Möller-Trumbore SDF (GPU)")
    parser.add_argument("--input_file", type=str, default="data/radio_0026.off")
    parser.add_argument("--fov", type=int, default=120)
    parser.add_argument("--num_rings", type=int, default=5)
    parser.add_argument("--rays_per_ring", type=int, default=10)
    parser.add_argument("--vertex_batch", type=int, default=64)
    parser.add_argument("--ray_batch", type=int, default=4096)
    args = parser.parse_args()

    try:
        mesh = trimesh.load(args.input_file, force='mesh')
        mesh.fix_normals()
        mesh.process()
        if len(mesh.vertices) == 0:
            raise ValueError("Mesh trống!")
    except Exception as e:
        print(f"Lỗi tải mesh: {e}")
        return

    sdf = compute_sdf_clipspace_gpu(
        mesh,
        fov_deg=args.fov,
        num_rings=args.num_rings,
        num_rays_per_ring=args.rays_per_ring,
        vertex_batch_size=args.vertex_batch,
        ray_batch_size=args.ray_batch
    )

    # Visualization — MeshLab Style
    pv_mesh = pv.wrap(mesh)
    pv_mesh.point_data['ClipSpace_MT_SDF'] = sdf

    plotter = pv.Plotter(title="Clip-Space Möller-Trumbore SDF (GPU)")
    plotter.add_mesh(
        pv_mesh,
        scalars='ClipSpace_MT_SDF',
        cmap='jet_r',
        smooth_shading=False,
        show_edges=False,
        scalar_bar_args={'title': "SDF (Clip-Space MT GPU)"}
    )
    plotter.set_background('white')
    plotter.add_axes()
    plotter.show()


if __name__ == "__main__":
    main()
