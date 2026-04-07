"""
hybrid_sdf_world_gpu.py — True Hybrid Pipeline: Camera-Space Rays + World-Space MT

Pipeline 4 bước (Mathematically Rigorous):
  ┌─────────────────────────────────────────────────────────────────┐
  │  BƯỚC 1 (Camera Space):                                        │
  │    V_mat = look_at(eye, target)                                │
  │    Sinh chùm tia hình nón trong Camera Space:                  │
  │    D_cam = normalize(px, py, -1)  ← Ring pattern               │
  │                                                                │
  │  BƯỚC 2 (Warp to World Space):                                 │
  │    V_rot = V_mat[:3, :3]        ← Khối xoay 3×3               │
  │    D_world = V_rot^T @ D_cam    ← Nghịch đảo (ortho → .T)     │
  │    O_world = eye                ← Gốc tia trong World Space    │
  │                                                                │
  │  BƯỚC 3 (Möller-Trumbore GPU):                                 │
  │    Bắn D_world từ O_world vào tam giác NGUYÊN BẢN World Space  │
  │    → Giải hệ Barycentric (u, v, t)                             │
  │                                                                │
  │  BƯỚC 4 (Exact Distance):                                      │
  │    SDF = mean(min_t)  ← t chính là khoảng cách chính xác       │
  │    (vì D_world đã normalize, ||D||=1 → distance = |t|)         │
  └─────────────────────────────────────────────────────────────────┘

ĐẢM BẢO TOÁN HỌC:
  - KHÔNG dùng face_centers → loại bỏ Quantization Error
  - KHÔNG dùng Perspective Divide → loại bỏ mọi tranh cãi phi tuyến
  - t từ Möller-Trumbore LÀ khoảng cách Euclid chính xác
    (vì V_rot là ma trận trực giao → V_rot^T bảo toàn chuẩn vector)
"""

import trimesh
import numpy as np
import pyvista as pv
import time
import torch
from tqdm import tqdm


# VIEW MATRIX — Camera Setup (fast_sdf.py)

def look_at_torch(eye, target, device='cuda'):
    """
    Ma trận View 4×4: World → Camera.
    Deterministic up vector để kết quả ổn định giữa các lần chạy.
    
    Tính chất quan trọng: khối xoay 3×3 phía trên-trái là MA TRẬN TRỰC GIAO.
    → Nghịch đảo = Chuyển vị:  R^{-1} = R^T
    """
    forward = target - eye
    forward = forward / (torch.norm(forward) + 1e-12)

    if torch.abs(forward[1]) > 0.99:
        world_up = torch.tensor([0.0, 0.0, 1.0], device=device)
    else:
        world_up = torch.tensor([0.0, 1.0, 0.0], device=device)

    right = torch.linalg.cross(forward, world_up)
    right = right / (torch.norm(right) + 1e-12)
    true_up = torch.linalg.cross(right, forward)

    # V_mat[:3,:3] = [right; true_up; -forward] — MA TRẬN TRỰC GIAO
    rotation = torch.zeros((4, 4), dtype=torch.float32, device=device)
    rotation[0, :3] = right
    rotation[1, :3] = true_up
    rotation[2, :3] = -forward
    rotation[3, 3] = 1.0

    translation = torch.eye(4, dtype=torch.float32, device=device)
    translation[0, 3] = -eye[0]
    translation[1, 3] = -eye[1]
    translation[2, 3] = -eye[2]

    return rotation @ translation


# SINH CHÙM TIA HÌNH NÓN TRONG CAMERA SPACE (Ring Pattern)
def generate_cone_rays_camera(fov_rad, num_rings=5, num_rays_per_ring=10, device='cuda'):
    """
    Sinh hướng tia hình nón trong Camera Space (nhìn theo -Z).
    Ring đồng tâm — logic giống fast_sdf.py Giáo sư.
    
    Returns:
        ray_dirs_cam: (R, 3) — ĐÃ NORMALIZE, ||D_cam|| = 1
    """
    tan_fov = torch.tan(fov_rad / 2).item()
    all_dirs = []
    for ring in range(num_rings):
        r = (ring + 0.5) / num_rings
        num_points = int(num_rays_per_ring * (ring + 1))
        angles = torch.linspace(0, 2 * torch.pi, num_points + 1, device=device)[:-1]
        x = r * torch.cos(angles) * tan_fov
        y = r * torch.sin(angles) * tan_fov
        z = -torch.ones_like(x)  # Camera nhìn theo -Z
        dirs = torch.stack([x, y, z], dim=-1)
        # ★ NORMALIZE: đảm bảo ||D_cam|| = 1 → t = distance
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        all_dirs.append(dirs)
    return torch.cat(all_dirs, dim=0)  # (R, 3)

# TRUE HYBRID ENGINE — 4 bước chính xác tuyệt đối

@torch.no_grad()
def compute_hybrid_sdf_world_gpu(mesh, fov_deg=120, num_rings=5, num_rays_per_ring=10,
                                  vertex_batch_size=64, ray_batch_size=4096):
    """
    True Hybrid SDF Pipeline:
      Bước 1: look_at → V_mat (Camera Setup)
      Bước 2: V_rot^T @ D_cam → D_world (Nghịch đảo không gian)
      Bước 3: Möller-Trumbore GPU trong World Space (tam giác nguyên bản)
      Bước 4: t chính xác → SDF
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  TRUE HYBRID PIPELINE: Camera Rays + World-Space MT (GPU)")
    print(f"  Thiết bị: {device}")
    print(f"{'='*60}")

    start_time = time.perf_counter()

    # --- Tải dữ liệu lên VRAM ---
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)

    N = len(vertices)
    F = len(faces)

    # Tam giác NGUYÊN BẢN trong World Space (KHÔNG biến đổi)
    V0 = vertices[faces[:, 0]]  # (F, 3)
    V1 = vertices[faces[:, 1]]
    V2 = vertices[faces[:, 2]]
    E1 = V1 - V0  # (F, 3)
    E2 = V2 - V0  # (F, 3)

    # Camera parameters
    fov_rad = torch.tensor(np.radians(fov_deg), dtype=torch.float32, device=device)

    # BƯỚC 1: Sinh chùm tia hình nón trong Camera Space (1 lần duy nhất)
    ray_dirs_cam = generate_cone_rays_camera(fov_rad, num_rings, num_rays_per_ring, device)
    R = len(ray_dirs_cam)  # Số tia / đỉnh

    print(f"[*] Mesh: {N} đỉnh, {F} tam giác")
    print(f"[*] Tia/đỉnh: {R} | FOV: {fov_deg}° | Rings: {num_rings}")

    sdf_values = torch.zeros(N, dtype=torch.float32, device=device)

    # VÒNG LẶP CHÍNH: Xử lý từng đỉnh
    for b_start in tqdm(range(0, N, vertex_batch_size),
                        desc="Hybrid World-MT Batches"):
        b_end = min(b_start + vertex_batch_size, N)

        for idx in range(b_start, b_end):
            # ===== BƯỚC 1: Camera Setup (fast_sdf.py) =====
            inward = -normals[idx]
            norm_val = torch.norm(inward)
            if norm_val < 1e-8:
                continue
            inward = inward / norm_val

            eye = vertices[idx] + 0.001 * inward  # Gốc tia World Space
            target = eye + inward
            V_mat = look_at_torch(eye, target, device)  # (4×4)

            # BƯỚC 2: Nghịch đảo không gian (V_rot^T)
            # V_rot = V_mat[:3,:3] là ma trận TRỰC GIAO (orthogonal)
            # → Nghịch đảo = Chuyển vị: V_rot^{-1} = V_rot^T
            V_rot_T = V_mat[:3, :3].T  # (3×3) — đây là V_rot^{-1}

            # D_world = V_rot^T @ D_cam
            # V_rot^T bảo toàn chuẩn → ||D_world|| = ||D_cam|| = 1
            ray_dirs_world = (V_rot_T @ ray_dirs_cam.T).T  # (R, 3)

            # O_world = eye (đã có sẵn)
            O_world = eye  # (3,)

            # Lọc Umbrella: loại mặt chứa chính đỉnh gốc
            is_umbrella = ((faces[:, 0] == idx) |
                           (faces[:, 1] == idx) |
                           (faces[:, 2] == idx))  # (F,)

            # ===== BƯỚC 3: Möller-Trumbore GPU trong WORLD SPACE =====
            # Tam giác: V0, V1, V2, E1, E2 — NGUYÊN BẢN, không biến đổi
            # Tia: O_world, ray_dirs_world
            
            best_t = torch.full((R,), float('inf'), device=device)

            for r_start in range(0, R, ray_batch_size):
                r_end = min(r_start + ray_batch_size, R)
                B = r_end - r_start

                D_batch = ray_dirs_world[r_start:r_end]  # (B, 3)

                # P = D × E2    (B, F, 3)
                P = torch.linalg.cross(
                    D_batch.unsqueeze(1).expand(B, F, 3),
                    E2.unsqueeze(0).expand(B, F, 3)
                )

                # a = E1 · P    (B, F)
                a = torch.sum(E1.unsqueeze(0) * P, dim=-1)
                face_ok = torch.abs(a) > 1e-8

                safe_a = torch.where(face_ok, a, torch.ones_like(a))
                f = 1.0 / safe_a

                # s = O - V0    (B, F, 3)
                s = O_world.unsqueeze(0).unsqueeze(0) - V0.unsqueeze(0)
                s = s.expand(B, F, 3)

                # u = f * (s · P)
                u_bary = f * torch.sum(s * P, dim=-1)  # (B, F)
                face_ok = face_ok & (u_bary >= 0.0) & (u_bary <= 1.0)

                # Q = s × E1
                Q = torch.linalg.cross(s, E1.unsqueeze(0).expand(B, F, 3))

                # v = f * (D · Q)
                v_bary = f * torch.sum(D_batch.unsqueeze(1) * Q, dim=-1)
                face_ok = face_ok & (v_bary >= 0.0) & (u_bary + v_bary <= 1.0)

                # ★ t = f * (E2 · Q)  ← KHOẢNG CÁCH CHÍNH XÁC
                # Vì ||D_world|| = 1, nên t = ||hit_point - O_world||
                t = f * torch.sum(E2.unsqueeze(0) * Q, dim=-1)  # (B, F)
                face_ok = face_ok & (t > 1e-4)  # Chỉ lấy phía trước tia

                # Umbrella filter
                face_ok = face_ok & (~is_umbrella.unsqueeze(0))

                t = torch.where(face_ok, t, torch.tensor(float('inf'), device=device))

                # Min t per ray — mặt gần nhất (first hit)
                min_t, _ = torch.min(t, dim=-1)  # (B,)

                # Cập nhật best_t
                update = min_t < best_t[r_start:r_end]
                best_t[r_start:r_end] = torch.where(update, min_t, best_t[r_start:r_end])

            # ===== BƯỚC 4: SDF = mean(t) =====
            # t CHÍNH LÀ khoảng cách Euclid (không cần face_centers)
            valid = best_t < float('inf')
            if valid.sum() > 0:
                sdf_values[idx] = best_t[valid].mean()

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    avg = sdf_values.mean().item()

    print(f"\n{'='*60}")
    print(f"  THỜI GIAN: {elapsed:.4f} giây")
    print(f"  SDF Trung bình: {avg:.6f}")
    print(f"{'='*60}\n")

    return sdf_values.cpu().numpy()

# MAIN
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="True Hybrid: Camera Rays + World-Space MT (GPU)")
    parser.add_argument("--input_file", type=str, default="data/radio_0026.off")
    parser.add_argument("--fov", type=int, default=120,
                        help="Góc FOV hình nón (MeshLab default: 120°)")
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

    sdf = compute_hybrid_sdf_world_gpu(
        mesh,
        fov_deg=args.fov,
        num_rings=args.num_rings,
        num_rays_per_ring=args.rays_per_ring,
        vertex_batch_size=args.vertex_batch,
        ray_batch_size=args.ray_batch
    )

    pv_mesh = pv.wrap(mesh)
    pv_mesh.point_data['Hybrid_World_SDF'] = sdf

    plotter = pv.Plotter(title="True Hybrid SDF (Camera + World-Space MT GPU)")
    plotter.add_mesh(
        pv_mesh,
        scalars='Hybrid_World_SDF',
        cmap='jet_r',
        smooth_shading=False,
        show_edges=False,
        scalar_bar_args={'title': "SDF (Hybrid World MT)"}
    )
    plotter.set_background('white')
    plotter.add_axes()
    plotter.show()


if __name__ == "__main__":
    main()
