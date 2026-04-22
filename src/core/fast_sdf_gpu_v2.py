import torch
import numpy as np
import pyvista as pv
import trimesh
import time
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# CÁC HÀM LÕI — GIỮ NGUYÊN Ý TƯỞNG CLIP SPACE
# ─────────────────────────────────────────────────────────────────────────────

def look_at_torch_batched(eyes, targets, device='cuda'):
    """✅ GIỮ NGUYÊN — View matrix batched"""
    B = eyes.shape[0]
    forward = targets - eyes
    forward = forward / (torch.norm(forward, dim=-1, keepdim=True) + 1e-12)

    world_up = torch.tensor([0.0, 1.0, 0.0], device=device).expand(B, 3).clone()
    mask = (torch.abs(forward[:, 1]) > 0.99)
    world_up[mask] = torch.tensor([0.0, 0.0, 1.0], device=device)

    right   = torch.linalg.cross(forward, world_up)
    right   = right / (torch.norm(right, dim=-1, keepdim=True) + 1e-12)
    true_up = torch.linalg.cross(right, forward)

    rotation = torch.zeros((B, 4, 4), dtype=torch.float32, device=device)
    rotation[:, 0, :3] = right
    rotation[:, 1, :3] = true_up
    rotation[:, 2, :3] = -forward
    rotation[:, 3, 3]  = 1.0

    translation = torch.eye(4, dtype=torch.float32, device=device).expand(B, 4, 4).clone()
    translation[:, 0, 3] = -eyes[:, 0]
    translation[:, 1, 3] = -eyes[:, 1]
    translation[:, 2, 3] = -eyes[:, 2]

    return rotation @ translation


def perspective_torch(fov_deg, near=0.001, far=5.0, device='cuda'):
    """✅ GIỮ NGUYÊN — Perspective projection matrix → Clip Space"""
    fov_rad = np.radians(fov_deg)
    cot = 1.0 / np.tan(fov_rad / 2.0)
    P = torch.zeros((4, 4), dtype=torch.float32, device=device)
    P[0, 0] = cot
    P[1, 1] = cot
    P[2, 2] = -far / (far - near)
    P[2, 3] = -(near * far) / (far - near)
    P[3, 2] = -1.0
    return P


def generate_rays_ndc(fov_deg, num_rings=5, num_rays_per_ring=10, device='cuda'):
    """✅ GIỮ NGUYÊN — Ray sampling trong NDC space"""
    tan_fov = np.tan(np.radians(fov_deg) / 2.0)
    cot     = 1.0 / tan_fov
    all_px, all_py = [], []
    for ring in range(num_rings):
        r          = (ring + 0.5) / num_rings
        num_points = int(num_rays_per_ring * (ring + 1))
        angles     = torch.linspace(0, 2 * torch.pi, num_points + 1, device=device)[:-1]
        x = r * torch.cos(angles) * tan_fov
        y = r * torch.sin(angles) * tan_fov
        all_px.append(cot * x)
        all_py.append(cot * y)
    return torch.cat(all_px), torch.cat(all_py)


# ─────────────────────────────────────────────────────────────────────────────
# HÀM CHÍNH — ĐÃ FIX VÀ OPTIMIZE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_sdf_gpu_optimized(mesh, fov_deg=90, num_rings=5,
                               num_rays_per_ring=10, face_batch_size=512):
    """
    Pipeline lõi được GIỮ NGUYÊN:
      World Space → Clip Space (perspective matrix)
      → NDC (perspective divide, W-clipping)
      → Barycentric test (2D cross product)
      → Z-buffer (tìm opposite surface)
      → Perspective-correct interpolation (1/W formula)
      → Euclidean distance = SDF value

    Optimization so với v1:
      - face_batch_size tăng từ 32 → 512 (GPU utilization cao hơn)
      - Precompute bbox TRƯỚC vòng lặp ray (tránh recompute R lần)
      - Vectorize R rays: stack tất cả rays thành tensor (R, B, F)
        thay vì loop R lần trong Python
      - Fix: px.expand() → torch.full() cho scalar tensor
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device       : {device}")
    print(f"  Face batch   : {face_batch_size}")
    print(f"  Num rings    : {num_rings}  |  Rays/ring : {num_rays_per_ring}")

    start_time = time.perf_counter()

    # ── Chuẩn bị dữ liệu ──────────────────────────────────────────────────
    vertices     = torch.tensor(mesh.vertices,     dtype=torch.float32, device=device)
    faces_t      = torch.tensor(mesh.faces,        dtype=torch.long,    device=device)
    normals      = torch.tensor(mesh.face_normals, dtype=torch.float32, device=device)

    F_count = len(faces_t)
    V_count = len(vertices)

    V0_world     = vertices[faces_t[:, 0]]                           # (F, 3)
    V1_world     = vertices[faces_t[:, 1]]
    V2_world     = vertices[faces_t[:, 2]]
    face_centers = (V0_world + V1_world + V2_world) / 3.0           # (F, 3)

    # ✅ GIỮ NGUYÊN: Homogeneous coordinates (4, V)
    V_homo_T = torch.cat(
        [vertices, torch.ones(V_count, 1, device=device)], dim=1
    ).T  # (4, V)

    # ✅ GIỮ NGUYÊN: Perspective matrix
    P_mat = perspective_torch(fov_deg, device=device)

    # ✅ GIỮ NGUYÊN: Ray generation trong NDC space
    px_rays, py_rays = generate_rays_ndc(fov_deg, num_rings, num_rays_per_ring, device)
    R = len(px_rays)
    print(f"  Total rays   : {R}")
    print(f"  Total faces  : {F_count}")

    sdf_values = torch.zeros(F_count, dtype=torch.float32, device=device)

    # ── Helper ────────────────────────────────────────────────────────────
    def cross2d(a, b):
        """2D cross product: a[...,0]*b[...,1] - a[...,1]*b[...,0]"""
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    # ── Vòng lặp chính theo face batch ────────────────────────────────────
    num_batches = (F_count + face_batch_size - 1) // face_batch_size

    for b_start in tqdm(range(0, F_count, face_batch_size), desc="SDF batches"):
        b_end = min(b_start + face_batch_size, F_count)
        B     = b_end - b_start   # số cameras trong batch này

        # ── BƯỚC 1: Tính View + Projection matrix cho B cameras ──────────
        # ✅ GIỮ NGUYÊN: inward direction từ face normal
        inward  = -normals[b_start:b_end]
        inward  = inward / (torch.norm(inward, dim=-1, keepdim=True) + 1e-12)
        eyes    = face_centers[b_start:b_end] + 0.0001 * inward
        targets = eyes + inward

        # ✅ GIỮ NGUYÊN: Look-at → View matrix
        V_mat  = look_at_torch_batched(eyes, targets, device)  # (B, 4, 4)
        PV_mat = P_mat @ V_mat                                 # (B, 4, 4)

        # ── BƯỚC 2: Transform tất cả vertices → Clip Space ───────────────
        # ✅ GIỮ NGUYÊN: einsum (B,4,4) @ (4,V) → (B,V,4)
        V_clip = torch.einsum('bij,jv->bvi', PV_mat, V_homo_T)  # (B, V, 4)

        # ✅ GIỮ NGUYÊN: W-clipping + Perspective divide → NDC
        W      = V_clip[..., 3]                                 # (B, V)
        W_safe = W.clamp(min=1e-8)
        NDC    = V_clip[..., :3] / W_safe.unsqueeze(-1)         # (B, V, 3)

        # ✅ GIỮ NGUYÊN: Tọa độ 3 đỉnh tam giác trong NDC
        V0_ndc = NDC[:, faces_t[:, 0], :]   # (B, F, 3)
        V1_ndc = NDC[:, faces_t[:, 1], :]
        V2_ndc = NDC[:, faces_t[:, 2], :]
        W0     = W[:, faces_t[:, 0]]         # (B, F)
        W1     = W[:, faces_t[:, 1]]
        W2     = W[:, faces_t[:, 2]]

        # ✅ GIỮ NGUYÊN: W-clipping mask (loại phantom faces)
        valid_w = (W0 > 0.01) & (W1 > 0.01) & (W2 > 0.01)      # (B, F)

        # ── BƯỚC 3: Precompute bbox VÀ area cho TẤT CẢ rays ─────────────
        # KEY OPTIMIZATION: Tính 1 lần, dùng cho tất cả R rays
        all_x  = torch.stack([V0_ndc[..., 0], V1_ndc[..., 0], V2_ndc[..., 0]], dim=-1)
        all_y  = torch.stack([V0_ndc[..., 1], V1_ndc[..., 1], V2_ndc[..., 1]], dim=-1)
        min_x  = all_x.min(dim=-1).values   # (B, F)
        max_x  = all_x.max(dim=-1).values
        min_y  = all_y.min(dim=-1).values
        max_y  = all_y.max(dim=-1).values

        # ✅ GIỮ NGUYÊN: Triangle area trong NDC (cho barycentric)
        area       = cross2d(
            V1_ndc[..., :2] - V0_ndc[..., :2],
            V2_ndc[..., :2] - V1_ndc[..., :2]
        )  # (B, F)
        valid_area = torch.abs(area) > 1e-6  # (B, F)

        # Combine static masks (không đổi theo ray)
        static_mask = valid_w & valid_area   # (B, F)

        # ── BƯỚC 4: Vectorize TẤT CẢ R rays ─────────────────────────────
        # FIX CHÍNH: Thay vì loop R lần trong Python,
        # stack rays thành tensor rồi xử lý song song

        # px_rays: (R,) → broadcast thành (R, B, F) để check bbox
        # min_x:   (B, F) → unsqueeze(0) → (1, B, F)

        px_r = px_rays.view(R, 1, 1)   # (R, 1, 1)
        py_r = py_rays.view(R, 1, 1)   # (R, 1, 1)

        # Bbox filter cho tất cả rays cùng lúc: (R, B, F)
        in_bbox = (
            (px_r >= min_x.unsqueeze(0)) & (px_r <= max_x.unsqueeze(0)) &
            (py_r >= min_y.unsqueeze(0)) & (py_r <= max_y.unsqueeze(0))
        ) & static_mask.unsqueeze(0)   # (R, B, F)

        # ✅ GIỮ NGUYÊN: Barycentric coords trong NDC
        # P_2d: (R, B, F, 2)
        P_2d = torch.stack([
            px_r.expand(R, B, F_count),
            py_r.expand(R, B, F_count)
        ], dim=-1)   # (R, B, F, 2)

        # V0,V1,V2 NDC 2D: (B, F, 2) → unsqueeze(0) → (1, B, F, 2)
        V0_2d = V0_ndc[..., :2].unsqueeze(0)   # (1, B, F, 2)
        V1_2d = V1_ndc[..., :2].unsqueeze(0)
        V2_2d = V2_ndc[..., :2].unsqueeze(0)

        # ✅ GIỮ NGUYÊN: 2D cross product cho barycentric
        w0 = cross2d(V2_2d - V1_2d, P_2d - V1_2d)   # (R, B, F)
        w1 = cross2d(V0_2d - V2_2d, P_2d - V2_2d)
        w2 = cross2d(V1_2d - V0_2d, P_2d - V0_2d)

        in_tri = in_bbox & (
            ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) |
            ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
        )  # (R, B, F)

        # ✅ GIỮ NGUYÊN: Tính u, v, w (barycentric)
        area_r = area.unsqueeze(0).expand(R, B, F_count)   # (R, B, F)
        u = torch.where(in_tri, w0 / area_r, torch.zeros_like(w0))
        v = torch.where(in_tri, w1 / area_r, torch.zeros_like(w1))
        w = 1.0 - u - v

        # ✅ GIỮ NGUYÊN: Z-buffer trong NDC
        Z0 = V0_ndc[..., 2].unsqueeze(0)   # (1, B, F)
        Z1 = V1_ndc[..., 2].unsqueeze(0)
        Z2 = V2_ndc[..., 2].unsqueeze(0)

        Z_hit = u * Z0 + v * Z1 + w * Z2   # (R, B, F)
        Z_inf = torch.where(in_tri, Z_hit,
                            torch.full_like(Z_hit, float('inf')))

        # ✅ GIỮ NGUYÊN: Tìm face gần nhất (Z nhỏ nhất) cho mỗi (ray, camera)
        best_z, hit_idx = torch.min(Z_inf, dim=2)   # (R, B)
        valid_ray = best_z < float('inf')            # (R, B)

        # ── BƯỚC 5: Perspective-correct interpolation ─────────────────────
        # ✅ GIỮ NGUYÊN: 1/W interpolation formula của GPU rasterizer
        # Gather barycentric coords tại hit triangle
        rb_idx = torch.arange(R, device=device).unsqueeze(1).expand(R, B)  # (R, B)
        b_idx  = torch.arange(B, device=device).unsqueeze(0).expand(R, B)  # (R, B)

        u_h = u[rb_idx, b_idx, hit_idx]   # (R, B)
        v_h = v[rb_idx, b_idx, hit_idx]
        w_h = 1.0 - u_h - v_h

        # Gather W của 3 đỉnh tại hit triangle
        W0_r = W0.unsqueeze(0).expand(R, B, F_count)   # (R, B, F)
        W1_r = W1.unsqueeze(0).expand(R, B, F_count)
        W2_r = W2.unsqueeze(0).expand(R, B, F_count)

        W0_h = W0_r[rb_idx, b_idx, hit_idx]   # (R, B)
        W1_h = W1_r[rb_idx, b_idx, hit_idx]
        W2_h = W2_r[rb_idx, b_idx, hit_idx]

        # ✅ GIỮ NGUYÊN: 1/W formula — perspective-correct
        inv_W   = u_h / W0_h + v_h / W1_h + w_h / W2_h   # (R, B)
        W_interp = 1.0 / (inv_W + 1e-12)                  # (R, B)

        # Gather world-space vertices tại hit triangle
        V0_r = V0_world.unsqueeze(0).unsqueeze(0).expand(R, B, F_count, 3)
        V1_r = V1_world.unsqueeze(0).unsqueeze(0).expand(R, B, F_count, 3)
        V2_r = V2_world.unsqueeze(0).unsqueeze(0).expand(R, B, F_count, 3)

        hit_idx_3d = hit_idx.unsqueeze(-1).expand(R, B, 3)   # (R, B, 3)
        V0_h = V0_r[rb_idx, b_idx, hit_idx]    # (R, B, 3)
        V1_h = V1_r[rb_idx, b_idx, hit_idx]
        V2_h = V2_r[rb_idx, b_idx, hit_idx]

        # ✅ GIỮ NGUYÊN: Reconstruct 3D hit point
        hit_point = W_interp.unsqueeze(-1) * (
            (u_h / W0_h).unsqueeze(-1) * V0_h +
            (v_h / W1_h).unsqueeze(-1) * V1_h +
            (w_h / W2_h).unsqueeze(-1) * V2_h
        )  # (R, B, 3)

        # ✅ GIỮ NGUYÊN: Euclidean distance = SDF value
        fc_batch = face_centers[b_start:b_end]              # (B, 3)
        dists    = torch.norm(
            hit_point - fc_batch.unsqueeze(0), dim=-1
        )  # (R, B)

        # Aggregate: trung bình các ray hợp lệ
        valid_f  = valid_ray.float()                        # (R, B)
        dist_sum = (dists * valid_f).sum(dim=0)             # (B,)
        hit_cnt  = valid_f.sum(dim=0)                       # (B,)

        sdf_values[b_start:b_end] = dist_sum / (hit_cnt + 1e-8)

    elapsed = time.perf_counter() - start_time
    print(f"\n✅ Hoàn thành trong {elapsed:.4f}s")
    return sdf_values.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SDF GPU Rasterizer v2 - Optimized")
    parser.add_argument("--input_file",      type=str,   default="data/bunny.obj")
    parser.add_argument("--fov",             type=int,   default=90)
    parser.add_argument("--face_batch_size", type=int,   default=512)
    parser.add_argument("--num_rings",       type=int,   default=5)
    parser.add_argument("--rays_per_ring",   type=int,   default=10)
    args = parser.parse_args()

    try:
        mesh = trimesh.load(args.input_file, force='mesh')
        mesh.fix_normals()
        mesh.process()
        print(f"Mesh loaded: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    sdf = compute_sdf_gpu_optimized(
        mesh,
        fov_deg        = args.fov,
        num_rings      = args.num_rings,
        num_rays_per_ring = args.rays_per_ring,
        face_batch_size = args.face_batch_size
    )

    # Visualize
    pv_mesh = pv.wrap(mesh)
    pv_mesh.cell_data['SDF_v2'] = sdf
    plotter = pv.Plotter(title="SDF GPU Rasterizer v2")
    plotter.add_mesh(pv_mesh, scalars='SDF_v2', cmap='jet_r', show_edges=True)
    plotter.set_background('white')
    plotter.show()


if __name__ == "__main__":
    main()