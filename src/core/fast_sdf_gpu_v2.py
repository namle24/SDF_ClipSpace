import torch
import numpy as np
import trimesh
import time
from tqdm import tqdm

def look_at_torch_batched(eyes, targets, device='cuda'):
    # GIỮ NGUYÊN - đây là ý tưởng lõi
    B = eyes.shape[0]
    forward = targets - eyes
    forward = forward / (torch.norm(forward, dim=-1, keepdim=True) + 1e-12)

    world_up = torch.tensor([0.0, 1.0, 0.0], device=device).expand(B, 3).clone()
    mask = (torch.abs(forward[:, 1]) > 0.99)
    world_up[mask] = torch.tensor([0.0, 0.0, 1.0], device=device)

    right = torch.linalg.cross(forward, world_up)
    right = right / (torch.norm(right, dim=-1, keepdim=True) + 1e-12)
    true_up = torch.linalg.cross(right, forward)

    rotation = torch.zeros((B, 4, 4), dtype=torch.float32, device=device)
    rotation[:, 0, :3] = right
    rotation[:, 1, :3] = true_up
    rotation[:, 2, :3] = -forward
    rotation[:, 3, 3] = 1.0

    translation = torch.eye(4, dtype=torch.float32, device=device).expand(B, 4, 4).clone()
    translation[:, 0, 3] = -eyes[:, 0]
    translation[:, 1, 3] = -eyes[:, 1]
    translation[:, 2, 3] = -eyes[:, 2]

    return rotation @ translation

def perspective_torch(fov_deg, near=0.001, far=5.0, device='cuda'):
    # GIỮ NGUYÊN - clip space projection matrix
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
    # GIỮ NGUYÊN - ray generation trong NDC space
    tan_fov = np.tan(np.radians(fov_deg) / 2.0)
    cot = 1.0 / tan_fov
    all_px, all_py = [], []
    for ring in range(num_rings):
        r = (ring + 0.5) / num_rings
        num_points = int(num_rays_per_ring * (ring + 1))
        angles = torch.linspace(0, 2 * torch.pi, num_points + 1, device=device)[:-1]
        x = r * torch.cos(angles) * tan_fov
        y = r * torch.sin(angles) * tan_fov
        all_px.append(cot * x)
        all_py.append(cot * y)
    return torch.cat(all_px), torch.cat(all_py)

@torch.no_grad()
def compute_sdf_gpu_optimized(mesh, fov_deg=90, num_rings=5, 
                               num_rays_per_ring=10, face_batch_size=512):
    """
    Giữ nguyên pipeline clip space → NDC → perspective-correct interpolation
    Optimize: loại bỏ Python loop, tăng parallelism trên GPU
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Face batch: {face_batch_size}")

    start_time = time.perf_counter()

    # ── Chuẩn bị dữ liệu ──────────────────────────────────────────────────
    vertices  = torch.tensor(mesh.vertices,      dtype=torch.float32, device=device)
    faces     = torch.tensor(mesh.faces,         dtype=torch.long,    device=device)
    normals   = torch.tensor(mesh.face_normals,  dtype=torch.float32, device=device)

    F_count = len(faces)
    V_count = len(vertices)

    V0_world = vertices[faces[:, 0]]   # (F, 3)
    V1_world = vertices[faces[:, 1]]
    V2_world = vertices[faces[:, 2]]
    face_centers = (V0_world + V1_world + V2_world) / 3.0  # (F, 3)

    # Homogeneous vertices - GIỮ NGUYÊN ý tưởng
    V_homo_T = torch.cat(
        [vertices, torch.ones(V_count, 1, device=device)], dim=1
    ).T  # (4, V)

    # GIỮ NGUYÊN: Perspective matrix
    P_mat = perspective_torch(fov_deg, device=device)

    # GIỮ NGUYÊN: Ray generation trong NDC
    px_rays, py_rays = generate_rays_ndc(
        fov_deg, num_rings, num_rays_per_ring, device
    )
    R = len(px_rays)  # tổng số rays

    sdf_values = torch.zeros(F_count, dtype=torch.float32, device=device)

    # ── OPTIMIZATION CHÍNH: Ray-major loop thay vì Face-major loop ────────
    # Thay vì: loop qua F faces (batch camera)
    # Mới:     loop qua R rays → mỗi ray check TẤT CẢ F faces cùng lúc
    #
    # Memory analysis cho 3060 12GB:
    # Tensor NDC:    face_batch × V × 3 = 512 × V × 3 × 4B → nhẹ
    # Tensor hits:   face_batch × F     = 512 × 10172 × 4B ≈ 20MB
    #
    # Với ray-major + face_batch=512:
    # → Mỗi iteration: 512 cameras × 1 ray × F faces
    # → Tổng iterations: R × ceil(F/512) thay vì ceil(F/32)
    # → Nhưng mỗi iteration nặng hơn nhiều → net speedup ~10-20x

    # Precompute: PV matrices cho TẤT CẢ F faces cùng lúc theo batch
    # GIỮ NGUYÊN: look_at + perspective projection

    for b_start in range(0, F_count, face_batch_size):
        b_end = min(b_start + face_batch_size, F_count)
        B = b_end - b_start

        # GIỮ NGUYÊN: inward direction từ face normal
        inward = -normals[b_start:b_end]
        inward = inward / (torch.norm(inward, dim=-1, keepdim=True) + 1e-12)
        eyes    = face_centers[b_start:b_end] + 0.0001 * inward
        targets = eyes + inward

        # GIỮ NGUYÊN: Look-at matrix
        V_mat  = look_at_torch_batched(eyes, targets, device)  # (B, 4, 4)
        PV_mat = P_mat @ V_mat  # (B, 4, 4)

        # GIỮ NGUYÊN: Transform vertices → Clip Space
        # einsum: (B,4,4) @ (4,V) → (B,V,4)
        V_clip = torch.einsum('bij,jv->bvi', PV_mat, V_homo_T)

        # GIỮ NGUYÊN: W-clipping + Perspective divide → NDC
        W     = V_clip[..., 3]                                    # (B, V)
        W_safe = W.clamp(min=1e-8)
        NDC   = V_clip[..., :3] / W_safe.unsqueeze(-1)           # (B, V, 3)

        # GIỮ NGUYÊN: Lấy tọa độ tam giác trong NDC
        V0_ndc = NDC[:, faces[:, 0], :]  # (B, F, 3)
        V1_ndc = NDC[:, faces[:, 1], :]
        V2_ndc = NDC[:, faces[:, 2], :]
        W0     = W[:, faces[:, 0]]       # (B, F)
        W1     = W[:, faces[:, 1]]
        W2     = W[:, faces[:, 2]]

        # GIỮ NGUYÊN: W-clipping mask
        valid_w = (W0 > 0.01) & (W1 > 0.01) & (W2 > 0.01)  # (B, F)

        # OPTIMIZATION: Precompute bbox một lần cho tất cả rays
        min_x = torch.min(torch.stack(
            [V0_ndc[...,0], V1_ndc[...,0], V2_ndc[...,0]], dim=-1), dim=-1).values
        max_x = torch.max(torch.stack(
            [V0_ndc[...,0], V1_ndc[...,0], V2_ndc[...,0]], dim=-1), dim=-1).values
        min_y = torch.min(torch.stack(
            [V0_ndc[...,1], V1_ndc[...,1], V2_ndc[...,1]], dim=-1), dim=-1).values
        max_y = torch.max(torch.stack(
            [V0_ndc[...,1], V1_ndc[...,1], V2_ndc[...,1]], dim=-1), dim=-1).values
        # Tất cả shape (B, F)

        # OPTIMIZATION: Loop qua R rays — nhưng tất cả B cameras song song
        # Đây là điểm mấu chốt: B cameras × 1 ray × F triangles
        # Thay vì: B cameras × R rays × F triangles (OOM)

        dist_sum   = torch.zeros(B, device=device)
        hit_count  = torch.zeros(B, device=device)

        def cross2d(a, b):
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

        # Precompute triangle area một lần
        area = cross2d(
            V1_ndc[..., :2] - V0_ndc[..., :2],
            V2_ndc[..., :2] - V1_ndc[..., :2]
        )  # (B, F)
        valid_area = torch.abs(area) > 1e-6  # (B, F)

        for r_idx in range(R):
            # GIỮ NGUYÊN: Ray trong NDC space
            px = px_rays[r_idx]  # scalar
            py = py_rays[r_idx]  # scalar

            # Bbox filter: (B, F)
            in_bbox = (
                (px >= min_x) & (px <= max_x) &
                (py >= min_y) & (py <= max_y)
            ) & valid_w & valid_area  # (B, F)

            # GIỮ NGUYÊN: Barycentric coords trong NDC
            # P_2d shape: (B, F, 2) via broadcast
            P = torch.stack([
                px.expand(B, len(faces)),
                py.expand(B, len(faces))
            ], dim=-1)  # (B, F, 2)

            w0 = cross2d(V2_ndc[..., :2] - V1_ndc[..., :2],
                         P - V1_ndc[..., :2])   # (B, F)
            w1 = cross2d(V0_ndc[..., :2] - V2_ndc[..., :2],
                         P - V2_ndc[..., :2])
            w2 = cross2d(V1_ndc[..., :2] - V0_ndc[..., :2],
                         P - V0_ndc[..., :2])

            in_tri = in_bbox & (
                ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) |
                ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
            )  # (B, F)

            # Barycentric coords
            u = torch.where(in_tri, w0 / area, torch.zeros_like(w0))
            v = torch.where(in_tri, w1 / area, torch.zeros_like(w1))
            w = 1.0 - u - v

            # Z-buffer: tìm hit gần nhất
            Z_hit = (u * V0_ndc[..., 2] +
                     v * V1_ndc[..., 2] +
                     w * V2_ndc[..., 2])   # (B, F)

            Z_inf = torch.where(in_tri, Z_hit,
                                torch.full_like(Z_hit, float('inf')))
            best_z, hit_idx = torch.min(Z_inf, dim=1)  # (B,)
            valid_ray = best_z < float('inf')            # (B,)

            # GIỮ NGUYÊN: Perspective-correct interpolation
            u_h = u[torch.arange(B), hit_idx]   # (B,)
            v_h = v[torch.arange(B), hit_idx]
            w_h = 1.0 - u_h - v_h

            W0_h = W0[torch.arange(B), hit_idx]  # (B,)
            W1_h = W1[torch.arange(B), hit_idx]
            W2_h = W2[torch.arange(B), hit_idx]

            # GIỮ NGUYÊN: 1/W interpolation formula
            inv_W = u_h / W0_h + v_h / W1_h + w_h / W2_h
            W_interp = 1.0 / (inv_W + 1e-12)

            V0_h = V0_world[hit_idx]   # (B, 3)
            V1_h = V1_world[hit_idx]
            V2_h = V2_world[hit_idx]

            hit_point = W_interp.unsqueeze(-1) * (
                (u_h / W0_h).unsqueeze(-1) * V0_h +
                (v_h / W1_h).unsqueeze(-1) * V1_h +
                (w_h / W2_h).unsqueeze(-1) * V2_h
            )  # (B, 3)

            dists = torch.norm(
                hit_point - face_centers[b_start:b_end], dim=-1
            )  # (B,)

            dist_sum  += torch.where(valid_ray, dists,
                                     torch.zeros_like(dists))
            hit_count += valid_ray.float()

        sdf_values[b_start:b_end] = dist_sum / (hit_count + 1e-8)

    elapsed = time.perf_counter() - start_time
    print(f"Hoàn thành trong {elapsed:.4f}s")
    return sdf_values.cpu().numpy()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ultimate SDF Rasterizer - Vectorized GPU Engine")
    parser.add_argument("--input_file", type=str, default="data/bunny1.obj")
    parser.add_argument("--fov", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    try:
        mesh = trimesh.load(args.input_file, force='mesh')
        mesh.fix_normals()
        mesh.process()
    except Exception as e:
        print(f"Error: {e}")
        return

    sdf = compute_fast_sdf_gpu(mesh, fov_deg=args.fov, batch_size=args.batch_size)

    pv_mesh = pv.wrap(mesh)
    pv_mesh.cell_data['SDF_Ultimate_Raster'] = sdf
    plotter = pv.Plotter(title="Ultimate GPU SDF Rasterizer")
    plotter.add_mesh(pv_mesh, scalars='SDF_Ultimate_Raster', cmap='jet_r', show_edges=True)
    plotter.set_background('white')
    plotter.show()

if __name__ == "__main__":
    main()
