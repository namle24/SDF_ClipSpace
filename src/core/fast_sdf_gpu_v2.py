import torch
import numpy as np
import pyvista as pv
import trimesh
import time
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# GIỮ NGUYÊN HOÀN TOÀN từ code gốc
# ─────────────────────────────────────────────────────────────────────────────

def look_at_torch_batched(eyes, targets, device='cuda'):
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
# V3: Giữ cấu trúc gốc (B, R, F), chỉ fix phần bottleneck repeat_interleave
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_sdf_gpu_v3(mesh, fov_deg=90, num_rings=5,
                       num_rays_per_ring=10, batch_size=32):
    """
    Giữ NGUYÊN cấu trúc code gốc hoạt động tốt.
    Chỉ thay đúng 1 chỗ bottleneck: repeat_interleave → gather trực tiếp.

    Thay đổi duy nhất:
        TRƯỚC: W0_hit = W0.repeat_interleave(R, dim=0).view(B*R, F)[...]
        SAU:   W0_hit = W0[b_idx, hit_idx]   (advanced indexing, O(1) memory)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device     : {device}")
    print(f"  Batch size : {batch_size}")

    start_time = time.perf_counter()

    # ── Chuẩn bị dữ liệu — GIỮ NGUYÊN ────────────────────────────────────
    vertices     = torch.tensor(mesh.vertices,     dtype=torch.float32, device=device)
    faces_t      = torch.tensor(mesh.faces,        dtype=torch.long,    device=device)
    normals      = torch.tensor(mesh.face_normals, dtype=torch.float32, device=device)

    F_count = len(faces_t)
    V_count = len(vertices)

    V0_world     = vertices[faces_t[:, 0]]
    V1_world     = vertices[faces_t[:, 1]]
    V2_world     = vertices[faces_t[:, 2]]
    face_centers = (V0_world + V1_world + V2_world) / 3.0

    V_homo_T = torch.cat(
        [vertices, torch.ones(V_count, 1, device=device)], dim=1
    ).T  # (4, V)

    P_mat = perspective_torch(fov_deg, device=device)

    px_rays, py_rays = generate_rays_ndc(fov_deg, num_rings, num_rays_per_ring, device)
    R = len(px_rays)

    # px/py → shape (1, R, 1) để broadcast với (B, R, F)
    px_batch = px_rays.view(1, R, 1)
    py_batch = py_rays.view(1, R, 1)

    sdf_values = torch.zeros(F_count, dtype=torch.float32, device=device)

    # ── Vòng lặp chính — GIỮ NGUYÊN cấu trúc gốc ─────────────────────────
    for b_start in tqdm(range(0, F_count, batch_size), desc="SDF GPU v3"):
        b_end = min(b_start + batch_size, F_count)
        B     = b_end - b_start

        # ── Clip Space pipeline — GIỮ NGUYÊN ─────────────────────────────
        inward  = -normals[b_start:b_end]
        inward  = inward / (torch.norm(inward, dim=-1, keepdim=True) + 1e-12)
        eyes    = face_centers[b_start:b_end] + 0.0001 * inward
        targets = eyes + inward

        V_mat  = look_at_torch_batched(eyes, targets, device)
        PV_mat = P_mat @ V_mat

        V_clip = torch.einsum('bij,jv->bvi', PV_mat, V_homo_T)  # (B, V, 4)

        W      = V_clip[..., 3]                                  # (B, V)
        W_safe = W.clamp(min=1e-8)
        NDC    = V_clip[..., :3] / W_safe.unsqueeze(-1)          # (B, V, 3)

        V0_ndc = NDC[:, faces_t[:, 0], :]   # (B, F, 3)
        V1_ndc = NDC[:, faces_t[:, 1], :]
        V2_ndc = NDC[:, faces_t[:, 2], :]
        W0     = W[:, faces_t[:, 0]]         # (B, F)
        W1     = W[:, faces_t[:, 1]]
        W2     = W[:, faces_t[:, 2]]

        # W-clipping — GIỮ NGUYÊN
        valid_w = (W0 > 0.01) & (W1 > 0.01) & (W2 > 0.01)

        # Bbox 2D — GIỮ NGUYÊN
        all_x = torch.stack([V0_ndc[..., 0], V1_ndc[..., 0], V2_ndc[..., 0]], dim=-1)
        all_y = torch.stack([V0_ndc[..., 1], V1_ndc[..., 1], V2_ndc[..., 1]], dim=-1)
        min_x = all_x.min(dim=-1).values    # (B, F)
        max_x = all_x.max(dim=-1).values
        min_y = all_y.min(dim=-1).values
        max_y = all_y.max(dim=-1).values

        in_bbox = (
            (px_batch >= min_x.unsqueeze(1)) & (px_batch <= max_x.unsqueeze(1)) &
            (py_batch >= min_y.unsqueeze(1)) & (py_batch <= max_y.unsqueeze(1))
        ) & valid_w.unsqueeze(1)  # (B, R, F)

        # Barycentric — GIỮ NGUYÊN
        V0_2d = V0_ndc[..., :2].unsqueeze(1)   # (B, 1, F, 2)
        V1_2d = V1_ndc[..., :2].unsqueeze(1)
        V2_2d = V2_ndc[..., :2].unsqueeze(1)
        P_2d  = torch.stack([px_batch, py_batch], dim=-1).expand(B, R, F_count, 2)

        def cross2d(a, b):
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

        area       = cross2d(V1_ndc[..., :2] - V0_ndc[..., :2],
                             V2_ndc[..., :2] - V1_ndc[..., :2])  # (B, F)
        valid_area = (torch.abs(area) > 1e-6).unsqueeze(1)        # (B, 1, F)

        w0 = cross2d(V2_2d - V1_2d, P_2d - V1_2d)   # (B, R, F)
        w1 = cross2d(V0_2d - V2_2d, P_2d - V2_2d)
        w2 = cross2d(V1_2d - V0_2d, P_2d - V0_2d)

        in_tri = in_bbox & valid_area & (
            ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) |
            ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
        )  # (B, R, F)

        area_br = area.unsqueeze(1).expand(B, R, F_count)
        u = torch.where(in_tri, w0 / area_br, torch.zeros_like(w0))  # (B, R, F)
        v = torch.where(in_tri, w1 / area_br, torch.zeros_like(w1))
        w_bary = 1.0 - u - v

        # Z-buffer — GIỮ NGUYÊN
        Z_hit = (u * V0_ndc[..., 2].unsqueeze(1) +
                 v * V1_ndc[..., 2].unsqueeze(1) +
                 w_bary * V2_ndc[..., 2].unsqueeze(1))   # (B, R, F)

        Z_inf = torch.where(in_tri, Z_hit,
                            torch.full_like(Z_hit, float('inf')))
        best_z, hit_idx = torch.min(Z_inf, dim=2)   # (B, R)
        valid_ray = best_z < float('inf')            # (B, R)

        # ── FIX DUY NHẤT: Thay repeat_interleave → gather trực tiếp ──────
        #
        # Code gốc (chậm):
        #   W0_hit = W0.repeat_interleave(R, dim=0).view(B*R, F)[f_idx, h_idx_flat]
        #   → Nhân bản W0 lên R=150 lần trong memory → B*R*F elements
        #
        # V3 (nhanh):
        #   b_idx[i,j] = i  →  chọn camera thứ i
        #   hit_idx[i,j]    →  chọn face bị hit bởi ray j của camera i
        #   → Gather trực tiếp, không cần nhân bản
        #
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, R)  # (B, R)

        u_h   = u[b_idx,   torch.arange(R, device=device).unsqueeze(0).expand(B, R), hit_idx]
        v_h   = v[b_idx,   torch.arange(R, device=device).unsqueeze(0).expand(B, R), hit_idx]
        # Gọn hơn với gather:
        u_h   = u.gather(2, hit_idx.unsqueeze(2)).squeeze(2)       # (B, R)
        v_h   = v.gather(2, hit_idx.unsqueeze(2)).squeeze(2)
        w_h   = 1.0 - u_h - v_h

        W0_h  = W0.gather(1, hit_idx.view(B, R))   # (B, R)  ← KHÔNG cần repeat
        W1_h  = W1.gather(1, hit_idx.view(B, R))
        W2_h  = W2.gather(1, hit_idx.view(B, R))

        # ── Perspective-correct interpolation — GIỮ NGUYÊN ────────────────
        inv_W    = u_h / W0_h + v_h / W1_h + w_h / W2_h   # (B, R)
        W_interp = 1.0 / (inv_W + 1e-12)

        # Gather world-space vertices tại hit face — dùng gather thay expand
        hit_idx_3d = hit_idx.unsqueeze(2).expand(B, R, 3)   # (B, R, 3)
        V0_h = V0_world[hit_idx.view(-1)].view(B, R, 3)     # (B, R, 3)
        V1_h = V1_world[hit_idx.view(-1)].view(B, R, 3)
        V2_h = V2_world[hit_idx.view(-1)].view(B, R, 3)

        hit_point = W_interp.unsqueeze(-1) * (
            (u_h / W0_h).unsqueeze(-1) * V0_h +
            (v_h / W1_h).unsqueeze(-1) * V1_h +
            (w_h / W2_h).unsqueeze(-1) * V2_h
        )  # (B, R, 3)

        # Euclidean distance — GIỮ NGUYÊN
        fc_b  = face_centers[b_start:b_end].unsqueeze(1)    # (B, 1, 3)
        dists = torch.norm(hit_point - fc_b, dim=-1)         # (B, R)

        valid_f  = valid_ray.float()
        dist_sum = (dists * valid_f).sum(dim=1)              # (B,)
        hit_cnt  = valid_f.sum(dim=1)                        # (B,)

        sdf_values[b_start:b_end] = dist_sum / (hit_cnt + 1e-8)

    elapsed = time.perf_counter() - start_time
    print(f"\n✅ Hoàn thành trong {elapsed:.4f}s")
    return sdf_values.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SDF GPU Rasterizer v3")
    parser.add_argument("--input_file",  type=str, default="data/bunny.obj")
    parser.add_argument("--fov",         type=int, default=120)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_rings",   type=int, default=15)
    parser.add_argument("--rays_per_ring", type=int, default=30)
    args = parser.parse_args()

    try:
        mesh = trimesh.load(args.input_file, force='mesh')
        mesh.fix_normals()
        mesh.process()
        print(f"Mesh: {len(mesh.faces)} faces | {len(mesh.vertices)} vertices")
    except Exception as e:
        print(f"Error: {e}")
        return

    sdf = compute_sdf_gpu_v3(
        mesh,
        fov_deg           = args.fov,
        num_rings         = args.num_rings,
        num_rays_per_ring = args.rays_per_ring,
        batch_size        = args.batch_size,
    )

    pv_mesh = pv.wrap(mesh)
    pv_mesh.cell_data['SDF_v3'] = sdf
    plotter = pv.Plotter(title="SDF GPU Rasterizer v3")
    plotter.add_mesh(pv_mesh, scalars='SDF_v3', cmap='jet_r', show_edges=True)
    plotter.set_background('white')
    plotter.show()


if __name__ == "__main__":
    main()