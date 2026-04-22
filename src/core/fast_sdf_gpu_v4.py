import torch
import numpy as np
import pyvista as pv
import trimesh
import time
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS — Clip-Space Pipeline (Preserved)
# ─────────────────────────────────────────────────────────────────────────────

def look_at_torch_batched(eyes, targets, device='cuda'):
    """Batched look-at view matrix construction."""
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
    """Perspective projection matrix → Clip Space."""
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
    """Generate ray sample points in NDC space using concentric rings."""
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
# V4: Memory-Optimized — Ray Chunking + Tile Culling + Soft Z-buffer
#
# Complexity: O(B × R_chunk × F) per iteration → eliminates OOM
# Key changes from v3:
#   1. Ray chunking:  inner loop over R_chunk rays (vs all R at once)
#   2. Scalar broadcast: cross2d via scalars, avoids (B,R,F,2) tensors
#   3. Tile culling:  16×16 NDC grid pre-filter replaces bbox
#   4. Soft Z-buffer: differentiable softmin (optional, alpha param)
#   5. Mixed precision: autocast for safe operations
#   6. Pre-allocated accumulators outside inner loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_sdf_gpu_v4(mesh, fov_deg=90, num_rings=5, num_rays_per_ring=10,
                       batch_size=32, ray_chunk_size=512, grid_size=16,
                       alpha=None):
    """
    Memory-optimized SDF via clip-space rasterization.

    Pipeline (preserved):
      World → Clip Space → NDC → Barycentric → Z-buffer
      → Perspective-correct 1/W interpolation → Euclidean distance

    Args:
        batch_size:      cameras (faces) per outer batch
        ray_chunk_size:  rays per inner chunk — controls peak GPU memory
        grid_size:       tile grid resolution for face culling (16 = 256 tiles)
        alpha:           soft Z-buffer temperature (None = hard min)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start_time = time.perf_counter()

    # ── [7] Data preparation — moved outside all loops ────────────────────
    vertices     = torch.tensor(mesh.vertices,     dtype=torch.float32, device=device)
    faces_t      = torch.tensor(mesh.faces,        dtype=torch.long,    device=device)
    normals      = torch.tensor(mesh.face_normals, dtype=torch.float32, device=device)

    F_total = len(faces_t)
    V_total = len(vertices)

    V0_world     = vertices[faces_t[:, 0]]                           # (F, 3)
    V1_world     = vertices[faces_t[:, 1]]
    V2_world     = vertices[faces_t[:, 2]]
    face_centers = (V0_world + V1_world + V2_world) / 3.0           # (F, 3)

    V_homo_T = torch.cat(
        [vertices, torch.ones(V_total, 1, device=device)], dim=1
    ).T  # (4, V)

    P_mat = perspective_torch(fov_deg, device=device)

    px_rays, py_rays = generate_rays_ndc(fov_deg, num_rings, num_rays_per_ring, device)
    R = len(px_rays)

    sdf_values = torch.zeros(F_total, dtype=torch.float32, device=device)

    num_cam_batches = (F_total + batch_size - 1) // batch_size
    num_ray_chunks  = (R + ray_chunk_size - 1) // ray_chunk_size

    # ── Print config ──────────────────────────────────────────────────────
    print(f"  Device         : {device}")
    print(f"  Faces          : {F_total}  |  Vertices : {V_total}")
    print(f"  Total rays     : {R}")
    print(f"  Batch size     : {batch_size}  (cameras/iter)")
    print(f"  Ray chunk      : {ray_chunk_size}  ({num_ray_chunks} chunks)")
    print(f"  Tile grid      : {grid_size}×{grid_size}  ({grid_size**2} tiles)")
    print(f"  Soft Z-buffer  : {'alpha=' + str(alpha) if alpha else 'OFF (hard min)'}")
    est_peak = batch_size * ray_chunk_size * F_total * 4 / 1e9
    print(f"  Est. peak/tensor: {est_peak:.2f} GB")

    # ── [7] Pre-allocate reusable index tensors ───────────────────────────
    gs = grid_size
    half_gs = gs * 0.5

    # ── Outer loop: camera batches ────────────────────────────────────────
    for b_start in tqdm(range(0, F_total, batch_size), desc="SDF v4"):
        b_end = min(b_start + batch_size, F_total)
        B     = b_end - b_start

        # ── Clip-space transform (preserved) ──────────────────────────────
        inward  = -normals[b_start:b_end]
        inward  = inward / (torch.norm(inward, dim=-1, keepdim=True) + 1e-12)
        eyes    = face_centers[b_start:b_end] + 0.0001 * inward
        targets = eyes + inward

        V_mat  = look_at_torch_batched(eyes, targets, device)
        PV_mat = P_mat @ V_mat                                     # (B, 4, 4)

        # [6] Mixed precision for the heavy einsum
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            V_clip = torch.einsum('bij,jv->bvi', PV_mat, V_homo_T).float()

        W      = V_clip[..., 3]                                    # (B, V)
        W_safe = W.clamp(min=1e-8)
        NDC    = V_clip[..., :3] / W_safe.unsqueeze(-1)            # (B, V, 3)

        # NDC triangle vertices
        V0_ndc = NDC[:, faces_t[:, 0], :]                          # (B, F, 3)
        V1_ndc = NDC[:, faces_t[:, 1], :]
        V2_ndc = NDC[:, faces_t[:, 2], :]
        W0     = W[:, faces_t[:, 0]]                               # (B, F)
        W1     = W[:, faces_t[:, 1]]
        W2     = W[:, faces_t[:, 2]]

        # W-clipping mask (preserved)
        valid_w = (W0 > 0.01) & (W1 > 0.01) & (W2 > 0.01)        # (B, F)

        # ── [3] Tile-based face culling: precompute face tile ranges ──────
        # Compute face bounding box in NDC (preserved logic)
        all_x  = torch.stack([V0_ndc[..., 0], V1_ndc[..., 0], V2_ndc[..., 0]], dim=-1)
        all_y  = torch.stack([V0_ndc[..., 1], V1_ndc[..., 1], V2_ndc[..., 1]], dim=-1)
        min_x  = all_x.min(dim=-1).values                         # (B, F)
        max_x  = all_x.max(dim=-1).values
        min_y  = all_y.min(dim=-1).values
        max_y  = all_y.max(dim=-1).values

        # Map face bbox to tile indices: NDC [-1,1] → tile [0, grid_size-1]
        ft_xmin = ((min_x + 1.0) * half_gs).long().clamp(0, gs - 1)  # (B, F)
        ft_xmax = ((max_x + 1.0) * half_gs).long().clamp(0, gs - 1)
        ft_ymin = ((min_y + 1.0) * half_gs).long().clamp(0, gs - 1)
        ft_ymax = ((max_y + 1.0) * half_gs).long().clamp(0, gs - 1)

        # Triangle area in NDC — computed ONCE per camera batch
        # [2] Scalar cross product avoids (B,F,2) intermediate
        e10_x = V1_ndc[..., 0] - V0_ndc[..., 0]                  # (B, F)
        e10_y = V1_ndc[..., 1] - V0_ndc[..., 1]
        e21_x = V2_ndc[..., 0] - V1_ndc[..., 0]
        e21_y = V2_ndc[..., 1] - V1_ndc[..., 1]
        area  = e10_x * e21_y - e10_y * e21_x                     # (B, F)
        valid_area = torch.abs(area) > 1e-6                        # (B, F)

        # Combined static mask (does not depend on rays)
        static_mask = valid_w & valid_area                         # (B, F)

        # ── Edge vectors for barycentric — precompute ONCE ────────────────
        # Edge V2→V1 (for w0)
        e_v2v1_x = V2_ndc[..., 0] - V1_ndc[..., 0]               # (B, F)
        e_v2v1_y = V2_ndc[..., 1] - V1_ndc[..., 1]
        # Edge V0→V2 (for w1)
        e_v0v2_x = V0_ndc[..., 0] - V2_ndc[..., 0]
        e_v0v2_y = V0_ndc[..., 1] - V2_ndc[..., 1]
        # Edge V1→V0 (for w2)
        e_v1v0_x = V1_ndc[..., 0] - V0_ndc[..., 0]
        e_v1v0_y = V1_ndc[..., 1] - V0_ndc[..., 1]

        # NDC Z per vertex (for Z-buffer interpolation)
        Z0 = V0_ndc[..., 2]                                       # (B, F)
        Z1 = V1_ndc[..., 2]
        Z2 = V2_ndc[..., 2]

        # NDC XY per vertex (for ray-vertex difference in barycentric)
        V0x, V0y = V0_ndc[..., 0], V0_ndc[..., 1]                 # (B, F)
        V1x, V1y = V1_ndc[..., 0], V1_ndc[..., 1]
        V2x, V2y = V2_ndc[..., 0], V2_ndc[..., 1]

        # ── [7] Pre-allocate accumulators for ray chunks ──────────────────
        dist_sum = torch.zeros(B, dtype=torch.float32, device=device)
        hit_cnt  = torch.zeros(B, dtype=torch.float32, device=device)

        # ── [1] Inner loop: ray chunking ──────────────────────────────────
        # Peak memory per chunk: O(B × R_chunk × F) instead of O(B × R × F)
        for r_start in range(0, R, ray_chunk_size):
            r_end = min(r_start + ray_chunk_size, R)
            Rc    = r_end - r_start

            px_sub = px_rays[r_start:r_end]                        # (Rc,)
            py_sub = py_rays[r_start:r_end]

            # ── [3] Tile culling: ray tile IDs ────────────────────────────
            # Map ray NDC position to tile index
            rt_x = ((px_sub + 1.0) * half_gs).long().clamp(0, gs - 1)  # (Rc,)
            rt_y = ((py_sub + 1.0) * half_gs).long().clamp(0, gs - 1)

            # Tile overlap test replaces per-pixel bbox check
            # (1,1,F) vs (1,Rc,1) → broadcasts to (B, Rc, F)
            # Uses integer tile comparisons (faster than float bbox)
            tile_hit = (
                (rt_x.view(1, Rc, 1) >= ft_xmin.unsqueeze(1)) &
                (rt_x.view(1, Rc, 1) <= ft_xmax.unsqueeze(1)) &
                (rt_y.view(1, Rc, 1) >= ft_ymin.unsqueeze(1)) &
                (rt_y.view(1, Rc, 1) <= ft_ymax.unsqueeze(1))
            ) & static_mask.unsqueeze(1)                           # (B, Rc, F)

            # ── [2] Barycentric via scalar broadcast ──────────────────────
            # KEY: avoid materializing (B, Rc, F, 2) by computing cross
            # products using scalar components directly.
            #
            # cross2d(edge, ray-vertex) = edge_x * dy - edge_y * dx
            # where dx = px - Vx, dy = py - Vy
            #
            # Shapes: px_sub (Rc,) → (1,Rc,1);  V1x (B,F) → (B,1,F)
            # Broadcast: (1,Rc,1) - (B,1,F) → (B,Rc,F) — no 4D tensor!

            px = px_sub.view(1, Rc, 1)                             # (1, Rc, 1)
            py = py_sub.view(1, Rc, 1)

            # w0 = cross2d(V2-V1, P-V1)
            dp1_x = px - V1x.unsqueeze(1)                         # (B, Rc, F)
            dp1_y = py - V1y.unsqueeze(1)
            w0 = e_v2v1_x.unsqueeze(1) * dp1_y - e_v2v1_y.unsqueeze(1) * dp1_x

            # w1 = cross2d(V0-V2, P-V2)
            dp2_x = px - V2x.unsqueeze(1)
            dp2_y = py - V2y.unsqueeze(1)
            w1 = e_v0v2_x.unsqueeze(1) * dp2_y - e_v0v2_y.unsqueeze(1) * dp2_x

            # w2 = cross2d(V1-V0, P-V0)
            dp0_x = px - V0x.unsqueeze(1)
            dp0_y = py - V0y.unsqueeze(1)
            w2 = e_v1v0_x.unsqueeze(1) * dp0_y - e_v1v0_y.unsqueeze(1) * dp0_x

            # Triangle inclusion (same winding test)
            in_tri = tile_hit & (
                ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) |
                ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
            )                                                      # (B, Rc, F)

            # Barycentric coordinates
            area_r = area.unsqueeze(1)                             # (B, 1, F) broadcasts
            u = torch.where(in_tri, w0 / area_r, torch.zeros_like(w0))
            v = torch.where(in_tri, w1 / area_r, torch.zeros_like(w1))
            w_bary = 1.0 - u - v

            # ── Z-buffer ──────────────────────────────────────────────────
            Z_hit = (u * Z0.unsqueeze(1) +
                     v * Z1.unsqueeze(1) +
                     w_bary * Z2.unsqueeze(1))                     # (B, Rc, F)

            Z_inf = torch.where(in_tri, Z_hit,
                                torch.full_like(Z_hit, float('inf')))

            # ── [5] Soft or Hard Z-buffer selection ───────────────────────
            if alpha is not None and alpha > 0:
                # Differentiable softmin: Z_soft = -logsumexp(-α·Z) / α
                neg_aZ = -alpha * Z_inf                            # (B, Rc, F)
                best_z = -torch.logsumexp(neg_aZ, dim=2) / alpha  # (B, Rc)
                # Soft weights for weighted hit-point reconstruction
                weights = torch.softmax(neg_aZ, dim=2)            # (B, Rc, F)
                valid_ray = best_z < 1e6
            else:
                # Hard min (original behavior)
                best_z, hit_idx = torch.min(Z_inf, dim=2)         # (B, Rc)
                valid_ray = best_z < float('inf')

            # ── Perspective-correct reconstruction ────────────────────────
            if alpha is not None and alpha > 0:
                # Soft reconstruction: weighted average over all faces
                # inv_W per face: u/W0 + v/W1 + w/W2
                W0_u = W0.unsqueeze(1)                             # (B, 1, F)
                W1_u = W1.unsqueeze(1)
                W2_u = W2.unsqueeze(1)
                inv_W_all = u / W0_u + v / W1_u + w_bary / W2_u   # (B, Rc, F)
                W_int_all = 1.0 / (inv_W_all + 1e-12)

                # Perspective-correct 3D point per face: (B, Rc, F, 3)
                # V0_world (F,3) → (1,1,F,3)
                hp_all = W_int_all.unsqueeze(-1) * (
                    (u / W0_u).unsqueeze(-1) * V0_world.unsqueeze(0).unsqueeze(0) +
                    (v / W1_u).unsqueeze(-1) * V1_world.unsqueeze(0).unsqueeze(0) +
                    (w_bary / W2_u).unsqueeze(-1) * V2_world.unsqueeze(0).unsqueeze(0)
                )  # (B, Rc, F, 3)

                fc_b = face_centers[b_start:b_end].view(B, 1, 1, 3)
                d_all = torch.norm(hp_all - fc_b, dim=-1)          # (B, Rc, F)

                # Weighted distance
                dists_chunk = (weights * d_all).sum(dim=2)         # (B, Rc)

            else:
                # Hard reconstruction at best face (original gather logic)
                u_h = u.gather(2, hit_idx.unsqueeze(2)).squeeze(2)     # (B, Rc)
                v_h = v.gather(2, hit_idx.unsqueeze(2)).squeeze(2)
                w_h = 1.0 - u_h - v_h

                W0_h = W0.gather(1, hit_idx)                          # (B, Rc)
                W1_h = W1.gather(1, hit_idx)
                W2_h = W2.gather(1, hit_idx)

                inv_W    = u_h / W0_h + v_h / W1_h + w_h / W2_h
                W_interp = 1.0 / (inv_W + 1e-12)

                # Gather world-space vertices at hit face
                V0_h = V0_world[hit_idx.view(-1)].view(B, Rc, 3)
                V1_h = V1_world[hit_idx.view(-1)].view(B, Rc, 3)
                V2_h = V2_world[hit_idx.view(-1)].view(B, Rc, 3)

                hit_point = W_interp.unsqueeze(-1) * (
                    (u_h / W0_h).unsqueeze(-1) * V0_h +
                    (v_h / W1_h).unsqueeze(-1) * V1_h +
                    (w_h / W2_h).unsqueeze(-1) * V2_h
                )  # (B, Rc, 3)

                fc_b = face_centers[b_start:b_end].unsqueeze(1)       # (B, 1, 3)
                dists_chunk = torch.norm(hit_point - fc_b, dim=-1)    # (B, Rc)

            # ── Accumulate across ray chunks ──────────────────────────────
            valid_f   = valid_ray.float()
            dist_sum += (dists_chunk * valid_f).sum(dim=1)         # (B,)
            hit_cnt  += valid_f.sum(dim=1)                         # (B,)

        # ── Final SDF = mean distance over valid rays ─────────────────────
        sdf_values[b_start:b_end] = dist_sum / (hit_cnt + 1e-8)

    elapsed = time.perf_counter() - start_time

    # ── Debug: memory report ──────────────────────────────────────────────
    if device.type == 'cuda':
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"\n  Peak GPU memory : {peak_mb:.0f} MB")
    print(f"  Completed in    : {elapsed:.4f}s")

    return sdf_values.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SDF GPU Rasterizer v4 — Memory-Optimized"
    )
    parser.add_argument("--input_file",     type=str,   default="data/bunny.obj")
    parser.add_argument("--fov",            type=int,   default=120)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--ray_chunk_size", type=int,   default=512,
                        help="Rays per chunk (controls peak GPU memory)")
    parser.add_argument("--num_rings",      type=int,   default=5)
    parser.add_argument("--rays_per_ring",  type=int,   default=10)
    parser.add_argument("--grid_size",      type=int,   default=16,
                        help="Tile grid resolution for face culling")
    parser.add_argument("--alpha",          type=float, default=0,
                        help="Soft Z-buffer temperature (0 = hard min)")
    args = parser.parse_args()

    try:
        mesh = trimesh.load(args.input_file, force='mesh')
        mesh.fix_normals()
        mesh.process()
        print(f"Mesh: {len(mesh.faces)} faces | {len(mesh.vertices)} vertices")
    except Exception as e:
        print(f"Error: {e}")
        return

    alpha = args.alpha if args.alpha > 0 else None

    sdf = compute_sdf_gpu_v4(
        mesh,
        fov_deg           = args.fov,
        num_rings         = args.num_rings,
        num_rays_per_ring = args.rays_per_ring,
        batch_size        = args.batch_size,
        ray_chunk_size    = args.ray_chunk_size,
        grid_size         = args.grid_size,
        alpha             = alpha,
    )

    pv_mesh = pv.wrap(mesh)
    pv_mesh.cell_data['SDF_v4'] = sdf
    plotter = pv.Plotter(title="SDF GPU Rasterizer v4")
    plotter.add_mesh(pv_mesh, scalars='SDF_v4', cmap='jet_r', show_edges=True)
    plotter.set_background('white')
    plotter.show()


if __name__ == "__main__":
    main()
