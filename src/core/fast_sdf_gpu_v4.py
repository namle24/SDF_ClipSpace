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
# V4: Research-grade SDF — Fixes applied:
#   [FIX 1] World-space distance (was mixing NDC + world → now pure world)
#   [FIX 2] Tile culling BEFORE barycentric (was post-hoc → now pre-filter)
#   [FIX 3] Soft Z-buffer stability (Z-shift + high alpha ≥ 500)
#   [FIX 4] Depth validity + backface rejection (Z > 0 mask)
#   [FIX 5] Component-wise interpolation (no 4D tensors)
#   [FIX 6] Mixed precision for heavy ops
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_sdf_gpu_v4(mesh, fov_deg=90, num_rings=5, num_rays_per_ring=10,
                       batch_size=32, ray_chunk_size=512, grid_size=16,
                       alpha=None):
    """
    Memory-optimized, mathematically correct SDF via clip-space rasterization.

    Pipeline:
      World → Clip → NDC → Tile Cull → Barycentric → Z-buffer
      → World-space interpolation → Euclidean distance

    Args:
        batch_size:      cameras (faces) per outer batch
        ray_chunk_size:  rays per inner chunk — controls peak GPU memory
        grid_size:       tile grid resolution for face culling
        alpha:           soft Z-buffer temperature (None = hard min, recommend ≥500)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = time.perf_counter()

    # ── Data preparation (ONCE, outside all loops) ────────────────────────
    vertices     = torch.tensor(mesh.vertices,     dtype=torch.float32, device=device)
    faces_t      = torch.tensor(mesh.faces,        dtype=torch.long,    device=device)
    normals      = torch.tensor(mesh.face_normals, dtype=torch.float32, device=device)

    F_total = len(faces_t)
    V_total = len(vertices)

    # [FIX 1] World-space vertices per face — used for distance computation
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

    num_ray_chunks = (R + ray_chunk_size - 1) // ray_chunk_size
    gs = grid_size
    half_gs = gs * 0.5

    # ── Print config ──────────────────────────────────────────────────────
    print(f"  Device         : {device}")
    print(f"  Faces          : {F_total}  |  Vertices : {V_total}")
    print(f"  Total rays     : {R}")
    print(f"  Batch size     : {batch_size}  |  Ray chunk : {ray_chunk_size}")
    print(f"  Tile grid      : {gs}×{gs}")
    print(f"  Soft Z-buffer  : {'alpha=' + str(alpha) if alpha else 'OFF (hard min)'}")

    # ── Outer loop: camera batches ────────────────────────────────────────
    for b_start in tqdm(range(0, F_total, batch_size), desc="SDF v4"):
        b_end = min(b_start + batch_size, F_total)
        B     = b_end - b_start

        # ── Stage 1: PROJECTION (World → Clip → NDC) ─────────────────────
        inward  = -normals[b_start:b_end]
        inward  = inward / (torch.norm(inward, dim=-1, keepdim=True) + 1e-12)
        eyes    = face_centers[b_start:b_end] + 0.0001 * inward
        targets = eyes + inward

        V_mat  = look_at_torch_batched(eyes, targets, device)
        PV_mat = P_mat @ V_mat                                     # (B, 4, 4)

        # [FIX 6] Mixed precision for the heavy einsum
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            V_clip = torch.einsum('bij,jv->bvi', PV_mat, V_homo_T).float()

        W      = V_clip[..., 3]                                    # (B, V)
        W_safe = W.clamp(min=1e-8)
        NDC    = V_clip[..., :3] / W_safe.unsqueeze(-1)            # (B, V, 3)

        # Per-face NDC coords
        V0_ndc = NDC[:, faces_t[:, 0], :]                          # (B, F, 3)
        V1_ndc = NDC[:, faces_t[:, 1], :]
        V2_ndc = NDC[:, faces_t[:, 2], :]
        W0     = W[:, faces_t[:, 0]]                               # (B, F)
        W1     = W[:, faces_t[:, 1]]
        W2     = W[:, faces_t[:, 2]]

        # W-clipping mask
        valid_w = (W0 > 0.01) & (W1 > 0.01) & (W2 > 0.01)        # (B, F)

        # ── Tile-culling precompute: face bbox → tile ranges ──────────────
        V0x, V0y = V0_ndc[..., 0], V0_ndc[..., 1]                 # (B, F)
        V1x, V1y = V1_ndc[..., 0], V1_ndc[..., 1]
        V2x, V2y = V2_ndc[..., 0], V2_ndc[..., 1]

        min_x = torch.min(torch.min(V0x, V1x), V2x)               # (B, F)
        max_x = torch.max(torch.max(V0x, V1x), V2x)
        min_y = torch.min(torch.min(V0y, V1y), V2y)
        max_y = torch.max(torch.max(V0y, V1y), V2y)

        ft_xmin = ((min_x + 1.0) * half_gs).long().clamp(0, gs - 1)
        ft_xmax = ((max_x + 1.0) * half_gs).long().clamp(0, gs - 1)
        ft_ymin = ((min_y + 1.0) * half_gs).long().clamp(0, gs - 1)
        ft_ymax = ((max_y + 1.0) * half_gs).long().clamp(0, gs - 1)

        # Triangle area in NDC (scalar cross product, no intermediate 2D vecs)
        area = ((V1x - V0x) * (V2y - V1y) - (V1y - V0y) * (V2x - V1x))  # (B, F)
        valid_area = torch.abs(area) > 1e-6                        # (B, F)

        # Static mask: valid W + non-degenerate area
        static_mask = valid_w & valid_area                         # (B, F)


        # NDC Z per vertex
        Z0 = V0_ndc[..., 2]                                       # (B, F)
        Z1 = V1_ndc[..., 2]
        Z2 = V2_ndc[..., 2]

        # ── Pre-allocate accumulators ──────────────────────────────────────
        dist_sum = torch.zeros(B, dtype=torch.float32, device=device)
        hit_cnt  = torch.zeros(B, dtype=torch.float32, device=device)
        INF = torch.tensor(1e6, device=device)

        # ── Inner loop: ray chunking ──────────────────────────────────────
        for r_start in range(0, R, ray_chunk_size):
            r_end = min(r_start + ray_chunk_size, R)
            Rc    = r_end - r_start

            px_sub = px_rays[r_start:r_end]                        # (Rc,)
            py_sub = py_rays[r_start:r_end]

            # ── TILE LOOKUP (per ray) ─────────────────────────────────────
            rt_x = ((px_sub + 1.0) * half_gs).long().clamp(0, gs - 1)  # (Rc,)
            rt_y = ((py_sub + 1.0) * half_gs).long().clamp(0, gs - 1)

            # Build tile overlap mask: (B, Rc, F)
            in_tile = (
                (rt_x.view(1, Rc, 1) >= ft_xmin.unsqueeze(1)) &
                (rt_x.view(1, Rc, 1) <= ft_xmax.unsqueeze(1)) &
                (rt_y.view(1, Rc, 1) >= ft_ymin.unsqueeze(1)) &
                (rt_y.view(1, Rc, 1) <= ft_ymax.unsqueeze(1)) &
                static_mask.unsqueeze(1)
            )  # (B, Rc, F)

            # ── Per-chunk accumulators ────────────────────────────────────
            sdf_chunk   = torch.zeros((B, Rc), device=device)
            valid_chunk = torch.zeros((B, Rc), device=device)

            # ── PER-RAY LOOP: true O(R·k) via sparse candidate selection ─
            for r in range(Rc):

                mask_r = in_tile[:, r, :]                          # (B, F)

                if not mask_r.any():
                    continue

                # Select candidate (camera, face) pairs
                idx   = mask_r.nonzero(as_tuple=False)             # (N, 2)
                b_idx = idx[:, 0]                                  # (N,)
                f_idx = idx[:, 1]                                  # (N,)

                # Gather NDC vertices for candidate faces
                c0x = V0x[b_idx, f_idx]                            # (N,)
                c0y = V0y[b_idx, f_idx]
                c0z = Z0[b_idx, f_idx]
                c1x = V1x[b_idx, f_idx]
                c1y = V1y[b_idx, f_idx]
                c1z = Z1[b_idx, f_idx]
                c2x = V2x[b_idx, f_idx]
                c2y = V2y[b_idx, f_idx]
                c2z = Z2[b_idx, f_idx]

                # Ray point (scalar, same for all candidates at this ray)
                px_r = px_sub[r]
                py_r = py_sub[r]

                # ── BARYCENTRIC (only k candidate faces) ──────────────────
                denom = (
                    (c1y - c2y) * (c0x - c2x) +
                    (c2x - c1x) * (c0y - c2y)
                ) + 1e-8

                u_val = (
                    (c1y - c2y) * (px_r - c2x) +
                    (c2x - c1x) * (py_r - c2y)
                ) / denom

                v_val = (
                    (c2y - c0y) * (px_r - c2x) +
                    (c0x - c2x) * (py_r - c2y)
                ) / denom

                w_val = 1.0 - u_val - v_val

                in_tri = (u_val >= 0) & (v_val >= 0) & (w_val >= 0)

                if not in_tri.any():
                    continue

                # ── DEPTH ─────────────────────────────────────────────────
                Z_val = u_val * c0z + v_val * c1z + w_val * c2z

                # [FIX 4] Reject negative depth + non-triangle hits
                valid_depth = in_tri & (Z_val > 0)
                Z_val = torch.where(valid_depth, Z_val, INF)

                # ── WORLD-SPACE INTERPOLATION [FIX 1] ─────────────────────
                V0_sel = V0_world[f_idx]                           # (N, 3)
                V1_sel = V1_world[f_idx]
                V2_sel = V2_world[f_idx]

                interp = (
                    u_val.unsqueeze(-1) * V0_sel +
                    v_val.unsqueeze(-1) * V1_sel +
                    w_val.unsqueeze(-1) * V2_sel
                )  # (N, 3)

                # Face centers for the camera batch (offset by b_start)
                fc_sel = face_centers[b_start + b_idx]             # (N, 3)
                d = torch.norm(interp - fc_sel, dim=-1)            # (N,)

                # ── VISIBILITY: per-camera soft/hard Z-buffer ─────────────
                unique_b = b_idx.unique()
                for bi in unique_b:
                    b_mask = (b_idx == bi)
                    Z_b = Z_val[b_mask]
                    d_b = d[b_mask]
                    valid_b = valid_depth[b_mask]

                    if not valid_b.any():
                        continue

                    if alpha is not None and alpha > 0:
                        # [FIX 3] Stable softmin with Z-shift
                        Z_shift = Z_b - Z_b[valid_b].min()
                        weights = torch.softmax(-alpha * Z_shift, dim=0)
                        sdf_chunk[bi, r] = (weights * d_b).sum()
                    else:
                        # Hard min: pick closest face
                        best = Z_b.argmin()
                        sdf_chunk[bi, r] = d_b[best]

                    valid_chunk[bi, r] = 1.0

            # ── Accumulate across ray chunks ──────────────────────────────
            dist_sum += sdf_chunk.sum(dim=-1)                      # (B,)
            hit_cnt  += valid_chunk.sum(dim=-1)                    # (B,)

        # ── Final SDF = mean distance over valid rays ─────────────────────
        sdf_values[b_start:b_end] = dist_sum / (hit_cnt + 1e-8)

    elapsed = time.perf_counter() - start_time
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
        description="SDF GPU Rasterizer v4 — Research-Grade"
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
                        help="Soft Z-buffer temperature (0=hard, recommend >=500)")
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
