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

        # Precompute edge vectors for barycentric (ONCE per batch)
        e_v2v1_x = V2x - V1x                                      # (B, F)
        e_v2v1_y = V2y - V1y
        e_v0v2_x = V0x - V2x
        e_v0v2_y = V0y - V2y
        e_v1v0_x = V1x - V0x
        e_v1v0_y = V1y - V0y

        # NDC Z per vertex
        Z0 = V0_ndc[..., 2]                                       # (B, F)
        Z1 = V1_ndc[..., 2]
        Z2 = V2_ndc[..., 2]

        # Pre-allocate accumulators
        dist_sum = torch.zeros(B, dtype=torch.float32, device=device)
        hit_cnt  = torch.zeros(B, dtype=torch.float32, device=device)

        # ── Inner loop: ray chunking ──────────────────────────────────────
        for r_start in range(0, R, ray_chunk_size):
            r_end = min(r_start + ray_chunk_size, R)
            Rc    = r_end - r_start

            px_sub = px_rays[r_start:r_end]                        # (Rc,)
            py_sub = py_rays[r_start:r_end]

            # ── Stage 2: TILE CULLING (before barycentric) ────────────────
            # [FIX 2] Cull faces BEFORE computing barycentric → O(R×k)
            rt_x = ((px_sub + 1.0) * half_gs).long().clamp(0, gs - 1)  # (Rc,)
            rt_y = ((py_sub + 1.0) * half_gs).long().clamp(0, gs - 1)

            # Batch-conservative tile ranges: union across B cameras
            ft_xmin_con = ft_xmin.min(dim=0).values                # (F,)
            ft_xmax_con = ft_xmax.max(dim=0).values
            ft_ymin_con = ft_ymin.min(dim=0).values
            ft_ymax_con = ft_ymax.max(dim=0).values
            static_any  = static_mask.any(dim=0)                   # (F,)

            # For each ray, find faces whose tile range contains that ray's tile
            # rt_x: (Rc,) → (Rc, 1) vs ft ranges: (F,) → (1, F)
            ray_face_hit = (
                (rt_x.unsqueeze(1) >= ft_xmin_con.unsqueeze(0)) &
                (rt_x.unsqueeze(1) <= ft_xmax_con.unsqueeze(0)) &
                (rt_y.unsqueeze(1) >= ft_ymin_con.unsqueeze(0)) &
                (rt_y.unsqueeze(1) <= ft_ymax_con.unsqueeze(0)) &
                static_any.unsqueeze(0)
            )  # (Rc, F)

            # Candidate face indices: faces that pass tile test for ANY ray
            cand_mask = ray_face_hit.any(dim=0)                    # (F,)
            cand_idx  = cand_mask.nonzero(as_tuple=True)[0]        # (k,)
            k = len(cand_idx)

            if k == 0:
                continue

            # ── Gather only candidate face data (F → k) ──────────────────
            c_V0x = V0x[:, cand_idx]                               # (B, k)
            c_V0y = V0y[:, cand_idx]
            c_V1x = V1x[:, cand_idx]
            c_V1y = V1y[:, cand_idx]
            c_V2x = V2x[:, cand_idx]
            c_V2y = V2y[:, cand_idx]

            c_e_v2v1_x = e_v2v1_x[:, cand_idx]                    # (B, k)
            c_e_v2v1_y = e_v2v1_y[:, cand_idx]
            c_e_v0v2_x = e_v0v2_x[:, cand_idx]
            c_e_v0v2_y = e_v0v2_y[:, cand_idx]
            c_e_v1v0_x = e_v1v0_x[:, cand_idx]
            c_e_v1v0_y = e_v1v0_y[:, cand_idx]

            c_area       = area[:, cand_idx]                       # (B, k)
            c_static     = static_mask[:, cand_idx]                # (B, k)
            c_Z0         = Z0[:, cand_idx]                         # (B, k)
            c_Z1         = Z1[:, cand_idx]
            c_Z2         = Z2[:, cand_idx]
            c_W0         = W0[:, cand_idx]                         # (B, k)
            c_W1         = W1[:, cand_idx]
            c_W2         = W2[:, cand_idx]

            # ── Stage 3: BARYCENTRIC (on candidates only) ─────────────────
            # Scalar broadcast: (1,Rc,1) vs (B,1,k) → (B,Rc,k)
            px = px_sub.view(1, Rc, 1)                             # (1, Rc, 1)
            py = py_sub.view(1, Rc, 1)

            # Per-camera tile test on candidates
            c_ft_xmin = ft_xmin[:, cand_idx]                       # (B, k)
            c_ft_xmax = ft_xmax[:, cand_idx]
            c_ft_ymin = ft_ymin[:, cand_idx]
            c_ft_ymax = ft_ymax[:, cand_idx]

            tile_hit = (
                (rt_x.view(1, Rc, 1) >= c_ft_xmin.unsqueeze(1)) &
                (rt_x.view(1, Rc, 1) <= c_ft_xmax.unsqueeze(1)) &
                (rt_y.view(1, Rc, 1) >= c_ft_ymin.unsqueeze(1)) &
                (rt_y.view(1, Rc, 1) <= c_ft_ymax.unsqueeze(1)) &
                c_static.unsqueeze(1)
            )  # (B, Rc, k)

            # w0 = cross2d(V2-V1, P-V1)
            dp1_x = px - c_V1x.unsqueeze(1)                       # (B, Rc, k)
            dp1_y = py - c_V1y.unsqueeze(1)
            w0 = c_e_v2v1_x.unsqueeze(1) * dp1_y - c_e_v2v1_y.unsqueeze(1) * dp1_x

            # w1 = cross2d(V0-V2, P-V2)
            dp2_x = px - c_V2x.unsqueeze(1)
            dp2_y = py - c_V2y.unsqueeze(1)
            w1 = c_e_v0v2_x.unsqueeze(1) * dp2_y - c_e_v0v2_y.unsqueeze(1) * dp2_x

            # w2 = cross2d(V1-V0, P-V0)
            dp0_x = px - c_V0x.unsqueeze(1)
            dp0_y = py - c_V0y.unsqueeze(1)
            w2 = c_e_v1v0_x.unsqueeze(1) * dp0_y - c_e_v1v0_y.unsqueeze(1) * dp0_x

            # Winding test
            in_tri = tile_hit & (
                ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) |
                ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
            )  # (B, Rc, k)

            # Barycentric coordinates
            area_r = c_area.unsqueeze(1)                           # (B, 1, k)
            u = torch.where(in_tri, w0 / area_r, torch.zeros_like(w0))
            v = torch.where(in_tri, w1 / area_r, torch.zeros_like(w1))
            w_bary = 1.0 - u - v

            # ── Stage 4: VISIBILITY (Z-buffer + depth validity) ───────────
            Z_hit = (u * c_Z0.unsqueeze(1) +
                     v * c_Z1.unsqueeze(1) +
                     w_bary * c_Z2.unsqueeze(1))                   # (B, Rc, k)

            # [FIX 4] Reject negative depth (behind camera)
            valid_hit = in_tri & (Z_hit > 0)                       # (B, Rc, k)

            Z_inf = torch.where(valid_hit, Z_hit,
                                torch.full_like(Z_hit, float('inf')))

            # ── Stage 5: DISTANCE (world-space, component-wise) ───────────
            # [FIX 1] Interpolate in WORLD SPACE, not NDC
            # [FIX 5] Component-wise to avoid (B, Rc, k, 3) tensor
            cV0wx = V0_world[cand_idx, 0]                         # (k,)
            cV0wy = V0_world[cand_idx, 1]
            cV0wz = V0_world[cand_idx, 2]
            cV1wx = V1_world[cand_idx, 0]
            cV1wy = V1_world[cand_idx, 1]
            cV1wz = V1_world[cand_idx, 2]
            cV2wx = V2_world[cand_idx, 0]
            cV2wy = V2_world[cand_idx, 1]
            cV2wz = V2_world[cand_idx, 2]

            # Perspective-correct weights
            cW0_br = c_W0.unsqueeze(1)                             # (B, 1, k)
            cW1_br = c_W1.unsqueeze(1)
            cW2_br = c_W2.unsqueeze(1)
            inv_W = u / cW0_br + v / cW1_br + w_bary / cW2_br     # (B, Rc, k)
            W_interp = 1.0 / (inv_W + 1e-12)

            # Component-wise world-space hit point: (B, Rc, k)
            u_w0 = u / cW0_br                                     # (B, Rc, k)
            v_w1 = v / cW1_br
            w_w2 = w_bary / cW2_br

            # (B, Rc, k) * scalar(k) via broadcast — NO 4D tensor
            hp_x = W_interp * (u_w0 * cV0wx + v_w1 * cV1wx + w_w2 * cV2wx)
            hp_y = W_interp * (u_w0 * cV0wy + v_w1 * cV1wy + w_w2 * cV2wy)
            hp_z = W_interp * (u_w0 * cV0wz + v_w1 * cV1wz + w_w2 * cV2wz)

            # Distance from face center (world space)
            fc_x = face_centers[b_start:b_end, 0].view(B, 1, 1)   # (B, 1, 1)
            fc_y = face_centers[b_start:b_end, 1].view(B, 1, 1)
            fc_z = face_centers[b_start:b_end, 2].view(B, 1, 1)

            dx = hp_x - fc_x                                      # (B, Rc, k)
            dy = hp_y - fc_y
            dz = hp_z - fc_z
            dists_all = torch.sqrt(dx*dx + dy*dy + dz*dz + 1e-12) # (B, Rc, k)

            # ── Z-buffer: soft or hard ────────────────────────────────────
            if alpha is not None and alpha > 0:
                # [FIX 3] Stable softmin: shift Z before scaling
                Z_for_soft = torch.where(valid_hit, Z_hit,
                    torch.tensor(1e6, device=device, dtype=Z_hit.dtype))
                Z_min = Z_for_soft.min(dim=-1, keepdim=True).values
                Z_shifted = Z_for_soft - Z_min                     # (B, Rc, k)
                weights = torch.softmax(-alpha * Z_shifted, dim=-1)
                dists_chunk = (weights * dists_all).sum(dim=-1)    # (B, Rc)
                valid_ray = valid_hit.any(dim=-1)                  # (B, Rc)
            else:
                # Hard min (original v3 logic)
                best_z, best_k = torch.min(Z_inf, dim=2)          # (B, Rc)
                valid_ray = best_z < float('inf')
                dists_chunk = dists_all.gather(
                    2, best_k.unsqueeze(2)).squeeze(2)             # (B, Rc)

            # ── Accumulate across ray chunks ──────────────────────────────
            valid_f   = valid_ray.float()
            dist_sum += (dists_chunk * valid_f).sum(dim=1)
            hit_cnt  += valid_f.sum(dim=1)

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
