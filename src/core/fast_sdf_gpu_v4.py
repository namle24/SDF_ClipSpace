import torch
import numpy as np
import trimesh
import time
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions — PRESERVED from v3
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


def cross2d_scalar(ax, ay, bx, by):
    """2D cross product using scalar components — avoids materializing (B,R,F,2)."""
    return ax * by - ay * bx


# ─────────────────────────────────────────────────────────────────────────────
# V4: Memory-optimized with ray chunking, tile culling, soft Z-buffer
# ─────────────────────────────────────────────────────────────────────────────

import torch
import math

def compute_sdf_gpu(
    V_world, F, face_centers, inward_normals,
    proj_matrices,
    num_rings=20,
    rays_per_ring=40,
    ray_chunk=512,
    grid_size=32,
    alpha=200.0,
    device="cuda"
):
    """
    Differentiable SDF via clip-space rasterization (optimized version)
    """

    B = inward_normals.shape[0]
    F_count = F.shape[0]

    # ─────────────────────────────────────────────
    # 1. Project vertices → clip space
    # ─────────────────────────────────────────────
    V = V_world.to(device)
    ones = torch.ones((V.shape[0], 1), device=device)
    V_h = torch.cat([V, ones], dim=1)  # (V, 4)

    clip = torch.matmul(proj_matrices, V_h.T).transpose(1, 2)  # (B, V, 4)

    cX, cY, cZ, cW = clip.unbind(-1)
    eps = 1e-6
    valid_w = cW.abs() > eps

    x_ndc = cX / (cW + eps)
    y_ndc = cY / (cW + eps)
    z_ndc = cZ / (cW + eps)

    # gather faces
    cV0 = torch.stack([x_ndc[:, F[:,0]], y_ndc[:, F[:,0]], z_ndc[:, F[:,0]]], dim=-1)
    cV1 = torch.stack([x_ndc[:, F[:,1]], y_ndc[:, F[:,1]], z_ndc[:, F[:,1]]], dim=-1)
    cV2 = torch.stack([x_ndc[:, F[:,2]], y_ndc[:, F[:,2]], z_ndc[:, F[:,2]]], dim=-1)

    cW0 = cW[:, F[:,0]]
    cW1 = cW[:, F[:,1]]
    cW2 = cW[:, F[:,2]]

    # ─────────────────────────────────────────────
    # 2. Face bounding boxes (NDC)
    # ─────────────────────────────────────────────
    min_x = torch.min(torch.stack([cV0[...,0], cV1[...,0], cV2[...,0]]), dim=0).values
    max_x = torch.max(torch.stack([cV0[...,0], cV1[...,0], cV2[...,0]]), dim=0).values
    min_y = torch.min(torch.stack([cV0[...,1], cV1[...,1], cV2[...,1]]), dim=0).values
    max_y = torch.max(torch.stack([cV0[...,1], cV1[...,1], cV2[...,1]]), dim=0).values

    # ─────────────────────────────────────────────
    # 3. Tile assignment (per face)
    # ─────────────────────────────────────────────
    def to_tile(x):
        return ((x + 1) * 0.5 * grid_size).long().clamp(0, grid_size - 1)

    face_tile_x0 = to_tile(min_x)
    face_tile_x1 = to_tile(max_x)
    face_tile_y0 = to_tile(min_y)
    face_tile_y1 = to_tile(max_y)

    # ─────────────────────────────────────────────
    # 4. Ray sampling (hemisphere)
    # ─────────────────────────────────────────────
    rays = []
    for r in range(1, num_rings + 1):
        for i in range(r * rays_per_ring):
            theta = (i / (r * rays_per_ring)) * 2 * math.pi
            radius = r / num_rings
            rays.append([radius * math.cos(theta), radius * math.sin(theta)])

    rays = torch.tensor(rays, device=device)  # (R, 2)
    R_total = rays.shape[0]

    px_all = rays[:,0].view(1, -1, 1)
    py_all = rays[:,1].view(1, -1, 1)

    fc = face_centers.to(device)

    INF = torch.tensor(1e6, device=device)

    sdf_accum = torch.zeros((B,), device=device)
    ray_count = 0

    # ─────────────────────────────────────────────
    # 5. Ray chunk loop
    # ─────────────────────────────────────────────
    for r_start in range(0, R_total, ray_chunk):
        r_end = min(r_start + ray_chunk, R_total)

        px = px_all[:, r_start:r_end]
        py = py_all[:, r_start:r_end]

        R_c = px.shape[1]

        # ─────────────────────────
        # Tile culling
        # ─────────────────────────
        ray_tx = to_tile(px)
        ray_ty = to_tile(py)

        ray_tx = ray_tx.view(1, R_c, 1)
        ray_ty = ray_ty.view(1, R_c, 1)

        ft_x0 = face_tile_x0.unsqueeze(1)
        ft_x1 = face_tile_x1.unsqueeze(1)
        ft_y0 = face_tile_y0.unsqueeze(1)
        ft_y1 = face_tile_y1.unsqueeze(1)

        in_tile = (
            (ray_tx >= ft_x0) & (ray_tx <= ft_x1) &
            (ray_ty >= ft_y0) & (ray_ty <= ft_y1)
        )

        # ─────────────────────────
        # Barycentric raster
        # ─────────────────────────
        px_br = px.unsqueeze(-1)
        py_br = py.unsqueeze(-1)

        denom = (
            (cV1[...,1] - cV2[...,1]) * (cV0[...,0] - cV2[...,0]) +
            (cV2[...,0] - cV1[...,0]) * (cV0[...,1] - cV2[...,1])
        ) + 1e-8

        u = (
            (cV1[...,1] - cV2[...,1]) * (px_br - cV2[...,0]) +
            (cV2[...,0] - cV1[...,0]) * (py_br - cV2[...,1])
        ) / denom

        v = (
            (cV2[...,1] - cV0[...,1]) * (px_br - cV2[...,0]) +
            (cV0[...,0] - cV2[...,0]) * (py_br - cV2[...,1])
        ) / denom

        w = 1 - u - v

        in_tri = (u >= 0) & (v >= 0) & (w >= 0)

        # combine mask
        mask = in_tri & in_tile

        # ─────────────────────────
        # Depth
        # ─────────────────────────
        Z = u * cV0[...,2] + v * cV1[...,2] + w * cV2[...,2]
        Z = torch.where(mask, Z, INF)

        # ─────────────────────────
        # Soft visibility
        # ─────────────────────────
        weights = torch.softmax(-alpha * Z, dim=-1)

        # ─────────────────────────
        # Distance (no 4D tensor)
        # ─────────────────────────
        interp_x = u * cV0[...,0] + v * cV1[...,0] + w * cV2[...,0]
        interp_y = u * cV0[...,1] + v * cV1[...,1] + w * cV2[...,1]
        interp_z = Z

        dx = interp_x - fc[:,0].view(B,1,1)
        dy = interp_y - fc[:,1].view(B,1,1)
        dz = interp_z - fc[:,2].view(B,1,1)

        dists = torch.sqrt(dx*dx + dy*dy + dz*dz)

        sdf_rays = (weights * dists).sum(dim=-1)

        valid = mask.any(dim=-1)
        sdf_rays = torch.where(valid, sdf_rays, 0.0)

        sdf_accum += sdf_rays.sum(dim=-1)
        ray_count += valid.sum(dim=-1)

    sdf = sdf_accum / (ray_count + 1e-6)

    return sdf


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import pyvista as pv

    parser = argparse.ArgumentParser(description="SDF GPU Rasterizer v4 (Memory-Optimized)")
    parser.add_argument("--input_file",     type=str,   default="data/bunny.obj")
    parser.add_argument("--fov",            type=int,   default=120)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--num_rings",      type=int,   default=15)
    parser.add_argument("--rays_per_ring",  type=int,   default=30)
    parser.add_argument("--ray_chunk_size", type=int,   default=512)
    parser.add_argument("--grid_size",      type=int,   default=16)
    parser.add_argument("--alpha",          type=float, default=100.0)
    parser.add_argument("--hard_zbuffer",   action="store_true")
    parser.add_argument("--no_amp",         action="store_true")
    parser.add_argument("--debug",          action="store_true")
    args = parser.parse_args()

    try:
        mesh = trimesh.load(args.input_file, force='mesh')
        mesh.fix_normals()
        mesh.process()
        print(f"Mesh: {len(mesh.faces)} faces | {len(mesh.vertices)} vertices")
    except Exception as e:
        print(f"Error: {e}")
        return

    sdf = compute_sdf_gpu_v4(
        mesh,
        fov_deg           = args.fov,
        num_rings         = args.num_rings,
        num_rays_per_ring = args.rays_per_ring,
        batch_size        = args.batch_size,
        ray_chunk_size    = args.ray_chunk_size,
        grid_size         = args.grid_size,
        alpha             = args.alpha,
        use_soft_zbuffer  = not args.hard_zbuffer,
        use_amp           = not args.no_amp,
        debug             = args.debug,
    )

    pv_mesh = pv.wrap(mesh)
    pv_mesh.cell_data['SDF_v4'] = sdf
    plotter = pv.Plotter(title="SDF GPU Rasterizer v4")
    plotter.add_mesh(pv_mesh, scalars='SDF_v4', cmap='jet_r', show_edges=True)
    plotter.set_background('white')
    plotter.show()


if __name__ == "__main__":
    main()
