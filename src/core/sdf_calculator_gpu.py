"""
sdf_calculator_gpu.py — Optimized: Clip-Space Möller-Trumbore SDF on GPU

Hybrid Architecture:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  PHASE 1 (View-Projection Transformation):                         │
  │    look_at(eye, target) → V          View Matrix  4×4               │
  │    perspective(fov, near, far) → P   Projection Matrix 4×4          │
  │    PV = P @ V                                                       │
  │    vertices_clip = (PV @ verts_homo.T).T   ← Perspective Divide     │
  │    ⇒ Perspective Cone in World Space                                │
  │       is straightened into PARALLEL rays along the Z-axis           │
  │       in Clipping Space.                                            │
  ├─────────────────────────────────────────────────────────────────────┤
  │  PHASE 2 (GPU Core):                                                │
  │    Generate parallel ray grid D = [0, 0, 1] on XY plane             │
  │    (Concentric rings pattern).                                      │
  │    Feed TRANSFORMED triangles + parallel rays into                 │
  │    Möller-Trumbore Batched GPU → find (u, v, t).                   │
  │    Use (u, v) to interpolate hit point in WORLD SPACE               │
  │    → Calculate exact Euclidean distance.                            │
  └─────────────────────────────────────────────────────────────────────┘
"""

import trimesh
import numpy as np
import pyvista as pv
import time
import torch
from tqdm import tqdm


# =====================================================================
# 1. CAMERA MATRICES — PyTorch GPU
# =====================================================================

def look_at_torch(eye, target, device='cuda'):
    """
    4x4 View Matrix: World Space → Camera Space.
    Implements Gram–Schmidt orthogonalization with a randomized up vector.
    """
    forward = target - eye
    forward = forward / (torch.norm(forward) + 1e-12)

    # Random up vector -> orthogonal projection -> right -> true_up (Gram-Schmidt)
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
    4x4 Projection Matrix: Camera Space → Clip Space.
    Standard perspective projection formula.
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
# 2. PARALLEL RAY GENERATION IN CLIP SPACE (Ring Pattern)
# =====================================================================

def generate_clip_space_rays(fov_rad, num_rings=5, num_rays_per_ring=10, device='cuda'):
    """
    Generates parallel rays D=[0,0,1] on the XY plane in Clip Space.
    Concentric ring pattern for even distribution.
    
    Returns:
        ray_origins_xy: (R, 2) — XY coordinates of ray origins in Clip Space
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
# 3. CLIP SPACE MÖLLER-TRUMBORE — GPU Batching Core
# =====================================================================

def moller_trumbore_clip_batch(ray_origins, ray_dir, V0c, V1c, V2c,
                               V0w, V1w, V2w,
                               umbrella_mask, ray_batch_size, device):
    """
    Batched Möller-Trumbore intersection on GPU in Clip Space.
    Parallel rays D=[0,0,1], origin at (px, py, -0.01).
    
    Returns (hit_world, hit_valid) for each ray.
    Uses Barycentric coordinates (u,v) to interpolate back to World Space.
    
    Args:
        ray_origins:    (R, 3) — Ray origins in Clip Space
        ray_dir:        (3,)   — Ray direction [0, 0, 1]
        V0c, V1c, V2c:  (Fv, 3) — Triangle vertices in Clip Space (filtered)
        V0w, V1w, V2w:  (Fv, 3) — Triangle vertices in World Space (for interpolation)
        umbrella_mask:  (Fv,) bool — True if face contains the source vertex
        ray_batch_size: int
    
    Returns:
        hit_world: (R, 3) — Euclidean World Space hit points
        hit_valid: (R,)   — Boolean mask of valid intersections
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
        
        # Möller-Trumbore: P = D × E2 (broadcast D over F faces)
        P = torch.linalg.cross(D.expand(Fv, 3), E2c)  # (Fv, 3)
        
        # a = E1 · P
        a = torch.sum(E1c * P, dim=-1)  # (Fv,)
        
        # Filter near-parallel faces
        face_valid = torch.abs(a) > 1e-8  # (Fv,)
        face_valid = face_valid & (~umbrella_mask)  # Exclude source faces
        
        safe_a = torch.where(face_valid, a, torch.ones_like(a))
        f = 1.0 / safe_a  # (Fv,)
        
        # s = O - V0 -> (B, Fv, 3)
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
        bary_valid = bary_valid & (t > 1e-4)  # Points ahead of ray
        
        # Set misses to inf
        t = torch.where(bary_valid, t, torch.tensor(float('inf'), device=device))
        
        # Find nearest face for each ray in batch
        min_t, min_idx = torch.min(t, dim=-1)  # (B,)
        
        # Update best if closer intersection found
        update_mask = min_t < best_t[i:end]
        best_t[i:end] = torch.where(update_mask, min_t, best_t[i:end])
        
        # Extract barycentric coords at intersection
        batch_arange = torch.arange(B, device=device)
        u_at_min = u_bary[batch_arange, min_idx]
        v_at_min = v_bary[batch_arange, min_idx]
        
        best_u[i:end] = torch.where(update_mask, u_at_min, best_u[i:end])
        best_v[i:end] = torch.where(update_mask, v_at_min, best_v[i:end])
        best_face[i:end] = torch.where(update_mask, min_idx, best_face[i:end])
    
    # Interpolate hit point in World Space using Barycentric coordinates
    hit_valid = best_t < float('inf')
    
    w_bary = 1.0 - best_u - best_v  # (R,)
    hit_world = (w_bary.unsqueeze(-1) * V0w[best_face] +
                 best_u.unsqueeze(-1) * V1w[best_face] +
                 best_v.unsqueeze(-1) * V2w[best_face])  # (R, 3)
    
    return hit_world, hit_valid


# =====================================================================
# 4. MAIN ENGINE — Clip-Space + Möller-Trumbore GPU
# =====================================================================

@torch.no_grad()
def compute_sdf_clipspace_gpu(mesh, fov_deg=90, num_rings=5, num_rays_per_ring=10,
                               vertex_batch_size=64, ray_batch_size=4096):
    """
    Calculates SDF values for all vertices using:
      - View-Projection transformation to Clip Space
      - GPU-accelerated Möller-Trumbore intersection
      - World-space distance calculation via Barycentric interpolation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  CLIP-SPACE MÖLLER-TRUMBORE SDF ENGINE (GPU)")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    start_time = time.perf_counter()

    # --- Data upload to VRAM ---
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)

    N = len(vertices)
    F = len(faces)

    # Pre-calculate Homogeneous coordinates
    ones = torch.ones((N, 1), dtype=torch.float32, device=device)
    verts_homo = torch.cat([vertices, ones], dim=-1)  # (N, 4)

    # World-Space triangle vertices (static — used for final interpolation)
    V0w = vertices[faces[:, 0]]  # (F, 3)
    V1w = vertices[faces[:, 1]]
    V2w = vertices[faces[:, 2]]

    # Camera parameters
    fov_rad = torch.tensor(np.radians(fov_deg), dtype=torch.float32, device=device)
    near = torch.tensor(0.001, dtype=torch.float32, device=device)
    bbox_diag = torch.norm(vertices.max(dim=0).values - vertices.min(dim=0).values)
    far = bbox_diag + 1.0

    # Projection Matrix (static for all viewpoints)
    P = perspective_torch(fov_rad, near, far, device)

    # Pre-generate parallel ray pattern
    ray_xy = generate_clip_space_rays(fov_rad, num_rings, num_rays_per_ring, device)
    R = len(ray_xy)
    
    # Ray origins in NDC space
    ray_origins_clip = torch.zeros((R, 3), dtype=torch.float32, device=device)
    ray_origins_clip[:, 0] = ray_xy[:, 0]
    ray_origins_clip[:, 1] = ray_xy[:, 1]
    ray_origins_clip[:, 2] = -0.01  # Slightly offset from near plane
    
    # Ray direction: D = [0, 0, 1] (along positive Z in NDC)
    ray_dir = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)

    print(f"[*] Mesh: {N} vertices, {F} triangles")
    print(f"[*] Rays/Vertex: {R} | FOV: {fov_deg}° | Near/Far: {near.item():.3f}/{far.item():.1f}")

    sdf_values = torch.zeros(N, dtype=torch.float32, device=device)

    # --- Vertex Batch Processing ---
    for b_start in tqdm(range(0, N, vertex_batch_size),
                        desc="Clip-Space MT Processing"):
        b_end = min(b_start + vertex_batch_size, N)

        for idx in range(b_start, b_end):
            # PHASE 1: View-Projection Transformation
            inward = -normals[idx]
            norm_val = torch.norm(inward)
            if norm_val < 1e-8:
                continue
            inward = inward / norm_val

            eye = vertices[idx] + 0.001 * inward
            target = eye + inward

            V_mat = look_at_torch(eye, target, device)  # (4, 4)
            PV = P @ V_mat  # (4, 4)

            # Transform ALL vertices to Clip Space
            verts_clip4 = (PV @ verts_homo.T).T  # (N, 4)
            w_clip = verts_clip4[:, 3:4]
            w_safe = torch.where(torch.abs(w_clip) > 1e-8, w_clip,
                                 torch.ones_like(w_clip) * 1e-8)
            
            # Perspective Divide: Transforms cone into parallel cylinder
            verts_ndc = verts_clip4[:, :3] / w_safe  # (N, 3)

            # Clip-Space triangle vertices
            V0c = verts_ndc[faces[:, 0]]  # (F, 3)
            V1c = verts_ndc[faces[:, 1]]
            V2c = verts_ndc[faces[:, 2]]

            # Visibility filtering: Z < 0 in Camera Space (ahead of camera)
            verts_cam = (V_mat @ verts_homo.T).T[:, :3]
            face_z = (verts_cam[faces[:, 0], 2] +
                      verts_cam[faces[:, 1], 2] +
                      verts_cam[faces[:, 2], 2]) / 3.0
            in_front = face_z < -0.005

            # Exclude source faces (containing the vertex itself)
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

            # PHASE 2: Batched GPU Möller-Trumbore
            hit_world, hit_valid = moller_trumbore_clip_batch(
                ray_origins_clip, ray_dir,
                V0c_f, V1c_f, V2c_f,
                V0w_f, V1w_f, V2w_f,
                umbrella_f, ray_batch_size, device
            )

            if hit_valid.sum() == 0:
                continue

            # Calculate exact Euclidean distance in World Space
            dists = torch.norm(hit_world - vertices[idx].unsqueeze(0), dim=-1)
            dists = torch.where(hit_valid, dists, torch.zeros_like(dists))

            sdf_values[idx] = dists.sum() / hit_valid.sum()

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    avg = sdf_values.mean().item()

    print(f"\n{'='*60}")
    print(f"  ELAPSED TIME: {elapsed:.4f} seconds")
    print(f"  Average SDF: {avg:.4f}")
    print(f"{'='*60}\n")

    return sdf_values.cpu().numpy()


# =====================================================================
# 5. ENTRY POINT
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Clip-Space Möller-Trumbore SDF (GPU Acceleration)")
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
            raise ValueError("Mesh is empty!")
    except Exception as e:
        print(f"Mesh loading error: {e}")
        return

    sdf = compute_sdf_clipspace_gpu(
        mesh,
        fov_deg=args.fov,
        num_rings=args.num_rings,
        num_rays_per_ring=args.rays_per_ring,
        vertex_batch_size=args.vertex_batch,
        ray_batch_size=args.ray_batch
    )

    # Visualization
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
