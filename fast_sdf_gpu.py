import torch
import numpy as np
import pyvista as pv
import trimesh
import time
from tqdm import tqdm

def look_at_torch_batched(eyes, targets, device='cuda'):
    B = eyes.shape[0]
    forward = targets - eyes
    forward = forward / (torch.norm(forward, dim=-1, keepdim=True) + 1e-12)

    # Đảm bảo Up vector ổn định (Deterministic)
    world_up = torch.tensor([0.0, 1.0, 0.0], device=device).expand(B, 3).clone()
    # Nếu hướng nhìn gần trùng với trục Y, dùng trục Z làm Up
    mask = (torch.abs(forward[:, 1]) > 0.99)
    world_up[mask] = torch.tensor([0.0, 0.0, 1.0], device=device)

    right = torch.linalg.cross(forward, world_up)
    right = right / (torch.norm(right, dim=-1, keepdim=True) + 1e-12)
    true_up = torch.linalg.cross(right, forward)

    # Xây dựng khối xoay (Rotate)
    rotation = torch.zeros((B, 4, 4), dtype=torch.float32, device=device)
    rotation[:, 0, :3] = right
    rotation[:, 1, :3] = true_up
    rotation[:, 2, :3] = -forward
    rotation[:, 3, 3] = 1.0

    # Xây dựng khối tịnh tiến (Translate)
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
    # Chiếu Z vào khoảng [0, 1] (Close=0, Far=1)
    P[2, 2] = -far / (far - near)
    P[2, 3] = -(near * far) / (far - near)
    P[3, 2] = -1.0
    return P

def generate_rays_ndc(fov_deg, num_rings=5, num_rays_per_ring=10, device='cuda'):
    tan_fov = np.tan(np.radians(fov_deg) / 2.0)
    cot = 1.0 / tan_fov
    
    all_px = []
    all_py = []
    for ring in range(num_rings):
        r = (ring + 0.5) / num_rings
        num_points = int(num_rays_per_ring * (ring + 1))
        angles = torch.linspace(0, 2 * torch.pi, num_points + 1, device=device)[:-1]
        
        x = r * torch.cos(angles) * tan_fov
        y = r * torch.sin(angles) * tan_fov
        
        # Chuyển từ Camera space sang NDC mặt phẳng XY
        px = cot * x
        py = cot * y
        
        all_px.append(px)
        all_py.append(py)
        
    return torch.cat(all_px), torch.cat(all_py)

@torch.no_grad()
def compute_fast_sdf_gpu(mesh, fov_deg=90, num_rings=5, num_rays_per_ring=10, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" VECTOR HÓA BATCH GPU")
    print(f"  Thiết bị: {device} | Batch Size: {batch_size}")

    start_time = time.perf_counter()

    # Chuẩn bị dữ liệu ban đầu
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    normals = torch.tensor(mesh.face_normals, dtype=torch.float32, device=device)

    V = len(vertices)
    F = len(faces)

    # Tính tâm mặt (Face Centers) để làm gốc bắn tia
    V0_world = vertices[faces[:, 0]]
    V1_world = vertices[faces[:, 1]]
    V2_world = vertices[faces[:, 2]]
    face_centers = (V0_world + V1_world + V2_world) / 3.0

    # Chuyển đỉnh sang Homogeneous Coordinate (4, V)
    V_homo_T = torch.cat([vertices, torch.ones(V, 1, device=device)], dim=1).T

    # Ma trận Perspective cố định
    P_mat = perspective_torch(fov_deg, device=device)
    
    # Sinh chùm tia NDC (px, py)
    px_rays, py_rays = generate_rays_ndc(fov_deg, num_rings, num_rays_per_ring, device)
    R = len(px_rays)
    px_batch = px_rays.view(1, R, 1)  # Kích thước (1, Ray, 1) để broadcast
    py_batch = py_rays.view(1, R, 1)

    sdf_values = torch.zeros(F, dtype=torch.float32, device=device)

    # Vòng lặp chính xử lý Lô (Batch) các Camera (Face-centric)
    for b_start in tqdm(range(0, F, batch_size), desc="GPU Rasterizing"):
        b_end = min(b_start + batch_size, F)
        B = b_end - b_start

        # Chuyển không gian Batched (View-Projection to NDC)

        # Hướng bắn tia đâm vào lòng vật thể
        inward_batch = -normals[b_start:b_end]
        inward_batch = inward_batch / (torch.norm(inward_batch, dim=-1, keepdim=True) + 1e-12)

        eyes = face_centers[b_start:b_end] + 0.0001 * inward_batch
        targets = eyes + inward_batch
        
        # Ma trận PV đa biến (B, 4, 4) - XỬ LÝ SONG SONG CẢ BATCH
        V_batch = look_at_torch_batched(eyes, targets, device)
        PV_batch = P_mat @ V_batch

        # Biến đổi toàn bộ Vertices bằng einsum: (B, 4, 4) @ (4, V) -> (B, V, 4)
        V_clip = torch.einsum('bij,jv->bvi', PV_batch, V_homo_T)

        # Trích xuất W và Perspective Divide
        W = V_clip[..., 3] 
        W_safe = torch.where(torch.abs(W) < 1e-8, torch.tensor(1e-8, device=device), W)
        NDC = V_clip[..., :3] / W_safe.unsqueeze(-1) # (B, V, 3)

        # Lấy tọa độ tam giác NDC và W chóp đỉnh
        V0_ndc = NDC[:, faces[:, 0], :] # (B, F, 3)
        V1_ndc = NDC[:, faces[:, 1], :]
        V2_ndc = NDC[:, faces[:, 2], :]
        W0 = W[:, faces[:, 0]] # (B, F)
        W1 = W[:, faces[:, 1]]
        W2 = W[:, faces[:, 2]]

        # W-CLIPPING & Lọc Bounding Box 2D

        # W-CLIPPING: Chỉ lấy tam giác có W dương (trước mặt camera) để tránh Phantom Hits
        valid_w_mask = (W0 > 0.01) & (W1 > 0.01) & (W2 > 0.01) # (B, F)

        # Tính toán BBox 2D trong không gian NDC XY
        all_x = torch.stack([V0_ndc[...,0], V1_ndc[...,0], V2_ndc[...,0]], dim=-1)
        all_y = torch.stack([V0_ndc[...,1], V1_ndc[...,1], V2_ndc[...,1]], dim=-1)
        min_x, _ = torch.min(all_x, dim=-1)
        max_x, _ = torch.max(all_x, dim=-1)
        min_y, _ = torch.min(all_y, dim=-1)
        max_y, _ = torch.max(all_y, dim=-1)

        # Mask lọc BBox (B, R, F)
        in_bbox = (px_batch >= min_x.unsqueeze(1)) & (px_batch <= max_x.unsqueeze(1)) & \
                  (py_batch >= min_y.unsqueeze(1)) & (py_batch <= max_y.unsqueeze(1))
        
        # Kết hợp với W-Clip Mask
        in_bbox = in_bbox & valid_w_mask.unsqueeze(1)

        # Point-in-Triangle & Z-Buffer (Z_ndc lọt [0, 1])
        V0_2d = V0_ndc[..., :2].unsqueeze(1) 
        V1_2d = V1_ndc[..., :2].unsqueeze(1)
        V2_2d = V2_ndc[..., :2].unsqueeze(1)
        P_2d = torch.stack([px_batch, py_batch], dim=-1).expand(B, R, F, 2)

        # Cross product 2D để tính Barycentric Coords
        # w0 = Area(P, V1, V2), w1 = Area(P, V2, V0), w2 = Area(P, V0, V1)
        def cross2d(a, b): return a[...,0]*b[...,1] - a[...,1]*b[...,0]
        
        area = cross2d(V1_ndc[...,:2] - V0_ndc[...,:2], V2_ndc[...,:2] - V1_ndc[...,:2])
        area = area.unsqueeze(1)
        # Kiểm tra area để tránh chia cho 0
        valid_area = torch.abs(area) > 1e-6

        w0 = cross2d(V2_2d - V1_2d, P_2d - V1_2d)
        w1 = cross2d(V0_2d - V2_2d, P_2d - V2_2d)
        w2 = cross2d(V1_2d - V0_2d, P_2d - V0_2d)

        # Kiểm tra u,v,w >= 0
        in_tri = in_bbox & valid_area & (
            ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
        )

        u = torch.where(in_tri, w0 / area, torch.zeros_like(w0))
        v = torch.where(in_tri, w1 / area, torch.zeros_like(w0))
        w = torch.where(in_tri, 1.0 - u - v, torch.zeros_like(w0))

        # Độ sâu Z_ndc (0=Near, 1=Far)
        Z_hit = u * V0_ndc[...,2].unsqueeze(1) + v * V1_ndc[...,2].unsqueeze(1) + w * V2_ndc[...,2].unsqueeze(1)
        
        # Z-Buffer Filtering: Tìm Z gần nhất (Z_ndc nhỏ nhất)
        Z_inf_hit = torch.where(in_tri, Z_hit, torch.tensor(float('inf'), device=device))
        best_z, hit_idx = torch.min(Z_inf_hit, dim=2) # (B, R)
        valid_ray = best_z < float('inf')

        # Perspective-Correct Interpolation (Euclidean Distance)

        # Flatten để truy cập nhanh
        f_idx = torch.arange(B * R, device=device)
        h_idx_flat = hit_idx.view(-1)
        
        u_hit = u.view(B*R, F)[f_idx, h_idx_flat].view(B, R)
        v_hit = v.view(B*R, F)[f_idx, h_idx_flat].view(B, R)
        w_hit = (1.0 - u_hit - v_hit)

        W0_hit = W0.repeat_interleave(R, dim=0).view(B*R, F)[f_idx, h_idx_flat].view(B, R)
        W1_hit = W1.repeat_interleave(R, dim=0).view(B*R, F)[f_idx, h_idx_flat].view(B, R)
        W2_hit = W2.repeat_interleave(R, dim=0).view(B*R, F)[f_idx, h_idx_flat].view(B, R)

        V0_w_hit = V0_world[hit_idx] # (B, R, 3)
        V1_w_hit = V1_world[hit_idx]
        V2_w_hit = V2_world[hit_idx]

        # Áp dụng công thức 1/W interpolation
        inv_W_interpolated = u_hit / W0_hit + v_hit / W1_hit + w_hit / W2_hit
        W_interpolated = 1.0 / (inv_W_interpolated + 1e-12)

        hit_point_3D = W_interpolated.unsqueeze(-1) * (
            (u_hit / W0_hit).unsqueeze(-1) * V0_w_hit +
            (v_hit / W1_hit).unsqueeze(-1) * V1_w_hit +
            (w_hit / W2_hit).unsqueeze(-1) * V2_w_hit
        )

        dists = torch.norm(hit_point_3D - face_centers[b_start:b_end].unsqueeze(1), dim=-1)
        valid_dists = torch.where(valid_ray, dists, torch.tensor(0.0, device=device))
        
        sdf_batch = valid_dists.sum(dim=1) / (valid_ray.float().sum(dim=1) + 1e-8)
        sdf_values[b_start:b_end] = sdf_batch

    end_time = time.perf_counter()
    print(f"\n Hoàn thành trong {end_time - start_time:.4f} s.")
    return sdf_values.cpu().numpy()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ultimate SDF Rasterizer - Vectorized GPU")
    parser.add_argument("--input_file", type=str, default="data/teapot.obj")
    parser.add_argument("--fov", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    try:
        mesh = trimesh.load(args.input_file, force='mesh')
        mesh.fix_normals()
        mesh.process()
    except Exception as e:
        print(f"Lỗi: {e}")
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
