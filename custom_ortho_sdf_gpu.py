import trimesh
import numpy as np
import pyvista as pv
import warnings
import time
import torch
from tqdm import tqdm

@torch.no_grad()
def compute_custom_ortho_sdf_gpu(mesh, num_rays=30, batch_size=256):
    """
    SDF Calculation bằng PyTorch Tensor GPU Vectorized Orthographic Rasterizer.
    Tối đa hóa hiệu năng bằng VRAM và CUDA Cores. Giữ nguyên toàn bộ bề mặt lưới (Mesh).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[*] Bắt đầu Custom Numpy Rasterizer. Thiết bị sử dụng: {device}")
    
    # 1. Chuyển toàn bộ dữ liệu Đỉnh + Pháp tuyến sang RAM VAG GPU (Tensor)
    origins = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32, device=device)
    
    mesh_vertices_tensor = origins # Vì ta tính trên chính các đỉnh ban đầu
    faces_tensor = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    
    N = len(origins)
    F = len(faces_tensor)
    print(f"[*] Mesh: {N} đỉnh, {F} tam giác. Đang khởi tạo bộ nhớ GPU...")
    
    # 2. Dynamic Radius: 1% đường chéo Bounding Box
    bounds = torch.tensor(mesh.bounds, dtype=torch.float32, device=device)
    bbox_diag = torch.norm(bounds[1] - bounds[0])
    radius = 0.01 * bbox_diag
    ray_epsilon = 1e-5 * bbox_diag  # Dynamic Epsilon để tránh self-intersection ổn định cho mọi Size Object
    print(f"[*] BBox Diagonal: {bbox_diag:.4f}, Bán kính chùm tia trực giao: {radius:.4f}, Ray Epsilon: {ray_epsilon:.6f}")
    
    face_vertices_world = mesh_vertices_tensor[faces_tensor]
    dist_matrix = torch.full((N, num_rays), float('nan'), device=device)
    
    # 3. Sinh lưới tia trực giao theo hình tròn
    r_rand = radius * torch.sqrt(torch.rand((N, num_rays), device=device))
    r_rand[:, 0] = 0.0 
    theta_rand = torch.rand((N, num_rays), device=device) * 2 * torch.pi
    
    ray_x = r_rand * torch.cos(theta_rand)
    ray_y = r_rand * torch.sin(theta_rand)
    
    # 4. View Camera Matrix cho từng đỉnh
    anti_normals = -normals
    norms = torch.norm(anti_normals, dim=1, keepdim=True)
    norms[norms == 0] = 1.0
    z_cam = anti_normals / norms
    
    temp = torch.zeros_like(z_cam)
    mask = torch.abs(z_cam[:, 0]) > 0.1
    temp[mask] = torch.tensor([0.0, 1.0, 0.0], device=device)
    temp[~mask] = torch.tensor([1.0, 0.0, 0.0], device=device)
    
    x_cam = torch.linalg.cross(temp, z_cam)
    x_norms = torch.norm(x_cam, dim=1, keepdim=True)
    x_norms[x_norms == 0] = 1.0
    x_cam = x_cam / x_norms
    
    y_cam = torch.linalg.cross(z_cam, x_cam)
    
    start_time = time.time()
    
    # 5. Batching xử lý hàng loạt trên CUDA
    for i in tqdm(range(0, N, batch_size), desc="GPU Rasterization Batches"):
        end = min(i + batch_size, N)
        B = end - i
        
        R = torch.zeros((B, 3, 3), device=device)
        R[:, 0, :] = x_cam[i:end]
        R[:, 1, :] = y_cam[i:end]
        R[:, 2, :] = z_cam[i:end]
        
        b_origins = origins[i:end]
        T = -torch.einsum('bij, bj -> bi', R, b_origins)
        
        # Transform Toàn bộ World 3D vào Camera Bằng Matrix PyTorch
        faces_cam = torch.einsum('bij, fvj -> bfvi', R, face_vertices_world) + T[:, None, None, :]
        
        v0 = faces_cam[:, :, 0, :] 
        v1 = faces_cam[:, :, 1, :]
        v2 = faces_cam[:, :, 2, :]
        
        v0_z = v0[:, :, 2].unsqueeze(1)
        v1_z = v1[:, :, 2].unsqueeze(1)
        v2_z = v2[:, :, 2].unsqueeze(1)
        
        v0_x = v0[:, :, 0].unsqueeze(1)
        v0_y = v0[:, :, 1].unsqueeze(1)
        v1_x = v1[:, :, 0].unsqueeze(1)
        v1_y = v1[:, :, 1].unsqueeze(1)
        v2_x = v2[:, :, 0].unsqueeze(1)
        v2_y = v2[:, :, 1].unsqueeze(1)
        
        px = ray_x[i:end].unsqueeze(2)
        py = ray_y[i:end].unsqueeze(2)
        
        # Point-in-Triangle 2D
        w0 = (v2_x - v1_x)*(py - v1_y) - (v2_y - v1_y)*(px - v1_x)
        w1 = (v0_x - v2_x)*(py - v2_y) - (v0_y - v2_y)*(px - v2_x)
        w2 = (v1_x - v0_x)*(py - v0_y) - (v1_y - v0_y)*(px - v0_x)
        
        has_neg = (w0 < 0) | (w1 < 0) | (w2 < 0)
        has_pos = (w0 > 0) | (w1 > 0) | (w2 > 0)
        inside = ~(has_neg & has_pos)
        
        # Khử Umbrella Faces (Che ô): Các bề mặt chứa bản thân của đỉnh đó
        vertex_indices = torch.arange(i, end, device=device).unsqueeze(1).unsqueeze(2) # (B, 1, 1)
        faces_expanded = faces_tensor.unsqueeze(0) # (1, F, 3)
        # Kiểm tra xem ID đỉnh chiếu phát đi có nằm trong 3 đỉnh ID hình thành nên mặt đó ko
        is_umbrella = torch.any(faces_expanded == vertex_indices, dim=2) # (B, F)
        is_umbrella = is_umbrella.unsqueeze(1) # Phát lên (B, 1, F)
        
        # 6. Nội suy Depth Z bằng toạ độ Barycentric (Z-Buffer Rasterizer)
        Area = (v1_x - v0_x)*(v2_y - v0_y) - (v1_y - v0_y)*(v2_x - v0_x)
        Z_hit = (w0 * v0_z + w1 * v1_z + w2 * v2_z) / (Area + 1e-12)
        
        # Kiểm định tia bắn hợp lệ (Nằm trong mặt, Phóng đi tới khoảng an toàn epsilon, Mặt ko suy biến, Mặt ko phải do mặt chứa đỉnh gốc)
        valid = inside & (Z_hit > ray_epsilon) & (torch.abs(Area) > 1e-12) & (~is_umbrella)
        Z_hit = torch.where(valid, Z_hit, torch.tensor(float('inf'), device=device))
        
        # Ray Tracing Tìm khoảng cách đâm xuyên gần nhất (Nearest hit Z-buffer)
        min_Z, _ = torch.min(Z_hit, dim=2)
        dist_matrix[i:end, :] = torch.where(torch.isinf(min_Z), torch.tensor(float('nan'), device=device), min_Z)
        
    print(f"[*] Hoàn thành Tensor Rasterization toàn bộ tia! Thời gian: {time.time() - start_time:.4f}s")
    
    # 7. Khử nhiễu Outlier IQR hoàn toàn bằng GPU PyTorch (100% Vectorized GPU)
    print("[*] Áp dụng lọc Outlier IQR bằng PyTorch Vectorized GPU...")
    
    # Nếu hàng full NaN, torch.nanquantile trả NaN tự động. Filter runtime warning ko cần thiết.
    q1 = torch.nanquantile(dist_matrix, 0.25, dim=1)
    q3 = torch.nanquantile(dist_matrix, 0.75, dim=1)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    valid_mask = (dist_matrix >= lower_bound.unsqueeze(1)) & (dist_matrix <= upper_bound.unsqueeze(1))
    
    filtered_dist_matrix = torch.where(valid_mask, dist_matrix, torch.tensor(float('nan'), device=device))
    
    # Tính mean và fill NaNs
    sdf_tensor = torch.nanmean(filtered_dist_matrix, dim=1)
    sdf_tensor = torch.nan_to_num(sdf_tensor, nan=0.0)
    
    # Đẩy lên CPU để PyVista có thể dùng làm scalars
    sdf_values = sdf_tensor.cpu().numpy()
    
    print(f"[*] SDF Trung bình toàn bộ đối tượng: {np.mean(sdf_values):.4f}")
    return sdf_values

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Custom Orthographic VO-SDF bằng Pytorch GPU để Visualization vùng mượt")
    parser.add_argument("--input_file", type=str, default="bunny.obj", help="Tên file hoặc Đường dẫn model 3D để render SDF lên bề mặt")
    parser.add_argument("--num_rays", type=int, default=30, help="Số tia bắn (chùm nón trực giao)")
    parser.add_argument("--batch_size", type=int, default=256, help="GPU Batch Size - Khuyến cáo <= 256 để tránh OOM.")
    args = parser.parse_args()
    
    model_path = args.input_file
    try:
        mesh = trimesh.load(model_path, force='mesh')
        if len(mesh.vertices) == 0:
            raise ValueError("Đồ thị (Mesh) trống hoặc hỏng.")
    except Exception as e:
        print(f"Không thể tải mesh {model_path}. Lỗi: {e}")
        return

    # Chạy Hàm SDF GPU
    t1 = time.time()
    sdf_values = compute_custom_ortho_sdf_gpu(mesh, num_rays=args.num_rays, batch_size=args.batch_size)
    print(f"--- TỔNG THỜI GIAN GPU: {time.time() - t1:.4f}s ---")

    # Hiển thị vùng màu mượt bằng PyVista (Mesh Render)
    pv_mesh = pv.wrap(mesh)
    pv_mesh.point_data['SDF_Thickness'] = sdf_values

    plotter = pv.Plotter(title="SDF Visualization - MeshLab Style")
    # Sử dụng 'jet_r' (Red-Yellow-Blue reversed) để giống MeshLab: Đỏ là mỏng, Xanh là dày
    plotter.add_mesh(
        pv_mesh, 
        scalars='SDF_Thickness', 
        cmap='jet_r', 
        smooth_shading=True,   # Giúp vùng màu trông mượt hơn
        show_edges=False,      # Tắt lưới để nhìn rõ vùng màu như MeshLab
        scalar_bar_args={'title': "SDF (Thickness)"}
    )
    plotter.set_background('white')
    # Thêm hướng nhìn trục tọa độ nhỏ ở góc
    plotter.add_axes()
    plotter.show()

if __name__ == "__main__":
    main()
