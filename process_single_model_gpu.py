import os
import time
import warnings
import numpy as np
import trimesh
import torch

class SingleModelSDFProcessorGPU:
    def __init__(self, num_points=1024, num_rays=30, batch_size=1024):
        self.num_points = num_points
        self.num_rays = num_rays
        self.batch_size = batch_size
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[*] Khởi tạo VO-SDF Processor. Thiết bị sử dụng: {self.device}")

    def sample_points_on_mesh(self, mesh):
        points, face_indices = trimesh.sample.sample_surface(mesh, self.num_points)
        normals = mesh.face_normals[face_indices]
        
        norms = np.linalg.norm(normals, axis=1)
        zero_masks = norms == 0
        if np.any(zero_masks):
            normals[zero_masks] = np.array([0.0, 0.0, 1.0])
            normals[zero_masks] = 1.0
        
        normals = normals / norms[:, None]
        return points, normals, face_indices

    @torch.no_grad()
    def compute_sdf(self, mesh, points, normals, face_indices):
        origins = torch.tensor(points, dtype=torch.float32, device=self.device)
        anti_normals = -torch.tensor(normals, dtype=torch.float32, device=self.device)
        
        mesh_vertices_tensor = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)
        faces_tensor = torch.tensor(mesh.faces, dtype=torch.long, device=self.device)
        face_indices_tensor = torch.tensor(face_indices, dtype=torch.long, device=self.device)
        
        N = len(origins)
        F = len(faces_tensor)
        
        bounds = torch.tensor(mesh.bounds, dtype=torch.float32, device=self.device)
        bbox_diag = torch.norm(bounds[1] - bounds[0])
        radius = 0.01 * bbox_diag
        
        face_vertices_world = mesh_vertices_tensor[faces_tensor]
        dist_matrix = torch.full((N, self.num_rays), float('nan'), device=self.device)
        
        r_rand = radius * torch.sqrt(torch.rand((N, self.num_rays), device=self.device))
        r_rand[:, 0] = 0.0 
        theta_rand = torch.rand((N, self.num_rays), device=self.device) * 2 * torch.pi
        
        ray_x = r_rand * torch.cos(theta_rand)
        ray_y = r_rand * torch.sin(theta_rand)
        
        norms = torch.norm(anti_normals, dim=1, keepdim=True)
        norms[norms == 0] = 1.0
        z_cam = anti_normals / norms
        
        temp = torch.zeros_like(z_cam)
        mask = torch.abs(z_cam[:, 0]) > 0.1
        temp[mask] = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        temp[~mask] = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        
        x_cam = torch.linalg.cross(temp, z_cam)
        x_norms = torch.norm(x_cam, dim=1, keepdim=True)
        x_norms[x_norms == 0] = 1.0
        x_cam = x_cam / x_norms
        
        y_cam = torch.linalg.cross(z_cam, x_cam)
        
        for i in range(0, N, self.batch_size):
            end = min(i + self.batch_size, N)
            B = end - i
            
            R = torch.zeros((B, 3, 3), device=self.device)
            R[:, 0, :] = x_cam[i:end]
            R[:, 1, :] = y_cam[i:end]
            R[:, 2, :] = z_cam[i:end]
            
            b_origins = origins[i:end]
            T = -torch.einsum('bij, bj -> bi', R, b_origins)
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
            
            w0 = (v2_x - v1_x)*(py - v1_y) - (v2_y - v1_y)*(px - v1_x)
            w1 = (v0_x - v2_x)*(py - v2_y) - (v0_y - v2_y)*(px - v2_x)
            w2 = (v1_x - v0_x)*(py - v0_y) - (v1_y - v0_y)*(px - v0_x)
            
            has_neg = (w0 < 0) | (w1 < 0) | (w2 < 0)
            has_pos = (w0 > 0) | (w1 > 0) | (w2 > 0)
            inside = ~(has_neg & has_pos)
            
            b_face_indices = face_indices_tensor[i:end]
            is_umbrella = (torch.arange(F, device=self.device).unsqueeze(0) == b_face_indices.unsqueeze(1))
            is_umbrella = is_umbrella.unsqueeze(1)
            
            Area = (v1_x - v0_x)*(v2_y - v0_y) - (v1_y - v0_y)*(v2_x - v0_x)
            Z_hit = (w0 * v0_z + w1 * v1_z + w2 * v2_z) / (Area + 1e-12)
            
            valid = inside & (Z_hit > 1e-4) & (torch.abs(Area) > 1e-12) & (~is_umbrella)
            Z_hit = torch.where(valid, Z_hit, torch.tensor(float('inf'), device=self.device))
            
            min_Z, _ = torch.min(Z_hit, dim=2)
            dist_matrix[i:end, :] = torch.where(min_Z == float('inf'), torch.tensor(float('nan'), device=self.device), min_Z)
            
        dist_matrix_np = dist_matrix.cpu().numpy()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            q1 = np.nanpercentile(dist_matrix_np, 25, axis=1)
            q3 = np.nanpercentile(dist_matrix_np, 75, axis=1)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            valid_mask = (dist_matrix_np >= lower_bound[:, None]) & (dist_matrix_np <= upper_bound[:, None])
            filtered_dist_matrix = np.where(valid_mask, dist_matrix_np, np.nan)
            
            sdf_values = np.nanmean(filtered_dist_matrix, axis=1)
            sdf_values = np.nan_to_num(sdf_values, nan=0.0)
            
        return sdf_values

    def process_file(self, file_path, output_file_path=None):
        if not os.path.exists(file_path):
            print(f"[LỖI] Không tìm thấy file đầu vào: {file_path}")
            return False
            
        if output_file_path is None:
            output_file_path = os.path.splitext(file_path)[0] + "_sdf_1024.npy"
            
        output_dir_path = os.path.dirname(os.path.abspath(output_file_path))
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path, exist_ok=True)
            
        print(f"\n--- Bắt đầu xử lý file: {os.path.basename(file_path)} ---")
        total_start_time = time.time()
            
        try:
            # 1. Đo thời gian tải Mesh
            t0 = time.time()
            mesh = trimesh.load(file_path, force='mesh')
            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                print(f"[LỖI] Mesh rỗng hoặc bị hỏng.")
                return False
            t1 = time.time()
            print(f"[1/4] Tải Mesh ({len(mesh.vertices)} đỉnh, {len(mesh.faces)} mặt): {t1 - t0:.4f} giây")
                
            # 2. Đo thời gian Sample
            points, normals, face_indices = self.sample_points_on_mesh(mesh)
            t2 = time.time()
            print(f"[2/4] Lấy ngẫu nhiên {self.num_points} điểm + Pháp tuyến: {t2 - t1:.4f} giây")
            
            # 3. Đo thời gian GPU SDF Rasterization
            print(f"[3/4] Chạy thuật toán VO-SDF bằng GPU PyTorch Tensor...")
            sdf_values = self.compute_sdf(mesh, points, normals, face_indices)
            t3 = time.time()
            print(f"      -> Hoàn tất Vectorized SDF: {t3 - t2:.4f} giây")
            
            # 4. Đo thời gian Lưu file
            output_data = np.hstack((points, sdf_values[:, None]))
            np.save(output_file_path, output_data)
            t4 = time.time()
            print(f"[4/4] Ghép Tensor ({self.num_points}, 4) và lưu thành file .npy: {t4 - t3:.4f} giây")
            
            print(f"--- TỔNG THỜI GIAN THỰC THI: {t4 - total_start_time:.4f} giây ---")
            print(f"[*] File đầu ra lưu tại: {output_file_path}")
            
            # 5. Hiển thị kết quả trực quan
            print(f"[*] Đang dựng hình điểm ảnh 3D để xem trước kết quả...")
            import pyvista as pv
            cloud = pv.PolyData(points)
            cloud.point_data["VO-SDF"] = sdf_values
            
            plotter = pv.Plotter(title="ModelNet VO-SDF 1024 Points Visualization")
            plotter.add_mesh(cloud, scalars="VO-SDF", cmap="jet", point_size=15, render_points_as_spheres=True, show_scalar_bar=True)
            
            # Thêm lớp Mesh mờ bên trong lót nền
            pv_mesh = pv.wrap(mesh)
            plotter.add_mesh(pv_mesh, color='white', opacity=0.15, show_edges=True)
            
            plotter.set_background('white')
            plotter.show()
            
            return True
            
        except Exception as e:
            print(f"[CẢNH BÁO] Lỗi không lường trước: {e}")
            return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Xử lý Đơn tệp VO-SDF GPU với Log Chi tiết")
    parser.add_argument("--input_file", type=str, required=True, help="Đường dẫn đến file 3D (.obj, .off, v.v.)")
    parser.add_argument("--output_file", type=str, default=None, help="Đường dẫn xuất file .npy (Mặc định: cùng chỗ file 3D)")
    parser.add_argument("--num_points", type=int, default=1024, help="Số điểm lấy mẫu")
    parser.add_argument("--num_rays", type=int, default=30, help="Số lượng tia bắn cho mỗi điểm")
    parser.add_argument("--batch_size", type=int, default=1024, help="Kích thước batch Vectorization cho GPU")
    
    args = parser.parse_args()
    
    processor = SingleModelSDFProcessorGPU(
        num_points=args.num_points,
        num_rays=args.num_rays,
        batch_size=args.batch_size
    )
    processor.process_file(args.input_file, args.output_file)
