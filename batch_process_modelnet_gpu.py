import os
import glob
import time
import warnings
import numpy as np
import trimesh
import torch
from tqdm import tqdm

class ModelNetSDFProcessorGPU:
    def __init__(self, input_dir="ModelNet40", output_dir="ModelNet40_SDF_1024", num_points=1024, num_rays=30, batch_size=256):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_points = num_points
        self.num_rays = num_rays
        self.batch_size = batch_size # Max 256 để chống tràn VRAM khi Face > 100k
        
        # Chọn Device ưu tiên GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[*] Khởi tạo VO-SDF Processor. Thiết bị sử dụng: {self.device}")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def sample_points_on_mesh(self, mesh):
        """
        Lấy mẫu ngẫu nhiên num_points điểm trên bề mặt mesh và nội suy pháp tuyến
        Khâu này sử dụng CPU (trimesh) vì nó rất nhẹ so với phần Ray Rasterization.
        """
        points, face_indices = trimesh.sample.sample_surface(mesh, self.num_points)
        
        normals = mesh.face_normals[face_indices]
        
        norms = np.linalg.norm(normals, axis=1)
        zero_masks = norms == 0
        if np.any(zero_masks):
            normals[zero_masks] = np.array([0.0, 0.0, 1.0])
            norms[zero_masks] = 1.0
        
        normals = normals / norms[:, None]
        return points, normals, face_indices

    @torch.no_grad() # Tối ưu tối đa vì không cần tính gradient backprop
    def compute_sdf(self, mesh, points, normals, face_indices):
        """
        Tính toán SDF sử dụng VO-SDF PyTorch Tensor Cuda.
        100% Vectorized GPU Rasterizer - Hiệu suất siêu tốc.
        """
        # --- Di chuyển Dữ liệu Input vào VRAM Device ---
        origins = torch.tensor(points, dtype=torch.float32, device=self.device)
        anti_normals = -torch.tensor(normals, dtype=torch.float32, device=self.device)
        
        mesh_vertices_tensor = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)
        faces_tensor = torch.tensor(mesh.faces, dtype=torch.long, device=self.device)
        face_indices_tensor = torch.tensor(face_indices, dtype=torch.long, device=self.device)
        
        N = len(origins)
        F = len(faces_tensor)
        
        # --- 1. Dynamic Radius: 1% đường chéo Bounding Box ---
        bounds = torch.tensor(mesh.bounds, dtype=torch.float32, device=self.device)
        bbox_diag = torch.norm(bounds[1] - bounds[0])
        radius = 0.01 * bbox_diag
        ray_epsilon = 1e-5 * bbox_diag
        
        face_vertices_world = mesh_vertices_tensor[faces_tensor]
        dist_matrix = torch.full((N, self.num_rays), float('nan'), device=self.device)
        
        # --- 2. Sinh các tia song song trên Clip Space ---
        r_rand = radius * torch.sqrt(torch.rand((N, self.num_rays), device=self.device))
        r_rand[:, 0] = 0.0 
        theta_rand = torch.rand((N, self.num_rays), device=self.device) * 2 * torch.pi
        
        ray_x = r_rand * torch.cos(theta_rand)
        ray_y = r_rand * torch.sin(theta_rand)
        
        # --- 3. Xây dựng Camera Axis bằng Broadcasting Cross Product ---
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
        
        # --- 4. Hashing Batches trên GPU ---
        # Tuỳ thuộc vào RAM GPU (VD: RTX 3090/A100), ta có thể đẩy batch size lên rất cao
        for i in range(0, N, self.batch_size):
            end = min(i + self.batch_size, N)
            B = end - i
            
            R = torch.zeros((B, 3, 3), device=self.device)
            R[:, 0, :] = x_cam[i:end]
            R[:, 1, :] = y_cam[i:end]
            R[:, 2, :] = z_cam[i:end]
            
            b_origins = origins[i:end]
            
            # View Transformation
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
            
            # --- 5. Point-in-Triangle (Edge Function 2D) ---
            w0 = (v2_x - v1_x)*(py - v1_y) - (v2_y - v1_y)*(px - v1_x)
            w1 = (v0_x - v2_x)*(py - v2_y) - (v0_y - v2_y)*(px - v2_x)
            w2 = (v1_x - v0_x)*(py - v0_y) - (v1_y - v0_y)*(px - v0_x)
            
            has_neg = (w0 < 0) | (w1 < 0) | (w2 < 0)
            has_pos = (w0 > 0) | (w1 > 0) | (w2 > 0)
            inside = ~(has_neg & has_pos)
            
            # Lọc nhiễu cục bộ: Umbrella Filter
            b_face_indices = face_indices_tensor[i:end]
            is_umbrella = (torch.arange(F, device=self.device).unsqueeze(0) == b_face_indices.unsqueeze(1))
            is_umbrella = is_umbrella.unsqueeze(1) # B, 1, F
            
            # --- 6. Barycentric Depth Mapping ---
            Area = (v1_x - v0_x)*(v2_y - v0_y) - (v1_y - v0_y)*(v2_x - v0_x)
            Z_hit = (w0 * v0_z + w1 * v1_z + w2 * v2_z) / (Area + 1e-12)
            
            valid = inside & (Z_hit > ray_epsilon) & (torch.abs(Area) > 1e-12) & (~is_umbrella)
            Z_hit = torch.where(valid, Z_hit, torch.tensor(float('inf'), device=self.device))
            
            # Nearest depth selection
            min_Z, _ = torch.min(Z_hit, dim=2)
            dist_matrix[i:end, :] = torch.where(torch.isinf(min_Z), torch.tensor(float('nan'), device=self.device), min_Z)
            
        # --- 7. Statistical IQR Outlier Removal ---
        # --- 7. Statistical IQR Outlier Removal Pytorch GPU ---
        q1 = torch.nanquantile(dist_matrix, 0.25, dim=1)
        q3 = torch.nanquantile(dist_matrix, 0.75, dim=1)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        valid_mask = (dist_matrix >= lower_bound.unsqueeze(1)) & (dist_matrix <= upper_bound.unsqueeze(1))
        filtered_dist_matrix = torch.where(valid_mask, dist_matrix, torch.tensor(float('nan'), device=self.device))
        
        sdf_tensor = torch.nanmean(filtered_dist_matrix, dim=1)
        sdf_tensor = torch.nan_to_num(sdf_tensor, nan=0.0)
        
        sdf_values = sdf_tensor.cpu().numpy()
        return sdf_values

    def process_file(self, file_path):
        rel_path = os.path.relpath(file_path, self.input_dir)
        output_file_path = os.path.join(self.output_dir, rel_path)
        output_file_path = os.path.splitext(output_file_path)[0] + '.npy'
        
        output_dir_path = os.path.dirname(output_file_path)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path, exist_ok=True)
            
        if os.path.exists(output_file_path):
            return True 
            
        try:
            mesh = trimesh.load(file_path, force='mesh')
            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                print(f"[Cảnh báo] MESH rỗng hoặc lỗi cấu trúc: {file_path}")
                return False
                
            points, normals, face_indices = self.sample_points_on_mesh(mesh)
            
            # Tính toán SDF trên Pytorch GPU
            sdf_values = self.compute_sdf(mesh, points, normals, face_indices)
            
            output_data = np.hstack((points, sdf_values[:, None]))
            
            np.save(output_file_path, output_data)
            return True
            
        except Exception as e:
            print(f"[Cảnh báo] Ngoại lệ khi xử lý file {file_path}: {e}")
            return False

    def run(self):
        print(f"Bắt đầu quét thư mục {self.input_dir}...")
        
        search_pattern_off = os.path.join(self.input_dir, '**', '*.off')
        search_pattern_obj = os.path.join(self.input_dir, '**', '*.obj')
        
        files = glob.glob(search_pattern_off, recursive=True) + glob.glob(search_pattern_obj, recursive=True)
        
        if len(files) == 0:
            print("Không tìm thấy file .off hoặc .obj nào trong thư mục đầu vào!")
            return
            
        print(f"Tổng số file cần xử lý: {len(files)}")
        
        success_count = 0
        error_count = 0
        skip_count = 0
        
        for file_path in tqdm(files, desc="Batch Processing ModelNet40 SDF (GPU)"):
            rel_path = os.path.relpath(file_path, self.input_dir)
            out_path = os.path.join(self.output_dir, rel_path)
            out_path = os.path.splitext(out_path)[0] + '.npy'
            
            if os.path.exists(out_path):
                skip_count += 1
                continue
                
            if self.process_file(file_path):
                success_count += 1
            else:
                error_count += 1
                
        print("\n--- TỔNG KẾT ---")
        print(f"Số file bỏ qua (đã có kết quả): {skip_count}")
        print(f"Số file xử lý thành công     : {success_count}")
        print(f"Số file gặp lỗi              : {error_count}")
        print(f"Tổng                        : {len(files)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPU-Accelerated ModelNet40 VO-SDF Data Preprocessing")
    parser.add_argument("--input_dir", type=str, default="ModelNet40", help="Đường dẫn đến thư mục root của ModelNet40")
    parser.add_argument("--output_dir", type=str, default="ModelNet40_SDF_1024", help="Thư mục xuất file .npy")
    parser.add_argument("--num_points", type=int, default=1024, help="Số điểm lấy mẫu")
    parser.add_argument("--num_rays", type=int, default=30, help="Số lượng tia bắn cho mỗi điểm")
    # Tăng Default batch_size lên 256 để chống tràn VRAM
    parser.add_argument("--batch_size", type=int, default=256, help="Kích thước batch Vectorization cho GPU")
    
    args = parser.parse_args()
    
    processor = ModelNetSDFProcessorGPU(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_points=args.num_points,
        num_rays=args.num_rays,
        batch_size=args.batch_size
    )
    processor.run()
