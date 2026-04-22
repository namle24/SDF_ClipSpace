import os
import glob
import time
import warnings
import numpy as np
import trimesh
from tqdm import tqdm

class ModelNetSDFProcessor:
    def __init__(self, input_dir="ModelNet40", output_dir="ModelNet40_SDF_1024", num_points=1024, num_rays=30, batch_size=100):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_points = num_points
        self.num_rays = num_rays
        self.batch_size = batch_size
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def sample_points_on_mesh(self, mesh):
        """
        Lấy mẫu ngẫu nhiên num_points điểm trên bề mặt mesh và nội suy pháp tuyến
        Sử dụng trimesh.sample.sample_surface
        """
        points, face_indices = trimesh.sample.sample_surface(mesh, self.num_points)
        
        # Lấy pháp tuyến của mặt cho các điểm tương ứng. 
        normals = mesh.face_normals[face_indices]
        
        # Nếu normals có độ dài bằng 0 (mặt suy biến), thay bằng vector mặc định
        norms = np.linalg.norm(normals, axis=1)
        zero_masks = norms == 0
        if np.any(zero_masks):
            normals[zero_masks] = np.array([0.0, 0.0, 1.0])
            norms[zero_masks] = 1.0
        
        normals = normals / norms[:, None]
        
        return points, normals, face_indices

    def compute_sdf(self, mesh, points, normals, face_indices):
        """
        Tính toán SDF sử dụng VO-SDF (Numpy Vectorized Orthographic Rasterizer) 
        chỉ cho số lượng điểm được chỉ định.
        """
        origins = points
        N = len(origins)
        
        mesh_vertices = mesh.vertices
        faces = mesh.faces
        F = len(faces)
        
        # 1. Dynamic Radius: 1% đường chéo Bounding Box
        bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
        radius = 0.01 * bbox_diag
        
        face_vertices_world = mesh_vertices[faces]
        dist_matrix = np.full((N, self.num_rays), np.nan)
        
        # 2. Sinh các tia song song trên Clip Space
        r_rand = radius * np.sqrt(np.random.uniform(0, 1, size=(N, self.num_rays)))
        r_rand[:, 0] = 0.0 
        theta_rand = np.random.uniform(0, 2 * np.pi, size=(N, self.num_rays))
        
        ray_x = r_rand * np.cos(theta_rand)
        ray_y = r_rand * np.sin(theta_rand)
        
        anti_normals = -normals
        norms = np.linalg.norm(anti_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        z_cam = anti_normals / norms
        
        temp = np.zeros_like(z_cam)
        mask = np.abs(z_cam[:, 0]) > 0.1
        temp[mask] = [0, 1, 0]
        temp[~mask] = [1, 0, 0]
        
        x_cam = np.cross(temp, z_cam)
        x_norms = np.linalg.norm(x_cam, axis=1, keepdims=True)
        x_norms[x_norms == 0] = 1.0
        x_cam = x_cam / x_norms
        
        y_cam = np.cross(z_cam, x_cam)
        
        # 4. Batching Numpy Vectorization
        for i in range(0, N, self.batch_size):
            end = min(i + self.batch_size, N)
            B = end - i
            
            R = np.zeros((B, 3, 3))
            R[:, 0, :] = x_cam[i:end]
            R[:, 1, :] = y_cam[i:end]
            R[:, 2, :] = z_cam[i:end]
            
            b_origins = origins[i:end]
            T = -np.einsum('bij, bj -> bi', R, b_origins)
            
            faces_cam = np.einsum('bij, fvj -> bfvi', R, face_vertices_world) + T[:, None, None, :]
            
            v0 = faces_cam[:, :, 0, :] 
            v1 = faces_cam[:, :, 1, :]
            v2 = faces_cam[:, :, 2, :]
            
            v0_z = v0[:, :, 2][:, None, :]
            v1_z = v1[:, :, 2][:, None, :]
            v2_z = v2[:, :, 2][:, None, :]
            
            v0_x = v0[:, :, 0][:, None, :]
            v0_y = v0[:, :, 1][:, None, :]
            v1_x = v1[:, :, 0][:, None, :]
            v1_y = v1[:, :, 1][:, None, :]
            v2_x = v2[:, :, 0][:, None, :]
            v2_y = v2[:, :, 1][:, None, :]
            
            px = ray_x[i:end][:, :, None]
            py = ray_y[i:end][:, :, None]
            
            # 5. Point-in-Triangle (Edge Function)
            w0 = (v2_x - v1_x)*(py - v1_y) - (v2_y - v1_y)*(px - v1_x)
            w1 = (v0_x - v2_x)*(py - v2_y) - (v0_y - v2_y)*(px - v2_x)
            w2 = (v1_x - v0_x)*(py - v0_y) - (v1_y - v0_y)*(px - v0_x)
            
            has_neg = (w0 < 0) | (w1 < 0) | (w2 < 0)
            has_pos = (w0 > 0) | (w1 > 0) | (w2 > 0)
            inside = ~(has_neg & has_pos)
            
            # Khử nhiễu: Bỏ qua tam giác mà điểm được sample ra (Umbrella Filter)
            b_face_indices = face_indices[i:end]
            is_umbrella = (np.arange(F)[None, :] == b_face_indices[:, None])
            is_umbrella = is_umbrella[:, None, :] # B, 1, F
            
            # 6. Nội suy Depth bằng Z-Buffer Barycentric interpolation
            Area = (v1_x - v0_x)*(v2_y - v0_y) - (v1_y - v0_y)*(v2_x - v0_x)
            Z_hit = (w0 * v0_z + w1 * v1_z + w2 * v2_z) / (Area + 1e-12)
            
            valid = inside & (Z_hit > 1e-4) & (np.abs(Area) > 1e-12) & (~is_umbrella)
            Z_hit = np.where(valid, Z_hit, np.inf)
            
            min_Z = np.min(Z_hit, axis=2)
            dist_matrix[i:end, :] = np.where(min_Z == np.inf, np.nan, min_Z)
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            q1 = np.nanpercentile(dist_matrix, 25, axis=1)
            q3 = np.nanpercentile(dist_matrix, 75, axis=1)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            valid_mask = (dist_matrix >= lower_bound[:, None]) & (dist_matrix <= upper_bound[:, None])
            filtered_dist_matrix = np.where(valid_mask, dist_matrix, np.nan)
            
            sdf_values = np.nanmean(filtered_dist_matrix, axis=1)
            sdf_values = np.nan_to_num(sdf_values, nan=0.0)
            
        return sdf_values

    def process_file(self, file_path):
        """
        Xử lý 1 file 3D, lấy output lưu thành .npy format (N, 4)
        """
        rel_path = os.path.relpath(file_path, self.input_dir)
        output_file_path = os.path.join(self.output_dir, rel_path)
        output_file_path = os.path.splitext(output_file_path)[0] + '.npy'
        
        output_dir_path = os.path.dirname(output_file_path)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path, exist_ok=True)
            
        if os.path.exists(output_file_path):
            return True # Đã xử lý (Skip)
            
        try:
            # 1. Load file an toàn
            mesh = trimesh.load(file_path, force='mesh')
            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                print(f"[Cảnh báo] MESH rỗng hoặc lỗi cấu trúc: {file_path}")
                return False
                
            # 2. Sample 1024 điểm
            points, normals, face_indices = self.sample_points_on_mesh(mesh)
            
            # 3. Tính toán phương pháp VO-SDF
            sdf_values = self.compute_sdf(mesh, points, normals, face_indices)
            
            # 4. Lưu .npy shape (1024, 4) (X, Y, Z, SDF)
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
        
        # Hỗ trợ đếm cả file .off và .obj
        files = glob.glob(search_pattern_off, recursive=True) + glob.glob(search_pattern_obj, recursive=True)
        
        if len(files) == 0:
            print("Không tìm thấy file .off hoặc .obj nào trong thư mục đầu vào!")
            return
            
        print(f"Tổng số file cần xử lý: {len(files)}")
        
        success_count = 0
        error_count = 0
        skip_count = 0
        
        for file_path in tqdm(files, desc="Batch Processing ModelNet40 SDF"):
            # Kiểm tra skip
            rel_path = os.path.relpath(file_path, self.input_dir)
            out_path = os.path.join(self.output_dir, rel_path)
            out_path = os.path.splitext(out_path)[0] + '.npy'
            
            if os.path.exists(out_path):
                # tqdm skip output without executing process_file completely to keep progress fast
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
    parser = argparse.ArgumentParser(description="ModelNet40 VO-SDF Data Preprocessing Pipeline")
    parser.add_argument("--input_dir", type=str, default="ModelNet40", help="Đường dẫn đến thư mục root của ModelNet40")
    parser.add_argument("--output_dir", type=str, default="ModelNet40_SDF_1024", help="Thư mục xuất file .npy")
    parser.add_argument("--num_points", type=int, default=1024, help="Số điểm lấy mẫu")
    parser.add_argument("--num_rays", type=int, default=30, help="Số lượng tia bắn cho mỗi điểm")
    parser.add_argument("--batch_size", type=int, default=100, help="Kích thước batch Vectorization để tối ưu RAM")
    
    args = parser.parse_args()
    
    processor = ModelNetSDFProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_points=args.num_points,
        num_rays=args.num_rays,
        batch_size=args.batch_size
    )
    processor.run()
