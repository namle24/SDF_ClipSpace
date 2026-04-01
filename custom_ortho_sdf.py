import trimesh
import numpy as np
import pyvista as pv
import warnings
import time
from tqdm import tqdm

def compute_custom_ortho_sdf(mesh, num_rays=30, batch_size=100):
    """
    SDF Calculation using a fully custom Numpy Vectorized Orthographic Rasterizer.
    No raytracing engines are used. Everything is based on View Matrix, Point-in-Triangle 2D, 
    and Barycentric Depth Interpolation.
    """
    origins = mesh.vertices
    normals = mesh.vertex_normals
    faces = mesh.faces
    N = len(origins)
    F = len(faces)
    
    print(f"Bắt đầu Custom Numpy Rasterizer. Mesh: {N} đỉnh, {F} tam giác.")
    
    # 1. Dynamic Radius: 1% đường chéo Bounding Box
    bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    radius = 0.01 * bbox_diag
    print(f"BBox Diagonal: {bbox_diag:.4f}, Bán kính chùm tia: {radius:.4f}")
    
    # Tọa độ World của tất cả các mặt
    face_vertices_world = origins[faces] # (F, 3, 3)
    
    # Khởi tạo Ma trận kết quả
    dist_matrix = np.full((N, num_rays), np.nan)
    
    # 2. Sinh các tia song song trên Clip Space (hoặc Camera Space)
    # Vì Orthographic nên tia bắn thẳng theo trục Z, tọa độ xy không đổi.
    r_rand = radius * np.sqrt(np.random.uniform(0, 1, size=(N, num_rays)))
    r_rand[:, 0] = 0.0 # Tia trung tâm phải giữ nguyên XY=0
    theta_rand = np.random.uniform(0, 2 * np.pi, size=(N, num_rays))
    
    ray_x = r_rand * np.cos(theta_rand) # (N, num_rays)
    ray_y = r_rand * np.sin(theta_rand) # (N, num_rays)

    # 3. Vectorized Xây dựng Hệ tọa độ Camera cho tất cả các đỉnh
    # Trục Z của Camera hướng thẳng theo chiều bắn tia (-normal)
    anti_normals = -normals
    norms = np.linalg.norm(anti_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    z_cam = anti_normals / norms
    
    # Trục X của Camera
    temp = np.zeros_like(z_cam)
    mask = np.abs(z_cam[:, 0]) > 0.1
    temp[mask] = [0, 1, 0]
    temp[~mask] = [1, 0, 0]
    
    x_cam = np.cross(temp, z_cam)
    x_norms = np.linalg.norm(x_cam, axis=1, keepdims=True)
    x_norms[x_norms == 0] = 1.0
    x_cam = x_cam / x_norms
    
    # Trục Y của Camera
    y_cam = np.cross(z_cam, x_cam)
    
    start_time = time.time()
    
    # 4. Batching Numpy Vectorization: chống tràn RAM cho N x F
    for i in tqdm(range(0, N, batch_size), desc="Rasterization Batches"):
        end = min(i + batch_size, N)
        B = end - i
        
        # Lấy View Rotation (R) và Translation (T) cho mẻ Batch hiện tại
        # Các vector trên là Trục của Camera trong không gian World
        # Ma trận R (World -> Camera) có các hàng chính là các trục này.
        R = np.zeros((B, 3, 3))
        R[:, 0, :] = x_cam[i:end]
        R[:, 1, :] = y_cam[i:end]
        R[:, 2, :] = z_cam[i:end]
        
        b_origins = origins[i:end] # (B, 3)
        
        # Translation T = - R * origin
        T = -np.einsum('bij, bj -> bi', R, b_origins) # (B, 3)
        
        # Transform toàn bộ mesh vào Camera Space của từng đỉnh trong Batch
        # R có shape (B, 3, 3), face_vertices_world có shape (F, 3, 3)
        # Kết quả (B, F, 3, 3): [Batch, Faces, V_index(0,1,2), XYZ(0,1,2)]
        faces_cam = np.einsum('bij, fvj -> bfvi', R, face_vertices_world) + T[:, None, None, :]
        
        # Tọa độ XYZ của v0, v1, v2
        v0 = faces_cam[:, :, 0, :] 
        v1 = faces_cam[:, :, 1, :]
        v2 = faces_cam[:, :, 2, :]
        
        # Tọa độ Z
        v0_z = v0[:, :, 2][:, None, :] # (B, 1, F)
        v1_z = v1[:, :, 2][:, None, :]
        v2_z = v2[:, :, 2][:, None, :]
        
        # Bỏ qua các tam giác nằm hoàn toàn đằng sau Camera (Z < epsilon)
        # Để tăng tốc độ, có thể sinh depth_mask, nhưng vector hóa sẽ tự xử lý
        
        # Tọa độ X, Y để kiểm tra Point-in-Triangle 2D
        v0_x = v0[:, :, 0][:, None, :] # (B, 1, F)
        v0_y = v0[:, :, 1][:, None, :]
        v1_x = v1[:, :, 0][:, None, :]
        v1_y = v1[:, :, 1][:, None, :]
        v2_x = v2[:, :, 0][:, None, :]
        v2_y = v2[:, :, 1][:, None, :]
        
        px = ray_x[i:end][:, :, None] # (B, num_rays, 1)
        py = ray_y[i:end][:, :, None]
        
        # 5. Point-in-Triangle (Edge Function) Vectorized
        w0 = (v2_x - v1_x)*(py - v1_y) - (v2_y - v1_y)*(px - v1_x)
        w1 = (v0_x - v2_x)*(py - v2_y) - (v0_y - v2_y)*(px - v2_x)
        w2 = (v1_x - v0_x)*(py - v0_y) - (v1_y - v0_y)*(px - v0_x)
        
        # Nếu bộ trọng số Barcentric cùng dấu, điểm đó nằm TRONG tam giác
        has_neg = (w0 < 0) | (w1 < 0) | (w2 < 0)
        has_pos = (w0 > 0) | (w1 > 0) | (w2 > 0)
        inside = ~(has_neg & has_pos) # (B, num_rays, F)
        
        # Phương pháp loại bỏ hoàn toàn nhiễu từ Umbrella (các mặt chứa chính đỉnh gốc)
        # faces_expanded lưu Vertex IDs, mặt nào chứa ID của đỉnh đang phóng tia -> Bỏ qua
        vertex_indices = np.arange(i, end)[:, None, None] # (B, 1, 1)
        is_umbrella = np.any(faces[None, :, :] == vertex_indices, axis=2) # (B, F) boolean
        is_umbrella = is_umbrella[:, None, :] # Phát thành (B, 1, F)
        
        # 6. Nội suy Depth bằng Z-Buffer Barycentric interpolation
        # Diện tích Area
        Area = (v1_x - v0_x)*(v2_y - v0_y) - (v1_y - v0_y)*(v2_x - v0_x)
        # Chia w0, w1, w2 cho Area sẽ ra chính xác độ sâu. Dù Diện tích Âm, W cũng âm nên chia nhau ra dương
        Z_hit = (w0 * v0_z + w1 * v1_z + w2 * v2_z) / (Area + 1e-12)
        
        # Validate các tia Hit thành công
        # 1. Nằm trong tam giác
        # 2. Bắn đi về phía trước camera (Z_hit > 1e-4)
        # 3. Diện tích tam giác mặt cắt hợp lệ
        valid = inside & (Z_hit > 1e-4) & (np.abs(Area) > 1e-12) & (~is_umbrella)
        
        Z_hit = np.where(valid, Z_hit, np.inf)
        
        # Tìm tia đâm gần nhất (nearest hit)
        min_Z = np.min(Z_hit, axis=2) # (B, num_rays)
        dist_matrix[i:end, :] = np.where(min_Z == np.inf, np.nan, min_Z)
        
    print(f"Hoàn thành Rasterization toàn bộ tia! Thời gian: {time.time() - start_time:.2f}s")
    
    # 7. Khử nhiễu IQR và Trung bình hóa Vectorized
    print("Đang áp dụng bộ lọc IQR Outlier Rejection và tính SDF...")
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
    
    print(f"SDF trung bình: {np.mean(sdf_values):.4f}")
    return sdf_values

def main():
    model_path = r"C:\Users\Admin\Downloads\grandPiano_V1_L1.123c7f869173-7960-449a-b773-09c1f8708734\grandPiano_V1_L1.123c7f869173-7960-449a-b773-09c1f8708734\10384_GrandPiano.obj"  # Thay tên model tuỳ ý
    try:
        mesh = trimesh.load(model_path, force='mesh')
    except Exception as e:
        print(f"Không thể tải mesh {model_path}. Lỗi: {e}")
        return

    sdf_values = compute_custom_ortho_sdf(mesh, num_rays=30, batch_size=100)

    # Visualization bằng PyVista
    pv_mesh = pv.wrap(mesh)
    pv_mesh.point_data['Custom_Ortho_SDF'] = sdf_values

    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars='Custom_Ortho_SDF', cmap='jet', show_scalar_bar=True)
    plotter.set_background('white')
    plotter.show()

if __name__ == "__main__":
    main()
