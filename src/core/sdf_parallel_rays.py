import trimesh
import numpy as np
import pyvista as pv
from tqdm import tqdm
import warnings

def compute_sdf_parallel(mesh, num_rays=30):
    origins = mesh.vertices
    normals = mesh.vertex_normals
    eps = 1e-4
    N = len(origins)
    
    # 1. Dynamic Radius: 1% đường chéo Bounding Box
    bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    radius = 0.01 * bbox_diag
    
    print(f"BBox Diagonal: {bbox_diag:.4f}, Dynamic Radius: {radius:.4f}")
    
    # 2. Vectorized Ray Generation (Sinh tia song song siêu nhanh bằng Numpy)
    print("Đang tạo chùm tia SONG SONG (Orthographic) siêu nhanh bằng Numpy...")
    
    # Hướng ngược pháp tuyến cho mỗi đỉnh
    anti_normals = -normals
    norms = np.linalg.norm(anti_normals, axis=1, keepdims=True)
    # Tránh chia cho 0
    norms[norms == 0] = 1.0
    w = anti_normals / norms
    
    # Tạo hệ tọa độ cục bộ (u, v) trên mặt phẳng Disk mặt cắt tia
    temp = np.zeros_like(w)
    mask = np.abs(w[:, 0]) > 0.1
    temp[mask] = [0, 1, 0]
    temp[~mask] = [1, 0, 0]
    
    u = np.cross(temp, w)
    u_norms = np.linalg.norm(u, axis=1, keepdims=True)
    u_norms[u_norms == 0] = 1.0
    u = u / u_norms
    v = np.cross(w, u)
    
    # Sinh thông số r, theta ngẫu nhiên trên đĩa tròn
    r = radius * np.sqrt(np.random.uniform(0, 1, size=(N, num_rays)))
    r[:, 0] = 0.0  # Tia gốc đầu tiên (trung tâm) không dịch chuyển
    theta = np.random.uniform(0, 2 * np.pi, size=(N, num_rays))
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Disk Origins: Mảng 3D shape (N, num_rays, 3) tính tọa độ dịch chuyển của disk
    disk_origins = origins[:, None, :] + x[:, :, None] * u[:, None, :] + y[:, :, None] * v[:, None, :]
    
    # Thụt tia vào trong bề mặt bằng epsilon, đề phòng tự va chạm. R(t) = O_safe + t.D
    safe_origins = disk_origins - normals[:, None, :] * eps
    
    # Trải phẳng thành mảng 2D cho raytracer
    all_ray_origins = safe_origins.reshape(-1, 3)
    # Hướng của từng tia vẫn hoàn toàn song song nhau trong chùm
    all_ray_directions = np.repeat(w, num_rays, axis=0)
    
    print(f"Tổng số tia song song: {len(all_ray_origins)} tia. Đang giao cho Engine C++ bắn...")
    
    # PyEmbree Vectorized raycasting (multiple_hits=False dùng để lấy giao điểm trong đầu tiên)
    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=all_ray_origins,
        ray_directions=all_ray_directions,
        multiple_hits=False
    )
    
    print("Đang áp dụng bộ lọc IQR Outlier Rejection và tính SDF...")
    
    # 3. Tính độ xa Euclid từ điểm gốc khởi tạo đến điểm chạm
    # "Ensure the Euclidean distance from the ray origin to the exact intersection point"
    hit_ray_origins = all_ray_origins[index_ray]
    hit_distances = np.linalg.norm(locations - hit_ray_origins, axis=1)
    
    # 4. Outlier Rejection IQR
    # Sắp xếp các đoạn chạm vào matrix (N, num_rays) sử dụng index_ray
    dist_matrix = np.full((N, num_rays), np.nan)
    hit_vertex_indices = index_ray // num_rays
    local_ray_idx = index_ray % num_rays
    
    dist_matrix[hit_vertex_indices, local_ray_idx] = hit_distances
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        # Bỏ qua NaN để tìm Q1, Q3
        q1 = np.nanpercentile(dist_matrix, 25, axis=1)
        q3 = np.nanpercentile(dist_matrix, 75, axis=1)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Chỉ giữ lại values nằm trong lower_bound, upper_bound
        valid_mask = (dist_matrix >= lower_bound[:, None]) & (dist_matrix <= upper_bound[:, None])
        filtered_dist_matrix = np.where(valid_mask, dist_matrix, np.nan)
        
        # Tính SDF: trung bình của các tia không bị loại bỏ
        sdf_values = np.nanmean(filtered_dist_matrix, axis=1)
        
        # Khắc phục các đỉnh không có rays nào chạm được
        sdf_values = np.nan_to_num(sdf_values, nan=0.0)
    
    print(f"Hoàn tất! SDF trung bình sau khi lọc: {np.mean(sdf_values):.4f}")
    return sdf_values

def main():
    model_path = 'bunny1.obj'  # Thay tên model tuỳ ý
    try:
        mesh = trimesh.load(model_path, force='mesh')
    except Exception as e:
        print(f"Không thể tải mesh {model_path}. Lỗi: {e}")
        return

    # Khởi chạy thuật toán Tia song song
    sdf_values = compute_sdf_parallel(mesh, num_rays=30)

    # Visualization
    pv_mesh = pv.wrap(mesh)
    pv_mesh.point_data['SDF_Parallel'] = sdf_values

    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars='SDF_Parallel', cmap='jet', show_scalar_bar=True)
    plotter.set_background('white')
    plotter.show()

if __name__ == "__main__":
    main()
