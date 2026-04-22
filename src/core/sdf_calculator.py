import trimesh
import numpy as np
import pyvista as pv
import time
from tqdm import tqdm


def generate_cone_rays(origin, normal, num_rays=30, cone_angle=120):
    """
    Tạo một chùm tia hình nón ngược hướng pháp tuyến
    """
    # Xử lý trường hợp pháp tuyến bị lỗi (norm = 0)
    norm_val = np.linalg.norm(normal)
    if norm_val < 1e-8:
        # Nếu không có pháp tuyến, bắn theo hướng ngẫu nhiên hoặc trục Z tạm thời
        anti_normal = np.array([0.0, 0.0, -1.0])
    else:
        anti_normal = -normal / norm_val
        
    max_angle = np.radians(cone_angle / 2)

    # Tạo hệ tọa độ cục bộ (u, v, w)
    w = anti_normal
    if abs(w[0]) > 0.1:
        temp = np.array([0, 1, 0])
    else:
        temp = np.array([1, 0, 0])
        
    u = np.cross(temp, w)
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-8:
        return np.zeros((num_rays, 3)) # Trả về mảng 0 để lọc bỏ
    
    u = u / u_norm
    v = np.cross(w, u)

    rays_dir = []
    for _ in range(num_rays):
        phi = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(np.cos(max_angle), 1.0)
        r = np.sqrt(max(0, 1 - z ** 2))
        x = r * np.cos(phi)
        y = r * np.sin(phi)

        ray_dir = x * u + y * v + z * w
        ray_dir_norm = np.linalg.norm(ray_dir)
        if ray_dir_norm > 1e-8:
            rays_dir.append(ray_dir / ray_dir_norm)
        else:
            rays_dir.append(w) # Fallback

    return np.array(rays_dir)


def compute_sdf_cone(mesh, num_rays=30, cone_angle=120):
    origins = mesh.vertices
    normals = mesh.vertex_normals
    eps = 1e-4

    print(f"Đang tạo chùm tia (Cone of Rays) cho {len(origins)} đỉnh...")
    all_ray_origins = []
    all_ray_directions = []
    ray_to_vertex_idx = []

    for i in tqdm(range(len(origins)), desc="Generating rays"):
        origin = origins[i]
        normal = normals[i]
        cone_dirs = generate_cone_rays(origin, normal, num_rays, cone_angle)
        safe_origin = origin - normal * eps

        for ray_dir in cone_dirs:
            # Chỉ thêm các tia hợp lệ (không chứa NaN/Inf)
            if not np.any(np.isnan(ray_dir)) and not np.any(np.isnan(safe_origin)):
                all_ray_origins.append(safe_origin)
                all_ray_directions.append(ray_dir)
                ray_to_vertex_idx.append(i)

    all_ray_origins = np.array(all_ray_origins)
    all_ray_directions = np.array(all_ray_directions)
    ray_to_vertex_idx = np.array(ray_to_vertex_idx)

    if len(all_ray_origins) == 0:
        print("Lỗi: Không tạo được tia hợp lệ nào!")
        return np.zeros(len(origins))

    print(f"Tổng số tia chuẩn bị bắn: {len(all_ray_origins)} tia. Đang bắn theo đợt (Batching)...")
    batch_size = 256
    all_locations = []
    all_index_ray = []

    for i in tqdm(range(0, len(all_ray_origins), batch_size), desc="Bắn tia theo Batch"):
        end = min(i + batch_size, len(all_ray_origins))
        batch_origins = all_ray_origins[i:end]
        batch_dirs = all_ray_directions[i:end]
        
        locs, idx_ray, _ = mesh.ray.intersects_location(
            ray_origins=batch_origins,
            ray_directions=batch_dirs,
            multiple_hits=False
        )
        
        if len(locs) > 0:
            all_locations.append(locs)
            all_index_ray.append(idx_ray + i)

    if len(all_locations) > 0:
        locations = np.vstack(all_locations)
        index_ray = np.concatenate(all_index_ray)
    else:
        print("Cảnh báo: Không có tia nào trúng mesh!")
        return np.zeros(len(origins))

    print("Đang tính trung bình độ dày (SDF)...")
    hit_origins = all_ray_origins[index_ray]
    hit_distances = np.linalg.norm(locations - hit_origins, axis=1)
    hit_vertex_indices = ray_to_vertex_idx[index_ray]

    sdf_values = np.zeros(len(origins))
    for i in range(len(origins)):
        mask = (hit_vertex_indices == i)
        if np.any(mask):
            sdf_values[i] = np.mean(hit_distances[mask])

    print(f"Hoàn tất! SDF trung bình: {np.mean(sdf_values):.4f}")
    return sdf_values


def main():
    # PATH ĐẾN MODEL CỦA BẠN
    model_path = "data/spot.obj"
    try:
        mesh = trimesh.load(model_path, force='mesh')
        # Tự động sửa lỗi mesh (gộp đỉnh, xóa mặt lỗi, xóa mặt trùng...)
        mesh.fix_normals()
        mesh.process()
    except Exception as e:
        print(f"Lỗi load mesh: {e}")
        return

    # Chạy thuật toán chùm tia SDF (Shape Diameter Function chuẩn MeshLab)
    print("\n[BẮT ĐẦU SÚNG BẮN TIA - SHAPE DIAMETER FUNCTION (SDF)]")
    start_time = time.perf_counter()
    sdf_values = compute_sdf_cone(mesh, num_rays=30, cone_angle=120)
    end_time = time.perf_counter()

    print(f"\n=======================================================")
    print(f"[*] THỜI GIAN TÍNH TOÁN SDF (CPU NATIVE): {end_time - start_time:.4f} giây")
    print(f"=======================================================\n")

    # Hiển thị vùng màu mượt (MeshLab Style)
    pv_mesh = pv.wrap(mesh)
    pv_mesh.point_data['SDF_Thickness_Cone'] = sdf_values
    
    plotter = pv.Plotter(title="SDF Cone Visualization - MeshLab Style")
    # Sử dụng 'jet_r' (Red-Yellow-Blue reversed) để giống MeshLab: Đỏ là mỏng, Xanh là dày
    plotter.add_mesh(
        pv_mesh, 
        scalars='SDF_Thickness_Cone', 
        cmap='jet_r', 
        smooth_shading=False,
        show_edges=False,
        scalar_bar_args={'title': "SDF (Thickness)"}
    )
    plotter.set_background('white')
    plotter.add_axes()
    plotter.show()


if __name__ == "__main__":
    main()