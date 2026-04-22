import trimesh
import numpy as np
import pyvista as pv
import sys

def visualize_single_vertex_rays_improved(model_path, vertex_idx=-1, num_rays=30):
    try:
        mesh = trimesh.load(model_path, force='mesh')
        mesh.fix_normals() # Sửa lỗi pháp tuyến nếu có
        if len(mesh.vertices) == 0:
            print("Lỗi: Mesh rỗng!")
            return
    except Exception as e:
        print(f"Lỗi khi tải mesh: {e}")
        return
    
    # 1. Chọn đỉnh
    if vertex_idx == -1:
        vertex_idx = np.random.randint(0, len(mesh.vertices))
        
    origin = mesh.vertices[vertex_idx]
    normal = mesh.vertex_normals[vertex_idx]
    
    # 2. Phóng to bán kính (5%) ĐỂ DỄ NHÌN HƠN TRÊN MÀN HÌNH
    bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    visual_radius = 0.05 * bbox_diag 
    
    # 3. HỆ TỌA ĐỘ CAMERA TRỰC GIAO
    z_cam = -normal # Hướng vào trong lòng mesh
    z_cam = z_cam / (np.linalg.norm(z_cam) + 1e-8)
    
    temp = np.array([0.0, 1.0, 0.0]) if abs(z_cam[0]) > 0.1 else np.array([1.0, 0.0, 0.0])
    x_cam = np.cross(temp, z_cam)
    x_cam = x_cam / np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    
    # 4. SINH CHÙM TIA BÊN TRONG HÌNH TRỤ
    r_rand = visual_radius * np.sqrt(np.random.uniform(0, 1, num_rays))
    r_rand[0] = 0.0 # Tia trung tâm
    theta_rand = np.random.uniform(0, 2 * np.pi, num_rays)
    
    ray_starts = origin + np.outer(r_rand * np.cos(theta_rand), x_cam) + np.outer(r_rand * np.sin(theta_rand), y_cam)
    
    # Dịch gốc tia vào trong lòng mesh một chút (epsilon) để không tự đâm vào mặt phẳng chứa chính nó
    eps = 1e-4
    ray_starts_offset = ray_starts + z_cam * eps
    ray_dirs = np.tile(z_cam, (num_rays, 1))
    
    # --- ĐIỂM ĂN TIỀN: Bắn tia thật để tìm điểm chạm vách đối diện ---
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_starts_offset,
        ray_directions=ray_dirs,
        multiple_hits=False # Chỉ lấy điểm chạm đầu tiên (vách đối diện)
    )
    
    # 5. GIAO DIỆN PYVISTA
    plotter = pv.Plotter(title=f"Minh hoạ Chùm tia Trực giao đâm xuyên tại Đỉnh {vertex_idx}")
    
    # Mesh nền mờ
    pv_mesh = pv.wrap(mesh)
    plotter.add_mesh(pv_mesh, color='white', opacity=0.3, show_edges=True)
    
    # --- THÊM 2 DÒNG NÀY ĐỂ VẼ KHỐI LẬP PHƯƠNG (BOUNDING BOX) ---
    bbox = pv_mesh.outline()
    plotter.add_mesh(bbox, color='cyan', line_width=2, label='Khối Bounding Box')
    # -------------------------------------------------------------
    
    # Đỉnh đang xét
    plotter.add_mesh(pv.Sphere(radius=visual_radius*0.1, center=origin), color='red')
    
    # Vẽ mặt phẳng Camera (Một cái đĩa ảo)
    disk = pv.Disc(center=origin, normal=z_cam, inner=0, outer=visual_radius)
    plotter.add_mesh(disk, color='yellow', opacity=0.5, label='Mặt phẳng Trực giao')
    
    # Vẽ các tia (Chỉ vẽ từ gốc đến điểm chạm đối diện)
    for i in range(len(locations)):
        ray_idx = index_ray[i]
        start_pt = ray_starts[ray_idx]
        hit_pt = locations[i]
        
        # Vẽ tia
        line = pv.Line(start_pt, hit_pt)
        plotter.add_mesh(line, color='blue', line_width=2)
        
        # Điểm chạm (Hit point ở vách đối diện)
        plotter.add_mesh(pv.Sphere(radius=visual_radius*0.05, center=hit_pt), color='orange')
        
    plotter.add_legend()
    plotter.add_axes()
    plotter.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        visualize_single_vertex_rays_improved(model_path, num_rays=50)
    else:
        print("Vui lòng cung cấp file 3D:")
        print("Ví dụ: python visualize_ortho_rays.py data/bunny.obj")
