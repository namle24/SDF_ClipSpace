import trimesh
import numpy as np
import pyvista as pv
import sys

def visualize_single_vertex_rays(model_path, vertex_idx=-1, num_rays=30):
    try:
        mesh = trimesh.load(model_path, force='mesh')
        if len(mesh.vertices) == 0:
            print("Lỗi: Mesh rỗng!")
            return
    except Exception as e:
        print(f"Lỗi khi tải mesh: {e}")
        return
    
    # Nếu không chỉ định đỉnh, chọn ngẫu nhiên 1 đỉnh bất kỳ
    if vertex_idx == -1:
        vertex_idx = np.random.randint(0, len(mesh.vertices))
        
    origin = mesh.vertices[vertex_idx]
    normal = mesh.vertex_normals[vertex_idx]
    
    # Tính Bán kính chùm tia (1% BBox Diagonal) giống như logic thuật toán
    bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    radius = 0.01 * bbox_diag
    
    # =========================================================
    # 1. MO MÔ HÌNH HÓA TOÁN HỌC CAMERA (Trích xuất từ code SDF)
    # =========================================================
    anti_normal = -normal
    norm_val = np.linalg.norm(anti_normal)
    z_cam = anti_normal / (norm_val if norm_val > 0 else 1.0)
    
    # Ma trận Camera: Tính toán Trục X và Trục Y giả lập
    temp = np.array([0.0, 1.0, 0.0]) if abs(z_cam[0]) > 0.1 else np.array([1.0, 0.0, 0.0])
    x_cam = np.cross(temp, z_cam)
    x_cam = x_cam / np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    
    # =========================================================
    # 2. SINH TIA SONG SONG BÊN TRONG HÌNH TRỤ KẾT CẤU BBOX
    # =========================================================
    r_rand = radius * np.sqrt(np.random.uniform(0, 1, num_rays))
    r_rand[0] = 0.0 # Bắt buộc có 1 tia ngay chính tâm đỉnh (tọa độ gốc)
    theta_rand = np.random.uniform(0, 2 * np.pi, num_rays)
    
    ray_x = r_rand * np.cos(theta_rand)
    ray_y = r_rand * np.sin(theta_rand)
    
    # Dịch chuyển tia ra không gian 3D Thế giới (World Space)
    ray_starts = origin + np.outer(ray_x, x_cam) + np.outer(ray_y, y_cam)
    
    # Giả lập tia phóng sâu vào lòng lưới 1 khoảng (Tùy chọn hiển thị)
    ray_length = 0.2 * bbox_diag
    ray_ends = ray_starts + z_cam * ray_length
    
    # =========================================================
    # 3. PYVISTA TẠO GIAO DIỆN HIỂN THỊ (VISUALIZER)
    # =========================================================
    plotter = pv.Plotter(title=f"Minh hoạ Góc quét Tia Ortho trực giao tại Đỉnh số {vertex_idx}")
    
    # Mesh nền (Giảm độ đục opacity để nhìn thấu tia đâm bên trong)
    pv_mesh = pv.wrap(mesh)
    plotter.add_mesh(pv_mesh, color='lightblue', opacity=0.3, show_edges=True)
    
    # Chấm Đỏ: Đỉnh đang lấy làm trung tâm Camera
    plotter.add_mesh(pv.Sphere(radius=radius*0.3, center=origin), color='red')
    
    # Mũi tên Xanh lá: Pháp tuyến (Hướng ra ngoài không gian)
    normal_arrow = pv.Arrow(start=origin, direction=normal, scale=ray_length*0.5)
    plotter.add_mesh(normal_arrow, color='green', label='Vector Pháp Tuyến (Ngoài)')
    
    # Mũi tên Đỏ: Trục Z_Camera (Ray đâm xuyên vào trong lòng Mesh)
    anti_normal_arrow = pv.Arrow(start=origin, direction=z_cam, scale=ray_length*0.5)
    plotter.add_mesh(anti_normal_arrow, color='red', label='Z-Cam (-Normal) Xuyên thấu')
    
    # Các Tia (Lines) song song tạo thành Khối trụ hình học
    for i in range(num_rays):
        line = pv.Line(ray_starts[i], ray_ends[i])
        plotter.add_mesh(line, color='blue', line_width=4 if i==0 else 2, 
                         label='Chùm tia Ortho' if i==0 else None)
        
    plotter.add_legend()
    plotter.add_axes()
    plotter.show()

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else None
    if not model:
        print("Vui lòng cung cấp file 3D:")
        print("Ví dụ: python visualize_ortho_rays.py data/bunny.obj")
    else:
        visualize_single_vertex_rays(model, num_rays=50)
