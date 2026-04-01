import os
import argparse
import numpy as np
import pyvista as pv

def main():
    parser = argparse.ArgumentParser(description="Công cụ Tự động xem Point Cloud & SDF từ file .npy")
    parser.add_argument("--input_file", type=str, required=True, help="Đường dẫn đến file SDF (.npy)")
    parser.add_argument("--point_size", type=float, default=15.0, help="Kích thước của hạt (sphere) 3D")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"[LỖI XÓT FILE] Không thể xác định file: {args.input_file}")
        return
        
    print(f"[*] Đang tải ma trận Tensor từ: {args.input_file}")
    data = np.load(args.input_file)
    
    if len(data.shape) != 2 or data.shape[1] < 4:
        print(f"[CẢNH BÁO] Hệ thống không nhận diện được kiến trúc (N, 4). Cấu trúc file của bạn là: {data.shape}")
        return
        
    # Tách dữ liệu
    points = data[:, 0:3]       # 3 Hệ tọa độ X, Y, Z
    sdf_values = data[:, 3]     # Cột cuối cùng là trường SDF
    
    print(f"[*] Load hình học thành công: {len(points)} Phân tử Điểm ảnh.\n[*] Khởi động PyVista Shader...")
    
    # ------------------ Dựng hình PyVista ------------------
    cloud = pv.PolyData(points)
    cloud.point_data["SDF Value"] = sdf_values
    
    title_name = os.path.basename(args.input_file)
    plotter = pv.Plotter(title=f"Đồ họa xem trước - {title_name}")
    
    # Tuỳ chọn rực rỡ và tối ưu hình hạt ngọc dâng cao chất lượng hiển thị
    plotter.add_mesh(
        cloud, 
        scalars="SDF Value", 
        cmap="jet",                   # Bảng màu cầu vồng đặc trưng của bài toán Heatmap / SDF
        point_size=args.point_size, 
        render_points_as_spheres=True, # Hiển thị là hình cầu (Spheres) 3D bóng bẩy thay vì ô vuông pixel
        show_scalar_bar=True,
        scalar_bar_args={"title": "Độ dày Không gian (SDF)"}
    )
    
    # Trục tọa độ vĩ mô định hướng thị giác
    plotter.add_axes()
    plotter.set_background('white')
    plotter.show()

if __name__ == "__main__":
    main()
