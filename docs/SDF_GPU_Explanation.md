# Báo Cáo Phân Tích Kỹ Thuật: `fast_sdf_gpu.py`

Tài liệu này giải thích chi tiết cấu trúc, các thành phần, thư viện được sử dụng, và đặc biệt là logic toán học cốt lõi để xác định giao điểm tia và tính độ dày (Shape Diameter) trong file `fast_sdf_gpu.py`.

---

## 1. Các Thư Viện Được Sử Dụng

### `torch` (PyTorch)

- **Mục đích:** Thư viện tính toán Tensor trên GPU. Thay vì dùng CPU chậm chạp, PyTorch cho phép tính toán song song hàng chục ngàn phép toán ma trận cùng lúc.
- **Hàm quan trọng:**
  - `torch.einsum`: Nhân ma trận Tensor đa chiều hàng loạt (Batched Matrix Multiplication). Dùng để chiếu toàn bộ đỉnh của 3D model vào không gian của hàng trăm Camera cùng lúc.
  - `torch.linalg.cross`: Tính tích có hướng (Cross Product) giữa 2 vector. Dùng để tính trục của Camera và diện tích Barycentric 2D.
  - `torch.where`, `torch.min`: Các hàm chọn lọc và tìm giá trị nhỏ nhất trên GPU mà không cần vòng lặp (if/else).
  - `torch.gather`: Trích xuất các giá trị tại các chỉ số (index) cụ thể ra khỏi một Tensor lớn (VD: Trích tọa độ của điểm chạm đầu tiên).

### `numpy` (NumPy)

- **Mục đích:** Xử lý các phép toán vô hướng cơ bản hoặc chuyển đổi góc độ trong lúc khởi tạo.
- **Hàm quan trọng:**
  - `np.radians`, `np.tan`: Chuyển đổi FOV (Field of View) từ độ sang radian và tính tang góc để tạo ma trận Perspective.

### `pyvista`

- **Mục đích:** Tiện ích Render và Visualizer 3D.
- **Hàm quan trọng:**
  - `pv.wrap`: Bọc dữ liệu điểm lưới thành format PyVista PolyData.
  - `pv_mesh.cell_data['SDF_Ultimate_Raster']`: Gán mảng giá trị SDF vừa tính được lên từng bề mặt (Face) của lưới để tô màu hiển thị.
  - `pv.Plotter`: Giao diện cửa sổ 3D để xem tương tác.

### `trimesh`

- **Mục đích:** Load các file 3D (.obj, .off) và lấy thông tin hình học thô nhanh chóng.
- **Hàm quan trọng:**
  - `trimesh.load`: Load file 3D.
  - `mesh.vertices`, `mesh.faces`, `mesh.face_normals`: Truy xuất cấu trúc Đỉnh, Mặt, và Vector Pháp Tuyến bề mặt của khối 3D.

---

## 2. Các Hàm (Functions) Cốt Lõi

1.  **`look_at_torch_batched(eyes, targets, device)`**
    - _Ý nghĩa:_ Tính ma trận View ($V$) cho một lô (Batch) các Camera. Nó tạo một hệ tọa độ máy ảnh cho mỗi mặt tam giác (đặt Camera tại tâm mặt, nhìn đâm vào trong lưới). Nó tạo ra ma trận dịch chuyển và xoay toàn thế giới về góc nhìn của mặt đó.
2.  **`perspective_torch(fov_deg, near, far, device)`**
    - _Ý nghĩa:_ Tính ma trận Phối cảnh ($P$). Nó nén và làm méo không gian 3D dạng hình chóp đứt khúc (Frustum) thành một hộp vuông vức NDC (Normalized Device Coordinates). Điều này biến các tia bắn tỏa ra thành các tia song song.
3.  **`generate_rays_ndc(fov_deg, num_rings, num_rays_per_ring, device)`**
    - _Ý nghĩa:_ Sinh ra lưới các tia ảo (Rays). Thay vì bắn tia 3D, hàm này quy đổi chùm nón tia thành các tạo độ điểm 2D $(p_x, p_y)$ nằm trên màn hình NDC chuẩn.
4.  **`compute_fast_sdf_gpu(mesh, ...)`**
    - _Ý nghĩa:_ Hàm trọng tâm điều phối toàn bộ Pipeline.

---

## 3. Quá Trình Nhận Diện Va Chạm (Ray Hitting & Z-Buffer)

Đây là kỹ thuật tinh hoa trong Đồ họa máy tính: Bắn tia song song trong không gian Màn Hình (Screen/NDC Space).

### Tia đâm vào tam giác như thế nào? (Barycentric Point-in-Triangle 2D)

Thay vì dùng phương pháp Möller-Trumbore 3D rất tốn kém, chúng ta chiếu mọi thứ lên không gian NDC. Khi đó:

- Tia nhìn từ Camera biến thành phương **song song với trục Z** (tia vuông góc đâm thẳng vào màn hình tại điểm $p_x, p_y$).
- Việc kiểm tra "Tia có đâm trúng Tam giác không" trở thành bài toán phẳng (2D): Điểm $(p_x, p_y)$ có nằm gọn bên trong hình chiếu 2D của Tam giác trên màn hình hay không?
- **Toán học áp dụng:** Thuật toán tính trọng số Barycentric $(u, v, w)$ bằng Tích có hướng 2D (`cross2d`). Nếu cả $u \ge 0$, $v \ge 0$, và $w \ge 0$, điều đó xác nhận điểm ảnh nằm bên trong tam giác. (Biến `in_tri` = True).

### Xác định tam giác bị đâm đầu tiên? (Phật toán Z-Buffer)

Khi Tia đâm thẳng vào khối 3D, nó sẽ xuyên qua nhiều lớp (nhiều tam giác chồng lên nhau trên màn hình 2D).

1.  **Nội suy độ sâu (Z-interpolation):** Tại điểm đâm $(p_x, p_y)$ của mỗi tam giác bị trúng, chúng ta dùng trọng số $(u, v, w)$ nội suy ra tọa độ độ sâu $Z_{hit}$ (giá trị Z trong không gian NDC).
2.  **Tái tạo Tọa độ 3D Giao điểm (World Space):**
    Một khi có $W_{hit\_real}$, dùng hệ số này nhân ngược với cụm nội suy tọa độ Đỉnh thực 3D ($V_{world}$):
    ```python
    hit_point_3D = W_interpolated * ( (u / W0)*V0 + (v / W1)*V1 + (w / W2)*V2 )
    ```
3.  **Tính Khoảng Cách Cuối Cùng:**
    Đo khoảng cách Euclid gốc `torch.norm` (L2 norm) từ đỉnh Camera (`face_centers` - nơi tia khởi phát) đến toạ độ `hit_point_3D`. Tổng hợp trung bình (`mean`) chiều dài của cụm các tia bắn ra sẽ tạo thành độ dày SDF của khối tại vị trí đó!
