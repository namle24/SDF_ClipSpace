# Nền Tảng Nghiên Cứu: Thuật Toán Tính Độ Dày Dựa Trên Bộ Đệm Độ Sâu (Z-Buffer Rasterization SDF) Bằng PyTorch GPU

Tài liệu này tổng hợp các luận điểm học thuật, so sánh kỹ thuật và những cải tiến cốt lõi của phương pháp **Custom Orthographic SDF** so với phương pháp Dò tia (Raytracing) truyền thống để phục vụ cho việc viết báo cáo nghiên cứu (Research Paper).

---

## 1. Khái Niệm Đột Phá: Perspective Cone vs. Orthographic Cylinder

### 1.1 Khuyết Điểm Của Phương Pháp Truyền Thống (Raytracing SDF)

- Kỹ thuật đo độ dày hình học Shape Diameter Function (SDF) truyền thống thường sử dụng phương thức **Perspective Ray-casting**.
- Thuật toán sẽ phát ra một **chùm tia hình nón (Cone)** lấy dải góc ngược hướng với Pháp tuyến bề mặt (Negative Normal).
- **Vấn đề toán học:** Chùm tia hình nón có xu hướng mở rộng bán kính vùng quét khi đi sâu vào lòng mô hình. Điều này khiến thuật toán thu thập kết quả khoảng cách từ những vùng "không liên quan" ở mặt đối diện (Đặc biệt trên các mô hình 3D phức tạp, có mức độ cong và gấp khúc cao). Kết quả độ dày thường bị nhiễu do thu thập quá nhiều giao điểm nằm ngoài tiết diện ban đầu.

### 1.2 Ý Tưởng Đề Xuất (Ortho Z-Buffer SDF)

- Xây dựng một **Camera Trực Giao (Orthographic)** tại mỗi đỉnh thay vì dùng hướng bắn tự do.
- Chùm tia bắn đi song song với nhau, bị giới hạn hình học thành một **Hình Trụ (Cylinder)** có bán kính chủ động bằng `1% Bounding Box Diagonal`.
- **Ưu điểm:** Phương pháp Ortho duy trì **diện tích tiết diện cắt ngang hoàn toàn không đổi** bất kể độ sâu của giao điểm mặt trong. Ý tưởng này mô phỏng một "Mũi khoan thẳng" khoan xuyên qua vật thể, đảm bảo đo đạc độ dày theo quy chuẩn cục bộ (Local Thickness) đồng nhất và có độ tiệm cận chính xác cao hơn tại các vùng góc hẹp.

---

## 2. Phân Tích Hiện Tượng "Cổ Chai" Khởi Tạo Trên Môi Trường Python

Trong các thực nghiệm chứng minh vì sao hệ thống cũ viết bằng thư viện Python (`trimesh` + `pyembree`) lại chậm hơn phần mềm viết bằng C++ (như MeshLab) xử lý SDF, có **3 nguyên nhân thiết kế căn bản** để đưa vào bài báo:

1. **The Python Global Interpreter Lock (GIL):**
   - Vòng lặp `for` để tính toán hướng tia và nạp vào mảng List trong quá trình chuẩn bị (Generating Rays) trên Python hoàn toàn mang tính tuần tự, chạy trên 1 nhân CPU đơn lẻ, gây thất thoát thời gian trầm trọng do GIL khóa các luồng xử lý đồng thời.
2. **Memory Transfer Overhead (Độ trễ truyền tải bộ nhớ):**
   - Không giống với C++ (thao tác trực tiếp với con trỏ), Python buộc phải sao chép hàng triệu tọa độ mảng `numpy` qua lại với thư viện dò tia viết bằng ngôn ngữ `C/C++` biên dịch sẵn (Embree/BVH fallback). Quá trình Allocate/Deallocate cấp phát bộ nhớ động liên tục tạo ra thắt cổ chai ở bộ nhớ RAM.
3. **BVH (Bounding Volume Hierarchy) Fallback:**
   - Khi không có sẵn nhân tăng tốc dò tia của Intel (`pyembree`), Trimesh sẽ thực thi cơ chế RTree bằng Python rất chậm, thay đổi cấu trúc cây với hàm tỷ lệ thời gian không tối ưu.

---

## 3. Các Cải Tiến Cốt Lõi Trên Pytorch GPU (Song Song Hóa Hoàn Toàn)

Sự ra đời của `custom_ortho_sdf_gpu.py` khắc phục toàn bộ khuyết điểm trên và đẩy tốc độ lên tối đa thông qua các nâng cấp cấu trúc sau:

### 3.1 100% Vectorized Tensor Computing (Loại Bỏ BVH & Raytracer Truyền Thống)

- Chuyển đổi toàn bộ quá trình đâm tia thành **phép biến đổi tọa độ Ma Trận (Matrix Transformation)** và **Kiểm tra tọa độ trọng tâm (Barycentric Z-Buffer)**.
- Thay vì dò cây BVH, hệ thống lợi dụng chức năng Broadcasting của Pytorch trên CUDA Cores, cho phép ném trực tiếp toàn bộ dữ liệu Mặt lưới (Faces) dạng Tensor vào Hàm Xét Đâm Xuyên (Point-in-Triangle 2D) xử lý song song cả triệu tam giác qua một chu kỳ Clock. Lập trình vector hóa (Vectorized) hoàn toàn dỡ bỏ hạn chế của Python GIL.

### 3.2 Dynamic Vectorized Epsilon (Giải Quyết Lỗi Self-Intersection Tự Động)

- Quá trình bắn tia luôn phát sinh lỗi **Self-Intersection** (Tia đâm ngay vào bề mặt do sai số dấu phẩy động Float32).
- Thay vì sử dụng Epsilon nội xạ cứng ngắc (vd: `1e-4`, vốn phá hỏng lớp ngụy trang bề mặt trên các Model scale siêu phân giải hoặc quá vi mô), thuật toán tự động nội suy **Dynamic Epsilon** linh động:
  `ray_epsilon = 1e-5 * bbox_diag`
- Điều này giúp hệ thống là **Scale-Invariant**, bất hoại với tỷ lệ phóng to/thu nhỏ của mô hình, đảm bảo tính ổn định đo lường trong nghiên cứu thực nghiệm.

### 3.3 Stable Tensor Batching (Phòng Trống Tràn Bộ Nhớ VRAM)

- Phát huy tối đa GPU nhưng không giới hạn cấu hình phần cứng: Thông qua việc phân mảnh mảng Đỉnh thành các batch (VD: `batch_size = 256`), hệ thống quản lý chủ động cực đại ma trận sinh ra trong không gian chiếu `(Batch_size × Num_rays × Faces)`, biến nó thành thông số an toàn chống OOM (Out Of Memory) ngay cả với các lưới có mật độ dày đặc hàng trăm ngàn mặt.

### 3.4 Quy Trình Thanh Lọc Nhiễu 100% On-GPU

- Quá trình lọc IQR (Interquartile Range) phục hồi tín hiệu đã được tích hợp hoàn toàn trong Pytorch (`torch.nanquantile`). Mảng nhiễu Outlier (Những tia bắn trượt ra ngoài) được khử thẳng trên GPU mà không có bất kì giao tiếp nghẽn cổ chai nào truyền qua cổng PCIe về RAM CPU, duy trì băng thông bộ đệm đồ họa liên tục cho đến kết quả điểm SDF phân giải cuối cùng.

---

## 4. Định Hướng Viết Bài Thực Nghiệm (Experiments Setup)

Để hoàn thiện cho báo cáo học thuật, các bước xác thực sau nên được triển khai để vẽ biểu đồ so sánh:

1. **Hiệu Suất (Performance Table):**
   - Đo lường thời gian (Render Time) giữa tính năng SDF của thư viện Python Trimesh vs Hệ Thống Pytorch Tensor Đề Xuất (trên cùng quy mô tia đâm và số lượng lưới).
2. **Độ Mượt Heatmap (Visual Quality):**
   - Khảo sát sự thay đổi biên độ màu (Smoothness) bằng PyVista qua các góc phức tạp (Tai thỏ Stanford, Bánh xe ô tô) – so sánh giữa dải lưới nhiễu của tia hình nón (Cone) và sự ổn định của tia Trực giao (Orthographic Cylinder).
3. **Phân Tích Cường Độ VRAM (Ablation Study):**
   - Đánh giá đánh đổi (Trade-off) giữa tham số `batch_size` vs lượng bộ nhớ VRAM sử dụng và Tốc độ suy luận.
