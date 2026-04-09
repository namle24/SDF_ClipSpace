# ĐIỀU KHIỂN HITTING POINT (CONTROL LOGIC)

*Báo cáo kỹ thuật về cơ chế kiểm soát giao điểm trong thuật toán Vectorized Möller-Trumbore trên GPU.*

> [!IMPORTANT]
> **Đính chính về Không gian tính toán (Warping Space vs World Space):**
> Trong các phiên bản thực nghiệm trước, nhóm đã thử nghiệm chuyển khối mesh sang **Clipping Space** (qua Perspective Divide). Tuy nhiên, phép chia phối cảnh (Perspective Divide) là một phép biến đổi phi tuyến (non-linear). Nó làm bóp méo các tam giác thành hình cong, khiến hệ tọa độ Barycentric của Möller-Trumbore bị sai lệch nghiêm trọng. 
> Do đó, ở file `sdf_meshlab_gpu.py` hiện tại (phiên bản chuẩn MeshLab 100%), thuật toán đã được dời toàn bộ về tính toán trực tiếp trong **World Space** (hoặc Camera Space trực giao). Mọi phương trình Hitting Point dưới đây hiện đang chạy trong World Space để đảm bảo độ chính xác tuyệt đối.

---

### 1. Phép biến đổi không gian (Warping Space) nằm ở đâu?
Trong phiên bản `sdf_meshlab_gpu.py` hiện tại, thay vì bóp méo toàn bộ Mesh sang Clipping Space (gây sai số), chúng ta sử dụng một phép Warping thông minh hơn: **Warping Chùm Tia (Ray Warping)** ngay trong Vector Space thay vì Warping Mesh.

- **Dòng 48 - 57:** Thiết lập hệ trục toà độ cục bộ `(u, v, w)` ngay tại không gian của từng đỉnh (World Space).
- **Dòng 70 - 73:** Phép Warping chùm tia được thực hiện thông qua Tổ hợp tuyến tính (Linear Combination) để nắn chùm tia hình nón vào đúng gốc pháp tuyến của từng đỉnh:
```python
# Biến đổi tổ hợp tuyến tính x*u + y*v + z*w (Vectorized)
ray_dirs = (x.unsqueeze(-1) * u.unsqueeze(1) + 
            y.unsqueeze(-1) * v.unsqueeze(1) + 
            rand_z.unsqueeze(-1) * w.unsqueeze(1))
```
*(Nếu là phiên bản `hybrid_sdf_world_gpu.py`, phép Warping bằng cấu trúc nghịch đảo không gian được thể hiện ở dòng lệnh `ray_dirs_world = (V_rot_T @ ray_dirs_cam.T).T`)*.

---

### 2. Möller-Trumbore xác nhận đâm trúng bằng Barycentric như thế nào?
Để xác định tia quang học có xuyên qua tam giác hay không, thuật toán giải hệ phương trình tuyến tính tìm ra 3 ẩn số: $t$ (khoảng cách), $u$ và $v$ (hệ tọa độ Barycentric).

Tia chỉ thực sự đâm trúng bề mặt tam giác khi điểm chạm nằm **BÊN TRONG** viền tam giác. Điều kiện Barycentric bắt buộc là: $u \ge 0$, $v \ge 0$, và $u + v \le 1$.

Trong file `sdf_meshlab_gpu.py`, logic culling (lọc) tia này nằm ở:
- **Dòng 122 - 123:** Tính biến `u_bary` (chính là $u$) và kiểm tra $0 \le u \le 1$.
- **Dòng 129 - 130:** Tính biến `v_bary` (chính là $v$) và kiểm tra $v \ge 0$ kèm theo điều kiện lõi $u + v \le 1$.
```python
u_bary = f * torch.sum(s * P, dim=-1)
face_ok = face_ok & (u_bary >= 0.0) & (u_bary <= 1.0)
...
v_bary = f * torch.sum(D_batch.unsqueeze(1) * Q, dim=-1)
face_ok = face_ok & (v_bary >= 0.0) & (u_bary + v_bary <= 1.0)
```
Hai biến `u_bary` và `v_bary` này tạo nên lõi điều khiển (Control Logic) kiên cố nhất của hệ thống GPU Möller-Trumbore.

---

### 3. Tọa độ chính xác của Hitting Point ($H = O + tD$) lấy ở đâu?
Hiện tại thuật toán đã được tối ưu để chỉ xuất riêng biến khoảng cách `$t$` (vì biến $t$ đại diện trực tiếp cho giá trị SDF khi vector hướng $D$ đã normalize về độ dài 1). 
Khoảng cách $t$ được tính toán tại **Dòng 133**:
```python
t = f * torch.sum(E2.unsqueeze(0) * Q, dim=-1)
```

Nếu Giáo sư yêu cầu **trích xuất Tọa độ 3D cụ thể** để vẽ lại vị trí điểm bị đâm trúng lên không gian thực (World Space), ta chỉ việc thêm công thức $H = O + tD$ ngay trong thân vòng lặp, áp dụng Broadcast Tensor:
```python
# H = O + t * D
hitting_points_3D = O_batch + t.unsqueeze(-1) * D_batch
```

---

### 4. Lọc bề mặt gần nhất khi đâm xuyên nhiều lớp (Z-Buffer logic)
Khi một tia đâm xuyên qua thân của đối tượng (ví dụ: đâm qua cả mặt trước, mặt giữa và mặt lưng của một hình đa giác phức tạp), ta sẽ thu được một ma trận chứa vô số các giá trị $t$ giao cắt.

Để chọn đúng mặt phẳng **gần nhất** (để mô phỏng khoảng cách Shape Diameter thực tế), chúng ta dùng hàm kiểm soát **Dòng 143**:
```python
# Tìm khoảng cách nhỏ nhất (đâm vào màng chắn đầu tiên)
min_t, _ = torch.min(t, dim=-1)
```
Hàm `torch.min(..., dim=-1)` của PyTorch đảm đương toàn bộ trọng trách của kỹ thuật **Z-Buffer** (hoặc Depth-Buffer). Nó sẽ quét theo chiều ngang của Tensor (chiều Faces) và ném bỏ toàn bộ những khoảng cách lớn hơn (các lớp bị che khuất đằng sau), chỉ giữ lại duy nhất giá trị giao nhau có chỉ số chiều dài cực tiểu nhất chiếu theo tia tọa độ.
