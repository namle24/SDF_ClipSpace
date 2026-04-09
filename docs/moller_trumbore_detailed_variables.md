# BÁO CÁO CỐT LÕI: MÖLLER-TRUMBORE VÀ BARYCENTRIC LOGIC

*(Tài liệu giải mã định hướng dành cho Khóa luận - Giải thích rõ ràng ý nghĩa toán học của từng biến số trong thuật toán xác định Hitting Point)*

Thuật toán xác định Hitting Point (Giao điểm) trong file `sdf_meshlab_gpu.py` được xây dựng hoàn toàn từ con số 0 trên nền tảng **Möller-Trumbore Ray-Triangle Intersection Algorithm**. 
Mục tiêu của thuật toán này là giải tự động hệ phương trình tuyến tính cho hàng triệu cặp (Tia, Tam giác) cùng một lúc.

## I. XÂY DỰNG PHƯƠNG TRÌNH

Bản chất của giao điểm là nơi **Tia sáng đâm thủng Mặt phẳng Tam giác**. Tại điểm đâm đó, tọa độ 3D nội suy từ Tia phải bằng đúng với tọa độ 3D nội suy từ bản thân bề mặt Tam giác.
- **Phương trình Tia sáng:** $H = O + t \times D$
  - `O`: Gốc tia `safe_origins` (Dòng 83: `O_flat`).
  - `t`: Khoảng cách độ dài tia bay đi (Biến cần đi tìm).
  - `D`: Hướng vector của tia `ray_dirs` (Dòng 84: `D_flat`).
- **Phương trình Tam giác:** $H = V_0 + u \times E_1 + v \times E_2$
  - `V0`: Đỉnh thứ nhất của tam giác (Dòng 29: `V0`).
  - `E1` và `E2`: Lần lượt là 2 cạnh xuất phát từ V0 (`E1 = V1 - V0`, `E2 = V2 - V0` tại Dòng 32, 33).
  - $u, v$: Gọi là **Hệ Tọa Độ Trọng Tâm (Barycentric Coordinates)**. Giống như trục X, Y nhưng bị bóp méo dính lên mặt phẳng của tam giác. Nó biểu thị điểm đâm đó dịch chuyển bao nhiêu phần trăm theo cạnh $E_1$, và bao nhiêu phần trăm theo cạnh $E_2$.

Cân bằng 2 phương trình trên ta có: $O + tD = V_0 + uE_1 + vE_2$
Nhiệm vụ của Möller-Trumbore là giải ra 3 biến $t, u, v$ dựa vào Định lý Cramer (Đại số tuyến tính).

---

## II. GIẢI MÃ Ý NGHĨA CÁC BIẾN (Dòng 106 - 134)

Dưới đây là giải nghĩa từng biến được khai báo trong vòng lặp GPU:

**1. Biến `P` (Dòng 107):** `P = torch.linalg.cross(D, E2)`
- Tích có hướng (Cross Product) của Tia sáng và Cạnh tam giác nảy ra một vector trực giao. Nó dùng để làm tử số kiểm tra góc độ.

**2. Biến `a` (Dòng 110):** `a = torch.sum(E1 * P, dim=-1)`
- Tích vô hướng (Dot Product). Đây chính là cái **Định thức (Determinant)** của mảng phương trình.
- Rất quan trọng: Nếu `a` tiến dần bằng 0, nghĩa là tia của bạn đang bay *song song* tạt qua mặt phẳng tam giác. Không bao giờ xuyên lủng được. Góc giới hạn lệch `1e-8` kiểm duyệt điều này ở dòng `113`.

**3. Biến `f` (Dòng 116):** `f = 1.0 / a`
- Thay vì đem từng kết quả chia cho định thức $a$ tốn tài nguyên GPU, ta nghịch đảo nó thành $f$ rồi đem nhân cho mọi biến phụ ở dưới.

**4. Biến `s` (Dòng 119):** `s = O - V0`
- Dời trục hệ tọa độ. Dịch Gốc toa độ không gian dính chặt về ngay trên điểm neo V0 của Tam Giác.

**5. HỆ SỐ BARYCENTRIC Số 1 (`u_bary` - Dòng 122):** `u_bary = f * (s · P)`
- Điểm chạm di chuyển bao xa dọc theo cạnh viền E1?
- Nếu $u < 0$ (Rớt ra ngoài mép trái) hoặc $u > 1$ (Rớt quá mép phải) $\rightarrow$ Tia bay ra ngoài tam giác! Dòng `123` dùng để Culling (khóa) lại.

**6. Biến `Q` (Dòng 126):** `Q = torch.linalg.cross(s, E1)`
- Chuẩn bị biến phụ để tính $v$.

**7. HỆ SỐ BARYCENTRIC Số 2 (`v_bary` - Dòng 129):** `v_bary = f * (D · Q)`
- Điểm chạm di chuyển bao xa dọc theo cạnh viền E2?
- **Logic Tuyệt Vời Nhất:** Dòng `130` kiểm tra `v_bary >= 0` VÀ quan trọng nhất là `u_bary + v_bary <= 1.0`.
  Tam giác là một nửa của hình bình hành. Nếu $u+v > 1$, điểm rơi sẽ trượt ra nửa ngoài của hình bình hành (phần không thuộc tam giác). Thuật toán gạt bỏ!

**8. TOẠ ĐỘ KHOẢNG CÁCH CHÍNH XÁC `t` (Dòng 133):** `t = f * (E2 · Q)`
- Đây chính xác là lời giải từ Định lý Cramer mô tả Khoảng Cách vật lý (Euclid Distance) từ mắt Camera (Origin) đến điểm đâm thủng bề mặt tam giác.
- Nếu $t < 0$, mặt phẳng nằm ở sau lưng chúng ta, không hợp lệ! Nên phải có dòng khóa biên độ `(t > 1e-4)`.

---

## III. TÌM TAM GIÁC OUTING POINT NẰM Ở ĐÂU? (Dòng 143)

Sau khi bộ lọc chạy xong, mỗi 1 tia bắn ra thu về một Ma Trận khổng lồ, ví dụ: 
`Tia_001_trúng_các_mặt_phẳng = [Vô cực, Vô cực, 2.4 mét, Vô cực, 10.6 mét, Vô cực]`
*(Tương ứng Tia_001 đâm trượt Tam giác 1, Trượt 2, Trúng 3, Trúng 5).*

Khối lệnh ở **Dòng 143**:
```python
min_t, _ = torch.min(t, dim=-1)
```
> **Logic Z-Buffer:** PyTorch quét cái mảng khổng lồ đó, nhặt ra con số nhỏ nhất (Tức là khoảng cách gần nhất so với người bắn). 
> Ở đây `min_t` sẽ chọn $2.4$ mét thay vì $10.6$ mét. Bức tường $2.4$ mét ấy TÊN LÀ Tam giác 3. 
> Tam giác 3 chính là vỏ áo tiếp theo mà tia đâm phải sau khi băng qua khoảng không gian trống trong bụng vật thể (Chính là Outing Point). 

Độ dày $2.4$ mét được lưu giữ lại làm kết quả của tia đo đạc đó, đưa ra bộ cộng hàm ở dưới đáy file code để tính trung bình cộng SDF. Trọn vẹn quá trình Hình học không gian hoàn toàn dùng Tensor.
