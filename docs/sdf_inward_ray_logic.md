# LUỒNG LOGIC TÌM KIẾM ĐỈNH ĐÂM VÀ TÍNH SDF: TỪ TAM GIÁC A SANG B

Để hiểu thấu đáo thuật toán hoạt động ra sao nếu ta bắn 1 chùm 30 tia vào lòng một vật thể, ta đi theo hành trình tuyến tính của Cụm PyTorch Tensor trong file `sdf_meshlab_gpu.py`.

### Bước 1: 30 tia xuất phát từ đâu? (Tam giác A)
SDF là "hàm đo đường kính độ dày" bên trong lòng vật thể. Tức là ta bắn tia **từ vỏ của vật ném vào trong bụng vật**, cho đến khi nó chạm vào cái vỏ ở bờ tường đối diện phía bên kia.
* **Gốc bắn (Origin `O`):** Tại 1 đỉnh bất kỳ trên vỏ (thuộc tam giác A), pháp tuyến của nó chỉ thẳng ra ngoài không gian. Ta đảo ngược pháp tuyến (`anti_normals`) để nó chĩa vào trong.
* **Điểm neo an toàn:** Để mũi tên vừa sinh ra không đâm ngược và mắc kẹt ngay lập tức vào chính Tam giác A (nơi nó vừa sinh ra), ở dòng 79-80 ta tịnh tiến Gốc tia nhích nhẹ vào trong bụng vật thể 1 li (`eps = 1e-4`):
  `safe_origins = vertices - normals * eps` 
  Đây là điểm khởi đầu thực sự $O$ của 30 tia.

### Bước 2: 30 tia đâm toán học diễn ra thế nào?
File code **KHÔNG DÙNG vòng lặp for cho từng tam giác**, mà dùng **Đại số Tuyến tính Tensor**.
Giả sử ta có 1 đỉnh phát ra 30 tia $D$ hướng vào trong bụng vật, và cái rổ chứa toàn tuyến hình (Mesh) của ta có $10.000$ mặt tam giác. 
PyTorch sẽ giãn nở (Broadcast) ra một Ma Trận khổng lồ, kiểm tra 30 tia này đối đầu trực diện với cả $10.000$ tam giác *cùng lúc*.

**Phương trình Mặt phẳng - Tia (Dòng 106 - 134):**
- **$P, Q, s, f$**: Là các biến tạm trong Đại số Cramer để giải nén công thức nội suy điểm cắt trên tam giác.
- Mọi mặt phẳng đều bị đem ra xét điểm giao cắt kéo dài $t$.
- Biến $u, v$ giúp ta gạt bỏ (culling) $9.900$ tam giác nằm ngoài rìa. Chỉ còn lại (ví dụ) $100$ mặt phẳng cản đường bắn thực sự.
- Các tam giác không ngáng đường sẽ bị gạt thành khoảng cách $t = \infty$ (Vô cực) do bộ lặp chặn logic ở dòng `140:` `t = torch.where(face_ok, t, float('inf'))`.

### Bước 3: Đâm xuyên qua nhiều lớp thì nhận diện Tam Giác B ra sao? (Dòng 143)
Tưởng tượng tia đi xuyên qua vùng bụng con Thỏ, chạm nhẹ vào bức tường bên kia, sau đó đi xuyên ra ngoài không khí lặn tiếp vào cái tai con Thỏ. (Một đường thẳng xuyên qua nhiều lớp vỏ).

Trong PyTorch, 1 tia đâm này sẽ trả về cho bạn 1 danh sách khoảng cách với các mặt phẳng cản đường (Ví dụ: `t = [1.5, 4.3, 8.2, ∞, ∞...]`).
Thuật toán làm sao biết vỏ nào là vỏ bụng con Thỏ (Điểm cản đầu tiên)?
Chính là **Dòng 143**:
```python
min_t, _ = torch.min(t, dim=-1)
```
Hàm `min` gạt bỏ đi $4.3$ hay $8.2$, lôi cổ đúng anh **chặn đầu đầu tiên ($1.5$)**. Khoảng cách $t = 1.5$ ấy chính là khoảng cách từ bức tường trong của Tam giác A đến mặt phẳng đầu tiên nó gặp: **Tam giác B (The Outing Point / Điểm xuất kích)**.

### Bước 4: Khoảng cách thật được chốt đơn ra sao?
Bởi vì vector tia điện quang 3D ($D$) của 30 tia đã được ép (Normalize) thành độ dài bằng 1 ở lệnh:
`ray_dirs / torch.norm(ray_dirs)`
Nên biến tỷ lệ $t$ lấy từ Möller-Trumbore KHÔNG CHỈ LÀ THAM SỐ, NÓ CHÍNH LÀ ĐỘ DÀI VẬT LÝ KHOẢNG CÁCH.

- **Vị trí 3D của Outing Point (Tam giác B) được định vị thế nào?**
Nếu bạn muốn chấm một cục màu tròn hiển thị tọa độ cái điểm kết thúc của dòng tia, áp dụng phương trình đường thẳng tại điểm kết thúc $t = 1.5$:
$H_{outing\_point} = O + 1.5 \times D$
Do $D$ là Vector hướng (Chiều dài = 1), nhân độ dài $1.5$, cộng cho vị trí gốc tên lò, sẽ trả đích xác vị trí va chạm tại thành hang bên kia.

- **Trích xuất 30 tia gom thành SDF:**
Lúc này 1 Đỉnh phóng ra 30 tia, trả về 30 giá trị `min_t` (tức là 30 Outing Point nằm ở đâu đó trên các lớp thành hang bên kia). Có tia vô cực vì bắn ra vùng hư không lọt ra khe hở (`float('inf')`).
Ở **Dòng 147 - 158**, thuật toán loại bỏ tất cả khoảng cách Vô Cực, sau đó **Lấy Trung Bình Cộng** của các khoảng cách chạm thành tổ ong hợp lệ kia. 
Trung bình cộng đó chính là: **Shape Diameter ($SDF$)** của Tam giác A.
