"""
hybrid_sdf_gpu.py — Hybrid View-Projection SDF trên GPU

Kết hợp:
  - Tư duy toán học của Giáo sư: look_at + perspective → Clip Space
    (chùm tia nón phối cảnh được "nắn thẳng" thành chùm tia song song dọc trục Z)
  - Sức mạnh GPU PyTorch: Toàn bộ phép biến đổi, lọc Bounding-Box 2D,
    Point-in-Triangle 2D, Z-Buffer Interpolation đều chạy trên Tensor CUDA.

Ý tưởng cốt lõi (trích dẫn cho Paper):
  1. Với mỗi đỉnh (hoặc face center), tạo ma trận View V = look_at(eye, target)
     và ma trận Projection P = perspective(fov, near, far).
  2. Nhân PV @ vertices_homo.T  →  chuyển MỌI đỉnh sang Clip Space cùng lúc.
  3. Sau Perspective Divide (xyz /= w), chùm tia hình nón trong World Space
     biến thành chùm tia SONG SONG dọc trục Z trong Clip Space.
  4. Sinh lưới tia đều trên mặt phẳng XY (vòng tròn đồng tâm) trong Clip Space.
  5. Dùng Bounding-Box 2D + Point-in-Triangle 2D Vectorized để tìm mặt bị đâm.
  6. Nội suy Z bằng Barycentric → tìm điểm chạm gần nhất.
  7. Tính khoảng cách Euclid trong World Space → SDF.
"""

import trimesh
import numpy as np
import pyvista as pv
import time
import torch
from tqdm import tqdm


# 1. CAMERA MATRICES — Dịch nguyên bản Toán học Giáo sư sang PyTorch

def look_at_torch(eye, target, device='cuda'):
    """
    Xây dựng ma trận View 4×4 (World → Camera) trên GPU.
    Giữ nguyên logic Gram–Schmidt: forward → random up → right → true_up.
    """
    forward = target - eye
    forward = forward / (torch.norm(forward) + 1e-12)

    # Sinh vector up ngẫu nhiên rồi chiếu vuông góc với forward (Gram–Schmidt)
    up = torch.randn(3, device=device, dtype=torch.float32)
    up = up - torch.dot(up, forward) * forward  # Gram-Schmidt
    up = up / (torch.norm(up) + 1e-12)

    right = torch.linalg.cross(forward, up)
    right = right / (torch.norm(right) + 1e-12)

    true_up = torch.linalg.cross(right, forward)

    rotation = torch.tensor([
        [right[0],    right[1],    right[2],    0],
        [true_up[0],  true_up[1],  true_up[2],  0],
        [-forward[0], -forward[1], -forward[2], 0],
        [0,           0,           0,           1]
    ], dtype=torch.float32, device=device)

    translation = torch.eye(4, dtype=torch.float32, device=device)
    translation[0, 3] = -eye[0]
    translation[1, 3] = -eye[1]
    translation[2, 3] = -eye[2]

    return rotation @ translation


def perspective_torch(fov, near, far, device='cuda'):
    """
    Ma trận Projection 4×4 (Camera → Clip) trên GPU.
    """
    cot = 1.0 / torch.tan(fov / 2)
    P = torch.zeros((4, 4), dtype=torch.float32, device=device)
    P[0, 0] = cot
    P[1, 1] = cot
    P[2, 2] = -far / (far - near)
    P[2, 3] = -(near * far) / (far - near)
    P[3, 2] = -1.0
    return P


# 2. SINH LƯỚI TIA VÒNG TRÒN ĐỒNG TÂM — Giữ nguyên logic Ring của Thầy

def generate_ring_rays_torch(fov, num_rings=5, num_rays_per_ring=10, device='cuda'):
    """
    Sinh chùm tia dạng vòng tròn đồng tâm bên trong Clip Space.
    Giữ nguyên logic ring/angle nhưng output là Tensor.
    """
    tan_fov = torch.tan(fov / 2)
    all_xy = []
    for ring in range(num_rings):
        r = (ring + 0.5) / num_rings
        num_points = int(num_rays_per_ring * (ring + 1))
        angles = torch.linspace(0, 2 * torch.pi, num_points + 1, device=device)[:-1]
        x = r * torch.cos(angles) * tan_fov
        y = r * torch.sin(angles) * tan_fov
        all_xy.append(torch.stack([x, y], dim=-1))
    return torch.cat(all_xy, dim=0)  # (R, 2)


# 3. ENGINE CHÍNH — Hybrid View-Projection + GPU Vectorized Rasterizer
@torch.no_grad()
def compute_hybrid_sdf_gpu(mesh, fov_deg=90, num_rings=5, num_rays_per_ring=10,
                           vertex_batch_size=64):
    """
    Tính SDF trên từng đỉnh bằng phương pháp Hybrid:
      - View-Projection camera (Giáo sư)
      - Point-in-Triangle 2D + Z-Buffer trên GPU (Pipeline chúng ta)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[BẮT ĐẦU HYBRID VIEW-PROJECTION SDF]")
    print(f"Thiết bị xử lý lõi: {device}")

    start_time = time.perf_counter()

    # --- Tải dữ liệu lên VRAM ---
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)  # (V, 3)
    normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)  # (F, 3)

    V_count = len(vertices)
    F_count = len(faces)
    print(f"[*] Mesh: {V_count} đỉnh, {F_count} tam giác → vRAM.")

    # Homogeneous coordinates cho toàn bộ đỉnh (tính 1 lần duy nhất)
    ones = torch.ones((V_count, 1), dtype=torch.float32, device=device)
    verts_homo = torch.cat([vertices, ones], dim=-1)  # (V, 4)

    # Tham số camera
    fov = torch.tensor(np.radians(fov_deg), dtype=torch.float32, device=device)
    near = torch.tensor(0.001, dtype=torch.float32, device=device)
    # far tự động = Đường chéo BBox + 1
    bbox_diag = torch.norm(vertices.max(dim=0).values - vertices.min(dim=0).values)
    far = bbox_diag + 1.0

    # Sinh lưới tia 1 lần (dùng lại cho mọi đỉnh)
    ray_xy = generate_ring_rays_torch(fov, num_rings, num_rays_per_ring, device)  # (R, 2)
    R = len(ray_xy)
    print(f"[*] Số tia / đỉnh: {R}  |  FOV: {fov_deg}°  |  Near/Far: {near.item():.3f}/{far.item():.1f}")

    # Ma trận Projection (chung cho tất cả — chỉ View thay đổi theo đỉnh)
    P = perspective_torch(fov, near, far, device)  # (4, 4)

    sdf_values = torch.zeros(V_count, dtype=torch.float32, device=device)

    # --- Xử lý theo từng Batch đỉnh ---
    for b_start in tqdm(range(0, V_count, vertex_batch_size),
                        desc="Hybrid VP-SDF Batches"):
        b_end = min(b_start + vertex_batch_size, V_count)

        for idx in range(b_start, b_end):
            # ===== BƯỚC A: Xây Camera cho đỉnh này (giống Thầy) =====
            inward = -normals[idx]
            norm_val = torch.norm(inward)
            if norm_val < 1e-8:
                continue
            inward = inward / norm_val

            eye = vertices[idx] + 0.001 * inward
            target = eye + inward
            V_mat = look_at_torch(eye, target, device)  # (4, 4)
            PV = P @ V_mat  # (4, 4)

            # ===== BƯỚC B: Chuyển MỌI đỉnh sang Clip Space 1 phát =====
            verts_clip = (PV @ verts_homo.T).T  # (V, 4)
            w_clip = verts_clip[:, 3:4]
            # Perspective Divide  —  Chùm nón nắn thành chùm song song
            w_safe = torch.where(torch.abs(w_clip) > 1e-8, w_clip,
                                 torch.tensor(1e-8, device=device))
            verts_ndc = verts_clip[:, :3] / w_safe  # (V, 3)

            # ===== BƯỚC C: Lấy tọa độ tam giác trong NDC =====
            tri_v0 = verts_ndc[faces[:, 0]]  # (F, 3)
            tri_v1 = verts_ndc[faces[:, 1]]
            tri_v2 = verts_ndc[faces[:, 2]]

            # Bounding Box 2D cho mỗi tam giác ♦ Vectorized
            tri_min_x = torch.minimum(torch.minimum(tri_v0[:, 0], tri_v1[:, 0]), tri_v2[:, 0])
            tri_max_x = torch.maximum(torch.maximum(tri_v0[:, 0], tri_v1[:, 0]), tri_v2[:, 0])
            tri_min_y = torch.minimum(torch.minimum(tri_v0[:, 1], tri_v1[:, 1]), tri_v2[:, 1])
            tri_max_y = torch.maximum(torch.maximum(tri_v0[:, 1], tri_v1[:, 1]), tri_v2[:, 1])

            # Lọc mặt nằm phía trước Camera (z < 0 trong Camera Space)
            verts_cam = (V_mat @ verts_homo.T).T[:, :3]
            face_center_z = (verts_cam[faces[:, 0], 2] +
                             verts_cam[faces[:, 1], 2] +
                             verts_cam[faces[:, 2], 2]) / 3.0
            in_front = face_center_z < -0.005  # (F,) bool

            # Lọc Umbrella (mặt chứa chính đỉnh gốc)
            is_umbrella = ((faces[:, 0] == idx) |
                           (faces[:, 1] == idx) |
                           (faces[:, 2] == idx))
            valid_face_mask = in_front & (~is_umbrella)  # (F,)

            if valid_face_mask.sum() == 0:
                continue

            # Chỉ giữ lại mặt hợp lệ
            vf_idx = torch.where(valid_face_mask)[0]  # indices
            vf_v0 = tri_v0[vf_idx]  # (Fv, 3)
            vf_v1 = tri_v1[vf_idx]
            vf_v2 = tri_v2[vf_idx]
            vf_min_x = tri_min_x[vf_idx]
            vf_max_x = tri_max_x[vf_idx]
            vf_min_y = tri_min_y[vf_idx]
            vf_max_y = tri_max_y[vf_idx]
            Fv = len(vf_idx)

            # ===== BƯỚC D: Bắn tia song song trong Clip Space =====
            # ray_xy shape (R, 2).  Broadcast so sánh BBox: (R, Fv)
            px = ray_xy[:, 0].unsqueeze(1)  # (R, 1)
            py = ray_xy[:, 1].unsqueeze(1)

            in_bbox = ((px >= vf_min_x.unsqueeze(0)) &
                       (px <= vf_max_x.unsqueeze(0)) &
                       (py >= vf_min_y.unsqueeze(0)) &
                       (py <= vf_max_y.unsqueeze(0)))  # (R, Fv)

            # ===== BƯỚC E: Point-in-Triangle 2D (Edge Function) =====
            # Chỉ chạy trên các cặp (ray, face) qua BBox
            v0x = vf_v0[:, 0].unsqueeze(0)  # (1, Fv)
            v0y = vf_v0[:, 1].unsqueeze(0)
            v1x = vf_v1[:, 0].unsqueeze(0)
            v1y = vf_v1[:, 1].unsqueeze(0)
            v2x = vf_v2[:, 0].unsqueeze(0)
            v2y = vf_v2[:, 1].unsqueeze(0)

            w0 = (v2x - v1x) * (py - v1y) - (v2y - v1y) * (px - v1x)
            w1 = (v0x - v2x) * (py - v2y) - (v0y - v2y) * (px - v2x)
            w2 = (v1x - v0x) * (py - v0y) - (v1y - v0y) * (px - v0x)

            has_neg = (w0 < 0) | (w1 < 0) | (w2 < 0)
            has_pos = (w0 > 0) | (w1 > 0) | (w2 > 0)
            inside = ~(has_neg & has_pos)  # (R, Fv)

            hit_mask = in_bbox & inside  # (R, Fv)

            # ===== BƯỚC F: Z-Buffer Interpolation (Barycentric) =====
            v0z = vf_v0[:, 2].unsqueeze(0)  # (1, Fv)
            v1z = vf_v1[:, 2].unsqueeze(0)
            v2z = vf_v2[:, 2].unsqueeze(0)

            Area = (v1x - v0x) * (v2y - v0y) - (v1y - v0y) * (v2x - v0x)
            Z_hit = (w0 * v0z + w1 * v1z + w2 * v2z) / (Area + 1e-12)  # (R, Fv)

            # Validate: Nằm trong tam giác, Area hợp lệ, Z > 0 (phía trước)
            valid = hit_mask & (torch.abs(Area) > 1e-12) & (Z_hit > 0) & (Z_hit < 1)
            Z_hit = torch.where(valid, Z_hit, torch.tensor(float('inf'), device=device))

            # Tìm mặt gần nhất cho từng tia
            min_Z, min_idx = torch.min(Z_hit, dim=1)  # (R,)

            # ===== BƯỚC G: Tính khoảng cách World Space =====
            # Lấy face index thực tế trong mảng gốc
            hit_face_global = vf_idx[min_idx]  # (R,)
            hit_valid = min_Z < float('inf')

            if hit_valid.sum() == 0:
                continue

            # Tâm của face bị đâm trúng (trong World Space)
            hit_faces_v0 = vertices[faces[hit_face_global, 0]]
            hit_faces_v1 = vertices[faces[hit_face_global, 1]]
            hit_faces_v2 = vertices[faces[hit_face_global, 2]]
            hit_centers = (hit_faces_v0 + hit_faces_v1 + hit_faces_v2) / 3.0

            # Khoảng cách Euclid từ đỉnh gốc → Tâm mặt bị đâm
            dists = torch.norm(hit_centers - vertices[idx].unsqueeze(0), dim=1)
            dists = torch.where(hit_valid, dists, torch.tensor(0.0, device=device))

            valid_count = hit_valid.sum()
            if valid_count > 0:
                sdf_values[idx] = dists.sum() / valid_count

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    avg_sdf = sdf_values.mean().item()
    print(f"\n=======================================================")
    print(f"[*] THỜI GIAN TÍNH TOÁN HYBRID VP-SDF (GPU): {elapsed:.4f} giây")
    print(f"[*] SDF Trung bình: {avg_sdf:.4f}")
    print(f"=======================================================\n")

    return sdf_values.cpu().numpy()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Hybrid View-Projection SDF GPU (Giáo sư + PyTorch)")
    parser.add_argument("--input_file", type=str, default="data/radio_0026.off",
                        help="Đường dẫn model 3D")
    parser.add_argument("--fov", type=int, default=90, help="Góc FOV (độ)")
    parser.add_argument("--num_rings", type=int, default=5, help="Số vòng tia")
    parser.add_argument("--rays_per_ring", type=int, default=10,
                        help="Số tia / vòng (vòng ngoài nhân bội)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch đỉnh xử lý song song")
    args = parser.parse_args()

    try:
        mesh = trimesh.load(args.input_file, force='mesh')
        mesh.fix_normals()
        mesh.process()
        if len(mesh.vertices) == 0:
            raise ValueError("Mesh trống!")
    except Exception as e:
        print(f"Lỗi tải mesh: {e}")
        return

    sdf = compute_hybrid_sdf_gpu(
        mesh,
        fov_deg=args.fov,
        num_rings=args.num_rings,
        num_rays_per_ring=args.rays_per_ring,
        vertex_batch_size=args.batch_size
    )

    # Visualization — MeshLab Style
    pv_mesh = pv.wrap(mesh)
    pv_mesh.point_data['Hybrid_VP_SDF'] = sdf

    plotter = pv.Plotter(title="Hybrid View-Projection SDF (GPU) — MeshLab Style")
    plotter.add_mesh(
        pv_mesh,
        scalars='Hybrid_VP_SDF',
        cmap='jet_r',
        smooth_shading=False,
        show_edges=False,
        scalar_bar_args={'title': "SDF Thickness (Hybrid VP-GPU)"}
    )
    plotter.set_background('white')
    plotter.add_axes()
    plotter.show()


if __name__ == "__main__":
    main()
