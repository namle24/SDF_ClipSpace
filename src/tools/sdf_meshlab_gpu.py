import trimesh
import numpy as np
import pyvista as pv
import time
import torch
from tqdm import tqdm


@torch.no_grad()
def compute_sdf_meshlab_gpu(mesh, num_rays=30, cone_angle=120, ray_batch_size=4096):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  SDF MESHLAB CHUẨN (WORLD-SPACE MT GPU)")
    print(f"  Thiết bị lôgic: {device}")
    print(f"  Chế độ: {num_rays} tia ngẫu nhiên, góc mở {cone_angle}°")
    print(f"{'='*60}")

    start_time = time.perf_counter()

    # 1. Nạp thẳng dữ liệu lên GPU
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)

    N = len(vertices)
    F = len(faces)

    # Cache trước 3 đỉnh tam giác
    V0 = vertices[faces[:, 0]]
    V1 = vertices[faces[:, 1]]
    V2 = vertices[faces[:, 2]]
    E1 = V1 - V0
    E2 = V2 - V0

    print(f"[*] Đã tải Mesh [{N} đỉnh, {F} tam giác] vào vRAM.")

    # 2. Xây dựng trục tọa độ gốc cho phép xòe hình nón trên GPU
    # Đảo pháp tuyến để đâm vào TRONG
    anti_normals = -normals
    norms_val = torch.norm(anti_normals, dim=-1, keepdim=True)
    mask_zero = (norms_val.squeeze(-1) < 1e-8)
    # Xử lý các pháp tuyến bị hỏng
    anti_normals[mask_zero] = torch.tensor([0.0, 0.0, -1.0], device=device)
    norms_val[mask_zero] = 1.0
    w = anti_normals / norms_val  # Trục w chuẩn hóa

    # Tạo vector u vuông góc với w
    temp = torch.zeros_like(w)
    temp[:, 0] = torch.where(torch.abs(w[:, 0]) > 0.1, 0.0, 1.0)
    temp[:, 1] = torch.where(torch.abs(w[:, 0]) > 0.1, 1.0, 0.0)

    u = torch.linalg.cross(temp, w)
    u_norm = torch.norm(u, dim=-1, keepdim=True)
    u = u / (u_norm + 1e-8)
    
    # Tạo vector v vuông góc với u và w
    v = torch.linalg.cross(w, u)

    # 3. Phân bố Spherical Cap (ngẫu nhiên chuẩn MeshLab)
    max_angle_rad = np.radians(cone_angle / 2.0)
    cos_max = float(np.cos(max_angle_rad))

    rand_phi = torch.rand((N, num_rays), device=device) * 2.0 * torch.pi
    rand_z = torch.rand((N, num_rays), device=device) * (1.0 - cos_max) + cos_max
    rand_r = torch.sqrt(torch.clamp(1.0 - rand_z**2, min=0.0))

    x = rand_r * torch.cos(rand_phi)
    y = rand_r * torch.sin(rand_phi)

    # Biến đổi tổ hợp tuyến tính x*u + y*v + z*w (Vectorized)
    ray_dirs = (x.unsqueeze(-1) * u.unsqueeze(1) + 
                y.unsqueeze(-1) * v.unsqueeze(1) + 
                rand_z.unsqueeze(-1) * w.unsqueeze(1))
                
    # NORMALIZE CHUẨN: Để t sau này chính là khoảng cách Euclid
    ray_dirs = ray_dirs / (torch.norm(ray_dirs, dim=-1, keepdim=True) + 1e-12)

    # 4. Thiết lập gốc tia
    eps = 1e-4
    safe_origins = vertices - normals * eps #gốc tia, điểm bắn đầu bắn

    # Flatten thành chùm tia 1 chiều để vứt lên GPU
    O_flat = safe_origins.unsqueeze(1).expand(N, num_rays, 3).reshape(-1, 3)
    D_flat = ray_dirs.reshape(-1, 3)
    vertex_idx_flat = torch.arange(N, device=device).unsqueeze(1).expand(N, num_rays).reshape(-1)

    total_rays = len(O_flat)
    print(f"[*] Sẵn sàng bắn: {total_rays:,} tia trong Vector Space.")

    # 5. Möller-Trumbore Vectorized trên GPU
    hit_distances = torch.full((total_rays,), float('inf'), device=device)

    for i in tqdm(range(0, total_rays, ray_batch_size), desc="World-Space MT"):
        end = min(i + ray_batch_size, total_rays)
        B = end - i
        
        O_batch = O_flat[i:end]  # (B, 3)
        D_batch = D_flat[i:end]  # (B, 3)
        v_idx_batch = vertex_idx_flat[i:end].unsqueeze(1)  # (B, 1)

        # Lọc Umbrella để tia không va vào chính những tam giác kề với đỉnh phát ra tia
        is_umbrella = ((faces[None, :, 0] == v_idx_batch) | 
                       (faces[None, :, 1] == v_idx_batch) | 
                       (faces[None, :, 2] == v_idx_batch))  # (B, F)

        # P = D_batch x E2
        P = torch.linalg.cross(D_batch.unsqueeze(1).expand(B, F, 3), E2.unsqueeze(0).expand(B, F, 3))
        
        # a = E1 . P
        a = torch.sum(E1.unsqueeze(0) * P, dim=-1)
        
        # Culling the rays that are parallel to the triangle
        face_ok = torch.abs(a) > 1e-8
        
        safe_a = torch.where(face_ok, a, torch.ones_like(a))
        f = 1.0 / safe_a
        
        # s = O - V0
        s = O_batch.unsqueeze(1) - V0.unsqueeze(0)
        
        # u = f * (s . P)
        u_bary = f * torch.sum(s * P, dim=-1)
        face_ok = face_ok & (u_bary >= 0.0) & (u_bary <= 1.0)
        
        # Q = s x E1
        Q = torch.linalg.cross(s, E1.unsqueeze(0).expand(B, F, 3))
        
        # v = f * (D . Q)
        v_bary = f * torch.sum(D_batch.unsqueeze(1) * Q, dim=-1)
        face_ok = face_ok & (v_bary >= 0.0) & (u_bary + v_bary <= 1.0)

        # t = f * (E2 . Q) — T CHÍNH LÀ KHOẢNG CÁCH EUCLID TRONG WORLD SPACE CHÍNH XÁC NHẤT
        t = f * torch.sum(E2.unsqueeze(0) * Q, dim=-1)
        face_ok = face_ok & (t > 1e-4)
        
        # Áp Umbrella
        face_ok = face_ok & (~is_umbrella)

        # Nếu không trúng, gán t bằng vô cực
        t = torch.where(face_ok, t, torch.tensor(float('inf'), device=device))
        
        # Tìm khoảng cách nhỏ nhất (đâm vào màng chắn đầu tiên)
        min_t, _ = torch.min(t, dim=-1)
        hit_distances[i:end] = min_t

    # 6. Gom khoảng cách về lại đỉnh
    valid_hits = hit_distances < float('inf')
    
    safed_hits = torch.where(valid_hits, hit_distances, torch.tensor(0.0, device=device))
    
    sum_sdf = torch.zeros(N, device=device)
    sum_sdf.scatter_add_(0, vertex_idx_flat, safed_hits)
    
    count_hits = torch.zeros(N, device=device)
    count_hits.scatter_add_(0, vertex_idx_flat, valid_hits.float())
    
    # Tính trung bình khoảng cách SDF
    final_sdf = torch.where(count_hits > 0, sum_sdf / count_hits, torch.zeros_like(sum_sdf))
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    avg = final_sdf.mean().item()

    print(f"\n============================================================")
    print(f"  THỜI GIAN GPU: {elapsed:.4f} giây")
    print(f"  SDF Trung bình: {avg:.4f}")
    print(f"============================================================\n")

    return final_sdf.cpu().numpy()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/teapot.obj")
    parser.add_argument("--fov", type=int, default=120, help="Góc xòe của chùm tia nón")
    parser.add_argument("--num_rays", type=int, default=30, help="Số tia sinh ra cho mỗi đỉnh")
    parser.add_argument("--ray_batch", type=int, default=4096, help="Tối ưu dung lượng vRAM")
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

    sdf_values = compute_sdf_meshlab_gpu(
        mesh,
        num_rays=args.num_rays,
        cone_angle=args.fov,
        ray_batch_size=args.ray_batch
    )

    # Dựng hình
    pv_mesh = pv.wrap(mesh)
    pv_mesh.point_data['SDF_MeshLab_Exact'] = sdf_values

    plotter = pv.Plotter(title="SDF: Exact MeshLab Mathematics in CUDA")
    plotter.add_mesh(
        pv_mesh,
        scalars='SDF_MeshLab_Exact',
        cmap='jet_r',
        smooth_shading=False,
        show_edges=True,
        scalar_bar_args={'title': "Shape Diameter"}
    )
    plotter.set_background('white')
    plotter.add_axes()
    plotter.show()


if __name__ == "__main__":
    main()
