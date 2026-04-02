import trimesh
import numpy as np
import pyvista as pv
import time
import warnings
import torch
from tqdm import tqdm

@torch.no_grad()
def compute_sdf_cone_gpu(mesh, num_rays=30, cone_angle=120, ray_batch_size=2048):
    """
    SDF Calculation (Hình Nón - Shape Diameter Function chuẩn MeshLab).
    Loại bỏ Trimesh BVH CPU. Sử dụng thuật toán Möller-Trumbore Ray-Triangle Intersection 
    được Vector hóa 100% bằng PyTorch Tensor trên không gian CUDA.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[BẮT ĐẦU SÚNG BẮN TIA GPU - Möller–Trumbore RayTracer]")
    print(f"Thiết bị xử lý lõi: {device}")
    
    start_time = time.perf_counter()
    
    # 1. Tải lên VRAM toàn bộ dữ liệu hình học
    origins = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    
    N = len(origins)
    F = len(faces)
    
    V0 = origins[faces[:, 0]]
    V1 = origins[faces[:, 1]]
    V2 = origins[faces[:, 2]]
    
    E1 = V1 - V0 # (F, 3)
    E2 = V2 - V0 # (F, 3)
    
    print(f"[*] Đã tải Mesh [{N} vertices, {F} faces] vào vRAM.")
    
    # 2. Sinh chùm tia Toán học Spherical Cap Vectorized 100%
    anti_normals = -normals
    norms_val = torch.norm(anti_normals, dim=-1, keepdim=True)
    mask_zero = norms_val.squeeze(-1) < 1e-8
    anti_normals[mask_zero] = torch.tensor([0.0, 0.0, -1.0], device=device)
    norms_val[mask_zero] = 1.0
    w = anti_normals / norms_val
    
    temp = torch.zeros_like(w)
    temp[:, 0] = torch.where(torch.abs(w[:, 0]) > 0.1, 0.0, 1.0)
    temp[:, 1] = torch.where(torch.abs(w[:, 0]) > 0.1, 1.0, 0.0)
    
    u = torch.cross(temp, w, dim=-1)
    u = u / (torch.norm(u, dim=-1, keepdim=True) + 1e-8)
    v = torch.cross(w, u, dim=-1)
    
    max_angle_rad = np.radians(cone_angle / 2.0)
    cos_max = float(np.cos(max_angle_rad))
    
    r1 = torch.rand((N, num_rays), device=device)
    phi = r1 * 2 * torch.pi
    
    r2 = torch.rand((N, num_rays), device=device)
    z = r2 * (1.0 - cos_max) + cos_max
    r = torch.sqrt(torch.clamp(1.0 - z**2, min=0.0))
    
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    
    ray_dirs_cone = x.unsqueeze(-1) * u.unsqueeze(1) + y.unsqueeze(-1) * v.unsqueeze(1) + z.unsqueeze(-1) * w.unsqueeze(1)
    ray_dirs_cone = ray_dirs_cone / (torch.norm(ray_dirs_cone, dim=-1, keepdim=True) + 1e-8)
    
    # Offset origin tránh va chạm ngay ở bề mặt bắt đầu xuất phát
    eps = 1e-4
    safe_origins = origins - normals * eps
    
    # 3. Chuyển hóa toàn mảng thành mảng Tia 1 chiều
    ray_origins = safe_origins.unsqueeze(1).expand(N, num_rays, 3).reshape(-1, 3) # (N*R, 3)
    ray_dirs = ray_dirs_cone.reshape(-1, 3) # (N*R, 3)
    ray_to_vertex_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, num_rays).reshape(-1) # (N*R,)
    
    total_rays = len(ray_origins)
    print(f"[*] Tổng số lượng tia quang học: {total_rays:,}. Chế độ: Cone {cone_angle} độ.")
    
    hit_distances = torch.full((total_rays,), float('inf'), device=device)
    
    # 4. Kích hoạt Möller-Trumbore theo Batch (Tránh tràn VRAM do ma trận O(R*F))
    # Batch R rays đối đầu với toàn bộ F faces
    for i in tqdm(range(0, total_rays, ray_batch_size), desc="GPU Ray-Tracing"):
        end = min(i + ray_batch_size, total_rays)
        B = end - i
        
        O_batch = ray_origins[i:end] # (B, 3)
        D_batch = ray_dirs[i:end]    # (B, 3)
        V_idx_batch = ray_to_vertex_idx[i:end].unsqueeze(1) # (B, 1)
        
        # P = D x E2
        # Thay vì tốn RAM tạo bảng, ta cross 2 mảng cực lớn nhờ Broadcast
        P = torch.cross(D_batch.unsqueeze(1).expand(B, F, 3), E2.unsqueeze(0).expand(B, F, 3), dim=-1) # (B, F, 3)
        
        # a = E1 . P
        a = torch.sum(E1.unsqueeze(0) * P, dim=-1) # (B, F)
        
        # Lọc tia song song tam giác
        mask = torch.abs(a) > 1e-8
        
        # f = 1/a
        # Thay a bằng 1 ở những ô lỗi dể bàng quan chia không nổ tung
        safe_a = torch.where(mask, a, torch.tensor(1.0, device=device))
        f = 1.0 / safe_a
        
        s = O_batch.unsqueeze(1) - V0.unsqueeze(0) # (B, F, 3)
        
        # u = f * (s . P)
        u = f * torch.sum(s * P, dim=-1)
        mask = mask & (u >= 0.0) & (u <= 1.0)
        
        # Q = s x E1
        Q = torch.cross(s, E1.unsqueeze(0).expand(B, F, 3), dim=-1)
        
        # v = f * (D . Q)
        v = f * torch.sum(D_batch.unsqueeze(1) * Q, dim=-1)
        mask = mask & (v >= 0.0) & (u + v <= 1.0)
        
        # t = f * (E2 . Q)
        t = f * torch.sum(E2.unsqueeze(0) * Q, dim=-1)
        mask = mask & (t > 1e-4) # Chỉ lấy điểm phía xa (màng đối diện)
        
        # Áp dụng bộ lọc Umbrella khắt khe: 
        # Mặt bị đâm trúng không được phép chứa điểm nguồn bắn tia ra
        is_umbrella = (faces[None, :, 0] == V_idx_batch) | (faces[None, :, 1] == V_idx_batch) | (faces[None, :, 2] == V_idx_batch)
        mask = mask & (~is_umbrella)
        
        # Những điểm xịt thì Z là vô tận
        t = torch.where(mask, t, torch.tensor(float('inf'), device=device))
        
        # Lấy khoảng cách tia đâm gần ống kính góc nhất
        min_t, _ = torch.min(t, dim=-1) # (B,)
        hit_distances[i:end] = min_t
        
    print(f"[*] Thuật toán Möller-Trumbore kết thúc.")
    
    # 5. Lọc mảng tia lỗi và tính Trung bình (Mean Inverse Thickness) cho từng đỉnh
    valid_hits = hit_distances < float('inf')
    
    sdf_values = torch.zeros(N, device=device)
    
    # Thu gom các tia trúng về đúng đỉnh của nó
    # Có thể dùng scatter_add trung bình, nhưng có đỉnh không trúng tia nào
    counts = torch.zeros(N, device=device)
    safe_hits = torch.where(valid_hits, hit_distances, torch.tensor(0.0, device=device))
    
    sdf_values.scatter_add_(0, ray_to_vertex_idx, safe_hits)
    counts.scatter_add_(0, ray_to_vertex_idx, valid_hits.to(torch.float32))
    
    # Trị số trung bình chuẩn
    final_sdf = torch.where(counts > 0, sdf_values / counts, torch.tensor(0.0, device=device))
    
    end_time = time.perf_counter()
    print(f"\n=======================================================")
    print(f"[*] THỜI GIAN TÍNH TOÁN SDF (GPU MÖLLER-TRUMBORE): {end_time - start_time:.4f} giây")
    print(f"=======================================================\n")
    
    return final_sdf.cpu().numpy()

def main():
    model_path = "data/teapot.obj"
    try:
        mesh = trimesh.load(model_path, force='mesh')
        mesh.fix_normals()
        mesh.process()
    except Exception as e:
        print(f"Lỗi load mesh: {e}")
        return

    # Chạy thuật toán chùm tia (SDF MeshLab) TRÊN GPU NVIDIA
    sdf_values = compute_sdf_cone_gpu(mesh, num_rays=30, cone_angle=120, ray_batch_size=4096)

    # Hiển thị vùng màu mượt (MeshLab Style)
    pv_mesh = pv.wrap(mesh)
    pv_mesh.point_data['SDF_Thickness_Cone_GPU'] = sdf_values
    
    plotter = pv.Plotter(title="SDF Cone GPU Visualization - MeshLab Style")
    plotter.add_mesh(
        pv_mesh, 
        scalars='SDF_Thickness_Cone_GPU', 
        cmap='jet_r', 
        smooth_shading=False,
        show_edges=False,
        scalar_bar_args={'title': "SDF Thickness (GPU Cone)"}
    )
    plotter.set_background('white')
    plotter.add_axes()
    plotter.show()

if __name__ == "__main__":
    main()
