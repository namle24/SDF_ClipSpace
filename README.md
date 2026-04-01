# SDF_ClipSpace
Bộ công cụ tính toán SDF (Shape Diameter Function) sử dụng chùm tia song song (Orthographic Projection) trong Clip Space, tối ưu hóa bằng Numpy và PyTorch GPU.

## Tính năng
- `custom_ortho_sdf_gpu.py`: Rasterizer bằng PyTorch (hỗ trợ CUDA) để tính SDF siêu tốc.
- `sdf_calculator.py`: Thuật toán Cone of Rays truyền thống bằng Trimesh (Batching xử lý RAM thấp).
- `custom_ortho_sdf.py`: Pipeline đồ họa tùy chỉnh viết bằng Numpy (không dùng Raytracing engine).

## Cài đặt
```bash
pip install -r requirements.txt
```

## Cách dùng
```powershell
python custom_ortho_sdf_gpu.py --input_file path/to/model.obj --batch_size 32
```
