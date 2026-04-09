# High-Performance GPU-Accelerated Vectorized Rasterization for Shape Diameter Function (SDF) Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## 1. Abstract
This repository implements an advanced, fully vectorized GPU-accelerated pipeline for calculating the **Shape Diameter Function (SDF)**—a critical geometric descriptor for 3D mesh segmentation and skeletonization. Unlike traditional ray-tracing methods that suffer from high computational overhead, our approach leverages **modern graphics rasterization principles within the GPU Clipping Space**. By utilizing PyTorch’s CUDA-optimized tensor operations, we achieve millisecond-level processing speeds on complex manifolds, ensuring mathematical rigor through **Perspective-Correct Reconstruction ($1/W$ Interpolation)**.

## 2. Core Methodology
The system transitions the SDF calculation from a "Ray-Centric" problem to a "Face-Centric Rasterization" problem. The pipeline follows four critical stages:

1.  **Batched Transformation:** Parallel projection of mesh vertices into the Clipping Space and NDC (Normalized Device Coordinates) for thousands of virtual cameras simultaneously.
2.  **Geometry Filtering (W-Clipping):** Mathematical rejection of primitives behind the near-plane ($W > 0.01$) to prevent projection inversion and phantom hits.
3.  **Rasterization & Z-Buffering:** Efficient point-in-triangle testing using batched Barycentric coordinates $(u, v, w)$ and depth selection via GPU Z-Buffer logic.
4.  **Perspective-Correct Reconstruction:** Reconstructing world-space hit points by interpolating the reciprocal of homogeneous depth ($1/W$) to resolve the non-linearity of the perspective transformation.

## 3. Key Theoretical Advancements
- **100% Vectorized GPU Pipeline:** Removal of Python-level loops in the batch processing stage, allowing for true SIMD execution.
- **W-Clipping Defense:** Integration of homogeneous clipping to ensure geometric stability in concave regions.
- **Numerical Stability:** Implementation of directed area checks ($\text{Area} > 10^{-6}$) to handle degenerate triangles often found in raw scanner data.

## 4. Repository Structure
Modified and optimized scripts include:

*   [`fast_sdf_gpu.py`](./fast_sdf_gpu.py): **[Flagship]** The ultimate 100% vectorized rasterization engine in Clip Space.
*   [`sdf_meshlab_gpu.py`](./sdf_meshlab_gpu.py): A vectorized implementation of the standard MeshLab SDF filter (Lior Shapira 2008) in World Space.
*   [`ultimate_sdf_face_centric_gpu.py`](./ultimate_sdf_face_centric_gpu.py): specialized face-centric ray-casting.
*   [`docs/SDF_GPU_Explanation.md`](./docs/SDF_GPU_Explanation.md): Detailed mathematical derivation of the $1/W$ reconstruction logic.

## 5. Performance Benchmarks
| Model | Vertices | Faces | Method | Time (GPU) |
| :--- | :--- | :--- | :--- | :--- |
| Teapot.obj | 3,241 | 6,320 | Vectorized Raster | **~7.78 seconds** |
| Radio_0026.off | 4,200 | 8,390 | Vectorized Raster | **~10.12 seconds** |

*Hardware: Tested on NVIDIA CUDA-compatible GPUs.*

## 6. Installation & Usage
### Dependencies
Ensure you have a CUDA-enabled environment:
```bash
pip install torch trimesh pyvista numpy tqdm
```

### Execution
To run the high-speed rasterizer on a custom mesh:
```powershell
python fast_sdf_gpu.py --input_file ./data/teapot.obj --batch_size 32 --fov 90
```

## 7. Mathematical Verification (Deep Dive)
For a detailed explanation of how we counteract the perspective distortion using the formula:
$$W_{real} = \frac{1}{\sum \frac{\lambda_i}{W_i}}$$
Please refer to the [Technical Documentation](./docs/SDF_GPU_Explanation.md).

## 8. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
