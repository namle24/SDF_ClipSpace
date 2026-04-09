# RESEARCH ASSISTANT CONTEXT: 3D SHAPE DIAMETER FUNCTION (SDF) OPTIMIZATION

## 1. PROJECT OBJECTIVE

Develop a robust, highly optimized, and vectorized Python pipeline to calculate the Shape Diameter Function (SDF) for the ModelNet40 dataset. The output will be used as localized point features for training a 3D Point Cloud Transformer model.

## 2. BACKGROUND & CURRENT STATE

We are calculating the internal local thickness of 3D meshes using Ray-Casting.

- **Ground Truth:** The original SDF paper (Shapira et al., 2008) uses a "Cone of Rays" (Perspective projection, 30 rays, 120-degree angle) sent strictly opposite to the normal vector. This yields smooth, pose-invariant local thickness.
- **Professor's Hypothesis:** To speed up calculations and avoid perspective distortion (where rays diverge too much in thick parts of the mesh), the Professor suggested using "Parallel Rays" (Orthographic projection) inspired by Camera-to-Clip-Space rasterization techniques.
- **Tech Stack:** Python, `trimesh` (backed by `pyembree` for fast C++ BVH ray-triangle intersections), `numpy`, `pyvista`.

## 3. THE PROBLEM (CURRENT BUG)

We implemented the Professor's idea by generating a cylinder of parallel rays (sampled uniformly on a local disk around each vertex) and shooting them along the anti-normal direction.

- **Observation:** The result is highly noisy and inaccurate compared to the Cone of Rays approach.
- **Suspected Issues:** 1. **Scale Mismatch:** The sampling radius of the disk might not be normalized to the bounding box of the specific mesh. 2. **Grazing Angles:** Parallel rays might be hitting internal faces at extreme grazing angles, returning near-infinite distances. 3. **Lack of Volumetric Exploration:** Unlike a cone that expands to average a wider target area, a strict cylinder of parallel rays is too easily trapped by local self-intersections.

## 4. AGENT TASKS & DIRECTIVES

As an AI Computer Graphics Expert, your tasks are:

**Task A: Mathematical Debugging**
Analyze why the parallel ray setup fails to produce smooth thickness values. Propose a mathematical fix for the disk sampling radius (e.g., dynamic radius based on local curvature or global bounding box) to ensure stable averaging.

**Task B: Hybrid Algorithm Design**
The Professor wants the speed and linear nature of Orthographic projection (Parallel rays), but we need the smoothness of Perspective projection (Cone of rays). Formulate a hybrid algorithm or suggest a filtering technique (e.g., median filtering, outlier rejection using $1.5 \times IQR$, or bilateral filtering on the mesh) to clean the parallel ray results.

**Task C: Optimization for Batch Processing**
The final code must run efficiently over 12,000 `.obj` files.

- Strictly avoid $O(N^2)$ Python loops.
- Leverage `trimesh.ray.intersects_location(multiple_hits=False)` with massive vectorized arrays.
- Ensure the mathematical equation for rays $R(t) = O + tD$ accurately captures the Euclidean distance from the ray origin to the exact intersection point, NOT the distance between face centers.

## 5. OUTPUT EXPECTATIONS

- Provide updated, clean, and fully commented Python code.
- Explain the geometric reasoning behind any parameter changes (especially sampling radius and outlier rejection).

## For env:

conda activate cross_sdf
