You are a senior GPU / PyTorch / Computer Graphics engineer helping optimize a research prototype for a differentiable Shape Diameter Function (SDF) computation pipeline.

## 🧩 Context

The current implementation computes SDF using a **clip-space rasterization pipeline** in PyTorch:

* Transform mesh vertices → clip space → NDC
* Generate rays in NDC
* Perform rasterization-like intersection using:

  * bounding box filtering
  * barycentric coordinates
  * Z-buffer (min depth)
* Compute SDF as average ray distance

The current implementation is:

* Fully vectorized
* Differentiable (important constraint)
* GPU-based

However, it suffers from:

* **Severe memory explosion: O(B × R × F)**
* **Out-of-memory errors on GPU**
* Poor scalability when increasing ray count

⚠️ IMPORTANT:
You MUST preserve:

* Differentiability (no external C++/CUDA kernels)
* PyTorch tensor pipeline
* Clip-space rasterization logic (DO NOT replace with ray-triangle CPU or BVH methods)

---

## 🎯 Your Goal

Refactor and optimize the code to:

1. Eliminate OOM issues
2. Improve scalability to support 10k–20k rays
3. Maintain or improve numerical correctness
4. Keep the pipeline differentiable
5. Preserve the original mathematical logic (clip-space rasterization)

---

## 🔧 REQUIRED MODIFICATIONS (MANDATORY)

### 1. Ray Chunking (CRITICAL)

Replace all full-ray operations:

```
(B, R, F)
```

with chunked processing:

```
(B, R_chunk, F)
```

Implementation requirements:

* Introduce a parameter: `ray_chunk_size` (e.g., 256–1024)
* Loop over rays:
  for r_start in range(0, R, ray_chunk_size):
* Use px_sub, py_sub slices
* Accumulate final SDF results correctly across chunks

---

### 2. Remove ALL Large Tensor Expansions

Specifically eliminate patterns like:

```
expand(B, R, F, 2)
```

Replace with:

* broadcasting
* unsqueeze only when necessary

Goal:

* No tensor of shape (B, R, F, 2) or larger should be materialized

---

### 3. Tile-Based Face Culling (CORE OPTIMIZATION)

Implement screen-space binning:

1. Define grid resolution:
   grid_size = 16 or 32

2. For each triangle:

   * Compute bounding box in NDC
   * Map to tile indices

3. For each ray:

   * Compute its tile index

4. Filter candidate faces:

   Only keep faces whose tile overlaps with the ray tile

This should:

* Replace or augment existing `in_bbox`
* Reduce effective F → k (k ≪ F)

---

### 4. Keep Barycentric + Raster Logic

DO NOT:

* Replace with ray-triangle intersection
* Remove clip-space pipeline

You must:

* Keep barycentric coordinate computation
* Keep perspective-correct interpolation
* Keep current geometric logic intact

---

### 5. Optional but Recommended: Soft Z-buffer

Replace hard min:

```
torch.min(Z, dim=...)
```

with differentiable softmin:

```
Z_soft = -logsumexp(-alpha * Z) / alpha
```

Requirements:

* Add parameter alpha (e.g., 50–200)
* Ensure numerical stability

---

### 6. Mixed Precision (Memory Optimization)

Wrap heavy computation with:

```
torch.cuda.amp.autocast()
```

Optional:

* allow float16 tensors where safe

---

### 7. Avoid Redundant Tensor Creation

Move repeated allocations outside loops:

* torch.arange
* constants
* grids

---

## ⚙️ EXPECTED OUTPUT

You must return:

1. ✅ Refactored function (clean, readable)
2. ✅ Clearly marked sections:

   * Ray chunking
   * Tile culling
   * Memory optimization
3. ✅ Comments explaining:

   * WHY each change improves performance
   * Complexity reduction
4. ✅ Ensure code runs end-to-end

---

## 🚫 DO NOT DO

* Do NOT switch to CPU methods
* Do NOT use external geometry libraries (PyMesh, etc.)
* Do NOT remove differentiability
* Do NOT rewrite into a completely different algorithm

---

## 🧪 EXTRA (IF POSSIBLE)

Add optional debug outputs:

* memory usage estimate
* number of candidate faces per ray

---

## 🧠 Mental Model You Should Follow

Original:
O(R × F)

Target:
O(R × k), where k ≪ F

---

## 🎯 Final Goal

The final implementation should:

* Handle large meshes (100k+ faces)
* Support 10k+ rays
* Avoid GPU OOM
* Remain suitable for integration into deep learning pipelines

---

Now refactor the provided code accordingly.
