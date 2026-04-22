# ─────────────────────────────────────────────
# PRECOMPUTE WORLD VERTICES (OUTSIDE LOOP)
# ─────────────────────────────────────────────
V0w = V_world[F[:,0]].to(device)  # (F,3)
V1w = V_world[F[:,1]].to(device)
V2w = V_world[F[:,2]].to(device)

INF = torch.tensor(1e6, device=device)

# ─────────────────────────────────────────────
# RAY CHUNK LOOP (CORE)
# ─────────────────────────────────────────────
for r_start in range(0, R_total, ray_chunk):
    r_end = min(r_start + ray_chunk, R_total)

    px = px_all[:, r_start:r_end]   # (1, R_c, 1)
    py = py_all[:, r_start:r_end]

    R_c = px.shape[1]

    # ─────────────────────────
    # TILE LOOKUP (PER RAY)
    # ─────────────────────────
    ray_tx = ((px + 1) * 0.5 * grid_size).long().clamp(0, grid_size-1)
    ray_ty = ((py + 1) * 0.5 * grid_size).long().clamp(0, grid_size-1)

    # (B, R_c, F)
    ray_tx = ray_tx.view(1, R_c, 1)
    ray_ty = ray_ty.view(1, R_c, 1)

    ft_x0 = face_tile_x0.unsqueeze(1)
    ft_x1 = face_tile_x1.unsqueeze(1)
    ft_y0 = face_tile_y0.unsqueeze(1)
    ft_y1 = face_tile_y1.unsqueeze(1)

    in_tile = (
        (ray_tx >= ft_x0) & (ray_tx <= ft_x1) &
        (ray_ty >= ft_y0) & (ray_ty <= ft_y1)
    )  # (B, R_c, F)

    # ─────────────────────────
    # KEY IDEA: REDUCE F PER RAY
    # ─────────────────────────
    sdf_chunk = torch.zeros((B, R_c), device=device)
    valid_chunk = torch.zeros((B, R_c), device=device)

    # LOOP PER RAY (critical for true O(R·k))
    for r in range(R_c):

        mask_r = in_tile[:, r, :]   # (B, F)

        if not mask_r.any():
            continue

        # select candidate faces (per batch)
        idx = mask_r.nonzero(as_tuple=False)  # (N, 2)

        b_idx = idx[:,0]
        f_idx = idx[:,1]

        # gather only needed faces
        c0 = cV0[b_idx, f_idx]
        c1 = cV1[b_idx, f_idx]
        c2 = cV2[b_idx, f_idx]

        # ray point
        px_r = px[:, r, 0]
        py_r = py[:, r, 0]

        px_r = px_r[b_idx]
        py_r = py_r[b_idx]

        # ─────────────────────────
        # BARYCENTRIC (ONLY k faces)
        # ─────────────────────────
        denom = (
            (c1[:,1] - c2[:,1]) * (c0[:,0] - c2[:,0]) +
            (c2[:,0] - c1[:,0]) * (c0[:,1] - c2[:,1])
        ) + 1e-8

        u = (
            (c1[:,1] - c2[:,1]) * (px_r - c2[:,0]) +
            (c2[:,0] - c1[:,0]) * (py_r - c2[:,1])
        ) / denom

        v = (
            (c2[:,1] - c0[:,1]) * (px_r - c2[:,0]) +
            (c0[:,0] - c2[:,0]) * (py_r - c2[:,1])
        ) / denom

        w = 1 - u - v

        in_tri = (u >= 0) & (v >= 0) & (w >= 0)

        if not in_tri.any():
            continue

        # ─────────────────────────
        # DEPTH
        # ─────────────────────────
        Z = u * c0[:,2] + v * c1[:,2] + w * c2[:,2]
        Z = torch.where(in_tri, Z, INF)

        # ─────────────────────────
        # SOFT VISIBILITY
        # ─────────────────────────
        Z_shift = Z - Z.min()
        weights = torch.softmax(-alpha * Z_shift, dim=0)

        # ─────────────────────────
        # WORLD INTERPOLATION
        # ─────────────────────────
        V0_sel = V0w[f_idx]
        V1_sel = V1w[f_idx]
        V2_sel = V2w[f_idx]

        interp = (
            u.unsqueeze(-1) * V0_sel +
            v.unsqueeze(-1) * V1_sel +
            w.unsqueeze(-1) * V2_sel
        )

        fc_sel = face_centers[b_idx]

        d = torch.norm(interp - fc_sel, dim=-1)

        sdf_val = (weights * d).sum()

        # scatter back
        sdf_chunk[b_idx, r] = sdf_val
        valid_chunk[b_idx, r] = 1

    sdf_accum += sdf_chunk.sum(dim=-1)
    ray_count += valid_chunk.sum(dim=-1)