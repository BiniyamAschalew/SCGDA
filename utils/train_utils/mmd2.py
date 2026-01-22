import torch
""" A safer implementation of MMD with multi-kernel RBF and unbiased estimate of MMD^2. """

def gaussian_kernel_rbf(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | torch.Tensor | None = None,
    sigma_heuristic: str = "median",  # "median" or "mean"
    eps: float = 1e-12,
):
    """
    Multi-kernel RBF kernel matrix K over concatenated [source; target].

    Notes:
      - Uses detach() for bandwidth computation (safe for autograd).
      - Supports median/mean heuristic for sigma when fix_sigma is None.
    """
    total = torch.cat([source, target], dim=0)
    n_total = total.size(0)

    # Pairwise squared Euclidean distances: [n_total, n_total]
    # (Avoid expand-heavy code; broadcasting is enough.)
    diff = total.unsqueeze(0) - total.unsqueeze(1)
    l2 = (diff * diff).sum(dim=-1)

    if fix_sigma is not None:
        bandwidth = torch.as_tensor(fix_sigma, device=total.device, dtype=total.dtype)
    else:
        # Use upper triangle (excluding diagonal) for heuristic
        l2_det = l2.detach()
        triu_idx = torch.triu_indices(n_total, n_total, offset=1, device=total.device)
        vals = l2_det[triu_idx[0], triu_idx[1]]

        if vals.numel() == 0:
            # Fallback for degenerate cases (e.g., n_total < 2)
            bandwidth = torch.tensor(1.0, device=total.device, dtype=total.dtype)
        else:
            if sigma_heuristic.lower() == "median":
                bandwidth = vals.median()
            elif sigma_heuristic.lower() == "mean":
                bandwidth = vals.mean()
            else:
                raise ValueError(f"Unknown sigma_heuristic: {sigma_heuristic}")

        bandwidth = bandwidth.clamp_min(eps)

    # Center bandwidth for multi-kernel scaling (as in many DA implementations)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))

    kernel_sum = 0.0
    for i in range(kernel_num):
        bw = bandwidth * (kernel_mul ** i)
        kernel_sum = kernel_sum + torch.exp(-l2 / bw.clamp_min(eps))

    return kernel_sum


def get_MMD2_unbiased(
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | torch.Tensor | None = None,
    sigma_heuristic: str = "median",
    eps: float = 1e-12,
):
    """
    Unbiased estimate of MMD^2 using multi-kernel RBF.
    Excludes diagonal terms in XX and YY and normalizes by m(m-1), n(n-1).
    """
    m = source_feat.size(0)
    n = target_feat.size(0)
    if m < 2 or n < 2:
        raise ValueError("Need at least 2 samples in each domain for unbiased MMD^2.")

    K = gaussian_kernel_rbf(
        source_feat,
        target_feat,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
        sigma_heuristic=sigma_heuristic,
        eps=eps,
    )

    K_xx = K[:m, :m]
    K_yy = K[m:, m:]
    K_xy = K[:m, m:]

    # Remove diagonals for unbiased estimator
    sum_xx = (K_xx.sum() - K_xx.diag().sum()) / (m * (m - 1))
    sum_yy = (K_yy.sum() - K_yy.diag().sum()) / (n * (n - 1))
    sum_xy = K_xy.mean()  # = (1/(mn)) sum_{i,j} k(x_i, y_j)

    mmd2 = sum_xx + sum_yy - 2.0 * sum_xy
    return mmd2


def MMD2(
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    sampling_num: int = 1000,
    times: int = 5,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | torch.Tensor | None = None,
    sigma_heuristic: str = "median",
    no_replacement: bool = True,
    eps: float = 1e-12,
):
    """
    Monte-Carlo estimate of unbiased MMD^2 with optional subsampling.

    Updates vs your original:
      - Unbiased estimator (diagonal removed, correct normalization).
      - No .data usage (uses detach internally).
      - No-replacement sampling when possible.
      - Indices created on the same device as features.
    """
    device = source_feat.device
    source_num = source_feat.size(0)
    target_num = target_feat.size(0)

    if source_num < 2 or target_num < 2:
        raise ValueError("Need at least 2 samples per domain to compute unbiased MMD^2.")

    # Choose effective k (cannot exceed domain size if no-replacement)
    if no_replacement:
        k_s = min(sampling_num, source_num)
        k_t = min(sampling_num, target_num)
    else:
        k_s = sampling_num
        k_t = sampling_num

    mmd2 = 0.0
    for _ in range(times):
        if no_replacement:
            idx_s = torch.randperm(source_num, device=device)[:k_s]
            idx_t = torch.randperm(target_num, device=device)[:k_t]
        else:
            idx_s = torch.randint(source_num, (k_s,), device=device)
            idx_t = torch.randint(target_num, (k_t,), device=device)

        xs = source_feat[idx_s]
        yt = target_feat[idx_t]

        mmd2 = mmd2 + get_MMD2_unbiased(
            xs,
            yt,
            kernel_mul=kernel_mul,
            kernel_num=kernel_num,
            fix_sigma=fix_sigma,
            sigma_heuristic=sigma_heuristic,
            eps=eps,
        )

    return mmd2 / times
