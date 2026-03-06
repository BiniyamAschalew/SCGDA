import torch




def _cheb_basis_matrix(order: int, dtype=None, device=None) -> torch.Tensor:
    """Return basis matrix B where T_k(x) = sum_j B[k, j] x^j."""
    if order <= 0:
        raise ValueError(f"order must be positive, got {order}")

    B = torch.zeros((order, order), dtype=dtype, device=device)
    B[0, 0] = 1.0
    if order > 1:
        B[1, 1] = 1.0

    for k in range(2, order):
        # T_k(x) = 2 x T_{k-1}(x) - T_{k-2}(x)
        B[k, 1:] += 2.0 * B[k - 1, :-1]
        B[k, :] -= B[k - 2, :]

    return B


def cheb_to_monomial(cheb_coeffs: torch.Tensor) -> torch.Tensor:
    """Convert Chebyshev coefficients c_k to monomial coefficients a_j.

    h(x) = sum_k c_k T_k(x) = sum_j a_j x^j
    """
    cheb_coeffs = torch.as_tensor(cheb_coeffs)
    if cheb_coeffs.dim() != 1:
        raise ValueError(f"Expected 1D cheb_coeffs, got shape {tuple(cheb_coeffs.shape)}")

    order = cheb_coeffs.numel()
    B = _cheb_basis_matrix(order, dtype=cheb_coeffs.dtype, device=cheb_coeffs.device)
    return B.transpose(0, 1) @ cheb_coeffs


def monomial_to_cheb(monomial_coeffs: torch.Tensor) -> torch.Tensor:
    """Convert monomial coefficients a_j to Chebyshev coefficients c_k.

    h(x) = sum_j a_j x^j = sum_k c_k T_k(x)
    """
    monomial_coeffs = torch.as_tensor(monomial_coeffs)
    if monomial_coeffs.dim() != 1:
        raise ValueError(
            f"Expected 1D monomial_coeffs, got shape {tuple(monomial_coeffs.shape)}"
        )

    order = monomial_coeffs.numel()
    B = _cheb_basis_matrix(order, dtype=monomial_coeffs.dtype, device=monomial_coeffs.device)
    A = B.transpose(0, 1)
    return torch.linalg.solve(A, monomial_coeffs)


def tensor_to_float_list(values: torch.Tensor, precision: int = 6):
    """Utility for stable human-readable logging of 1D tensors."""
    values = torch.as_tensor(values).detach().cpu().flatten().tolist()
    return [round(float(v), precision) for v in values]


def pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances between row-vectors in x and y."""
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True).transpose(0, 1)
    return torch.clamp(x_norm + y_norm - 2.0 * (x @ y.transpose(0, 1)), min=0.0)


def median_bandwidth(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Median heuristic bandwidth for RBF kernels."""
    all_feat = torch.cat([x, y], dim=0)
    dists = pairwise_sq_dist(all_feat, all_feat).detach()
    n = dists.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=dists.device)
    vals = dists[mask]
    if vals.numel() == 0:
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)
    return vals.median().clamp_min(eps)


def mmd_rbf(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Multi-kernel RBF MMD."""
    bandwidth = (
        torch.as_tensor(fix_sigma, device=source.device, dtype=source.dtype)
        if fix_sigma is not None
        else median_bandwidth(source, target, eps=eps)
    )
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))

    xx = pairwise_sq_dist(source, source)
    yy = pairwise_sq_dist(target, target)
    xy = pairwise_sq_dist(source, target)

    k_xx = 0.0
    k_yy = 0.0
    k_xy = 0.0
    for i in range(kernel_num):
        bw = (bandwidth * (kernel_mul**i)).clamp_min(eps)
        k_xx = k_xx + torch.exp(-xx / bw)
        k_yy = k_yy + torch.exp(-yy / bw)
        k_xy = k_xy + torch.exp(-xy / bw)
    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


def conditional_mmd(
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    source_y: torch.Tensor,
    target_y: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | None = None,
) -> torch.Tensor:
    """Class-conditional MMD averaged over classes present in both domains."""
    classes = torch.unique(torch.cat([source_y, target_y], dim=0))
    vals = []
    for cls in classes:
        src_mask = source_y == cls
        tgt_mask = target_y == cls
        if int(src_mask.sum().item()) == 0 or int(tgt_mask.sum().item()) == 0:
            continue
        vals.append(
            mmd_rbf(
                source_feat[src_mask],
                target_feat[tgt_mask],
                kernel_mul=kernel_mul,
                kernel_num=kernel_num,
                fix_sigma=fix_sigma,
            )
        )
    if not vals:
        return torch.tensor(0.0, device=source_feat.device, dtype=source_feat.dtype)
    return torch.stack(vals).mean()


def make_gaussian_probe( # combine the source and target feature, compute mean and std, and create a probe
    source_x: torch.Tensor,
    target_x: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample source/target probe features from combined mean/std."""
    all_real = torch.cat([source_x, target_x], dim=0)
    probe_mean = all_real.mean(dim=0, keepdim=True)
    probe_std = all_real.std(dim=0, keepdim=True).clamp_min(eps)
    source_probe = torch.randn_like(source_x) * probe_std + probe_mean
    target_probe = torch.randn_like(target_x) * probe_std + probe_mean
    return source_probe, target_probe
