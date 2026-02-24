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
