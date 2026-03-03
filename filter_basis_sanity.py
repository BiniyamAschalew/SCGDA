"""Sanity check: mono/cheb/bern filters produce identical outputs for equivalent coefficients."""

import torch
from torch_geometric.utils import get_laplacian

from models.__filters.bern import BernProp
from models.ours.fda.objective import FDAFilterAlignObjective
from utils.filter_utils import monomial_to_cheb


def build_toy_graph() -> torch.Tensor:
    """5-node undirected toy graph edge_index."""
    edges = [
        (0, 1), (1, 0),
        (1, 2), (2, 1),
        (2, 3), (3, 2),
        (3, 4), (4, 3),
        (0, 4), (4, 0),
        (1, 3), (3, 1),
    ]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def dense_normalized_adjacency(edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype) -> torch.Tensor:
    ei, ew = get_laplacian(
        edge_index,
        edge_weight=None,
        normalization="sym",
        dtype=dtype,
        num_nodes=num_nodes,
    )
    lambda_max = torch.tensor(2.0, dtype=dtype, device=ei.device)
    ew = (2.0 * ew) / lambda_max
    ew.masked_fill_(ew == float("inf"), 0.0)
    loop_mask = ei[0] == ei[1]
    ew[loop_mask] -= 1.0
    ew = -ew
    return torch.sparse_coo_tensor(ei, ew, (num_nodes, num_nodes), dtype=dtype).coalesce().to_dense()


def bern_from_mono(mono_coeffs: torch.Tensor) -> torch.Tensor:
    k = int(mono_coeffs.numel()) - 1
    basis_cols = []
    for i in range(k + 1):
        unit = torch.zeros(k + 1, dtype=mono_coeffs.dtype, device=mono_coeffs.device)
        unit[i] = 1.0
        basis_cols.append(BernProp.to_polynomial(unit))
    basis = torch.stack(basis_cols, dim=1)
    return torch.linalg.solve(basis, mono_coeffs)


def polynomial_apply_dense(adj: torch.Tensor, x: torch.Tensor, mono_coeffs: torch.Tensor) -> torch.Tensor:
    out = mono_coeffs[0] * x
    cur = x
    for k in range(1, int(mono_coeffs.numel())):
        cur = adj @ cur
        out = out + mono_coeffs[k] * cur
    return out


def main() -> None:
    torch.manual_seed(0)

    edge_index = build_toy_graph()
    num_nodes = 5
    x = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [0.5, 1.0, 0.0],
            [2.0, 1.0, 1.0],
            [0.0, 2.0, 1.5],
            [1.0, 1.0, 0.5],
        ],
        dtype=torch.float32,
    )

    mono_coeffs = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    cheb_expected = torch.tensor([1.5, 1.0, 0.5, 0.0], dtype=torch.float32)

    cheb_from_mono = monomial_to_cheb(mono_coeffs)
    bern_coeffs = bern_from_mono(mono_coeffs)

    print("Edge index:\n", edge_index)
    adj = dense_normalized_adjacency(edge_index, num_nodes=num_nodes, dtype=x.dtype)
    print("Normalized adjacency A:\n", adj)

    print("\nMonomial coeffs:", mono_coeffs.tolist())
    print("Cheb expected (user guess):", cheb_expected.tolist())
    print("Cheb from monomial:", cheb_from_mono.tolist())
    print("Bern from monomial:", bern_coeffs.tolist())

    mono_head = FDAFilterAlignObjective(filter_name="mono")
    cheb_head = FDAFilterAlignObjective(filter_name="cheb")
    bern_head = FDAFilterAlignObjective(filter_name="bern")

    y_mono = mono_head.apply_filter(x, edge_index, params=mono_coeffs)
    y_cheb_expected = cheb_head.apply_filter(x, edge_index, params=cheb_expected)
    y_cheb = cheb_head.apply_filter(x, edge_index, params=cheb_from_mono)
    y_bern = bern_head.apply_filter(x, edge_index, params=bern_coeffs)
    y_dense = polynomial_apply_dense(adj, x, mono_coeffs)

    def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).abs().max().item())

    print("\nMax |mono - dense|:", max_abs(y_mono, y_dense))
    print("Max |cheb(expected) - mono|:", max_abs(y_cheb_expected, y_mono))
    print("Max |cheb(from_mono) - mono|:", max_abs(y_cheb, y_mono))
    print("Max |bern(from_mono) - mono|:", max_abs(y_bern, y_mono))

    print("\nSample output (mono):\n", y_mono)
    print("\nSample output (cheb from mono):\n", y_cheb)
    print("\nSample output (bern from mono):\n", y_bern)


if __name__ == "__main__":
    main()
