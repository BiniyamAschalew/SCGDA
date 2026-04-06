"""Small sanity check for mono/cheb/bern filter equivalence."""

import torch

from Learn.Clean_SCGDA.models.__filters.bern import BernProp
from Learn.Clean_SCGDA.models.__filters.cheb import ChebProp
from Learn.Clean_SCGDA.models.__filters.mono import MonoProp
from Learn.Clean_SCGDA.utils.filter_utils import monomial_to_cheb


def _toy_edge_index() -> torch.Tensor:
    edges = [
        (0, 1), (1, 0),
        (1, 2), (2, 1),
        (2, 3), (3, 2),
        (3, 4), (4, 3),
        (0, 4), (4, 0),
    ]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _bern_from_mono(mono: torch.Tensor) -> torch.Tensor:
    k = mono.numel() - 1
    basis_cols = []
    for i in range(k + 1):
        unit = torch.zeros(k + 1, dtype=mono.dtype)
        unit[i] = 1.0
        basis_cols.append(BernProp.to_polynomial(unit))
    basis = torch.stack(basis_cols, dim=1)
    return torch.linalg.solve(basis, mono)


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(5, 3)
    edge_index = _toy_edge_index()

    mono = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    cheb = monomial_to_cheb(mono)
    bern = _bern_from_mono(mono)

    print("mono coeffs:", mono.tolist())
    print("cheb coeffs (computed):", cheb.tolist())
    print("cheb coeffs (guess):", [1.5, 1.0, 0.5, 0.0])
    print("bern coeffs (computed):", bern.tolist())

    mono_prop = MonoProp()
    cheb_prop = ChebProp()
    bern_prop = BernProp()

    y_mono = mono_prop(x, edge_index, mono)
    y_cheb = cheb_prop(x, edge_index, cheb)
    y_bern = bern_prop(x, edge_index, bern)

    print("max |mono - cheb|:", float((y_mono - y_cheb).abs().max()))
    print("max |mono - bern|:", float((y_mono - y_bern).abs().max()))


if __name__ == "__main__":
    main()
