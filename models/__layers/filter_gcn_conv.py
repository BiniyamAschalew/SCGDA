from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import add_remaining_self_loops

from  models.__layers.prop_gcn_conv import gcn_norm

from  models.__filters.build_filter import build_filter


class FilterGCNConv(torch.nn.Module):
    """GCN-style linear projection + runtime polynomial filter propagation.

    The propagation operator is chosen once at layer construction via `filter_type`
    (`mono`, `cheb`, or `bern`), while filter coefficients are provided at forward
    time via `filter_param`.

    For `mono`, default behavior prepends a zero coefficient so parameter
    `[c1, c2, ...]` means `c1*A + c2*A^2 + ...` over the projected features,
    matching "hop-based" interpretation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_type: str = "mono",
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
        prepend_zero_to_filter: bool | None = None,
        bern_enforce_nonneg: bool = False,
        **kwargs,
    ):
        super(FilterGCNConv, self).__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        # Kept only for API compatibility with old conv signatures.
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.filter_type = str(filter_type).lower()
        self.filter_operator = build_filter(self.filter_type)()

        self.prepend_zero_to_filter = (
            self.filter_type == "mono" if prepend_zero_to_filter is None else bool(prepend_zero_to_filter)
        )
        self.bern_enforce_nonneg = bool(bern_enforce_nonneg)

        self.lin = Linear(
            self.in_channels,
            self.out_channels,
            bias=False,
            weight_initializer="glorot",
        )

        if bias:
            self.bias = Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def _prepare_params(self, x: Tensor, filter_param: Tensor | None) -> Tensor:
        if filter_param is None:
            # Default to one-hop propagation.
            params = x.new_ones(1)
        else:
            params = torch.as_tensor(filter_param, dtype=x.dtype, device=x.device).view(-1)
            if params.numel() == 0:
                raise ValueError("filter_param must have at least one coefficient.")

        # Mono is interpreted as hop weights (A, A^2, ...), so prepend zero
        # unless explicitly disabled.
        if self.filter_type == "mono" and self.prepend_zero_to_filter:
            params = torch.cat([params.new_zeros(1), params], dim=0)

        return params

    @staticmethod
    def _propagate_once(x: Tensor, edge_index: Tensor, edge_weight: Tensor | None) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        if edge_weight is None:
            msg = x[src]
        else:
            msg = x[src] * edge_weight.view(-1, 1)
        out = x.new_zeros((x.size(0), x.size(1)))
        out.index_add_(0, dst, msg)
        return out

    def _mono_forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor | None, params: Tensor) -> Tensor:
        if self.normalize:
            edge_index, edge_weight = gcn_norm(
                edge_index,
                edge_weight,
                x.size(0),
                self.improved,
                self.add_self_loops,
                dtype=x.dtype,
            )
            if not isinstance(edge_index, Tensor):
                raise TypeError("FilterGCNConv mono path currently expects Tensor edge_index.")
        elif self.add_self_loops:
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index,
                edge_weight,
                fill_value=1.0,
                num_nodes=x.size(0),
            )

        out = params[0] * x
        cur = x
        for k in range(1, int(params.numel())):
            cur = self._propagate_once(cur, edge_index, edge_weight)
            out = out + params[k] * cur
        return out

    def forward(
        self,
        x: Tensor,
        edge_index,
        edge_weight: Tensor | None = None,
        filter_param: Tensor | None = None,
    ) -> Tensor:
        if not isinstance(edge_index, Tensor):
            raise TypeError(
                "FilterGCNConv currently expects dense COO edge_index Tensor "
                "(SparseTensor is not supported)."
            )

        out = self.lin(x)
        params = self._prepare_params(out, filter_param)

        if self.filter_type == "mono":
            out = self._mono_forward(out, edge_index, edge_weight, params)
        elif self.filter_type == "bern":
            out = self.filter_operator(
                out,
                edge_index,
                params,
                edge_weight=edge_weight,
                enforce_nonneg=self.bern_enforce_nonneg,
            )
        else:
            out = self.filter_operator(out, edge_index, params, edge_weight=edge_weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"{self.in_channels}, {self.out_channels}, filter_type='{self.filter_type}')"
        )
