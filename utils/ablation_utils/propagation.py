import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from models.__components.chebprop import ChebProp


class Propagation(MessagePassing):
    """GCN-style normalized one-hop propagation."""

    def __init__(self, aggr: str = "add"):
        super().__init__(aggr=aggr)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.0)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j


class WeightedPropagation(MessagePassing):
    """GCN-style one-hop propagation with explicit edge weights."""

    def __init__(self, aggr: str = "add"):
        super().__init__(aggr=aggr)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=x.size(0)
        )
        row, col = edge_index
        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        deg = deg.index_add(0, col, edge_weight.to(dtype=x.dtype))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.0)
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j


def apply_cheb_once(
    x: torch.Tensor,
    data,
    cheb_prop: ChebProp,
    temp: torch.Tensor,
    edge_weight: torch.Tensor,
    lambda_max: float,
) -> torch.Tensor:
    return cheb_prop(
        x,
        data.edge_index,
        edge_weight=edge_weight,
        batch=getattr(data, "batch", None),
        lambda_max=lambda_max,
        temp=temp,
    )


def propagate_layers(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    max_layers: int,
    prop: Propagation,
) -> list[torch.Tensor]:
    layers = [x]
    cur = x
    for _ in range(max_layers):
        cur = prop(cur, edge_index)
        layers.append(cur)
    return layers


def propagate_weighted_layers(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    max_layers: int,
    prop: WeightedPropagation,
) -> list[torch.Tensor]:
    layers = [x]
    cur = x
    for _ in range(max_layers):
        cur = prop(cur, edge_index, edge_weight)
        layers.append(cur)
    return layers


def uniform_cheb_temp(cheb_k: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if cheb_k <= 0:
        raise ValueError(f"cheb_k must be positive, got {cheb_k}")
    return torch.full((int(cheb_k),), 1.0 / float(cheb_k), device=device, dtype=dtype)


def propagate_fixed_cheb_layers(
    x: torch.Tensor,
    data,
    max_layers: int,
    cheb_k: int,
    lambda_max: float,
    temp: torch.Tensor | None = None,
    edge_weight: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Repeatedly apply a fixed Chebyshev filter on one graph."""
    dtype = x.dtype
    device = x.device
    cheb_prop = ChebProp(int(cheb_k), is_source_domain=False).to(device)
    temp_vec = (
        uniform_cheb_temp(int(cheb_k), device=device, dtype=dtype)
        if temp is None
        else temp.to(device=device, dtype=dtype)
    )
    edge_w = (
        torch.ones(data.edge_index.size(1), device=device, dtype=dtype)
        if edge_weight is None
        else edge_weight.to(device=device, dtype=dtype)
    )

    layers = [x]
    cur = x
    for _ in range(max_layers):
        cur = apply_cheb_once(
            cur,
            data,
            cheb_prop,
            temp_vec,
            edge_w,
            lambda_max,
        )
        layers.append(cur)
    return layers


def induced_operator_matrix(
    data,
    cheb_prop: ChebProp,
    temp: torch.Tensor,
    edge_weight: torch.Tensor,
    lambda_max: float,
    chunk_size: int,
) -> torch.Tensor:
    """Compute exact induced operator matrix H via H(I) with chunked identity blocks."""
    n = int(data.x.size(0))
    device = data.x.device
    dtype = data.x.dtype if data.x.dtype.is_floating_point else torch.float32
    h_cpu = torch.empty((n, n), dtype=dtype, device="cpu")

    if chunk_size <= 0:
        chunk_size = n

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        width = end - start

        basis = torch.zeros((n, width), device=device, dtype=dtype)
        rows = torch.arange(start, end, device=device)
        cols = torch.arange(width, device=device)
        basis[rows, cols] = 1.0

        out = apply_cheb_once(
            basis,
            data,
            cheb_prop,
            temp,
            edge_weight,
            lambda_max,
        )
        h_cpu[:, start:end] = out.detach().to("cpu")

    return h_cpu


def operator_to_edges(
    operator_h: torch.Tensor,
    edge_threshold: float,
    topk: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert dense induced operator into sparse edge list + positive weights."""
    h = operator_h.abs()
    n = int(h.size(0))

    if topk > 0 and topk < n:
        vals, idx = torch.topk(h, k=topk, dim=1)
        row = torch.arange(n).unsqueeze(1).expand(-1, topk).reshape(-1)
        col = idx.reshape(-1)
        weight = vals.reshape(-1)
    else:
        row, col = torch.nonzero(h > edge_threshold, as_tuple=True)
        weight = h[row, col]

    if edge_threshold > 0:
        keep = weight > edge_threshold
        row = row[keep]
        col = col[keep]
        weight = weight[keep]

    edge_index = torch.stack([row, col], dim=0).long().to(device)
    edge_weight = weight.to(device=device, dtype=torch.float32)
    return edge_index, edge_weight
