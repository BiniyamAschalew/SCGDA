"""Monomial filter."""

import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import get_laplacian


def _adj_norm(edge_index, edge_weight, num_nodes, dtype, lambda_max=2.0):
    edge_index, edge_weight = get_laplacian(
        edge_index,
        edge_weight,
        normalization="sym",
        dtype=dtype,
        num_nodes=num_nodes,
    )
    if not isinstance(lambda_max, torch.Tensor):
        lambda_max = torch.tensor(lambda_max, dtype=dtype, device=edge_index.device)
    edge_weight = (2.0 * edge_weight) / lambda_max
    edge_weight.masked_fill_(edge_weight == float("inf"), 0.0)
    loop_mask = edge_index[0] == edge_index[1]
    edge_weight[loop_mask] -= 1.0
    return edge_index, -edge_weight  # normalized adjacency A


class MonoProp(MessagePassing):
    def forward(self, x, edge_index, parameters, edge_weight=None, lambda_max=2.0):
        params = torch.as_tensor(parameters, device=x.device, dtype=x.dtype).view(-1)
        if params.numel() == 0:
            raise ValueError("parameters must have at least one coefficient.")

        edge_index, norm = _adj_norm(edge_index, edge_weight, x.size(self.node_dim), x.dtype, lambda_max)
        out = params[0] * x
        cur = x
        for k in range(1, params.numel()):
            cur = self.propagate(edge_index, x=cur, norm=norm, size=None)
            out = out + params[k] * cur
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    @staticmethod
    def to_polynomial(parameters):
        return torch.as_tensor(parameters)
