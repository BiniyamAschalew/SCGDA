"""Bernstein filter."""

import math

import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian


class BernProp(MessagePassing):
    def forward(self, x, edge_index, parameters, edge_weight=None, enforce_nonneg=False):
        params = torch.as_tensor(parameters, device=x.device, dtype=x.dtype).view(-1)
        if enforce_nonneg:
            params = F.relu(params)
        k = params.numel() - 1
        if k < 0:
            raise ValueError("parameters must have at least one coefficient.")

        edge_index1, norm1 = get_laplacian(
            edge_index,
            edge_weight,
            normalization="sym",
            dtype=x.dtype,
            num_nodes=x.size(self.node_dim),
        )
        edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=2.0, num_nodes=x.size(self.node_dim))

        tmp = [x]
        for _ in range(k):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        out = (math.comb(k, 0) / (2.0**k)) * params[0] * tmp[k]
        for i in range(k):
            x = tmp[k - i - 1]
            for _ in range(i + 1):
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            out = out + (math.comb(k, i + 1) / (2.0**k)) * params[i + 1] * x
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    @staticmethod
    def to_polynomial(parameters, enforce_nonneg=False):
        params = torch.as_tensor(parameters)
        if enforce_nonneg:
            params = F.relu(params)
        k = params.numel() - 1
        out = torch.zeros(k + 1, dtype=params.dtype, device=params.device)

        for i in range(k + 1):
            scale = (math.comb(k, i) / (2.0**k)) * params[i]
            for p in range(i + 1):
                for q in range(k - i + 1):
                    deg = p + q
                    out[deg] = out[deg] + scale * (math.comb(i, p) * ((-1.0) ** p) * math.comb(k - i, q))
        return out
