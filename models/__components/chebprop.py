"""Chebyshev polynomial based spectral filter """

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import get_laplacian


class ChebProp(MessagePassing):
   
    def __init__(self, K, is_source_domain=True, bias=True, **kwargs):
        super(ChebProp, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.is_source_domain = is_source_domain
        self.cached_terms = None
        self.cached_coefs = None
        # Match PyG ChebConv semantics: K basis terms, indexed 0..K-1.
        self.temp = nn.Parameter(torch.Tensor(self.K), requires_grad=is_source_domain)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.is_source_domain:
                self.temp.zero_()
                self.temp[0] = 1.0
            else:
                self.temp.zero_()
                self.temp[0] = 1.0

    def get_filter(self):
        if self.cached_terms is None or self.cached_coefs is None:
            raise RuntimeError("cached_terms/cached_coefs must be set before calling get_filter().")

        TEMP = F.relu(self.temp)
        H = 0

        for k in range(self.K):
            H = H + TEMP[k] * self.cached_coefs[k] * self.cached_terms[k]

        return H

    def __norm__(self, edge_index, num_nodes, edge_weight, lambda_max=None, dtype=None, batch=None):
        """Build normalized propagation weights for Chebyshev recurrence."""
        edge_index, edge_weight = get_laplacian(
            edge_index,
            edge_weight,
            normalization='sym',
            dtype=dtype,
            num_nodes=num_nodes,
        )
        assert edge_weight is not None

        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype, device=edge_index.device)

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        loop_mask = edge_index[0] == edge_index[1]
        edge_weight[loop_mask] -= 1.0

        # ChebConv-style scaled operator gives L_hat.
        # We flip sign to use normalized adjacency A_norm as the base operator
        # (for lambda_max=2, this becomes exactly A_norm).
        edge_weight = -edge_weight

        return edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None, batch=None, lambda_max=None, temp=None):

        if temp is None:
            TEMP = F.relu(self.temp)
        else:
            if temp.numel() != self.K:
                raise ValueError(f"Expected temp with {self.K} elements, got {temp.numel()}.")
            TEMP = F.relu(temp.to(dtype=x.dtype, device=x.device))

        edge_index1, norm1 = self.__norm__(
            edge_index,
            x.size(self.node_dim),
            edge_weight,
            lambda_max=lambda_max,
            dtype=x.dtype,
            batch=batch,
        )

        # T_0(x)
        t0 = x
        out = TEMP[0] * t0

        if self.K > 1:
            # T_1(x) = L_hat x
            t1 = self.propagate(edge_index1, x=t0, norm=norm1, size=None)
            out = out + TEMP[1] * t1

            # T_k(x) = 2 L_hat T_{k-1}(x) - T_{k-2}(x)
            for k in range(2, self.K):
                t2 = self.propagate(edge_index1, x=t1, norm=norm1, size=None)
                t2 = 2.0 * t2 - t0
                out = out + TEMP[k] * t2
                t0, t1 = t1, t2

        return out


    def message(self, x_j, norm):

        return norm.view(-1, 1) * x_j

    def __repr__(self):

        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K, self.temp)
