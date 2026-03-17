"""Chebyshev polynomial based spectral filter """

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import get_laplacian
from scipy.special import comb


class MonoProp(MessagePassing):
   
    def __init__(self, K, is_source_domain=True, bias=True, **kwargs):
        super(MonoProp, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.is_source_domain = is_source_domain
        self.cached_terms = None
        self.cached_coefs = None

        self.coef = nn.Parameter(torch.Tensor(self.K), requires_grad=is_source_domain)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.is_source_domain:
                self.coef.zero_()
                self.coef[0] = 1.0
            else:
                self.coef.zero_()
                self.coef[0] = 1.0

    def get_filter(self):
        if self.cached_terms is None or self.cached_coefs is None:
            raise RuntimeError("cached_terms/cached_coefs must be set before calling get_filter().")

        TEMP = self.coef
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
            lambda_max = torch.tensor(2.0, dtype=dtype, device=edge_index.device)
        elif not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype, device=edge_index.device)

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        lambda_max = lambda_max.clamp_min(1e-12)

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight = edge_weight.clone()
        edge_weight.masked_fill_(~torch.isfinite(edge_weight), 0.0)

        loop_mask = edge_index[0] == edge_index[1]
        edge_weight[loop_mask] -= 1.0

        # ChebConv-style scaled operator gives L_hat.
        # We flip sign to use normalized adjacency A_norm as the base operator
        # (for lambda_max=2, this becomes exactly A_norm).
        edge_weight = -edge_weight

        return edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None, batch=None, lambda_max=None, temp=None):

        if temp is None:
            TEMP = self.coef
        else:
            if temp.numel() != self.K:
                raise ValueError(f"Expected temp with {self.K} elements, got {temp.numel()}.")
            TEMP = temp.to(dtype=x.dtype, device=x.device)

        edge_index1, norm1 = self.__norm__(
            edge_index,
            x.size(self.node_dim),
            edge_weight,
            lambda_max=lambda_max,
            dtype=x.dtype,
            batch=batch,
        )

        # we are using plain monomial basis (GPRGNN style), so the k-th term is simply A^k x
        out = 0
        h = x
        for k in range(self.K):

            out = out + TEMP[k] * h
            h = self.propagate(edge_index1, x=h, norm=norm1, size=None)
        return out


    def message(self, x_j, norm):

        return norm.view(-1, 1) * x_j

    def to_polynomial(self):
        """already polynomial"""

        coefs = self.coef.detach().cpu().numpy().tolist()
        return coefs

    def __repr__(self):

        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K, self.coef)
