"""Monomial filter: weighted sum of powers of the adjacency matrix"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import get_laplacian


class MonomialProp(MessagePassing):

    def __init__(self, K, is_source_domain=True, bias=True, **kwargs):
        super(MonomialProp, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.is_source_domain = is_source_domain
        self.cached_terms = None
        self.temp = nn.Parameter(torch.Tensor(self.K + 1), requires_grad=is_source_domain)
        self.reset_parameters()

    def reset_parameters(self):

        if self.is_source_domain:
            self.temp.data.fill_(1)
        else:
            self.temp.data = torch.linspace(1, 0, self.K + 1)

    def get_filter(self):

        TEMP = F.relu(self.temp)
        H = 0

        for k in range(self.K + 1):
            H = H + TEMP[k] * self.cached_terms[k]

        return H

    def forward(self, x, edge_index, edge_weight=None):

        TEMP = F.relu(self.temp)

        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                           num_nodes=x.size(self.node_dim))
        edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=2., num_nodes=x.size(self.node_dim))

        tmp = []
        tmp.append(x)
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        out = torch.empty_like(x)
        for k in range(self.K + 1):
            out = out + TEMP[k] * tmp[k]

        return out
    

    