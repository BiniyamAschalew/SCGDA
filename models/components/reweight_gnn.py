import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor
import torch_geometric.nn as pyg_nn
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor
from typing import Optional
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor
from torch_sparse import matmul as torch_sparse_matmul


def spmm(src: Adj, other: Tensor, reduce: str = "sum") -> Tensor:

    assert reduce in ['sum', 'add', 'mean', 'min', 'max']

    if isinstance(src, SparseTensor):
        return torch_sparse_matmul(src, other, reduce)

    if reduce in ['sum', 'add']:
        return torch.sparse.mm(src, other)

    # TODO: Support `mean` reduction for PyTorch SparseTensor
    raise ValueError(f"`{reduce}` reduction is not supported for "
                     f"`torch.sparse.Tensor`.")


class GCN_reweight(pyg_nn.MessagePassing):

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = False,
        normalize: bool = True,
        bias: bool = True,
        **kwargs
        ):

        super(GCN_reweight, self).__init__(aggr=aggr, flow ="target_to_source")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        if aggr == "add":
            self.normalize = False
        else:
            self.normalize = True
        
        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x, edge_index, edge_weight, lmda):
       
        edge_rw = edge_weight
        edge_weight = torch.ones_like(edge_rw)
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, size=None, x=x, edge_weight=edge_weight, edge_rw=edge_rw, lmda=lmda)

        if self.bias is not None:
            out = out + self.bias

        return out


    def message(self, x_j, edge_index, edge_weight, edge_rw, lmda):
        
        x_j = (edge_weight.view(-1, 1) * x_j)
        x_j = (1-lmda) * x_j + (lmda) * (edge_rw.view(-1, 1) * x_j)
        return x_j


    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:

        return spmm(adj_t, x, reduce=self.aggr)


class GS_reweight(pyg_nn.MessagePassing):

    def __init__(self, in_channels, out_channels, reducer, normalize_embedding=False):
        super(GS_reweight, self).__init__(aggr=reducer, flow ="target_to_source")
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.agg_lin = torch.nn.Linear(out_channels + in_channels, out_channels)

        self.normalize_emb = normalize_embedding

    def forward(self, x, edge_index, edge_weight, lmda):

        num_nodes = x.size(0)
        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x, edge_weight=edge_weight, lmda=lmda)

    def message(self, x_j, edge_index, edge_weight, lmda):

        x_j = self.lin(x_j)
        x_j = (1-lmda) * x_j + (lmda) * (edge_weight.view(-1, 1) * x_j)

        return x_j

    def update(self, aggr_out, x):

        aggr_out = torch.cat((aggr_out, x), dim=-1)
        aggr_out = self.agg_lin(aggr_out)
        aggr_out = F.relu(aggr_out)

        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)

        return aggr_out


class ReweightGNN(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        gnn_dim,
        output_dim,
        cls_dim,
        gnn_layers=3,
        cls_layers=2,
        backbone='GS',
        pooling='mean',
        dropout=0.5,
        bn=False,
        rw_lmda=1.0,
        # **kwargs
        ):
        super(ReweightGNN, self).__init__()
        if backbone == 'GCN':
            self.prop_input = GCN_reweight(input_dim, gnn_dim, pooling)
            self.prop_hidden = GCN_reweight(gnn_dim, gnn_dim, pooling)
        elif backbone == 'GS':
            self.prop_input = GS_reweight(input_dim, gnn_dim, pooling)
            self.prop_hidden = GS_reweight(gnn_dim, gnn_dim, pooling)

        self.dropout = dropout
        self.bn = bn
        self.lmda = rw_lmda

        # conv layers
        self.conv = nn.ModuleList()
        self.conv.append(self.prop_input)
        for i in range(gnn_layers - 1):
            self.conv.append(self.prop_hidden)

        #bn layers
        self.bns = nn.ModuleList()
        for i in range(gnn_layers - 1):
            self.bns.append(nn.BatchNorm1d(gnn_dim))

        self.bn_mlp = nn.BatchNorm1d(cls_dim)

        # classification layer
        self.mlp_classify = nn.ModuleList()
        if cls_layers == 1:
            self.mlp_classify.append(nn.Linear(gnn_dim, output_dim))
        else:
            self.mlp_classify.append(nn.Linear(gnn_dim, cls_dim))
            for i in range(cls_layers - 2):
                self.mlp_classify.append(nn.Linear(cls_dim, cls_dim))
            self.mlp_classify.append(nn.Linear(cls_dim, output_dim))

    def forward(self, data, h):

        x, edge_index, edge_weight = h, data.edge_index, data.edge_weight
        for i, layer in enumerate(self.conv):
            x = layer(x, edge_index, edge_weight, self.lmda)
            # if self.bn and (i != len(self.conv) - 1):
            #     x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        y = x
        for i in range(len(self.mlp_classify)):
            y = self.mlp_classify[i](y)
            if i != (len(self.mlp_classify) - 1):
                if self.bn: 
                    y = self.bn_mlp(y)
                y = F.relu(y)
            #y = F.dropout(y, p=self.dropout)
        return x, y
