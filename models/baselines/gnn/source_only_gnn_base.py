import torch
from torch import nn
import torch.nn.functional as F

from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from Learn.Clean_SCGDA.models.__layers.build_layer import build_activation


class SourceOnlyGNNBase(nn.Module):
    """Explicit linear/activation/propagation GNN used for layerwise analysis."""

    def __init__(self, config: dict):
        super(SourceOnlyGNNBase, self).__init__()

        self.in_dim = config["model"]["in_dim"]
        self.hid_dim = config["model"]["hid_dim"]
        self.num_classes = config["model"]["num_classes"]
        self.num_layers = config["model"]["num_layers"]
        self.dropout = config["model"]["dropout_ratio"]

        self.gnn_type = config["model"]["gnn"]
        self.act_type = config["model"]["activation"]
        self.mode = config["model"]["mode"]

        if self.gnn_type.lower() != "gcn":
            raise ValueError(
                f"SourceOnlyGNNBase currently supports gnn='gcn' only, got {self.gnn_type}."
            )

        self.act = build_activation(self.act_type)
        self.lins = nn.ModuleList()

        self.lins.append(Linear(self.in_dim, self.hid_dim))
        for _ in range(self.num_layers - 1):
            self.lins.append(Linear(self.hid_dim, self.hid_dim))

        self.cls = Linear(self.hid_dim, self.num_classes)

    @staticmethod
    def _propagate_once(x, edge_index, edge_weight):
        if edge_index.numel() == 0:
            return x

        src, dst = edge_index[0], edge_index[1]
        if edge_weight is None:
            msg = x[src]
        else:
            msg = x[src] * edge_weight.view(-1, 1)

        out = x.new_zeros((x.size(0), x.size(1)))
        out.index_add_(0, dst, msg)
        return out

    def _normalized_graph(self, x, edge_index, edge_weight=None):
        norm_edge_index, norm_edge_weight = gcn_norm(
            edge_index,
            edge_weight,
            x.size(0),
            improved=False,
            add_self_loops=True,
            dtype=x.dtype,
        )
        if not isinstance(norm_edge_index, torch.Tensor):
            raise TypeError("SourceOnlyGNNBase expects dense COO edge_index tensors.")
        return norm_edge_index, norm_edge_weight

    def feat_bottleneck(
        self,
        x,
        edge_index,
        edge_weight=None,
        batch=None,
        return_snapshots=False,
        include_raw_input=False,
    ):
        norm_edge_index, norm_edge_weight = self._normalized_graph(x, edge_index, edge_weight)

        snapshots = {}
        if include_raw_input:
            snapshots["raw_input"] = x

        for layer_idx, lin in enumerate(self.lins, start=1):
            after_linear = self.act(lin(x))
            snapshots[f"after_linear{layer_idx}"] = after_linear

            after_mp = self._propagate_once(after_linear, norm_edge_index, norm_edge_weight)
            snapshots[f"after_mp{layer_idx}"] = after_mp

            x = after_mp
            if layer_idx < len(self.lins):
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.mode == "graph":
            if batch is None:
                raise ValueError("batch must be provided when mode='graph'.")
            x = global_mean_pool(x, batch)

        if return_snapshots:
            return x, snapshots
        return x

    def feat_classifier(self, x, edge_index=None, edge_weight=None):
        return self.cls(x)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        x = self.feat_bottleneck(x, edge_index, edge_weight=edge_weight, batch=batch)
        x = self.feat_classifier(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)
