import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from models.__layers.filter_gcn_conv import FilterGCNConv
from models.__layers.ppmi_conv import PPMIConv


class GNN(torch.nn.Module):
    """AdaGCN encoder with optional learnable filter propagation."""

    def __init__(
        self,
        in_dim,
        hid_dim,
        gnn_type="filter_mono",
        num_layers=3,
        act=F.relu,
        dropout=0.1,
        filter_depth=4,
        **kwargs,
    ):
        super(GNN, self).__init__()

        self.gnn_type = str(gnn_type).lower()
        self.act = act
        self.num_layers = num_layers
        self.filter_depth = int(filter_depth)

        self.conv_layers = nn.ModuleList()

        if self.gnn_type == "gcn":
            self.conv_layers.append(GCNConv(in_dim, hid_dim))
            for _ in range(1, self.num_layers):
                self.conv_layers.append(GCNConv(hid_dim, hid_dim))
        elif self.gnn_type == "ppmi":
            self.conv_layers.append(PPMIConv(in_dim, hid_dim))
            for _ in range(1, self.num_layers):
                self.conv_layers.append(PPMIConv(hid_dim, hid_dim))
        else:
            # Filter path: default to monomial and use direct basis semantics.
            filter_type = "mono"
            if self.gnn_type.startswith("filter_"):
                filter_type = self.gnn_type.split("_", 1)[1]
            self.conv_layers.append(
                FilterGCNConv(
                    in_dim,
                    hid_dim,
                    filter_type=filter_type,
                    prepend_zero_to_filter=False,
                )
            )
            for _ in range(1, self.num_layers):
                self.conv_layers.append(
                    FilterGCNConv(
                        hid_dim,
                        hid_dim,
                        filter_type=filter_type,
                        prepend_zero_to_filter=False,
                    )
                )

        self.dropout = nn.Dropout(dropout)

    def _default_filter_param(self, x: torch.Tensor) -> torch.Tensor:
        param = torch.zeros(self.filter_depth, dtype=x.dtype, device=x.device)
        idx = 1 if self.filter_depth > 1 else 0
        param[idx] = 1.0
        return param

    def forward(self, x, edge_index, batch, mode="node", filter_param=None):
        use_filter = not (self.gnn_type in ("gcn", "ppmi"))
        if use_filter and filter_param is None:
            filter_param = self._default_filter_param(x)

        for i, conv_layer in enumerate(self.conv_layers):
            if use_filter:
                x = conv_layer(x, edge_index, filter_param=filter_param)
            else:
                x = conv_layer(x, edge_index)
            if i < len(self.conv_layers) - 1:
                x = self.act(x)
                x = self.dropout(x)

        if mode == "graph":
            x = global_mean_pool(x, batch)

        return x


class FiltADABase(nn.Module):
    """AdaGCNBase structure with filter-aware encoder."""

    def __init__(
        self,
        in_dim,
        hid_dim,
        num_classes,
        num_layers=3,
        dropout=0.1,
        act=F.relu,
        gnn_type="filter_mono",
        mode="node",
        filter_depth=4,
        **kwargs,
    ):
        super(FiltADABase, self).__init__()

        self.encoder = GNN(
            in_dim=in_dim,
            hid_dim=hid_dim,
            gnn_type=gnn_type,
            act=act,
            num_layers=num_layers,
            dropout=dropout,
            filter_depth=filter_depth,
        )
        self.cls_model = nn.Sequential(nn.Linear(hid_dim, num_classes))
        self.mode = mode
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, data, filter_param=None):
        if self.mode == "node":
            x, edge_index, batch = data.x, data.edge_index, None
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.encoder(x, edge_index, batch, mode=self.mode, filter_param=filter_param)
        return x

