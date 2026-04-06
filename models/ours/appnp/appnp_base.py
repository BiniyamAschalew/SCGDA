import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import APPNP

from  models.__layers.build_layer import build_activation


class APPNPBase(nn.Module):
    """MLP + APPNP propagation, matching the repo's node-GNN backbone API."""

    def __init__(self, config: dict):
        super(APPNPBase, self).__init__()

        self.in_dim = int(config["model"]["in_dim"])
        self.hid_dim = int(config["model"]["hid_dim"])
        self.num_classes = int(config["model"]["num_classes"])
        self.num_layers = int(config["model"]["num_layers"])
        self.dropout = float(config["model"]["dropout_ratio"])
        self.mode = config["model"]["mode"]
        self.act = build_activation(config["model"]["activation"])

        if self.mode != "node":
            raise ValueError(f"APPNPBase currently supports mode='node' only, got {self.mode}.")
        if self.num_layers < 1:
            raise ValueError(f"APPNPBase expects num_layers >= 1, got {self.num_layers}.")

        self.prop_steps = int(config["model"].get("K", 10))
        self.alpha = float(config["model"].get("alpha", 0.1))
        self.prop_dropout = float(
            config["model"].get("prop_dropout", config["model"].get("dprate", 0.0))
        )

        self.hidden_lins = nn.ModuleList()
        if self.num_layers > 1:
            self.hidden_lins.append(Linear(self.in_dim, self.hid_dim))
            for _ in range(self.num_layers - 2):
                self.hidden_lins.append(Linear(self.hid_dim, self.hid_dim))
            cls_in_dim = self.hid_dim
        else:
            cls_in_dim = self.in_dim

        self.cls = Linear(cls_in_dim, self.num_classes)
        self.prop = APPNP(K=self.prop_steps, alpha=self.alpha, dropout=self.prop_dropout)

    def reset_parameters(self):
        for lin in self.hidden_lins:
            lin.reset_parameters()
        self.cls.reset_parameters()
        self.prop.reset_parameters()

    def feat_bottleneck(self, x):
        for lin in self.hidden_lins:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.act(lin(x))
        return x

    def feat_classifier(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.cls(x)
        return self.prop(x, edge_index, edge_weight=edge_weight)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        del batch
        x = self.feat_bottleneck(x)
        x = self.feat_classifier(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)
