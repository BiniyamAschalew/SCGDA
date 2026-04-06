from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from Learn.Clean_SCGDA.models.__layers.build_layer import build_activation
from Learn.Clean_SCGDA.models.__components.bernprop import BernProp



class BernNetBase(nn.Module):

    def __init__(self, config: dict):
        super(BernNetBase, self).__init__()

        self.in_dim = config["model"]["in_dim"]
        self.hid_dim = config["model"]["hid_dim"]
        self.num_classes = config["model"]["num_classes"]
        self.num_layers = config["model"]["num_layers"]
        self.dropout = config["model"]["dropout_ratio"]
        self.mode = config["model"]["mode"]

        self.bern_k = config["model"].get("bern_k", config["model"].get("K", 10))
        self.act = build_activation(config["model"]["activation"])

        self.lin = nn.Linear(self.in_dim, self.hid_dim)
        self.props = nn.ModuleList([BernProp(self.bern_k) for _ in range(self.num_layers)])

        if self.mode == "node":
            self.cls = nn.Linear(self.hid_dim, self.num_classes)
            self.prop_out = BernProp(self.bern_k)
        elif self.mode == "graph":
            self.cls = nn.Linear(self.hid_dim, self.num_classes)
            self.prop_out = None
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        x = self.feat_bottleneck(x, edge_index, edge_weight, batch)
        x = self.feat_classifier(x, edge_index, edge_weight)
        x = F.log_softmax(x, dim=1)
        return x

    def feat_bottleneck(self, x, edge_index, edge_weight=None, batch=None):
        x = self.lin(x)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, prop in enumerate(self.props):
            x = prop(x, edge_index, edge_weight=edge_weight)
            if i < len(self.props) - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.mode == "graph":
            x = global_mean_pool(x, batch)

        return x

    def feat_classifier(self, x, edge_index, edge_weight=None):
        if self.mode == "node":
            x = self.cls(x)
            if self.prop_out is not None:
                x = self.prop_out(x, edge_index, edge_weight=edge_weight)
        else:
            x = self.cls(x)
        return x

    def filter_parameters(self):
        params = []
        for prop in self.props:
            params.extend(list(prop.parameters()))
        return params
