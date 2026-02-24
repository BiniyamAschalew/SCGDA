import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from models.__layers.build_layer import build_activation
from models.__components.chebprop import ChebProp


class TestBase(nn.Module):

    def __init__(self, config: dict):
        super(TestBase, self).__init__()

        self.in_dim = config["model"]["in_dim"]
        self.hid_dim = config["model"]["hid_dim"]
        self.num_classes = config["model"]["num_classes"]
        self.num_layers = config["model"]["num_layers"]
        self.dropout = config["model"]["dropout_ratio"]
        self.mode = config["model"]["mode"]

        self.cheb_k = config["model"].get(
            "cheb_k",
            config["model"].get("cheb_K", config["model"].get("K", 3)),
        )
        self.act = build_activation(config["model"]["activation"])

        self.lin = nn.Linear(self.in_dim, self.hid_dim)
        self.props = nn.ModuleList(
            [ChebProp(self.cheb_k, is_source_domain=False) for _ in range(self.num_layers)]
        )

        if self.mode == "node":
            self.cls = nn.Linear(self.hid_dim, self.num_classes)
            self.prop_out = ChebProp(self.cheb_k, is_source_domain=False)
        elif self.mode == "graph":
            self.cls = nn.Linear(self.hid_dim, self.num_classes)
            self.prop_out = None
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # One shared Chebyshev approximation per domain, reused in every layer.
        self.source_temp = nn.Parameter(self.lin.weight.new_empty(self.cheb_k))
        self.target_temp = nn.Parameter(self.lin.weight.new_empty(self.cheb_k))
        self.reset_filter_parameters()

    def reset_filter_parameters(self):
        init_value = 1.0 / float(self.cheb_k)
        with torch.no_grad():
            self.source_temp.fill_(init_value)
            self.target_temp.fill_(init_value)

    def _get_domain_temp(self, domain):
        if domain == "source":
            return self.source_temp
        if domain == "target":
            return self.target_temp
        raise ValueError(f"Invalid domain: {domain}")

    def forward(self, x, edge_index, edge_weight=None, batch=None, domain="source"):
        x = self.feat_bottleneck(x, edge_index, edge_weight=edge_weight, batch=batch, domain=domain)
        x = self.feat_classifier(x, edge_index, edge_weight=edge_weight, domain=domain)
        x = F.log_softmax(x, dim=1)
        return x

    def feat_bottleneck(self, x, edge_index, edge_weight=None, batch=None, domain="source"):
        temp = self._get_domain_temp(domain)

        x = self.lin(x)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, prop in enumerate(self.props):
            x = prop(x, edge_index, edge_weight=edge_weight, batch=batch, temp=temp)
            if i < len(self.props) - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.mode == "graph":
            x = global_mean_pool(x, batch)

        return x

    def feat_classifier(self, x, edge_index, edge_weight=None, domain="source"):
        temp = self._get_domain_temp(domain)

        if self.mode == "node":
            x = self.cls(x)
            if self.prop_out is not None:
                x = self.prop_out(x, edge_index, edge_weight=edge_weight, temp=temp)
        else:
            x = self.cls(x)
        return x

    def filter_parameters(self):
        return [self.source_temp, self.target_temp]
