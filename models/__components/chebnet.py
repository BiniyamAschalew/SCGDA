from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, global_mean_pool

from models.__layers.build_layer import build_activation

class ChebNetBase(nn.Module):

    def __init__(self, config: dict):
        super(ChebNetBase, self).__init__()

        self.in_dim = config["model"]["in_dim"]
        self.hid_dim = config["model"]["hid_dim"]
        self.num_classes = config["model"]["num_classes"]
        self.num_layers = config["model"]["num_layers"]
        self.dropout = config["model"]["dropout_ratio"]
        self.mode = config["model"]["mode"]

        self.cheb_k = config["model"].get("cheb_k", config["model"].get("K", 3))
        self.cheb_norm = config["model"].get("cheb_norm", "sym")
        self.cheb_bias = config["model"].get("cheb_bias", True)

        self.act = build_activation(config["model"]["activation"])

        self.convs = nn.ModuleList()
        self.convs.append(
            ChebConv(
                self.in_dim,
                self.hid_dim,
                K=self.cheb_k,
                normalization=self.cheb_norm,
                bias=self.cheb_bias,
            )
        )
        for _ in range(self.num_layers - 1):
            self.convs.append(
                ChebConv(
                    self.hid_dim,
                    self.hid_dim,
                    K=self.cheb_k,
                    normalization=self.cheb_norm,
                    bias=self.cheb_bias,
                )
            )

        if self.mode == "node":
            self.cls = ChebConv(
                self.hid_dim,
                self.num_classes,
                K=self.cheb_k,
                normalization=self.cheb_norm,
                bias=self.cheb_bias,
            )
        elif self.mode == "graph":
            self.cls = nn.Linear(self.hid_dim, self.num_classes)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _apply_cheb(self, conv, x, edge_index, edge_weight=None, lambda_max=None):
        if lambda_max is None:
            return conv(x, edge_index, edge_weight=edge_weight)
        return conv(x, edge_index, edge_weight=edge_weight, lambda_max=lambda_max)

    def forward(self, x, edge_index, edge_weight=None, batch=None, lambda_max=None):
        x = self.feat_bottleneck(x, edge_index, edge_weight, batch, lambda_max)
        x = self.feat_classifier(x, edge_index, edge_weight, lambda_max)
        x = F.log_softmax(x, dim=1)
        return x

    def feat_bottleneck(self, x, edge_index, edge_weight=None, batch=None, lambda_max=None):
        for i, conv in enumerate(self.convs):
            x = self._apply_cheb(conv, x, edge_index, edge_weight, lambda_max)
            if i < len(self.convs) - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.mode == "graph":
            x = global_mean_pool(x, batch)

        return x

    def feat_classifier(self, x, edge_index, edge_weight=None, lambda_max=None):
        if self.mode == "node":
            x = self._apply_cheb(self.cls, x, edge_index, edge_weight, lambda_max)
        else:
            x = self.cls(x)
        return x

    def filter_parameters(self):
        params = []
        for conv in self.convs:
            params.extend(list(conv.parameters()))
        return params
