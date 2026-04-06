import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from Learn.Clean_SCGDA.models.__layers.build_layer import build_activation


class GPRProp(MessagePassing):
    """Learnable generalized PageRank propagation from the original GPR-GNN."""

    def __init__(self, K, alpha, init="PPR", gamma=None, **kwargs):
        kwargs.setdefault("aggr", "add")
        super(GPRProp, self).__init__(**kwargs)

        self.K = int(K)
        self.alpha = float(alpha)
        self.init = init

        if gamma is not None:
            gamma = torch.as_tensor(gamma, dtype=torch.float)
            if gamma.numel() != self.K + 1:
                raise ValueError(f"Expected gamma with {self.K + 1} elements, got {gamma.numel()}.")
            self.gamma = gamma.clone().detach()
        else:
            self.gamma = None

        self.temp = Parameter(self._build_init_coefficients())

    def _build_init_coefficients(self):
        if self.init == "SGC":
            peak = int(round(self.alpha))
            if peak < 0 or peak > self.K:
                raise ValueError(f"SGC init expects integer alpha in [0, {self.K}], got {self.alpha}.")
            temp = torch.zeros(self.K + 1, dtype=torch.float)
            temp[peak] = 1.0
            return temp

        if self.init == "PPR":
            temp = self.alpha * (1.0 - self.alpha) ** torch.arange(self.K + 1, dtype=torch.float)
            temp[-1] = (1.0 - self.alpha) ** self.K
            return temp

        if self.init == "NPPR":
            temp = self.alpha ** torch.arange(self.K + 1, dtype=torch.float)
            return temp / temp.abs().sum().clamp_min(1e-12)

        if self.init == "Random":
            bound = math.sqrt(3.0 / float(self.K + 1))
            temp = torch.empty(self.K + 1, dtype=torch.float).uniform_(-bound, bound)
            return temp / temp.abs().sum().clamp_min(1e-12)

        if self.init == "WS":
            if self.gamma is None:
                raise ValueError("WS init requires a gamma coefficient list.")
            return self.gamma.to(dtype=torch.float)

        raise ValueError(f"Unsupported GPR initialization: {self.init}")

    def reset_parameters(self):
        with torch.no_grad():
            self.temp.copy_(self._build_init_coefficients().to(self.temp.device))

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index,
            edge_weight,
            num_nodes=x.size(0),
            dtype=x.dtype,
        )

        hidden = x * self.temp[0]
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            hidden = hidden + self.temp[k + 1] * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GPRGNNBase(nn.Module):
    """MLP + learnable generalized PageRank propagation."""

    def __init__(self, config: dict):
        super(GPRGNNBase, self).__init__()

        self.in_dim = int(config["model"]["in_dim"])
        self.hid_dim = int(config["model"]["hid_dim"])
        self.num_classes = int(config["model"]["num_classes"])
        self.num_layers = int(config["model"]["num_layers"])
        self.dropout = float(config["model"]["dropout_ratio"])
        self.mode = config["model"]["mode"]
        self.act = build_activation(config["model"]["activation"])

        if self.mode != "node":
            raise ValueError(f"GPRGNNBase currently supports mode='node' only, got {self.mode}.")
        if self.num_layers < 1:
            raise ValueError(f"GPRGNNBase expects num_layers >= 1, got {self.num_layers}.")

        self.prop_steps = int(config["model"].get("K", 10))
        self.alpha = float(config["model"].get("alpha", 0.1))
        self.prop_init = str(config["model"].get("Init", "PPR"))
        self.dprate = float(config["model"].get("dprate", 0.0))
        self.gamma = config["model"].get("Gamma", None)

        self.hidden_lins = nn.ModuleList()
        if self.num_layers > 1:
            self.hidden_lins.append(Linear(self.in_dim, self.hid_dim))
            for _ in range(self.num_layers - 2):
                self.hidden_lins.append(Linear(self.hid_dim, self.hid_dim))
            cls_in_dim = self.hid_dim
        else:
            cls_in_dim = self.in_dim

        self.cls = Linear(cls_in_dim, self.num_classes)
        self.prop = GPRProp(
            K=self.prop_steps,
            alpha=self.alpha,
            init=self.prop_init,
            gamma=self.gamma,
        )

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
        if self.dprate > 0.0:
            x = F.dropout(x, p=self.dprate, training=self.training)
        return self.prop(x, edge_index, edge_weight=edge_weight)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        del batch
        x = self.feat_bottleneck(x)
        x = self.feat_classifier(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)
