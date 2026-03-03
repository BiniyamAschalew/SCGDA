import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

from models.__layers.build_layer import build_layer, build_activation
from models.__layers.filter_gcn_conv import FilterGCNConv
from models.__layers.reverse_layer import GradReverse


class SimGDAFilterBase(nn.Module):
    """A2GNNBase-style backbone with monomial-basis filter propagation."""

    def __init__(self, config: dict):
        super(SimGDAFilterBase, self).__init__()

        self.in_dim = config["model"]["in_dim"]
        self.hid_dim = config["model"]["hid_dim"]

        self.num_classes = config["model"]["num_classes"]
        self.num_layers = config["model"]["num_layers"]
        self.adv = bool(config["model"].get("adv", False))

        self.dropout = config["model"]["dropout_ratio"]
        self.mode = config["model"]["mode"]
        self.act = build_activation(config["model"]["activation"])

        gnn_type = str(config["model"].get("gnn", "filter_mono")).lower()
        self.gnn_type = gnn_type if gnn_type.startswith("filter") else "filter_mono"

        self.k = int(config["model"].get("K", 4))
        self.cls_pnums = int(config["model"].get("cls_pnums", 1))

        self.convs = nn.ModuleList()
        self.convs.append(self._build_conv(self.in_dim, self.hid_dim))
        for _ in range(self.num_layers - 1):
            self.convs.append(self._build_conv(self.hid_dim, self.hid_dim))

        if self.mode == "node":
            self.cls = self._build_conv(self.hid_dim, self.num_classes)
        elif self.mode == "graph":
            self.cls = nn.Linear(self.hid_dim, self.num_classes)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        if self.adv:
            self.domain_discriminator = nn.Linear(self.hid_dim, 2)

    def _build_conv(self, in_dim: int, out_dim: int):
        if self.gnn_type.startswith("filter"):
            if self.gnn_type == "filter":
                filter_type = "mono"
            else:
                filter_type = self.gnn_type.split("_", 1)[1]
            return FilterGCNConv(
                in_dim,
                out_dim,
                filter_type=filter_type,
                prepend_zero_to_filter=False,  # basis index k maps to A^k
            )
        return build_layer(in_dim, out_dim, self.gnn_type)

    def _basis_one_hot(self, basis_idx: int, x: torch.Tensor) -> torch.Tensor:
        idx = int(basis_idx)
        if idx < 0 or idx >= self.k:
            raise ValueError(f"basis index out of range: idx={idx}, K={self.k}")
        basis = torch.zeros(self.k, device=x.device, dtype=x.dtype)
        basis[idx] = 1.0
        return basis

    def _to_filter_param(self, filter_param_or_idx, x: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(filter_param_or_idx):
            filter_param = filter_param_or_idx.to(device=x.device, dtype=x.dtype).view(-1)
            if int(filter_param.numel()) != self.k:
                raise ValueError(f"filter_param length mismatch: expected K={self.k}, got {int(filter_param.numel())}")
            return filter_param
        return self._basis_one_hot(int(filter_param_or_idx), x)

    def forward(self, data, filter_param):
        if self.mode == "node":
            x, edge_index, batch = data.x, data.edge_index, None
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.feat_bottleneck(x, edge_index, batch, filter_param=filter_param)
        x = self.feat_classifier(x, edge_index, batch, filter_param=self._basis_one_hot(self.cls_pnums, x))
        return x

    def feat_bottleneck(self, x, edge_index, batch, filter_param):
        filter_param = self._to_filter_param(filter_param, x)
        for conv in self.convs:
            x = conv(x, edge_index, filter_param=filter_param)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.mode == "graph":
            x = global_mean_pool(x, batch)

        return x

    def feat_classifier(self, x, edge_index, batch, filter_param):
        if self.mode == "node":
            filter_param = self._to_filter_param(filter_param, x)
            x = self.cls(x, edge_index, filter_param=filter_param)
        else:
            x = self.cls(x)

        return x

    def domain_classifier(self, x, alpha):
        x = GradReverse.apply(x, alpha)
        d_logit = self.domain_discriminator(x)
        return d_logit
