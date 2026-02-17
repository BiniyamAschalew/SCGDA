"""Models used for structural-shift experiments."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True).t()
    dist = x_norm + y_norm - 2.0 * (x @ y.t())
    return torch.clamp(dist, min=0.0)


def mmd_rbf(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    with torch.no_grad():
        all_feat = torch.cat([source.detach(), target.detach()], dim=0)
        dists = _pairwise_sq_dist(all_feat, all_feat)
        n = dists.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=dists.device)
        vals = dists[mask]
        if vals.numel() == 0:
            bandwidth = torch.tensor(1.0, device=source.device, dtype=source.dtype)
        else:
            bandwidth = vals.median().clamp_min(eps)
    bandwidth = bandwidth if fix_sigma is None else torch.as_tensor(fix_sigma, device=source.device, dtype=source.dtype)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))

    xx = _pairwise_sq_dist(source, source)
    yy = _pairwise_sq_dist(target, target)
    xy = _pairwise_sq_dist(source, target)

    k_xx = torch.zeros((), device=source.device, dtype=source.dtype)
    k_yy = torch.zeros_like(k_xx)
    k_xy = torch.zeros_like(k_xx)
    for i in range(kernel_num):
        bw = (bandwidth * (kernel_mul ** i)).clamp_min(eps)
        k_xx = k_xx + torch.exp(-xx / bw)
        k_yy = k_yy + torch.exp(-yy / bw)
        k_xy = k_xy + torch.exp(-xy / bw)

    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


class _DomainAlignMixin:
    def mmd_loss(self, source_embed: torch.Tensor, target_embed: torch.Tensor) -> torch.Tensor:
        return mmd_rbf(source_embed, target_embed)

    def encode(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class MLP(nn.Module, _DomainAlignMixin):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        mmd_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.encoder = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.classifier = nn.Linear(output_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.mmd_weight = mmd_weight

    def encode(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.encoder):
            h = layer(h)
            if i < len(self.encoder) - 1:
                h = self.dropout(F.relu(h))
        return h

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, adj)
        return self.classifier(z), z


class GNN(nn.Module, _DomainAlignMixin):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        add_self_loop: bool = True,
        mmd_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.classifier = nn.Linear(output_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.add_self_loop = add_self_loop
        self.mmd_weight = mmd_weight

    @staticmethod
    def _dense_adj(adj: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
        if adj.dim() != 2:
            raise ValueError("Expected a dense adjacency matrix for this GNN path.")
        if adj.shape[0] != adj.shape[1]:
            raise ValueError("Dense adjacency must be square.")
        if adj.shape[0] != num_nodes:
            raise ValueError("Adjacency size does not match number of nodes.")
        if not torch.is_floating_point(adj):
            adj = adj.float()
        return adj.to(device=device)

    @staticmethod
    def _norm_adj(adj: torch.Tensor, add_self_loop: bool) -> torch.Tensor:
        if add_self_loop:
            adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        deg = adj.sum(dim=1)
        inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
        return inv_sqrt[:, None] * adj * inv_sqrt[None, :]

    def encode(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        if adj is None:
            raise ValueError("GNN requires adjacency matrix for propagation")

        h = x
        a = self._norm_adj(self._dense_adj(adj, x.size(0), x.device), self.add_self_loop)
        for i, layer in enumerate(self.layers):
            h = a @ h
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.dropout(F.relu(h))
        return h

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, adj)
        return self.classifier(z), z
