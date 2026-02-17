"""Graph models for structural-shift alignment experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return squared Euclidean distance matrix."""
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True).t()
    return torch.clamp(x_norm + y_norm - 2.0 * (x @ y.t()), min=0.0)


def mmd_rbf(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return multi-kernel RBF MMD between source and target embeddings."""
    all_feat = torch.cat([source, target], dim=0)
    with torch.no_grad():
        dists = _pairwise_sq_dist(all_feat, all_feat)
        n = dists.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=dists.device)
        vals = dists[mask]
        bandwidth = (
            vals.median().clamp_min(eps)
            if vals.numel() > 0
            else torch.tensor(1.0, dtype=source.dtype, device=source.device)
        )
    if fix_sigma is not None:
        bandwidth = torch.as_tensor(fix_sigma, dtype=source.dtype, device=source.device)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))

    xx = _pairwise_sq_dist(source, source)
    yy = _pairwise_sq_dist(target, target)
    xy = _pairwise_sq_dist(source, target)

    k_xx = torch.zeros((), dtype=source.dtype, device=source.device)
    k_yy = torch.zeros_like(k_xx)
    k_xy = torch.zeros_like(k_xx)
    for i in range(kernel_num):
        bw = (bandwidth * (kernel_mul**i)).clamp_min(eps)
        k_xx = k_xx + torch.exp(-xx / bw)
        k_yy = k_yy + torch.exp(-yy / bw)
        k_xy = k_xy + torch.exp(-xy / bw)
    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


def normalize_dense_adj(adj: torch.Tensor, *, add_self_loop: bool = True, eps: float = 1e-12) -> torch.Tensor:
    """Return symmetric-normalized dense adjacency."""
    if add_self_loop:
        adj = adj + torch.eye(adj.size(0), dtype=adj.dtype, device=adj.device)
    deg = adj.sum(dim=1).clamp_min(eps)
    inv_sqrt = deg.pow(-0.5)
    return inv_sqrt[:, None] * adj * inv_sqrt[None, :]


def _as_dense_adj(adj: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    """Validate and move a square dense adjacency matrix."""
    if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Expected a square dense adjacency matrix.")
    if adj.shape[0] != num_nodes:
        raise ValueError("Adjacency size must match node count.")
    if not torch.is_floating_point(adj):
        adj = adj.float()
    return adj.to(device=device)


class PolynomialFilter(nn.Module):
    """Apply learnable polynomial filtering S(A)=sum_k alpha_k A^k."""

    def __init__(self, order: int) -> None:
        """Initialize polynomial filter with trainable logits."""
        super().__init__()
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = int(order)
        self.logits = nn.Parameter(torch.zeros(order + 1))

    def coefficients(self) -> torch.Tensor:
        """Return softmax-normalized polynomial coefficients."""
        return F.softmax(self.logits, dim=0)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        """Return filtered features after polynomial aggregation."""
        hops = [x]
        h = x
        for _ in range(self.order):
            h = norm_adj @ h
            hops.append(h)
        stacked = torch.stack(hops, dim=0)
        coeff = self.coefficients().view(-1, 1, 1)
        return (coeff * stacked).sum(dim=0)


class GNN(nn.Module):
    """Compute relu(S(A) X W) with fixed first-order filter S(A)=A_norm."""

    needs_adjacency = True

    def __init__(
        self,
        *,
        input_dim: int,
        embed_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        align_on: str = "weight",
        add_self_loop: bool = True,
    ) -> None:
        """Initialize dual-domain GNN with source-supervised head."""
        super().__init__()
        if align_on not in {"weight", "none"}:
            raise ValueError("GNN align_on must be one of {'weight', 'none'}")
        self.align_on = align_on
        self.add_self_loop = add_self_loop
        self.source_weight = nn.Linear(input_dim, embed_dim, bias=False)
        self.target_weight = nn.Linear(input_dim, embed_dim, bias=False)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("_fixed_coeff", torch.tensor([0.0, 1.0], dtype=torch.float32))

    def _encode(self, x: torch.Tensor, adj: torch.Tensor, weight: nn.Linear) -> torch.Tensor:
        """Encode one domain with fixed first-order graph filtering."""
        dense = _as_dense_adj(adj, x.size(0), x.device)
        norm = normalize_dense_adj(dense, add_self_loop=self.add_self_loop)
        return self.dropout(F.relu(weight(norm @ x)))

    def forward(
        self,
        source_x: torch.Tensor,
        source_adj: torch.Tensor,
        target_x: torch.Tensor,
        target_adj: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return logits and embeddings for source and target domains."""
        source_embed = self._encode(source_x, source_adj, self.source_weight)
        target_embed = self._encode(target_x, target_adj, self.target_weight)
        return {
            "source_embed": source_embed,
            "target_embed": target_embed,
            "source_logits": self.classifier(source_embed),
            "target_logits": self.classifier(target_embed),
        }

    def embedding_alignment_loss(self, source_embed: torch.Tensor, target_embed: torch.Tensor) -> torch.Tensor:
        """Return embedding alignment loss."""
        return mmd_rbf(source_embed, target_embed)

    def parameter_alignment_loss(self) -> torch.Tensor:
        """Return weight alignment loss when enabled."""
        if self.align_on == "weight":
            return F.mse_loss(self.source_weight.weight, self.target_weight.weight)
        return torch.zeros((), dtype=self.classifier.weight.dtype, device=self.classifier.weight.device)

    def filter_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return source and target filter coefficients."""
        coeff = self._fixed_coeff.to(dtype=self.classifier.weight.dtype, device=self.classifier.weight.device)
        return coeff, coeff

    def filter_gap(self) -> torch.Tensor:
        """Return L2 gap between source and target filters."""
        return torch.zeros((), dtype=self.classifier.weight.dtype, device=self.classifier.weight.device)

    def weight_gap(self) -> torch.Tensor:
        """Return L2 gap between source and target weights."""
        return torch.norm(self.source_weight.weight - self.target_weight.weight, p=2)


class SpectralGNN(nn.Module):
    """Compute relu(S(A) X W) with learnable polynomial spectral filter S(A)."""

    needs_adjacency = True

    def __init__(
        self,
        *,
        input_dim: int,
        embed_dim: int,
        num_classes: int,
        filter_order: int = 2,
        dropout: float = 0.0,
        align_on: str = "filter",
        add_self_loop: bool = True,
    ) -> None:
        """Initialize dual-domain SpectralGNN with source-supervised head."""
        super().__init__()
        if align_on not in {"filter", "weight", "none"}:
            raise ValueError("SpectralGNN align_on must be one of {'filter', 'weight', 'none'}")
        self.align_on = align_on
        self.add_self_loop = add_self_loop
        self.source_filter = PolynomialFilter(filter_order)
        self.target_filter = PolynomialFilter(filter_order)
        self.source_weight = nn.Linear(input_dim, embed_dim, bias=False)
        self.target_weight = nn.Linear(input_dim, embed_dim, bias=False)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def _encode(self, x: torch.Tensor, adj: torch.Tensor, filt: PolynomialFilter, weight: nn.Linear) -> torch.Tensor:
        """Encode one domain with learnable polynomial graph filtering."""
        dense = _as_dense_adj(adj, x.size(0), x.device)
        norm = normalize_dense_adj(dense, add_self_loop=self.add_self_loop)
        filtered = filt(x, norm)
        return self.dropout(F.relu(weight(filtered)))

    def forward(
        self,
        source_x: torch.Tensor,
        source_adj: torch.Tensor,
        target_x: torch.Tensor,
        target_adj: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return logits and embeddings for source and target domains."""
        source_embed = self._encode(source_x, source_adj, self.source_filter, self.source_weight)
        target_embed = self._encode(target_x, target_adj, self.target_filter, self.target_weight)
        return {
            "source_embed": source_embed,
            "target_embed": target_embed,
            "source_logits": self.classifier(source_embed),
            "target_logits": self.classifier(target_embed),
        }

    def embedding_alignment_loss(self, source_embed: torch.Tensor, target_embed: torch.Tensor) -> torch.Tensor:
        """Return embedding alignment loss."""
        return mmd_rbf(source_embed, target_embed)

    def parameter_alignment_loss(self) -> torch.Tensor:
        """Return either filter or weight alignment loss."""
        if self.align_on == "filter":
            return F.mse_loss(self.source_filter.logits, self.target_filter.logits)
        if self.align_on == "weight":
            return F.mse_loss(self.source_weight.weight, self.target_weight.weight)
        return torch.zeros((), dtype=self.classifier.weight.dtype, device=self.classifier.weight.device)

    def filter_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return source and target softmax filter coefficients."""
        return self.source_filter.coefficients(), self.target_filter.coefficients()

    def filter_gap(self) -> torch.Tensor:
        """Return L2 gap between source and target filters."""
        src, tgt = self.filter_weights()
        return torch.norm(src - tgt, p=2)

    def weight_gap(self) -> torch.Tensor:
        """Return L2 gap between source and target weights."""
        return torch.norm(self.source_weight.weight - self.target_weight.weight, p=2)
