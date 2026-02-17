from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from prob_form.synthetic_data import SyntheticDomainData


@dataclass
class TrainingConfig:
    hidden_dim: int = 64
    latent_dim: int = 8
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    snapshot_stride: int = 10
    seed: int = 7
    device: str = "cpu"


@dataclass
class TrainingArtifacts:
    regime: str
    history: list[dict[str, float]]
    snapshots: dict[int, dict[str, np.ndarray]]


class DomainAdaptationModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, n_classes: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.classifier = nn.Linear(latent_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits, z


def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True).transpose(0, 1)
    dist = x_norm + y_norm - 2.0 * (x @ y.transpose(0, 1))
    return torch.clamp(dist, min=0.0)


def _median_bandwidth(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    all_feat = torch.cat([x, y], dim=0)
    dists = _pairwise_sq_dist(all_feat, all_feat).detach()
    n = dists.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=dists.device)
    vals = dists[mask]
    if vals.numel() == 0:
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)
    return vals.median().clamp_min(eps)


def mmd_rbf(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    bandwidth = (
        torch.as_tensor(fix_sigma, device=source.device, dtype=source.dtype)
        if fix_sigma is not None
        else _median_bandwidth(source, target, eps=eps)
    )
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))

    xx = _pairwise_sq_dist(source, source)
    yy = _pairwise_sq_dist(target, target)
    xy = _pairwise_sq_dist(source, target)

    k_xx = 0.0
    k_yy = 0.0
    k_xy = 0.0
    for i in range(kernel_num):
        bw = (bandwidth * (kernel_mul**i)).clamp_min(eps)
        k_xx = k_xx + torch.exp(-xx / bw)
        k_yy = k_yy + torch.exp(-yy / bw)
        k_xy = k_xy + torch.exp(-xy / bw)

    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _snapshot_epochs(epochs: int, stride: int) -> list[int]:
    points = set(range(0, epochs + 1, max(stride, 1)))
    points.add(0)
    points.add(epochs)
    return sorted(points)


def _to_tensors(data: SyntheticDomainData, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "xs": torch.tensor(data.source_x, dtype=torch.float32, device=device),
        "ys": torch.tensor(data.source_y, dtype=torch.long, device=device),
        "xt": torch.tensor(data.target_x, dtype=torch.float32, device=device),
        "yt": torch.tensor(data.target_y, dtype=torch.long, device=device),
    }


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


def _conditional_mmd(
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    source_y: torch.Tensor,
    target_y: torch.Tensor,
) -> torch.Tensor:
    classes = torch.unique(torch.cat([source_y, target_y], dim=0))
    per_class: list[torch.Tensor] = []
    for cls in classes:
        src_mask = source_y == cls
        tgt_mask = target_y == cls
        if int(src_mask.sum().item()) == 0 or int(tgt_mask.sum().item()) == 0:
            continue
        per_class.append(mmd_rbf(source_feat[src_mask], target_feat[tgt_mask]))
    if not per_class:
        return torch.tensor(0.0, device=source_feat.device, dtype=source_feat.dtype)
    return torch.stack(per_class).mean()


def _evaluate(
    model: DomainAdaptationModel,
    tensors: dict[str, torch.Tensor],
    lambda_mmd: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    with torch.no_grad():
        logits_s, z_s = model(tensors["xs"])
        logits_t, z_t = model(tensors["xt"])

        cls_loss = F.cross_entropy(logits_s, tensors["ys"])
        mmd_loss = mmd_rbf(z_s, z_t)
        total = cls_loss + lambda_mmd * mmd_loss

        raw_mmd = mmd_rbf(tensors["xs"], tensors["xt"])
        cond_raw_mmd = _conditional_mmd(
            tensors["xs"], tensors["xt"], tensors["ys"], tensors["yt"]
        )
        cond_latent_mmd = _conditional_mmd(z_s, z_t, tensors["ys"], tensors["yt"])
        metrics = {
            "total_loss": float(total.item()),
            "cls_loss": float(cls_loss.item()),
            "mmd_loss": float(mmd_loss.item()),
            "raw_mmd": float(raw_mmd.item()),
            "conditional_raw_mmd": float(cond_raw_mmd.item()),
            "latent_mmd": float(mmd_loss.item()),
            "conditional_latent_mmd": float(cond_latent_mmd.item()),
            "source_acc": _accuracy(logits_s, tensors["ys"]),
            "target_acc": _accuracy(logits_t, tensors["yt"]),
        }

        snapshot = {
            "source_latent": z_s.detach().cpu().numpy(),
            "target_latent": z_t.detach().cpu().numpy(),
        }
    return metrics, snapshot


def run_training(
    *,
    regime: str,
    data: SyntheticDomainData,
    cfg: TrainingConfig,
    lambda_mmd: float,
) -> TrainingArtifacts:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    tensors = _to_tensors(data, device=device)

    model = DomainAdaptationModel(
        in_dim=data.source_x.shape[1],
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        n_classes=int(data.source_y.max()) + 1,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: list[dict[str, float]] = []
    snapshots: dict[int, dict[str, np.ndarray]] = {}
    capture_epochs = set(_snapshot_epochs(cfg.epochs, cfg.snapshot_stride))

    metrics, snap = _evaluate(model, tensors, lambda_mmd=lambda_mmd)
    metrics["epoch"] = 0
    history.append(metrics)
    if 0 in capture_epochs:
        snapshots[0] = snap

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits_s, z_s = model(tensors["xs"])
        _, z_t = model(tensors["xt"])
        cls_loss = F.cross_entropy(logits_s, tensors["ys"])
        mmd_loss = mmd_rbf(z_s, z_t)
        loss = cls_loss + lambda_mmd * mmd_loss

        loss.backward()
        optimizer.step()

        metrics, snap = _evaluate(model, tensors, lambda_mmd=lambda_mmd)
        metrics["epoch"] = epoch
        history.append(metrics)
        if epoch in capture_epochs:
            snapshots[epoch] = snap

    return TrainingArtifacts(regime=regime, history=history, snapshots=snapshots)


def combine_histories(artifacts: Iterable[TrainingArtifacts]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for item in artifacts:
        for row in item.history:
            out = dict(row)
            out["regime"] = item.regime
            rows.append(out)
    return rows
