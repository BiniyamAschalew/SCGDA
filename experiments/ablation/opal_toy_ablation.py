"""Expanded OPAL ablation on synthetic and real data.

This script has three connected goals:
1. Synthetic performance sweeps on a larger toy benchmark.
2. Fully visualizable small-graph structure diagnostics with weighted induced graphs.
3. Real-data OPAL ablations using the repo's normal config/dataset/model path.

The synthetic objective mirrors the OPAL split between feature alignment (`beta`)
and filter alignment (`delta`), while the separation of feature-vs-filter effects
is also informed by the spectral SimGDA warmup path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.build_dataset import build_dataset
from models.build_model import build_model
from utils.config_utils import build_config
from utils.expt_utils import set_seed as repo_set_seed
from utils.train_utils.metrics import BaseMetric

torch.set_num_threads(1)


ABLATION_PLAN = {
    "synthetic_large": {
        "purpose": "Measure performance and divergence trends across seeds on a stable toy DA benchmark.",
        "variants": ["source_only", "feat_only", "filter_only", "feat_plus_filter"],
        "focus_metric": "target_acc",
    },
    "synthetic_small": {
        "purpose": "Visualize how induced weighted structures change on fully viewable small graphs.",
        "motif_shift": "source uses sparse cycle communities, target uses dense clique communities",
        "structural_metrics": [
            "effective_density",
            "support_density",
            "triangle_count",
            "wedge_count",
            "transitivity",
            "probe_dirichlet_energy",
            "within_class_mass",
            "cross_class_mass",
        ],
    },
    "real_data": {
        "purpose": "Quantify whether filter alignment helps on top of feature alignment under the repo-native OPAL training path.",
        "model": "opal",
        "comparison": "feat_plus_filter minus feat_only",
        "reported_metrics": ["target_micro_f1", "target_macro_f1", "filter_coef_gap_l2"],
    },
    "degree_shift_fixed_source": {
        "purpose": "Hold the source filter fixed and learn only the target filter to see whether operator alignment mitigates a 3x source-target degree shift.",
        "setting": "source graph is 3x denser than target in expectation",
        "diagnostics": [
            "raw_degree_ratio",
            "induced_degree_ratio",
            "raw_degree_gap_l1",
            "induced_degree_gap_l1",
            "response_div_mmd",
            "response_div_mse",
        ],
    },
}

VARIANT_LABELS = {
    "source_only": "Source Only",
    "feat_only": "Feature Align",
    "filter_only": "Filter Align",
    "feat_plus_filter": "Feature + Filter",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_int_list(raw: str) -> list[int]:
    values = []
    for part in str(raw).split(","):
        token = part.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def save_json(path: Path, payload) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(_json_ready(payload), f, indent=2)


def save_rows_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["empty"])
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    payload[key] = json.dumps(_json_ready(value))
                else:
                    payload[key] = value
            writer.writerow(payload)


def summarize_variants(rows: list[dict], metrics: list[str], order: list[str]) -> list[dict]:
    grouped = {name: [] for name in order}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)

    summary_rows = []
    for variant in order:
        items = grouped.get(variant, [])
        if not items:
            continue
        payload = {"variant": variant, "variant_label": VARIANT_LABELS.get(variant, variant), "n": len(items)}
        for metric in metrics:
            vals = [float(item[metric]) for item in items if metric in item]
            payload[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
            payload[f"{metric}_std"] = float(np.std(vals)) if vals else float("nan")
        summary_rows.append(payload)
    return summary_rows


def compute_gain(
    summary_rows: list[dict],
    compare_variant: str,
    base_variant: str,
    metrics: list[str],
) -> dict[str, float | str]:
    by_variant = {row["variant"]: row for row in summary_rows}
    if compare_variant not in by_variant or base_variant not in by_variant:
        return {}

    comp = by_variant[compare_variant]
    base = by_variant[base_variant]
    gain = {
        "compare_variant": compare_variant,
        "base_variant": base_variant,
        "compare_label": VARIANT_LABELS.get(compare_variant, compare_variant),
        "base_label": VARIANT_LABELS.get(base_variant, base_variant),
    }
    for metric in metrics:
        gain[f"{metric}_delta"] = float(comp.get(f"{metric}_mean", float("nan")) - base.get(f"{metric}_mean", float("nan")))
    return gain


def set_local_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_binary_adj(num_nodes: int, undirected_edges: list[tuple[int, int]]) -> torch.Tensor:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for i, j in undirected_edges:
        if i == j:
            continue
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return adj


def make_sbm(labels: list[int], p_in: float, p_out: float) -> torch.Tensor:
    n = len(labels)
    adj = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(i + 1, n):
            prob = p_in if labels[i] == labels[j] else p_out
            if random.random() < prob:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    return adj


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    num_nodes = adj.shape[0]
    eye = torch.eye(num_nodes, dtype=adj.dtype, device=adj.device)
    a_tilde = adj + eye
    deg = a_tilde.sum(dim=1)
    inv_sqrt = deg.clamp(min=1e-8).pow(-0.5)
    d = torch.diag(inv_sqrt)
    return d @ a_tilde @ d


def symmetrize_matrix(mat: torch.Tensor) -> torch.Tensor:
    out = 0.5 * (mat + mat.t())
    out = out.clone()
    out.fill_diagonal_(0.0)
    return out


def positive_part(mat: torch.Tensor) -> torch.Tensor:
    return symmetrize_matrix(mat).clamp_min(0.0)


def raw_density(adj: torch.Tensor) -> float:
    num_nodes = adj.size(0)
    if num_nodes <= 1:
        return 0.0
    return float(adj.sum().item() / float(num_nodes * (num_nodes - 1)))


def effective_density(weighted_adj: torch.Tensor) -> float:
    num_nodes = weighted_adj.size(0)
    if num_nodes <= 1:
        return 0.0
    return float(weighted_adj.sum().item() / float(num_nodes * (num_nodes - 1)))


def support_adjacency(weighted_adj: torch.Tensor, support_ratio: float) -> torch.Tensor:
    weights = positive_part(weighted_adj)
    max_weight = float(weights.max().item()) if weights.numel() else 0.0
    if max_weight <= 0.0:
        return torch.zeros_like(weights)
    support = (weights >= max_weight * float(support_ratio)).float()
    support.fill_diagonal_(0.0)
    return support


def within_class_mass(mat: torch.Tensor, y: torch.Tensor) -> float:
    weights = mat.abs().clone()
    weights.fill_diagonal_(0.0)
    same = (y[:, None] == y[None, :]).float()
    return float(((weights * same).sum() / (weights.sum() + 1e-9)).item())


def cross_class_mass(mat: torch.Tensor, y: torch.Tensor) -> float:
    weights = mat.abs().clone()
    weights.fill_diagonal_(0.0)
    diff = (y[:, None] != y[None, :]).float()
    return float(((weights * diff).sum() / (weights.sum() + 1e-9)).item())


def triangle_count_binary(adj: torch.Tensor) -> int:
    tri = torch.trace(adj @ adj @ adj) / 6.0
    return int(round(float(tri.item())))


def wedge_count_binary(adj: torch.Tensor) -> int:
    deg = adj.sum(dim=1)
    triplets = float((deg * (deg - 1.0) * 0.5).sum().item())
    triangles = float(triangle_count_binary(adj))
    wedges = max(triplets - 3.0 * triangles, 0.0)
    return int(round(wedges))


def transitivity_binary(adj: torch.Tensor) -> float:
    deg = adj.sum(dim=1)
    triplets = float((deg * (deg - 1.0) * 0.5).sum().item())
    if triplets <= 0.0:
        return 0.0
    triangles = float(triangle_count_binary(adj))
    return float((3.0 * triangles) / triplets)


def dirichlet_energy(weighted_adj: torch.Tensor, features: torch.Tensor) -> float:
    weights = positive_part(weighted_adj)
    degree = torch.diag(weights.sum(dim=1))
    lap = degree - weights
    energy = torch.trace(features.t() @ lap @ features) / max(1, features.size(0))
    return float(energy.item())


def node_degrees(adj: torch.Tensor) -> torch.Tensor:
    return adj.detach().float().sum(dim=1)


def sorted_degree_gap(source_deg: torch.Tensor, target_deg: torch.Tensor) -> float:
    src = torch.sort(source_deg.detach().float())[0]
    tgt = torch.sort(target_deg.detach().float())[0]
    size = min(src.numel(), tgt.numel())
    if size == 0:
        return 0.0
    return float((src[:size] - tgt[:size]).abs().mean().item())


def safe_degree_ratio(source_deg: torch.Tensor, target_deg: torch.Tensor) -> float:
    denom = float(target_deg.detach().float().mean().item())
    if abs(denom) <= 1e-9:
        return float("inf")
    return float(source_deg.detach().float().mean().item() / denom)


def community_layout(labels: torch.Tensor) -> dict[int, tuple[float, float]]:
    labels = labels.detach().cpu().long()
    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        groups.setdefault(int(label), []).append(idx)

    centers = {
        0: (-1.25, 0.0),
        1: (1.25, 0.0),
        2: (0.0, 1.25),
        3: (0.0, -1.25),
    }
    pos = {}
    for label, nodes in groups.items():
        center_x, center_y = centers.get(label, (float(label), 0.0))
        radius = 0.55
        for i, node in enumerate(nodes):
            angle = (2.0 * math.pi * i) / max(1, len(nodes))
            pos[node] = (
                center_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle),
            )
    return pos


def _cycle_edges(offset: int, size: int) -> list[tuple[int, int]]:
    return [(offset + i, offset + ((i + 1) % size)) for i in range(size)]


def _clique_edges(offset: int, size: int) -> list[tuple[int, int]]:
    return [(offset + i, offset + j) for i in range(size) for j in range(i + 1, size)]


def _rotated_features(labels: torch.Tensor, d_in: int, shift_noise: float) -> tuple[torch.Tensor, torch.Tensor]:
    num_per_class = int((labels == labels[0]).sum().item())
    mu = torch.zeros(d_in)
    mu[0] = 2.0
    if d_in > 1:
        mu[1] = 0.75

    source_x = torch.cat(
        [
            torch.randn(num_per_class, d_in) * 0.65 - mu,
            torch.randn(num_per_class, d_in) * 0.65 + mu,
        ],
        dim=0,
    )
    q, _ = torch.linalg.qr(torch.randn(d_in, d_in))
    target_base = torch.cat(
        [
            torch.randn(num_per_class, d_in) * 0.90 - mu,
            torch.randn(num_per_class, d_in) * 0.90 + mu,
        ],
        dim=0,
    ) @ q
    target_x = target_base + shift_noise * torch.randn_like(target_base) * 0.35
    return source_x, target_x


def make_large_synthetic_data(
    n_per_class: int = 60,
    d_in: int = 8,
    feat_shift: float = 1.0,
    source_h: tuple[float, float] = (0.14, 0.02),
    target_h: tuple[float, float] = (0.24, 0.08),
) -> dict[str, torch.Tensor]:
    labels = torch.cat(
        [
            torch.zeros(n_per_class, dtype=torch.long),
            torch.ones(n_per_class, dtype=torch.long),
        ]
    )
    source_x, target_x = _rotated_features(labels, d_in=d_in, shift_noise=feat_shift)
    source_adj = make_sbm(labels.tolist(), *source_h)
    target_adj = make_sbm(labels.tolist(), *target_h)
    return {
        "name": "large_synthetic",
        "Xs": source_x,
        "Xt": target_x,
        "ys": labels,
        "yt": labels.clone(),
        "A_s": source_adj,
        "A_t": target_adj,
        "Ahat_s": normalize_adj(source_adj),
        "Ahat_t": normalize_adj(target_adj),
    }


def make_small_motif_data(
    n_per_class: int = 6,
    d_in: int = 8,
    feat_shift: float = 0.7,
) -> dict[str, torch.Tensor]:
    labels = torch.cat(
        [
            torch.zeros(n_per_class, dtype=torch.long),
            torch.ones(n_per_class, dtype=torch.long),
        ]
    )
    source_x, target_x = _rotated_features(labels, d_in=d_in, shift_noise=feat_shift)
    num_nodes = int(labels.numel())

    source_edges = _cycle_edges(0, n_per_class) + _cycle_edges(n_per_class, n_per_class)
    source_edges += [(1, n_per_class + 1), (4, n_per_class + 4)]

    target_edges = _clique_edges(0, n_per_class) + _clique_edges(n_per_class, n_per_class)
    target_edges += [(2, n_per_class + 2)]

    source_adj = make_binary_adj(num_nodes, source_edges)
    target_adj = make_binary_adj(num_nodes, target_edges)
    return {
        "name": "small_motif",
        "Xs": source_x,
        "Xt": target_x,
        "ys": labels,
        "yt": labels.clone(),
        "A_s": source_adj,
        "A_t": target_adj,
        "Ahat_s": normalize_adj(source_adj),
        "Ahat_t": normalize_adj(target_adj),
    }


def make_degree_shift_data(
    n_per_class: int = 24,
    d_in: int = 8,
    feat_shift: float = 0.35,
    target_h: tuple[float, float] = (0.10, 0.03),
    source_multiplier: float = 3.0,
) -> dict[str, torch.Tensor]:
    source_h = tuple(min(0.95, float(val) * float(source_multiplier)) for val in target_h)
    data = make_large_synthetic_data(
        n_per_class=n_per_class,
        d_in=d_in,
        feat_shift=feat_shift,
        source_h=source_h,
        target_h=target_h,
    )
    data["name"] = "degree_shift_fixed_source"
    data["source_multiplier"] = float(source_multiplier)
    return data


class MMDLoss(nn.Module):
    def __init__(self, sigmas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)):
        super().__init__()
        self.sigmas = sigmas

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist2 = torch.cdist(x, y, p=2).pow(2)
        out = 0.0
        for sigma in self.sigmas:
            out = out + torch.exp(-dist2 / (2.0 * sigma * sigma))
        return out / float(len(self.sigmas))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, x).mean() + self.kernel(y, y).mean() - 2.0 * self.kernel(x, y).mean()


class DenseChebFilter(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.K = int(k)
        self.coef = nn.Parameter(torch.zeros(self.K))
        with torch.no_grad():
            self.coef[0] = 1.0

    def forward(self, ahat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        coeff = self.coef
        t0 = x
        out = coeff[0] * t0
        if self.K > 1:
            t1 = ahat @ x
            out = out + coeff[1] * t1
            for k in range(2, self.K):
                t2 = 2.0 * (ahat @ t1) - t0
                out = out + coeff[k] * t2
                t0, t1 = t1, t2
        return out

    def realized_matrix(self, ahat: torch.Tensor) -> torch.Tensor:
        num_nodes = ahat.size(0)
        eye = torch.eye(num_nodes, device=ahat.device, dtype=ahat.dtype)
        coeff = self.coef
        t0 = eye
        out = coeff[0] * t0
        if self.K > 1:
            t1 = ahat
            out = out + coeff[1] * t1
            for k in range(2, self.K):
                t2 = 2.0 * (ahat @ t1) - t0
                out = out + coeff[k] * t2
                t0, t1 = t1, t2
        return out


class OPALToy(nn.Module):
    def __init__(self, d_in: int, d_h: int = 16, num_prop_layers: int = 2, k: int = 5):
        super().__init__()
        self.enc = nn.Linear(d_in, d_h)
        self.prop_weights = nn.ParameterList(
            [nn.Parameter(torch.randn(d_h, d_h) / math.sqrt(d_h)) for _ in range(num_prop_layers)]
        )
        self.src_filter = DenseChebFilter(k)
        self.tgt_filter = DenseChebFilter(k)
        self.cls = nn.Linear(d_h, 2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.enc(x))

    def propagate_hidden(self, h0: torch.Tensor, ahat: torch.Tensor, source: bool = True) -> torch.Tensor:
        filt = self.src_filter if source else self.tgt_filter
        h = h0
        for weight in self.prop_weights:
            h = F.relu(filt(ahat, h) @ weight)
        return h

    def forward(self, x: torch.Tensor, ahat: torch.Tensor, source: bool = True):
        h0 = self.encode(x)
        h = self.propagate_hidden(h0, ahat, source)
        return h0, h, self.cls(h)

    def realized(self, ahat_s: torch.Tensor, ahat_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.src_filter.realized_matrix(ahat_s), self.tgt_filter.realized_matrix(ahat_t)


def cheb_weighted_ilip(coeff: torch.Tensor) -> torch.Tensor:
    if coeff.numel() <= 1:
        return coeff.new_tensor(0.0)
    orders = torch.arange(1, coeff.numel(), device=coeff.device, dtype=coeff.dtype)
    return torch.sum((orders**2) * coeff[1:].abs())


def deterministic_reference_bank(
    hs0: torch.Tensor,
    ht0: torch.Tensor,
    mode: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = str(mode).lower()
    if mode == "pooled":
        bank = torch.cat([hs0.detach(), ht0.detach()], dim=0)
        idx_s = torch.arange(hs0.size(0), device=bank.device) % bank.size(0)
        idx_t = (torch.arange(ht0.size(0), device=bank.device) + hs0.size(0)) % bank.size(0)
        return bank[idx_s], bank[idx_t]
    if mode == "source":
        bank = hs0.detach()
        idx_s = torch.arange(hs0.size(0), device=bank.device) % bank.size(0)
        idx_t = torch.arange(ht0.size(0), device=bank.device) % bank.size(0)
        return bank[idx_s], bank[idx_t]
    if mode == "target":
        bank = ht0.detach()
        idx_s = torch.arange(hs0.size(0), device=bank.device) % bank.size(0)
        idx_t = torch.arange(ht0.size(0), device=bank.device) % bank.size(0)
        return bank[idx_s], bank[idx_t]
    if mode == "gaussian":
        bank = torch.cat([hs0.detach(), ht0.detach()], dim=0)
        mu = bank.mean(dim=0, keepdim=True)
        std = bank.std(dim=0, keepdim=True).clamp_min(1e-4)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        src_noise = torch.randn(hs0.shape, generator=gen, dtype=hs0.dtype)
        tgt_noise = torch.randn(ht0.shape, generator=gen, dtype=ht0.dtype)
        src_noise = src_noise.to(device=hs0.device)
        tgt_noise = tgt_noise.to(device=ht0.device)
        return mu + std * src_noise, mu + std * tgt_noise
    raise ValueError(f"Unsupported probe mode: {mode}")


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().item())


@dataclass
class ToyTrainConfig:
    beta: float = 0.5
    delta: float = 0.5
    alpha: float = 1e-4
    gamma: float = 0.0
    lr: float = 5e-3
    wd: float = 5e-4
    epochs: int = 120
    probe_mode: str = "pooled"
    hidden_dim: int = 16
    num_prop_layers: int = 2
    K: int = 5
    track_every: int = 5
    support_ratio: float = 0.25


@dataclass
class FixedSourceDegreeShiftConfig:
    epochs: int = 120
    lr: float = 1e-2
    wd: float = 0.0
    alpha: float = 1e-4
    mmd_weight: float = 1.0
    mse_weight: float = 0.25
    K: int = 5
    source_coef: tuple[float, ...] = (0.5, 0.5, 0.0, 0.0, 0.0)


def summarize_single_structure(
    raw_adj: torch.Tensor,
    induced_adj: torch.Tensor,
    labels: torch.Tensor,
    reference_bank: torch.Tensor,
    prefix: str,
    support_ratio: float,
) -> dict[str, float | int]:
    induced_pos = positive_part(induced_adj)
    support = support_adjacency(induced_pos, support_ratio=support_ratio)
    raw_triangles = triangle_count_binary(raw_adj)
    induced_triangles = triangle_count_binary(support)
    raw_wedges = wedge_count_binary(raw_adj)
    induced_wedges = wedge_count_binary(support)

    return {
        f"{prefix}_raw_density": raw_density(raw_adj),
        f"{prefix}_raw_triangles": raw_triangles,
        f"{prefix}_raw_wedges": raw_wedges,
        f"{prefix}_raw_transitivity": transitivity_binary(raw_adj),
        f"{prefix}_raw_within_mass": within_class_mass(raw_adj, labels),
        f"{prefix}_raw_cross_mass": cross_class_mass(raw_adj, labels),
        f"{prefix}_induced_effective_density": effective_density(induced_pos),
        f"{prefix}_induced_support_density": raw_density(support),
        f"{prefix}_induced_triangles": induced_triangles,
        f"{prefix}_induced_wedges": induced_wedges,
        f"{prefix}_induced_transitivity": transitivity_binary(support),
        f"{prefix}_induced_within_mass": within_class_mass(induced_pos, labels),
        f"{prefix}_induced_cross_mass": cross_class_mass(induced_pos, labels),
        f"{prefix}_probe_dirichlet_energy": dirichlet_energy(induced_pos, reference_bank),
        f"{prefix}_induced_weight_mean": float(induced_pos.mean().item()),
        f"{prefix}_induced_weight_max": float(induced_pos.max().item()),
    }


def collect_toy_state(
    model: OPALToy,
    data: dict[str, torch.Tensor],
    cfg: ToyTrainConfig,
    seed: int,
    include_tensors: bool = False,
) -> tuple[dict[str, float | int | list[float]], dict[str, torch.Tensor] | None]:
    xs = data["Xs"]
    xt = data["Xt"]
    ys = data["ys"]
    yt = data["yt"]
    ahat_s = data["Ahat_s"]
    ahat_t = data["Ahat_t"]
    raw_s = data["A_s"]
    raw_t = data["A_t"]

    div = MMDLoss()
    with torch.no_grad():
        hs0, hs, logits_s = model(xs, ahat_s, True)
        ht0, ht, logits_t = model(xt, ahat_t, False)
        bs, bt = deterministic_reference_bank(hs0, ht0, mode=cfg.probe_mode, seed=seed + 17)
        qs = model.src_filter(ahat_s, bs)
        qt = model.tgt_filter(ahat_t, bt)
        rs, rt = model.realized(ahat_s, ahat_t)

        ref_gen = torch.Generator(device="cpu")
        ref_gen.manual_seed(int(seed + 101))
        ref_bank = torch.randn((xs.size(0), hs0.size(1)), generator=ref_gen, dtype=hs0.dtype).to(xs.device)

        source_struct = summarize_single_structure(
            raw_adj=raw_s,
            induced_adj=rs,
            labels=ys,
            reference_bank=ref_bank,
            prefix="source",
            support_ratio=cfg.support_ratio,
        )
        target_struct = summarize_single_structure(
            raw_adj=raw_t,
            induced_adj=rt,
            labels=yt,
            reference_bank=ref_bank,
            prefix="target",
            support_ratio=cfg.support_ratio,
        )

        metrics = {
            "target_acc": accuracy(logits_t, yt),
            "source_acc_lab": accuracy(logits_s, ys),
            "feature_div": float(div(hs0, ht0).item()),
            "response_div": float(div(qs, qt).item()),
            "filter_gap_fro": float(torch.linalg.norm(rs - rt).item()),
            "src_coef_l2": float(torch.linalg.norm(model.src_filter.coef).item()),
            "tgt_coef_l2": float(torch.linalg.norm(model.tgt_filter.coef).item()),
            "src_tgt_coef_gap_l2": float(torch.linalg.norm(model.src_filter.coef - model.tgt_filter.coef).item()),
            "src_coef": model.src_filter.coef.detach().cpu().tolist(),
            "tgt_coef": model.tgt_filter.coef.detach().cpu().tolist(),
            **source_struct,
            **target_struct,
        }
        metrics["density_gap_raw"] = abs(metrics["source_raw_density"] - metrics["target_raw_density"])
        metrics["density_gap_induced_effective"] = abs(
            metrics["source_induced_effective_density"] - metrics["target_induced_effective_density"]
        )
        metrics["density_gap_induced_support"] = abs(
            metrics["source_induced_support_density"] - metrics["target_induced_support_density"]
        )
        metrics["triangle_gap_raw"] = abs(metrics["source_raw_triangles"] - metrics["target_raw_triangles"])
        metrics["triangle_gap_induced"] = abs(metrics["source_induced_triangles"] - metrics["target_induced_triangles"])
        metrics["transitivity_gap_induced"] = abs(
            metrics["source_induced_transitivity"] - metrics["target_induced_transitivity"]
        )
        metrics["probe_energy_gap"] = abs(
            metrics["source_probe_dirichlet_energy"] - metrics["target_probe_dirichlet_energy"]
        )

    snapshot = None
    if include_tensors:
        snapshot = {
            "A_s": raw_s.detach().cpu(),
            "A_t": raw_t.detach().cpu(),
            "R_s": rs.detach().cpu(),
            "R_t": rt.detach().cpu(),
            "ys": ys.detach().cpu(),
            "yt": yt.detach().cpu(),
            "Xs": xs.detach().cpu(),
            "Xt": xt.detach().cpu(),
        }
    return metrics, snapshot


def _label_budget(y: torch.Tensor) -> int:
    per_class = min(int((y == cls).sum().item()) for cls in y.unique(sorted=True))
    return min(15, max(2, per_class // 2))


def train_toy_once(
    seed: int,
    cfg: ToyTrainConfig,
    data_builder,
    return_traj: bool = False,
    include_snapshot: bool = False,
) -> tuple[dict, list[dict], dict | None]:
    set_local_seed(seed)
    data = data_builder()

    xs, xt = data["Xs"], data["Xt"]
    ys, yt = data["ys"], data["yt"]
    ahat_s, ahat_t = data["Ahat_s"], data["Ahat_t"]
    num_labels = _label_budget(ys)
    idx0 = torch.where(ys == 0)[0]
    idx1 = torch.where(ys == 1)[0]
    src_lab = torch.cat([idx0[torch.randperm(len(idx0))[:num_labels]], idx1[torch.randperm(len(idx1))[:num_labels]]])

    model = OPALToy(
        d_in=xs.size(1),
        d_h=cfg.hidden_dim,
        num_prop_layers=cfg.num_prop_layers,
        k=cfg.K,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    div = MMDLoss()

    traj_rows = []
    if return_traj:
        pre_metrics, _ = collect_toy_state(model, data, cfg, seed=seed)
        traj_rows.append({"epoch": 0, **pre_metrics})

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        hs0, hs, logits_s = model(xs, ahat_s, True)
        ht0, ht, logits_t = model(xt, ahat_t, False)

        cls_loss = F.cross_entropy(logits_s[src_lab], ys[src_lab])
        feat_loss = div(hs0, ht0)
        bs, bt = deterministic_reference_bank(hs0, ht0, mode=cfg.probe_mode, seed=seed + epoch)
        qs = model.src_filter(ahat_s, bs)
        qt = model.tgt_filter(ahat_t, bt)
        resp_loss = div(qs, qt)
        lip = cheb_weighted_ilip(model.src_filter.coef).pow(2) + cheb_weighted_ilip(model.tgt_filter.coef).pow(2)

        probs_t = F.softmax(logits_t, dim=1)
        entropy = -(probs_t * probs_t.clamp_min(1e-9).log()).sum(dim=1).mean()
        loss = cls_loss + cfg.beta * feat_loss + cfg.delta * resp_loss + cfg.alpha * lip + cfg.gamma * entropy

        opt.zero_grad()
        loss.backward()
        opt.step()

        if return_traj and (epoch % cfg.track_every == 0 or epoch == cfg.epochs):
            metrics, _ = collect_toy_state(model, data, cfg, seed=seed + epoch)
            traj_rows.append({"epoch": epoch, **metrics})

    final_metrics, snapshot = collect_toy_state(
        model,
        data,
        cfg,
        seed=seed + cfg.epochs,
        include_tensors=include_snapshot,
    )
    final_metrics["seed"] = int(seed)
    final_metrics["variant_cfg"] = asdict(cfg)
    final_metrics["data_name"] = data["name"]
    return final_metrics, traj_rows, snapshot


class FixedSourceDegreeShiftAligner(nn.Module):
    def __init__(self, k: int, source_coef: tuple[float, ...]):
        super().__init__()
        self.K = int(k)
        self.src_filter = DenseChebFilter(self.K)
        self.tgt_filter = DenseChebFilter(self.K)

        source_tensor = torch.zeros(self.K, dtype=torch.float32)
        src_vals = torch.tensor(list(source_coef), dtype=torch.float32)
        source_tensor[: min(self.K, src_vals.numel())] = src_vals[: min(self.K, src_vals.numel())]
        with torch.no_grad():
            self.src_filter.coef.copy_(source_tensor)
            self.tgt_filter.coef.zero_()
            self.tgt_filter.coef[0] = 1.0
        self.src_filter.coef.requires_grad_(False)

    def realized(self, ahat_s: torch.Tensor, ahat_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.src_filter.realized_matrix(ahat_s), self.tgt_filter.realized_matrix(ahat_t)


def collect_degree_shift_state(
    model: FixedSourceDegreeShiftAligner,
    data: dict[str, torch.Tensor],
    seed: int,
    include_snapshot: bool = False,
) -> tuple[dict[str, float | list[float]], dict | None]:
    ahat_s = data["Ahat_s"]
    ahat_t = data["Ahat_t"]
    raw_s = data["A_s"]
    raw_t = data["A_t"]
    d_in = int(data["Xs"].size(1))
    num_nodes = int(raw_s.size(0))

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed + 404))
    probes = torch.randn((num_nodes, d_in), generator=gen, dtype=raw_s.dtype)

    div = MMDLoss()
    with torch.no_grad():
        source_resp = model.src_filter(ahat_s, probes)
        target_resp = model.tgt_filter(ahat_t, probes)
        rs, rt = model.realized(ahat_s, ahat_t)

        raw_source_deg = node_degrees(raw_s)
        raw_target_deg = node_degrees(raw_t)
        induced_source_deg = node_degrees(positive_part(rs))
        induced_target_deg = node_degrees(positive_part(rt))

        metrics = {
            "response_div_mmd": float(div(source_resp, target_resp).item()),
            "response_div_mse": float(F.mse_loss(target_resp, source_resp).item()),
            "raw_source_mean_degree": float(raw_source_deg.mean().item()),
            "raw_target_mean_degree": float(raw_target_deg.mean().item()),
            "raw_degree_ratio": safe_degree_ratio(raw_source_deg, raw_target_deg),
            "raw_degree_gap_l1": sorted_degree_gap(raw_source_deg, raw_target_deg),
            "induced_source_mean_degree": float(induced_source_deg.mean().item()),
            "induced_target_mean_degree": float(induced_target_deg.mean().item()),
            "induced_degree_ratio": safe_degree_ratio(induced_source_deg, induced_target_deg),
            "induced_degree_gap_l1": sorted_degree_gap(induced_source_deg, induced_target_deg),
            "source_coef": model.src_filter.coef.detach().cpu().tolist(),
            "target_coef": model.tgt_filter.coef.detach().cpu().tolist(),
        }

    snapshot = None
    if include_snapshot:
        snapshot = {
            "raw_source_deg": raw_source_deg.detach().cpu(),
            "raw_target_deg": raw_target_deg.detach().cpu(),
            "induced_source_deg": induced_source_deg.detach().cpu(),
            "induced_target_deg": induced_target_deg.detach().cpu(),
            "R_s": rs.detach().cpu(),
            "R_t": rt.detach().cpu(),
            "A_s": raw_s.detach().cpu(),
            "A_t": raw_t.detach().cpu(),
        }
    return metrics, snapshot


def train_fixed_source_degree_shift(
    seed: int,
    cfg: FixedSourceDegreeShiftConfig,
    data_builder,
    return_traj: bool = False,
    include_snapshot: bool = False,
) -> tuple[dict, list[dict], dict | None]:
    set_local_seed(seed)
    data = data_builder()
    model = FixedSourceDegreeShiftAligner(k=cfg.K, source_coef=cfg.source_coef)
    optimizer = torch.optim.Adam([model.tgt_filter.coef], lr=cfg.lr, weight_decay=cfg.wd)
    div = MMDLoss()

    num_nodes = int(data["A_s"].size(0))
    d_in = int(data["Xs"].size(1))
    probe_gen = torch.Generator(device="cpu")
    probe_gen.manual_seed(int(seed + 777))
    probe_bank = torch.randn((num_nodes, d_in), generator=probe_gen, dtype=data["A_s"].dtype)

    traj_rows = []
    eval_seed = int(seed)
    before_metrics, _ = collect_degree_shift_state(model, data, seed=eval_seed, include_snapshot=False)
    if return_traj:
        traj_rows.append({"epoch": 0, **before_metrics})

    for epoch in range(1, cfg.epochs + 1):
        source_resp = model.src_filter(data["Ahat_s"], probe_bank)
        target_resp = model.tgt_filter(data["Ahat_t"], probe_bank)
        loss_mmd = div(source_resp, target_resp)
        loss_mse = F.mse_loss(target_resp, source_resp.detach())
        lip = cheb_weighted_ilip(model.tgt_filter.coef).pow(2)
        loss = cfg.mmd_weight * loss_mmd + cfg.mse_weight * loss_mse + cfg.alpha * lip

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if return_traj and (epoch % 5 == 0 or epoch == cfg.epochs):
            cur_metrics, _ = collect_degree_shift_state(model, data, seed=eval_seed, include_snapshot=False)
            traj_rows.append({"epoch": epoch, **cur_metrics})

    after_metrics, snapshot = collect_degree_shift_state(
        model,
        data,
        seed=eval_seed,
        include_snapshot=include_snapshot,
    )
    result = {
        "seed": int(seed),
        "data_name": str(data["name"]),
        "source_multiplier": float(data.get("source_multiplier", 3.0)),
        "response_div_mmd_before": float(before_metrics["response_div_mmd"]),
        "response_div_mmd_after": float(after_metrics["response_div_mmd"]),
        "response_div_mmd_reduction": float(before_metrics["response_div_mmd"] - after_metrics["response_div_mmd"]),
        "response_div_mse_before": float(before_metrics["response_div_mse"]),
        "response_div_mse_after": float(after_metrics["response_div_mse"]),
        "response_div_mse_reduction": float(before_metrics["response_div_mse"] - after_metrics["response_div_mse"]),
        "raw_degree_ratio": float(before_metrics["raw_degree_ratio"]),
        "induced_degree_ratio": float(after_metrics["induced_degree_ratio"]),
        "raw_degree_gap_l1": float(before_metrics["raw_degree_gap_l1"]),
        "induced_degree_gap_l1": float(after_metrics["induced_degree_gap_l1"]),
        "degree_gap_l1_reduction": float(before_metrics["raw_degree_gap_l1"] - after_metrics["induced_degree_gap_l1"]),
        "degree_ratio_improvement_to_one": float(abs(before_metrics["raw_degree_ratio"] - 1.0) - abs(after_metrics["induced_degree_ratio"] - 1.0)),
        "source_coef": after_metrics["source_coef"],
        "target_coef": after_metrics["target_coef"],
        "train_cfg": asdict(cfg),
    }
    return result, traj_rows, snapshot


def _select_plot_edges(
    matrix: torch.Tensor,
    support_ratio: float,
    max_edges: int,
    weighted: bool,
) -> list[tuple[int, int, float]]:
    mat = symmetrize_matrix(matrix.detach().cpu().float())
    if not weighted:
        edges = []
        for i in range(mat.size(0)):
            for j in range(i + 1, mat.size(1)):
                if mat[i, j] > 0:
                    edges.append((i, j, 1.0))
        return edges

    vals = []
    max_abs = float(mat.abs().max().item()) if mat.numel() else 0.0
    if max_abs <= 0.0:
        return vals

    threshold = max_abs * float(support_ratio)
    for i in range(mat.size(0)):
        for j in range(i + 1, mat.size(1)):
            weight = float(mat[i, j].item())
            if abs(weight) >= threshold:
                vals.append((i, j, weight))
    vals.sort(key=lambda item: abs(item[2]), reverse=True)
    return vals[: int(max_edges)]


def draw_graph(
    ax,
    matrix: torch.Tensor,
    labels: torch.Tensor,
    positions: dict[int, tuple[float, float]],
    title: str,
    support_ratio: float,
    max_edges: int,
    weighted: bool = False,
) -> None:
    edges = _select_plot_edges(matrix, support_ratio=support_ratio, max_edges=max_edges, weighted=weighted)
    max_abs = max((abs(w) for _, _, w in edges), default=1.0)

    for i, j, weight in edges:
        xi, yi = positions[i]
        xj, yj = positions[j]
        width = 0.8 if not weighted else 0.7 + 4.5 * abs(weight) / max_abs
        color = "#274c77" if weight >= 0 else "#c8553d"
        alpha = 0.55 if not weighted else 0.25 + 0.65 * abs(weight) / max_abs
        ax.plot([xi, xj], [yi, yj], color=color, linewidth=width, alpha=alpha, solid_capstyle="round")

    labels_np = labels.detach().cpu().numpy()
    palette = np.array(["#3a86ff", "#ff006e", "#8338ec", "#fb5607"])
    node_colors = [palette[int(v) % len(palette)] for v in labels_np]
    xs = [positions[i][0] for i in range(len(labels_np))]
    ys = [positions[i][1] for i in range(len(labels_np))]
    ax.scatter(xs, ys, c=node_colors, s=120, edgecolors="black", linewidths=0.8, zorder=3)

    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x, y, str(i), ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=4)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.set_frame_on(False)


def plot_small_graph_comparison(
    path: Path,
    labels: torch.Tensor,
    raw_source: torch.Tensor,
    raw_target: torch.Tensor,
    feat_snapshot: dict,
    full_snapshot: dict,
    support_ratio: float,
    max_edges: int,
) -> None:
    positions = community_layout(labels)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    draw_graph(axes[0, 0], raw_source, labels, positions, "Feature Align: Raw Source", support_ratio, max_edges, weighted=False)
    draw_graph(axes[0, 1], raw_target, labels, positions, "Feature Align: Raw Target", support_ratio, max_edges, weighted=False)
    draw_graph(
        axes[0, 2],
        feat_snapshot["R_s"],
        labels,
        positions,
        "Feature Align: Induced Source",
        support_ratio,
        max_edges,
        weighted=True,
    )
    draw_graph(
        axes[0, 3],
        feat_snapshot["R_t"],
        labels,
        positions,
        "Feature Align: Induced Target",
        support_ratio,
        max_edges,
        weighted=True,
    )

    draw_graph(axes[1, 0], raw_source, labels, positions, "Feature + Filter: Raw Source", support_ratio, max_edges, weighted=False)
    draw_graph(axes[1, 1], raw_target, labels, positions, "Feature + Filter: Raw Target", support_ratio, max_edges, weighted=False)
    draw_graph(
        axes[1, 2],
        full_snapshot["R_s"],
        labels,
        positions,
        "Feature + Filter: Induced Source",
        support_ratio,
        max_edges,
        weighted=True,
    )
    draw_graph(
        axes[1, 3],
        full_snapshot["R_t"],
        labels,
        positions,
        "Feature + Filter: Induced Target",
        support_ratio,
        max_edges,
        weighted=True,
    )

    fig.suptitle("Small synthetic motif graphs. Weighted induced edges use line thickness = |edge weight|.")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_toy_trajectories(path: Path, traj_by_variant: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.reshape(-1)
    metric_specs = [
        ("target_acc", "Target Accuracy"),
        ("feature_div", "Feature Divergence"),
        ("response_div", "Filter Response Divergence"),
        ("filter_gap_fro", "Operator Gap (Fro)"),
        ("probe_energy_gap", "Probe Energy Gap"),
        ("triangle_gap_induced", "Induced Triangle Gap"),
    ]
    colors = {
        "feat_only": "#ff7f0e",
        "feat_plus_filter": "#2a9d8f",
        "source_only": "#6c757d",
        "filter_only": "#8d5fd3",
    }

    for ax, (metric, title) in zip(axes, metric_specs):
        for variant, rows in traj_by_variant.items():
            xs = [int(row["epoch"]) for row in rows]
            ys = [float(row[metric]) for row in rows]
            ax.plot(xs, ys, linewidth=2.0, marker="o", markersize=3.5, label=VARIANT_LABELS.get(variant, variant), color=colors.get(variant))
        ax.set_title(title)
        ax.grid(alpha=0.3)

    for ax in axes[-3:]:
        ax.set_xlabel("Epoch")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Small synthetic trajectories")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_summary_bars(
    path: Path,
    summary_rows: list[dict],
    metric_specs: list[tuple[str, str]],
    title: str,
) -> None:
    if not summary_rows:
        return

    variants = [row["variant"] for row in summary_rows]
    labels = [VARIANT_LABELS.get(v, v) for v in variants]
    x = np.arange(len(variants))
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(5.2 * len(metric_specs), 4.5))
    if len(metric_specs) == 1:
        axes = [axes]

    colors = ["#6c757d", "#ff7f0e", "#8d5fd3", "#2a9d8f"]
    for ax, (metric, ylabel) in zip(axes, metric_specs):
        means = [float(row.get(f"{metric}_mean", float("nan"))) for row in summary_rows]
        stds = [float(row.get(f"{metric}_std", 0.0)) for row in summary_rows]
        ax.bar(x, means, yerr=stds, capsize=4, color=colors[: len(x)], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_gain_bars(path: Path, gain_rows: list[dict], title: str) -> None:
    if not gain_rows:
        return

    labels = [row["label"] for row in gain_rows]
    values = [float(row["value"]) for row in gain_rows]
    x = np.arange(len(labels))
    colors = ["#2a9d8f" if v >= 0 else "#d62828" for v in values]

    fig, ax = plt.subplots(1, 1, figsize=(max(7.0, 1.8 * len(labels)), 4.4))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.bar(x, values, color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Feature + Filter minus Feature Align")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_degree_shift_histograms(path: Path, snapshot: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].hist(snapshot["raw_source_deg"].numpy(), bins=12, alpha=0.65, label="source raw", color="#2a9d8f")
    axes[0, 0].hist(snapshot["raw_target_deg"].numpy(), bins=12, alpha=0.65, label="target raw", color="#e76f51")
    axes[0, 0].set_title("Raw degree distribution")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].hist(snapshot["induced_source_deg"].numpy(), bins=12, alpha=0.65, label="source induced", color="#264653")
    axes[0, 1].hist(snapshot["induced_target_deg"].numpy(), bins=12, alpha=0.65, label="target induced", color="#f4a261")
    axes[0, 1].set_title("Induced weighted degree distribution")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    raw_source_sorted = np.sort(snapshot["raw_source_deg"].numpy())
    raw_target_sorted = np.sort(snapshot["raw_target_deg"].numpy())
    induced_source_sorted = np.sort(snapshot["induced_source_deg"].numpy())
    induced_target_sorted = np.sort(snapshot["induced_target_deg"].numpy())
    x = np.arange(len(raw_source_sorted))

    axes[1, 0].plot(x, raw_source_sorted, linewidth=2.0, label="source raw", color="#2a9d8f")
    axes[1, 0].plot(x, raw_target_sorted, linewidth=2.0, label="target raw", color="#e76f51")
    axes[1, 0].set_title("Sorted raw degrees")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].plot(x, induced_source_sorted, linewidth=2.0, label="source induced", color="#264653")
    axes[1, 1].plot(x, induced_target_sorted, linewidth=2.0, label="target induced", color="#f4a261")
    axes[1, 1].set_title("Sorted induced degrees")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()

    fig.suptitle("Fixed-source degree shift: raw vs induced degree profiles")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_degree_shift_training(path: Path, traj_rows: list[dict]) -> None:
    if not traj_rows:
        return

    epochs = [int(row["epoch"]) for row in traj_rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(epochs, [float(row["response_div_mmd"]) for row in traj_rows], marker="o", linewidth=2.0, color="#264653")
    axes[0].set_title("Response MMD")
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, [float(row["response_div_mse"]) for row in traj_rows], marker="o", linewidth=2.0, color="#2a9d8f")
    axes[1].set_title("Response MSE")
    axes[1].grid(alpha=0.25)

    axes[2].plot(epochs, [float(row["induced_degree_gap_l1"]) for row in traj_rows], marker="o", linewidth=2.0, color="#e76f51", label="induced degree gap")
    axes[2].axhline(float(traj_rows[0]["raw_degree_gap_l1"]), color="#6c757d", linestyle="--", linewidth=1.8, label="raw degree gap")
    axes[2].set_title("Degree gap over training")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("Epoch")

    fig.suptitle("Fixed-source degree shift training")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_degree_shift_coefficients(path: Path, result: dict) -> None:
    source_coef = np.asarray(result["source_coef"], dtype=float)
    target_coef = np.asarray(result["target_coef"], dtype=float)
    x = np.arange(len(source_coef))
    width = 0.38

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.bar(x - width / 2.0, source_coef, width=width, label="fixed source", color="#264653")
    ax.bar(x + width / 2.0, target_coef, width=width, label="learned target", color="#f4a261")
    ax.set_xticks(x)
    ax.set_xlabel("Chebyshev order")
    ax.set_ylabel("Coefficient")
    ax.set_title("Fixed source filter vs learned target filter")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def summarize_single_payload(rows: list[dict], metrics: list[str]) -> dict[str, float]:
    payload = {"n": len(rows)}
    for metric in metrics:
        vals = [float(row[metric]) for row in rows if metric in row]
        payload[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
        payload[f"{metric}_std"] = float(np.std(vals)) if vals else float("nan")
    return payload


def run_degree_shift_suite(args, out_dir: Path) -> dict:
    ensure_dir(out_dir)
    seeds = parse_int_list(args.synthetic_seeds)
    cfg = FixedSourceDegreeShiftConfig(epochs=int(args.degree_shift_epochs))

    rows = []
    visual_snapshot = None
    visual_traj = []
    visual_result = None
    for seed in seeds:
        result, traj_rows, snapshot = train_fixed_source_degree_shift(
            seed=seed,
            cfg=cfg,
            data_builder=lambda: make_degree_shift_data(
                n_per_class=args.degree_shift_n_per_class,
                d_in=args.synthetic_d_in,
                feat_shift=args.degree_shift_feat_shift,
                target_h=(args.degree_shift_target_p_in, args.degree_shift_target_p_out),
                source_multiplier=args.degree_shift_source_multiplier,
            ),
            return_traj=(seed == int(args.visual_seed)),
            include_snapshot=(seed == int(args.visual_seed)),
        )
        rows.append(result)
        if seed == int(args.visual_seed):
            visual_snapshot = snapshot
            visual_traj = traj_rows
            visual_result = result

    summary = summarize_single_payload(
        rows,
        metrics=[
            "raw_degree_ratio",
            "induced_degree_ratio",
            "raw_degree_gap_l1",
            "induced_degree_gap_l1",
            "degree_gap_l1_reduction",
            "degree_ratio_improvement_to_one",
            "response_div_mmd_before",
            "response_div_mmd_after",
            "response_div_mse_before",
            "response_div_mse_after",
        ],
    )

    save_rows_csv(out_dir / "degree_shift_fixed_source_raw.csv", rows)
    save_json(out_dir / "degree_shift_fixed_source_summary.json", summary)

    if visual_snapshot is not None:
        plot_degree_shift_histograms(out_dir / "degree_shift_degree_profiles.png", visual_snapshot)
    if visual_traj:
        plot_degree_shift_training(out_dir / "degree_shift_training_curves.png", visual_traj)
    if visual_result is not None:
        plot_degree_shift_coefficients(out_dir / "degree_shift_filter_coefficients.png", visual_result)

    payload = {
        "plan": ABLATION_PLAN["degree_shift_fixed_source"],
        "summary": summary,
        "rows": rows,
        "visual_seed": int(args.visual_seed),
        "visual_result": visual_result,
    }
    save_json(out_dir / "degree_shift_fixed_source_results.json", payload)
    return payload


def toy_variants(epochs: int, support_ratio: float) -> dict[str, ToyTrainConfig]:
    common = {"epochs": int(epochs), "alpha": 1e-4, "gamma": 0.0, "support_ratio": float(support_ratio)}
    return {
        "source_only": ToyTrainConfig(beta=0.0, delta=0.0, **common),
        "feat_only": ToyTrainConfig(beta=0.5, delta=0.0, **common),
        "filter_only": ToyTrainConfig(beta=0.0, delta=0.5, **common),
        "feat_plus_filter": ToyTrainConfig(beta=0.5, delta=0.5, **common),
    }


def run_synthetic_suite(args, out_dir: Path) -> dict:
    ensure_dir(out_dir)
    seeds = parse_int_list(args.synthetic_seeds)
    degree_shift_payload = run_degree_shift_suite(args, out_dir=out_dir / "degree_shift_fixed_source")
    if args.degree_shift_only:
        payload = {
            "plan": {"degree_shift_fixed_source": ABLATION_PLAN["degree_shift_fixed_source"]},
            "degree_shift_fixed_source": degree_shift_payload,
        }
        save_json(out_dir / "synthetic_results.json", payload)
        return payload

    variants = toy_variants(args.synthetic_epochs, support_ratio=args.support_ratio)

    large_rows = []
    small_rows = []
    for variant_name, cfg in variants.items():
        for seed in seeds:
            large_metrics, _, _ = train_toy_once(
                seed=seed,
                cfg=cfg,
                data_builder=lambda: make_large_synthetic_data(
                    n_per_class=args.synthetic_n_per_class,
                    d_in=args.synthetic_d_in,
                    feat_shift=args.synthetic_feat_shift,
                ),
                return_traj=False,
                include_snapshot=False,
            )
            large_metrics["variant"] = variant_name
            large_metrics["suite"] = "synthetic_large"
            large_rows.append(large_metrics)

            small_metrics, _, _ = train_toy_once(
                seed=seed,
                cfg=cfg,
                data_builder=lambda: make_small_motif_data(
                    n_per_class=args.small_n_per_class,
                    d_in=args.synthetic_d_in,
                    feat_shift=args.small_feat_shift,
                ),
                return_traj=False,
                include_snapshot=False,
            )
            small_metrics["variant"] = variant_name
            small_metrics["suite"] = "synthetic_small"
            small_rows.append(small_metrics)

    summary_metrics = [
        "target_acc",
        "feature_div",
        "response_div",
        "filter_gap_fro",
        "probe_energy_gap",
        "density_gap_induced_effective",
        "triangle_gap_induced",
    ]
    large_summary = summarize_variants(large_rows, metrics=summary_metrics, order=list(variants.keys()))
    small_summary = summarize_variants(small_rows, metrics=summary_metrics, order=list(variants.keys()))

    save_rows_csv(out_dir / "synthetic_large_raw.csv", large_rows)
    save_rows_csv(out_dir / "synthetic_small_raw.csv", small_rows)
    save_rows_csv(out_dir / "synthetic_large_summary.csv", large_summary)
    save_rows_csv(out_dir / "synthetic_small_summary.csv", small_summary)

    plot_summary_bars(
        out_dir / "synthetic_large_performance.png",
        large_summary,
        metric_specs=[("target_acc", "Target Accuracy"), ("filter_gap_fro", "Operator Gap (Fro)")],
        title="Large synthetic performance and operator gap",
    )
    plot_summary_bars(
        out_dir / "synthetic_small_structure_summary.png",
        small_summary,
        metric_specs=[
            ("target_acc", "Target Accuracy"),
            ("probe_energy_gap", "Probe Energy Gap"),
            ("triangle_gap_induced", "Induced Triangle Gap"),
        ],
        title="Small synthetic performance and structure diagnostics",
    )

    visual_seed = int(args.visual_seed)
    visual_traj = {}
    visual_snaps = {}
    for variant_name in ("feat_only", "feat_plus_filter"):
        metrics, traj, snap = train_toy_once(
            seed=visual_seed,
            cfg=variants[variant_name],
            data_builder=lambda: make_small_motif_data(
                n_per_class=args.small_n_per_class,
                d_in=args.synthetic_d_in,
                feat_shift=args.small_feat_shift,
            ),
            return_traj=True,
            include_snapshot=True,
        )
        metrics["variant"] = variant_name
        visual_traj[variant_name] = traj
        visual_snaps[variant_name] = {
            "metrics": metrics,
            "snapshot": snap,
        }

    plot_toy_trajectories(out_dir / "synthetic_small_trajectories.png", visual_traj)
    plot_small_graph_comparison(
        out_dir / "synthetic_small_graphs.png",
        labels=visual_snaps["feat_plus_filter"]["snapshot"]["ys"],
        raw_source=visual_snaps["feat_plus_filter"]["snapshot"]["A_s"],
        raw_target=visual_snaps["feat_plus_filter"]["snapshot"]["A_t"],
        feat_snapshot=visual_snaps["feat_only"]["snapshot"],
        full_snapshot=visual_snaps["feat_plus_filter"]["snapshot"],
        support_ratio=args.support_ratio,
        max_edges=args.plot_max_edges,
    )

    large_gain = compute_gain(large_summary, compare_variant="feat_plus_filter", base_variant="feat_only", metrics=["target_acc"])
    small_gain = compute_gain(
        small_summary,
        compare_variant="feat_plus_filter",
        base_variant="feat_only",
        metrics=["target_acc", "probe_energy_gap", "triangle_gap_induced", "density_gap_induced_effective"],
    )
    plot_gain_bars(
        out_dir / "synthetic_filter_alignment_gain.png",
        gain_rows=[
            {"label": "Large target acc", "value": large_gain.get("target_acc_delta", float("nan"))},
            {"label": "Small target acc", "value": small_gain.get("target_acc_delta", float("nan"))},
            {"label": "Small energy gap", "value": -small_gain.get("probe_energy_gap_delta", float("nan"))},
            {"label": "Small triangle gap", "value": -small_gain.get("triangle_gap_induced_delta", float("nan"))},
            {"label": "Small density gap", "value": -small_gain.get("density_gap_induced_effective_delta", float("nan"))},
        ],
        title="Synthetic: does filter alignment help on top of feature alignment?",
    )

    payload = {
        "plan": ABLATION_PLAN["synthetic_large"] | {"synthetic_small": ABLATION_PLAN["synthetic_small"]},
        "large_summary": large_summary,
        "small_summary": small_summary,
        "large_gain": large_gain,
        "small_gain": small_gain,
        "degree_shift_fixed_source": degree_shift_payload,
        "visual_seed": visual_seed,
        "visual_snapshots": {
            name: {"metrics": payload["metrics"]} for name, payload in visual_snaps.items()
        },
    }
    save_json(out_dir / "synthetic_results.json", payload)
    return payload


def build_real_opal_config(args, seed: int) -> tuple[dict, bool]:
    config_setup = {"data": args.real_dataset, "expt": "default", "model": "opal"}
    common_update = {
        "expt": {
            "source": args.real_source,
            "target": args.real_target,
            "device": args.real_device,
            "seed": int(seed),
            "verbose": 0,
            "wandb_enabled": False,
            "early_stopping": False,
        },
        "model": {
            "epochs": int(args.real_epochs),
        },
    }

    borrow_model = str(args.real_borrow_model).strip().lower() or None
    used_tuned = False
    try:
        cfg = build_config(
            config_setup,
            update_config=common_update,
            borrow=borrow_model,
            use_tuned=int(args.real_use_tuned),
        )
        used_tuned = bool(int(args.real_use_tuned))
    except FileNotFoundError:
        cfg = build_config(config_setup, update_config=common_update, use_tuned=0)
        used_tuned = False
    cfg["model"]["epochs"] = int(args.real_epochs)
    return cfg, used_tuned


def apply_real_variant(cfg: dict, variant: str) -> dict:
    cfg = dict(cfg)
    cfg["model"] = dict(cfg["model"])
    cfg["expt"] = dict(cfg["expt"])

    base_beta = float(cfg["model"].get("beta", 0.5))
    base_delta = float(cfg["model"].get("delta", 0.1))
    cfg["model"]["gamma"] = 0.0

    if variant == "source_only":
        cfg["model"]["beta"] = 0.0
        cfg["model"]["delta"] = 0.0
    elif variant == "feat_only":
        cfg["model"]["beta"] = base_beta
        cfg["model"]["delta"] = 0.0
    elif variant == "filter_only":
        cfg["model"]["beta"] = 0.0
        cfg["model"]["delta"] = base_delta
    elif variant == "feat_plus_filter":
        cfg["model"]["beta"] = base_beta
        cfg["model"]["delta"] = base_delta
    else:
        raise ValueError(f"Unsupported real-data variant: {variant}")

    cfg["ablation_base_beta"] = base_beta
    cfg["ablation_base_delta"] = base_delta
    cfg["ablation_variant"] = variant
    return cfg


def run_real_once(args, variant: str, seed: int) -> dict:
    repo_set_seed(int(seed))
    cfg_base, used_tuned = build_real_opal_config(args, seed=seed)
    cfg = apply_real_variant(cfg_base, variant=variant)

    source_dataset, target_dataset = build_dataset(cfg)
    source_data = source_dataset[0].to(cfg["expt"]["device"])
    target_data = target_dataset[0].to(cfg["expt"]["device"])

    cfg["model"]["in_dim"] = int(source_data.x.shape[1])
    cfg["model"]["num_classes"] = int(source_data.y.unique().numel())

    model = build_model(cfg)
    model.fit(source_data, target_data)

    metric = BaseMetric(cfg)
    target_logits, target_labels = model.predict(target_data, source=False)
    target_result = metric(target_logits, target_labels)
    source_logits, source_labels = model.predict(source_data, source=True)
    source_result = metric(source_logits, source_labels)

    src_coef = model.opal.src_filter.coef.detach().cpu()
    tgt_coef = model.opal.tgt_filter.coef.detach().cpu()
    filter_gap = float(torch.linalg.norm(src_coef - tgt_coef).item())
    ilip = float(model.integral_lipschitz().detach().cpu().item())

    return {
        "variant": variant,
        "seed": int(seed),
        "dataset": args.real_dataset,
        "source": args.real_source,
        "target": args.real_target,
        "used_tuned": int(used_tuned),
        "source_micro_f1": float(source_result.get("micro_f1", float("nan"))),
        "source_macro_f1": float(source_result.get("macro_f1", float("nan"))),
        "target_micro_f1": float(target_result.get("micro_f1", float("nan"))),
        "target_macro_f1": float(target_result.get("macro_f1", float("nan"))),
        "filter_coef_gap_l2": filter_gap,
        "filter_ilip": ilip,
        "src_coef": src_coef.tolist(),
        "tgt_coef": tgt_coef.tolist(),
        "beta": float(cfg["model"]["beta"]),
        "delta": float(cfg["model"]["delta"]),
        "alpha": float(cfg["model"].get("alpha", 0.0)),
        "gamma": float(cfg["model"].get("gamma", 0.0)),
        "base_beta": float(cfg["ablation_base_beta"]),
        "base_delta": float(cfg["ablation_base_delta"]),
    }


def run_real_suite(args, out_dir: Path) -> dict:
    ensure_dir(out_dir)
    seeds = parse_int_list(args.real_seeds)
    variants = ["source_only", "feat_only", "filter_only", "feat_plus_filter"]

    rows = []
    for variant in variants:
        for seed in seeds:
            row = run_real_once(args, variant=variant, seed=seed)
            rows.append(row)

    summary = summarize_variants(
        rows,
        metrics=["target_micro_f1", "target_macro_f1", "filter_coef_gap_l2", "filter_ilip"],
        order=variants,
    )
    gain = compute_gain(
        summary,
        compare_variant="feat_plus_filter",
        base_variant="feat_only",
        metrics=["target_micro_f1", "target_macro_f1", "filter_coef_gap_l2"],
    )

    save_rows_csv(out_dir / "real_raw.csv", rows)
    save_rows_csv(out_dir / "real_summary.csv", summary)
    plot_summary_bars(
        out_dir / "real_performance.png",
        summary,
        metric_specs=[
            ("target_micro_f1", "Target Micro-F1"),
            ("target_macro_f1", "Target Macro-F1"),
            ("filter_coef_gap_l2", "Filter Coef Gap (L2)"),
        ],
        title=f"Real-data OPAL ablation: {args.real_dataset} {args.real_source}->{args.real_target}",
    )
    plot_gain_bars(
        out_dir / "real_filter_alignment_gain.png",
        gain_rows=[
            {"label": "Target micro-F1", "value": gain.get("target_micro_f1_delta", float("nan"))},
            {"label": "Target macro-F1", "value": gain.get("target_macro_f1_delta", float("nan"))},
            {"label": "Coef gap", "value": -gain.get("filter_coef_gap_l2_delta", float("nan"))},
        ],
        title="Real data: does filter alignment help on top of feature alignment?",
    )

    payload = {
        "plan": ABLATION_PLAN["real_data"],
        "summary": summary,
        "gain": gain,
        "rows": rows,
    }
    save_json(out_dir / "real_results.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Expanded OPAL ablation on synthetic and real data.")
    parser.add_argument("--mode", type=str, default="both", choices=["synthetic", "real", "both"])
    parser.add_argument("--out-dir", type=str, default="./__saved__/analysis/opal_toy_ablation")
    parser.add_argument("--support-ratio", type=float, default=0.25)
    parser.add_argument("--plot-max-edges", type=int, default=32)

    parser.add_argument("--synthetic-seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--visual-seed", type=int, default=0)
    parser.add_argument("--synthetic-epochs", type=int, default=120)
    parser.add_argument("--synthetic-n-per-class", type=int, default=60)
    parser.add_argument("--small-n-per-class", type=int, default=6)
    parser.add_argument("--synthetic-d-in", type=int, default=8)
    parser.add_argument("--synthetic-feat-shift", type=float, default=1.0)
    parser.add_argument("--small-feat-shift", type=float, default=0.7)
    parser.add_argument("--degree-shift-epochs", type=int, default=120)
    parser.add_argument("--degree-shift-n-per-class", type=int, default=24)
    parser.add_argument("--degree-shift-feat-shift", type=float, default=0.35)
    parser.add_argument("--degree-shift-target-p-in", type=float, default=0.10)
    parser.add_argument("--degree-shift-target-p-out", type=float, default=0.03)
    parser.add_argument("--degree-shift-source-multiplier", type=float, default=3.0)
    parser.add_argument("--degree-shift-only", action="store_true")

    parser.add_argument("--real-dataset", type=str, default="citation")
    parser.add_argument("--real-source", type=str, default="ACMv9")
    parser.add_argument("--real-target", type=str, default="Citationv1")
    parser.add_argument("--real-device", type=str, default="cuda:0")
    parser.add_argument("--real-seeds", type=str, default="0,1,2")
    parser.add_argument("--real-epochs", type=int, default=120)
    parser.add_argument("--real-use-tuned", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--real-borrow-model", type=str, default="")
    parser.add_argument("--skip-real-on-error", action="store_true")
    args = parser.parse_args()

    stamp = time.strftime("%m%d_%H%M%S")
    out_root = Path(args.out_dir) / stamp
    ensure_dir(out_root)
    save_json(out_root / "plan.json", {"plan": ABLATION_PLAN, "args": vars(args)})

    results = {"out_dir": str(out_root), "plan": ABLATION_PLAN}
    if args.mode in {"synthetic", "both"}:
        results["synthetic"] = run_synthetic_suite(args, out_dir=out_root / "synthetic")

    if args.mode in {"real", "both"}:
        try:
            results["real"] = run_real_suite(args, out_dir=out_root / "real")
        except Exception as exc:
            if not args.skip_real_on_error:
                raise
            results["real_error"] = str(exc)
            save_json(out_root / "real_error.json", {"error": str(exc)})

    save_json(out_root / "results.json", results)
    print(f"Saved ablation artifacts to: {out_root}")


if __name__ == "__main__":
    main()
