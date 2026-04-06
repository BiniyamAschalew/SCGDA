"""Adversarial monomial-filter motivation experiment with layer-wise transfer evaluation."""

from datetime import datetime
from collections import defaultdict
from itertools import product
from pathlib import Path
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch import nn

from  models.__filters.mono import MonoProp
from  utils.ablation_utils.common import load_pair, sample_idx
from  utils.ablation_utils.motivation_reporting import save_json, save_rows, to_plain_dict
from  utils.ablation_utils.propagation import Propagation
from  utils.ablation_utils.transfer import get_source_train_mask, train_eval_transfer_once
from  utils.config_utils import load_config
from  utils.expt_utils import set_seed
from  utils.filter_utils import mmd_rbf, median_bandwidth
from  utils.train_utils.mmd import Sinkhorn


SHIFT_CASE_SUMMARY_KEYS = [
    "mmd2_total_mean",
    "mmd2_total_std",
    "mmd2_norm_mean",
    "mmd2_norm_std",
    "mmd2_angle_mean",
    "mmd2_angle_std",
]

TRANSFER_CASE_SUMMARY_KEYS = [
    "source_micro_f1_mean",
    "source_micro_f1_std",
    "target_micro_f1_mean",
    "target_micro_f1_std",
    "delta_micro_f1_mean",
    "delta_micro_f1_std",
]


def _cfg_get(cfg, key: str, default):
    if hasattr(cfg, "get"):
        value = cfg.get(key, default)
        return default if value is None else value
    try:
        value = cfg[key]
    except Exception:
        return default
    return default if value is None else value


def _resolve_device(device: str) -> str:
    return "cpu" if str(device).startswith("cuda") and not torch.cuda.is_available() else str(device)


def _fit_lda(xs, ys, lda_dim, lda_eps):
    lda = LinearDiscriminantAnalysis(n_components=lda_dim, tol=lda_eps)
    lda.fit(xs.detach().cpu().numpy(), ys.detach().cpu().numpy())
    return torch.from_numpy(lda.scalings_).float().to(xs.device)


def _remap_labels_contiguous(ys: torch.Tensor, yt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    classes = torch.unique(torch.cat([ys.view(-1), yt.view(-1)], dim=0), sorted=True)
    mapping = {int(c.item()): i for i, c in enumerate(classes)}

    def _remap(y):
        return torch.tensor([mapping[int(v.item())] for v in y.view(-1)], device=y.device, dtype=torch.long)

    return _remap(ys), _remap(yt)


def _try_fit_joint_lda(xs, xt, ys, yt, lda_dim, lda_eps):
    n_classes = int(torch.unique(torch.cat([ys, yt], dim=0)).numel())
    out_dim = min(int(lda_dim), int(xs.size(1)), n_classes - 1)
    if out_dim <= 0:
        return None
    x = torch.cat([xs, xt], dim=0)
    y = torch.cat([ys, yt], dim=0)
    w = _fit_lda(x, y, out_dim, float(lda_eps))
    return w[:, :out_dim]


def _parse_fix_sigma(fix_sigma):
    if isinstance(fix_sigma, str):
        fix = fix_sigma.strip().lower()
        if fix in ("none", "null", ""):
            return None
        return float(fix)
    return fix_sigma


def _mmd(a: torch.Tensor, b: torch.Tensor, cfg, fix_sigma=None) -> float:
    if fix_sigma is None:
        fix_sigma = _parse_fix_sigma(_cfg_get(cfg, "fix_sigma", None))
    value = mmd_rbf(
        a,
        b,
        kernel_mul=float(_cfg_get(cfg, "kernel_mul", 2.0)),
        kernel_num=int(_cfg_get(cfg, "kernel_num", 5)),
        fix_sigma=fix_sigma,
    )
    return float(value.detach().cpu().item())


def _mmd_tensor(a: torch.Tensor, b: torch.Tensor, cfg) -> torch.Tensor:
    return mmd_rbf(
        a,
        b,
        kernel_mul=float(_cfg_get(cfg, "kernel_mul", 2.0)),
        kernel_num=int(_cfg_get(cfg, "kernel_num", 5)),
        fix_sigma=_parse_fix_sigma(_cfg_get(cfg, "fix_sigma", None)),
    )


def _alignment_tensor(a: torch.Tensor, b: torch.Tensor, cfg) -> torch.Tensor:
    """Probe pushforward alignment loss with switchable backend."""
    metric = str(_cfg_get(cfg, "alignment_metric", "mmd")).lower()
    if metric == "sinkhorn":
        sampling_num = int(_cfg_get(cfg, "sinkhorn_sampling_num", _cfg_get(cfg, "align_sample_size", 1024)))
        times = int(_cfg_get(cfg, "sinkhorn_times", 3))
        return Sinkhorn(
            a,
            b,
            sampling_num=max(1, sampling_num),
            times=max(1, times),
            blur=float(_cfg_get(cfg, "sinkhorn_blur", 0.05)),
            p=int(_cfg_get(cfg, "sinkhorn_p", 2)),
            scaling=float(_cfg_get(cfg, "sinkhorn_scaling", 0.9)),
            debias=bool(_cfg_get(cfg, "sinkhorn_debias", True)),
            backend=str(_cfg_get(cfg, "sinkhorn_backend", "auto")),
        )
    return _mmd_tensor(a, b, cfg)


def _edge_bpr_structure_loss(
    z: torch.Tensor,
    pos_edge_index: torch.Tensor,
    num_samples: int,
    margin: float = 0.0,
) -> torch.Tensor:
    """BPR-style structure preservation loss from positive edges vs random negatives."""
    if pos_edge_index.numel() == 0 or z.size(0) <= 1:
        return z.new_tensor(0.0)

    edge_count = int(pos_edge_index.size(1))
    k = int(min(max(1, num_samples), edge_count))
    pos_ids = torch.randint(edge_count, (k,), device=pos_edge_index.device)
    pos_u = pos_edge_index[0, pos_ids]
    pos_v = pos_edge_index[1, pos_ids]

    num_nodes = int(z.size(0))
    neg_u = pos_u
    neg_v = torch.randint(num_nodes, (k,), device=z.device)
    same = neg_u == neg_v
    neg_v = torch.where(same, (neg_v + 1) % num_nodes, neg_v)

    pos_dist = (z[pos_u] - z[pos_v]).pow(2).sum(dim=1)
    neg_dist = (z[neg_u] - z[neg_v]).pow(2).sum(dim=1)
    return F.softplus(float(margin) + pos_dist - neg_dist).mean()


def _dirichlet_energy(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Graph Dirichlet energy from edge-wise feature differences."""
    if edge_index.numel() == 0:
        return x.new_tensor(0.0)
    row, col = edge_index
    diff = x[row] - x[col]
    return diff.pow(2).sum(dim=1).mean()


def _spectral_component_loss(source_feat: torch.Tensor, target_feat: torch.Tensor, topk: int) -> torch.Tensor:
    """Match top covariance spectrum components between source and target."""
    k = max(1, int(topk))

    def _top_cov_eigs(x: torch.Tensor) -> torch.Tensor:
        x_centered = x - x.mean(dim=0, keepdim=True)
        denom = max(1, int(x_centered.size(0)) - 1)
        cov = (x_centered.t() @ x_centered) / float(denom)
        vals = torch.linalg.eigvalsh(cov).real
        vals = torch.flip(vals, dims=[0])
        return vals[: min(k, int(vals.numel()))]

    s_vals = _top_cov_eigs(source_feat)
    t_vals = _top_cov_eigs(target_feat)
    kk = min(int(s_vals.numel()), int(t_vals.numel()))
    if kk == 0:
        return source_feat.new_tensor(0.0)
    return F.l1_loss(s_vals[:kk], t_vals[:kk])


def _shared_bandwidth(a: torch.Tensor, b: torch.Tensor, cfg):
    fix_sigma = _parse_fix_sigma(_cfg_get(cfg, "fix_sigma", None))
    if fix_sigma is None:
        return median_bandwidth(a, b).item()
    return float(fix_sigma)


def _unit_normalize_rows(x: torch.Tensor, eps: float = 1e-8):
    norm = x.norm(p=2, dim=1, keepdim=True).clamp_min(float(eps))
    return x / norm, norm.view(-1)


def _compute_mmd_safe(hs: torch.Tensor, ht: torch.Tensor, cfg) -> tuple[float, float, float]:
    hs_cpu, ht_cpu = hs.detach().cpu(), ht.detach().cpu()
    shared_sigma = _shared_bandwidth(hs_cpu, ht_cpu, cfg)

    total_mmd = _mmd(hs_cpu, ht_cpu, cfg, fix_sigma=shared_sigma)
    source_unit, source_norm = _unit_normalize_rows(hs_cpu)
    target_unit, target_norm = _unit_normalize_rows(ht_cpu)
    source_norm_col = source_norm.unsqueeze(1)
    target_norm_col = target_norm.unsqueeze(1)
    norm_sigma = _shared_bandwidth(source_norm_col, target_norm_col, cfg)
    norm_mmd = _mmd(source_norm_col, target_norm_col, cfg, fix_sigma=norm_sigma)
    angle_sigma = _shared_bandwidth(source_unit, target_unit, cfg)
    angle_mmd = _mmd(source_unit, target_unit, cfg, fix_sigma=angle_sigma)

    return total_mmd, norm_mmd, angle_mmd


def _aggregate_case_layer(rows: list[dict], max_layers: int, metric_keys: list[str]) -> list[dict]:
    cases = sorted({str(r["case"]) for r in rows})
    out = []
    for case in cases:
        for layer in range(1, int(max_layers) + 1):
            layer_rows = [r for r in rows if str(r["case"]) == case and int(r["layer"]) == layer]
            if not layer_rows:
                continue
            row = {
                "case": case,
                "layer": float(layer),
                "n_trials": float(len(layer_rows)),
            }
            for key in metric_keys:
                vals = torch.tensor([float(r[key]) for r in layer_rows], dtype=torch.float32)
                row[f"{key}_mean"] = float(vals.mean().item())
                row[f"{key}_std"] = float(vals.std(unbiased=False).item())
            out.append(row)
    return out


def _average_case_layer_summaries(
    all_summaries: list[list[dict]],
    max_layers: int,
    metric_keys: list[str],
) -> list[dict]:
    cases = sorted({str(r["case"]) for rows in all_summaries for r in rows})
    out = []
    for case in cases:
        for layer in range(1, int(max_layers) + 1):
            layer_rows = [
                r
                for rows in all_summaries
                for r in rows
                if str(r.get("case", "")) == case and int(r["layer"]) == layer
            ]
            if not layer_rows:
                continue
            row = {
                "case": case,
                "layer": float(layer),
                "n_scenarios": float(len(layer_rows)),
            }
            for key in metric_keys:
                vals = torch.tensor([float(r[key]) for r in layer_rows], dtype=torch.float32)
                row[key] = float(vals.mean().item())
            out.append(row)
    return out


def _style_for_case(case: str) -> tuple[str, str]:
    if case == "adjacency":
        return "tab:blue", "o"
    if case == "mono_aligned":
        return "tab:orange", "s"
    if case == "mono_adv_aligned":
        return "tab:green", "^"
    return "tab:gray", "x"


def _plot_case_transfer_accuracy(path: Path, transfer_summary: list[dict], title: str) -> bool:
    if not transfer_summary:
        return False
    cases = sorted({str(r["case"]) for r in transfer_summary})
    row_map = {
        case: {int(r["layer"]): r for r in transfer_summary if str(r["case"]) == case}
        for case in cases
    }

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for case in cases:
        layers = sorted(row_map[case].keys())
        if not layers:
            continue
        color, marker = _style_for_case(case)
        y = [float(row_map[case][k]["target_micro_f1_mean"]) for k in layers]
        s = [float(row_map[case][k].get("target_micro_f1_std", 0.0)) for k in layers]
        ax.plot(layers, y, marker=marker, color=color, linewidth=2.0, label=case)
        ax.fill_between(layers, [a - b for a, b in zip(y, s)], [a + b for a, b in zip(y, s)], alpha=0.2, color=color)

    ax.set_xlabel("Propagation layer")
    ax.set_ylabel("Target transfer accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def _plot_case_shift_components(path: Path, shift_summary: list[dict], title: str) -> bool:
    if not shift_summary:
        return False
    cases = sorted({str(r["case"]) for r in shift_summary})
    row_map = {
        case: {int(r["layer"]): r for r in shift_summary if str(r["case"]) == case}
        for case in cases
    }

    metric_defs = [
        ("mmd2_total_mean", "mmd2_total_std", "MMD total"),
        ("mmd2_norm_mean", "mmd2_norm_std", "MMD norm"),
        ("mmd2_angle_mean", "mmd2_angle_std", "MMD angle"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)

    for axis, (mean_key, std_key, axis_title) in zip(axes, metric_defs):
        for case in cases:
            layers = sorted(row_map[case].keys())
            if not layers:
                continue
            color, marker = _style_for_case(case)
            y = [float(row_map[case][k][mean_key]) for k in layers]
            s = [float(row_map[case][k].get(std_key, 0.0)) for k in layers]
            axis.plot(layers, y, marker=marker, color=color, linewidth=1.8, label=case)
            axis.fill_between(layers, [a - b for a, b in zip(y, s)], [a + b for a, b in zip(y, s)], alpha=0.2, color=color)
        axis.set_title(axis_title)
        axis.set_xlabel("Propagation layer")
        axis.grid(alpha=0.3)

    axes[0].set_ylabel("Shift")
    axes[0].legend(loc="best")
    fig.suptitle(title)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def _plot_alignment_history(path: Path, nonadv_history: list[dict], adv_history: list[dict], title: str) -> bool:
    if not nonadv_history and not adv_history:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)

    def _draw(rows: list[dict], label: str, color: str):
        if not rows:
            return
        xs = [int(r["epoch"]) for r in rows]
        axes[0].plot(xs, [float(r["probe_mmd"]) for r in rows], color=color, linewidth=1.8, label=label)
        axes[1].plot(xs, [float(r["real_mmd2_total"]) for r in rows], color=color, linewidth=1.8, label=label)
        axes[2].plot(xs, [float(r["semantic_loss"]) for r in rows], color=color, linewidth=1.8, label=label)

    _draw(nonadv_history, "mono_aligned", "tab:orange")
    _draw(adv_history, "mono_adv_aligned", "tab:green")

    axes[0].set_title("Probe pushforward MMD")
    axes[1].set_title("Real-feature MMD total")
    axes[2].set_title("Source semantic loss")

    for axis in axes:
        axis.set_xlabel("Alignment epoch")
        axis.grid(alpha=0.3)

    axes[0].set_ylabel("Value")
    axes[0].legend(loc="best")

    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def _build_scenarios(transfer_settings):
    return {
        dataset: [(source, target) for source, targets in pairs.items() for target in targets]
        for dataset, pairs in transfer_settings.items()
    }


def _timestamped_out_dir(base_out_dir: str) -> str:
    base_path = Path(base_out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(base_path / f"motivation_{timestamp}")


def _effective_filter(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    t = max(float(temperature), 1e-6)
    return torch.softmax(logits / t, dim=0)


def _symmetric_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(float(eps))
    q = q.clamp_min(float(eps))
    kl_pq = (p * (p.log() - q.log())).sum()
    kl_qp = (q * (q.log() - p.log())).sum()
    return 0.5 * (kl_pq + kl_qp)


def _init_filter_logits(num_terms: int, init_mode: str, device: torch.device) -> torch.Tensor:
    mode = str(init_mode).lower()
    logits = torch.zeros(num_terms, device=device)
    if mode in {"adj", "adjacency", "a1"}:
        logits.fill_(-2.0)
        logits[1 if num_terms > 1 else 0] = 2.0
    elif mode in {"identity", "a0"}:
        logits.fill_(-2.0)
        logits[0] = 2.0
    return logits


def _probe_stats(source_x: torch.Tensor, target_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    all_feat = torch.cat([source_x.detach(), target_x.detach()], dim=0)
    mean = all_feat.mean(dim=0, keepdim=True)
    std = all_feat.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mean, std


def _sample_gaussian_probe_like(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return torch.randn_like(x) * std + mean


class LearnableProbeDistribution(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(1, feature_dim))
        self.log_scale = nn.Parameter(torch.zeros(1, feature_dim))

    def sample(self, base: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.log_scale) + 1e-4
        noise = torch.randn_like(base)
        return base + self.shift + scale * noise

    def regularizer(self) -> torch.Tensor:
        return self.shift.pow(2).mean() + self.log_scale.pow(2).mean()

    def diagnostics(self) -> dict[str, float]:
        with torch.no_grad():
            scale = F.softplus(self.log_scale) + 1e-4
            return {
                "shift_l2": float(self.shift.norm().detach().cpu().item()),
                "scale_mean": float(scale.mean().detach().cpu().item()),
            }


def _apply_monomial_once(
    mono_prop: MonoProp,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    coeff: torch.Tensor,
    lambda_max: float,
) -> torch.Tensor:
    return mono_prop(
        x,
        edge_index,
        parameters=coeff,
        edge_weight=None,
        lambda_max=float(lambda_max),
    )


def _propagate_monomial_layers(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    coeff: torch.Tensor,
    max_layers: int,
    lambda_max: float,
    mono_prop: MonoProp,
) -> list[torch.Tensor]:
    layers = [x]
    cur = x
    for _ in range(int(max_layers)):
        cur = _apply_monomial_once(
            mono_prop,
            cur,
            edge_index,
            coeff,
            lambda_max,
        )
        layers.append(cur)
    return layers


def _propagate_adjacency_layers(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    max_layers: int,
    prop: Propagation,
) -> list[torch.Tensor]:
    layers = [x]
    cur = x
    for _ in range(int(max_layers)):
        cur = prop(cur, edge_index)
        layers.append(cur)
    return layers


def _to_undirected_unique(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return unique undirected edges with shape [2, E], u < v."""
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    keep = row != col
    row, col = row[keep], col[keep]
    if row.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    lo = torch.minimum(row, col)
    hi = torch.maximum(row, col)
    key = lo * int(num_nodes) + hi
    uniq = torch.unique(key)
    out_lo = torch.div(uniq, int(num_nodes), rounding_mode="floor")
    out_hi = uniq % int(num_nodes)
    return torch.stack([out_lo.long(), out_hi.long()], dim=0)


def _induced_monomial_operator_matrix(
    data,
    coeff: torch.Tensor,
    lambda_max: float,
    chunk_size: int,
) -> torch.Tensor:
    """Compute induced linear operator matrix H by applying the monomial filter to I."""
    num_nodes = int(data.x.size(0))
    device = data.x.device
    dtype = data.x.dtype if data.x.dtype.is_floating_point else torch.float32
    mono_prop = MonoProp().to(device)
    out_cpu = torch.empty((num_nodes, num_nodes), dtype=torch.float32, device="cpu")

    width = max(1, int(chunk_size))
    for start in range(0, num_nodes, width):
        end = min(num_nodes, start + width)
        basis_w = end - start

        basis = torch.zeros((num_nodes, basis_w), device=device, dtype=dtype)
        rows = torch.arange(start, end, device=device)
        cols = torch.arange(basis_w, device=device)
        basis[rows, cols] = 1.0

        pushed = _apply_monomial_once(
            mono_prop=mono_prop,
            x=basis,
            edge_index=data.edge_index,
            coeff=coeff,
            lambda_max=lambda_max,
        )
        out_cpu[:, start:end] = pushed.detach().to("cpu", dtype=torch.float32)
    return out_cpu


def _operator_to_undirected_edges(
    operator_h: torch.Tensor,
    threshold: float,
    topk: int,
) -> torch.Tensor:
    """Convert dense operator into unique undirected induced graph edges."""
    h = operator_h.abs()
    h = 0.5 * (h + h.t())
    h.fill_diagonal_(0.0)
    num_nodes = int(h.size(0))

    if 0 < int(topk) < num_nodes:
        vals, idx = torch.topk(h, k=int(topk), dim=1)
        row = torch.arange(num_nodes).unsqueeze(1).expand(-1, int(topk)).reshape(-1)
        col = idx.reshape(-1)
        w = vals.reshape(-1)
        keep = w > float(threshold)
        edge_index = torch.stack([row[keep], col[keep]], dim=0).long()
    else:
        row, col = torch.nonzero(h > float(threshold), as_tuple=True)
        edge_index = torch.stack([row, col], dim=0).long()

    return _to_undirected_unique(edge_index, num_nodes)


def _degree_hist(edge_index: torch.Tensor, num_nodes: int, bins: int) -> torch.Tensor:
    if int(num_nodes) <= 1:
        return torch.ones(max(1, int(bins)), dtype=torch.float32) / float(max(1, int(bins)))
    undirected = _to_undirected_unique(edge_index, int(num_nodes)).to("cpu")
    deg = torch.zeros(int(num_nodes), dtype=torch.float32)
    if undirected.numel() > 0:
        deg.index_add_(0, undirected[0], torch.ones(undirected.size(1)))
        deg.index_add_(0, undirected[1], torch.ones(undirected.size(1)))
    deg_norm = deg / float(max(1, int(num_nodes) - 1))
    hist = torch.histc(deg_norm, bins=max(1, int(bins)), min=0.0, max=1.0)
    hist = hist / hist.sum().clamp_min(1e-8)
    return hist


def _motif_stats(edge_index: torch.Tensor, num_nodes: int, motif_max_nodes: int) -> dict[str, float]:
    undirected = _to_undirected_unique(edge_index, int(num_nodes)).to("cpu")
    graph = nx.Graph()
    graph.add_nodes_from(range(int(num_nodes)))
    if undirected.numel() > 0:
        graph.add_edges_from([(int(u), int(v)) for u, v in undirected.t().tolist()])

    n = float(num_nodes)
    m = float(graph.number_of_edges())
    density = 0.0 if n <= 1 else (2.0 * m) / (n * (n - 1.0))

    degrees = torch.tensor([d for _, d in graph.degree()], dtype=torch.float32)
    wedges = float((degrees * (degrees - 1.0) * 0.5).sum().item())
    triplets = max(1.0, n * (n - 1.0) * (n - 2.0) / 6.0)

    triangles = 0.0
    transitivity = 0.0
    if int(num_nodes) <= int(motif_max_nodes):
        tri_map = nx.triangles(graph)
        triangles = float(sum(tri_map.values()) / 3.0)
        transitivity = float(nx.transitivity(graph)) if wedges > 0 else 0.0

    return {
        "density": density,
        "wedge_density": wedges / triplets,
        "triangle_density": triangles / triplets,
        "transitivity": transitivity,
    }


def _structure_alignment_metrics(
    source_edge_index: torch.Tensor,
    target_edge_index: torch.Tensor,
    source_num_nodes: int,
    target_num_nodes: int,
    bins: int,
    motif_max_nodes: int,
) -> dict[str, float]:
    source_hist = _degree_hist(source_edge_index, source_num_nodes, bins)
    target_hist = _degree_hist(target_edge_index, target_num_nodes, bins)
    degree_l1 = float((source_hist - target_hist).abs().sum().item())

    source_motif = _motif_stats(source_edge_index, source_num_nodes, motif_max_nodes)
    target_motif = _motif_stats(target_edge_index, target_num_nodes, motif_max_nodes)
    motif_l1 = float(
        sum(abs(float(source_motif[k]) - float(target_motif[k])) for k in source_motif.keys())
    )

    row = {
        "degree_hist_l1": degree_l1,
        "motif_l1": motif_l1,
    }
    for key, val in source_motif.items():
        row[f"source_{key}"] = float(val)
    for key, val in target_motif.items():
        row[f"target_{key}"] = float(val)
    return row


def _induced_interpretability_rows(source_data, target_data, nonadv_state: dict, adv_state: dict, cfg) -> list[dict]:
    lambda_max = float(_cfg_get(cfg, "mono_lambda_max", 2.0))
    chunk_size = int(_cfg_get(cfg, "induced_chunk_size", 128))
    threshold = float(_cfg_get(cfg, "induced_edge_threshold", 1e-5))
    topk = int(_cfg_get(cfg, "induced_topk", 16))
    bins = int(_cfg_get(cfg, "degree_hist_bins", 20))
    motif_max_nodes = int(_cfg_get(cfg, "motif_max_nodes", 4000))

    rows = []
    source_adj = _to_undirected_unique(source_data.edge_index, int(source_data.x.size(0)))
    target_adj = _to_undirected_unique(target_data.edge_index, int(target_data.x.size(0)))
    base_metrics = _structure_alignment_metrics(
        source_adj,
        target_adj,
        int(source_data.x.size(0)),
        int(target_data.x.size(0)),
        bins,
        motif_max_nodes,
    )
    rows.append({"case": "adjacency", **base_metrics})

    for state in (nonadv_state, adv_state):
        source_op = _induced_monomial_operator_matrix(
            source_data,
            state["source_coeff"].detach(),
            lambda_max=lambda_max,
            chunk_size=chunk_size,
        )
        target_op = _induced_monomial_operator_matrix(
            target_data,
            state["target_coeff"].detach(),
            lambda_max=lambda_max,
            chunk_size=chunk_size,
        )
        source_edges = _operator_to_undirected_edges(source_op, threshold=threshold, topk=topk)
        target_edges = _operator_to_undirected_edges(target_op, threshold=threshold, topk=topk)
        metrics = _structure_alignment_metrics(
            source_edges,
            target_edges,
            int(source_data.x.size(0)),
            int(target_data.x.size(0)),
            bins,
            motif_max_nodes,
        )
        rows.append({"case": str(state["regime"]), **metrics})
    return rows


def _train_monomial_aligner(
    *,
    source_data,
    target_data,
    source_x: torch.Tensor,
    target_x: torch.Tensor,
    source_y: torch.Tensor,
    target_y: torch.Tensor,
    source_train_mask: torch.Tensor,
    cfg,
    adversarial: bool,
    seed_offset: int,
) -> dict:
    set_seed(int(_cfg_get(cfg, "seed", 0)) + int(seed_offset))

    device = source_x.device
    num_terms = int(_cfg_get(cfg, "mono_degree", 3)) + 1
    num_classes = int(torch.max(torch.cat([source_y, target_y], dim=0)).item()) + 1

    mono_prop = MonoProp().to(device)
    source_filter_logits = nn.Parameter(
        _init_filter_logits(num_terms, _cfg_get(cfg, "mono_init", "uniform"), device)
    )
    target_filter_logits = nn.Parameter(
        _init_filter_logits(num_terms, _cfg_get(cfg, "mono_init", "uniform"), device)
    )
    semantic_head = nn.Linear(source_x.size(1), num_classes).to(device)

    filter_optimizer = torch.optim.Adam(
        [source_filter_logits, target_filter_logits] + list(semantic_head.parameters()),
        lr=float(_cfg_get(cfg, "align_lr", 1e-2)),
        weight_decay=float(_cfg_get(cfg, "align_weight_decay", 0.0)),
    )

    source_dist = None
    target_dist = None
    adv_optimizer = None
    if adversarial:
        source_dist = LearnableProbeDistribution(int(source_x.size(1))).to(device)
        target_dist = LearnableProbeDistribution(int(target_x.size(1))).to(device)
        adv_optimizer = torch.optim.Adam(
            list(source_dist.parameters()) + list(target_dist.parameters()),
            lr=float(_cfg_get(cfg, "adv_lr", 3e-3)),
            weight_decay=float(_cfg_get(cfg, "adv_weight_decay", 0.0)),
        )

    probe_mean, probe_std = _probe_stats(source_x, target_x)

    align_epochs = int(_cfg_get(cfg, "align_epochs", 120))
    align_log_interval = max(1, int(_cfg_get(cfg, "align_log_interval", 10)))
    align_sample_size = int(_cfg_get(cfg, "align_sample_size", 1024))

    filter_temperature = float(_cfg_get(cfg, "mono_filter_temperature", 1.0))
    lambda_max = float(_cfg_get(cfg, "mono_lambda_max", 2.0))
    alignment_metric = str(_cfg_get(cfg, "alignment_metric", "mmd")).lower()
    probe_mmd_weight = float(_cfg_get(cfg, "probe_mmd_weight", 1.0))
    source_semantic_weight = float(_cfg_get(cfg, "source_semantic_weight", 1.0))
    filter_alignment_weight = float(_cfg_get(cfg, "filter_alignment_weight", 0.1))
    filter_coeff_reg_weight = float(_cfg_get(cfg, "filter_coeff_reg_weight", 0.0))
    structure_bpr_weight = float(_cfg_get(cfg, "structure_bpr_weight", 0.0))
    structure_bpr_samples = int(_cfg_get(cfg, "structure_bpr_samples", 2048))
    structure_bpr_margin = float(_cfg_get(cfg, "structure_bpr_margin", 0.0))
    spectral_component_weight = float(_cfg_get(cfg, "spectral_component_weight", 0.0))
    spectral_component_topk = int(_cfg_get(cfg, "spectral_component_topk", 8))
    spectral_dirichlet_weight = float(_cfg_get(cfg, "spectral_dirichlet_weight", 0.0))
    adv_distribution_reg_weight = float(_cfg_get(cfg, "adv_distribution_reg_weight", 1e-3))
    adv_steps = max(1, int(_cfg_get(cfg, "adv_steps", 1)))
    source_struct_edges = source_data.edge_index
    target_struct_edges = target_data.edge_index

    history_rows = []

    for epoch in range(1, align_epochs + 1):
        adv_probe_mmd_val = 0.0
        adv_objective_val = 0.0
        adv_reg_val = 0.0

        if adversarial and adv_optimizer is not None and source_dist is not None and target_dist is not None:
            for _ in range(adv_steps):
                adv_optimizer.zero_grad()

                with torch.no_grad():
                    source_coeff_adv = _effective_filter(source_filter_logits, filter_temperature)
                    target_coeff_adv = _effective_filter(target_filter_logits, filter_temperature)

                source_probe_base = _sample_gaussian_probe_like(source_x, probe_mean, probe_std)
                target_probe_base = _sample_gaussian_probe_like(target_x, probe_mean, probe_std)
                source_probe_adv = source_dist.sample(source_probe_base)
                target_probe_adv = target_dist.sample(target_probe_base)

                source_push_adv = _apply_monomial_once(
                    mono_prop,
                    source_probe_adv,
                    source_data.edge_index,
                    source_coeff_adv,
                    lambda_max,
                )
                target_push_adv = _apply_monomial_once(
                    mono_prop,
                    target_probe_adv,
                    target_data.edge_index,
                    target_coeff_adv,
                    lambda_max,
                )

                idx_s_adv = sample_idx(source_push_adv.size(0), align_sample_size, device)
                idx_t_adv = sample_idx(target_push_adv.size(0), align_sample_size, device)
                adv_probe_mmd = _alignment_tensor(
                    source_push_adv[idx_s_adv],
                    target_push_adv[idx_t_adv],
                    cfg,
                )
                adv_reg = adv_distribution_reg_weight * (
                    source_dist.regularizer() + target_dist.regularizer()
                )
                adv_objective = adv_probe_mmd - adv_reg
                loss_adv = -adv_objective
                loss_adv.backward()
                adv_optimizer.step()

                adv_probe_mmd_val = float(adv_probe_mmd.detach().cpu().item())
                adv_objective_val = float(adv_objective.detach().cpu().item())
                adv_reg_val = float(adv_reg.detach().cpu().item())

        filter_optimizer.zero_grad()
        source_coeff = _effective_filter(source_filter_logits, filter_temperature)
        target_coeff = _effective_filter(target_filter_logits, filter_temperature)

        source_probe_base = _sample_gaussian_probe_like(source_x, probe_mean, probe_std)
        target_probe_base = _sample_gaussian_probe_like(target_x, probe_mean, probe_std)
        if adversarial and source_dist is not None and target_dist is not None:
            with torch.no_grad():
                source_probe = source_dist.sample(source_probe_base)
                target_probe = target_dist.sample(target_probe_base)
        else:
            source_probe = source_probe_base
            target_probe = target_probe_base

        source_push = _apply_monomial_once(
            mono_prop,
            source_probe,
            source_data.edge_index,
            source_coeff,
            lambda_max,
        )
        target_push = _apply_monomial_once(
            mono_prop,
            target_probe,
            target_data.edge_index,
            target_coeff,
            lambda_max,
        )
        idx_s = sample_idx(source_push.size(0), align_sample_size, device)
        idx_t = sample_idx(target_push.size(0), align_sample_size, device)
        probe_mmd = _alignment_tensor(source_push[idx_s], target_push[idx_t], cfg)

        source_real = _apply_monomial_once(
            mono_prop,
            source_x,
            source_data.edge_index,
            source_coeff,
            lambda_max,
        )
        target_real = _apply_monomial_once(
            mono_prop,
            target_x,
            target_data.edge_index,
            target_coeff,
            lambda_max,
        )
        source_logits = semantic_head(source_real)
        semantic_loss = F.cross_entropy(
            source_logits[source_train_mask],
            source_y[source_train_mask],
        )

        filter_alignment_loss = _symmetric_kl(source_coeff, target_coeff)
        coeff_reg = filter_coeff_reg_weight * (
            source_coeff.pow(2).mean() + target_coeff.pow(2).mean()
        )
        source_bpr = _edge_bpr_structure_loss(
            source_real,
            source_struct_edges,
            num_samples=structure_bpr_samples,
            margin=structure_bpr_margin,
        )
        target_bpr = _edge_bpr_structure_loss(
            target_real,
            target_struct_edges,
            num_samples=structure_bpr_samples,
            margin=structure_bpr_margin,
        )
        structure_loss = structure_bpr_weight * (source_bpr + target_bpr)
        spectral_component_loss = spectral_component_weight * _spectral_component_loss(
            source_real,
            target_real,
            topk=spectral_component_topk,
        )
        spectral_dirichlet_loss = spectral_dirichlet_weight * torch.abs(
            _dirichlet_energy(source_real, source_data.edge_index)
            - _dirichlet_energy(target_real, target_data.edge_index)
        )

        total_loss = (
            probe_mmd_weight * probe_mmd
            + source_semantic_weight * semantic_loss
            + filter_alignment_weight * filter_alignment_loss
            + coeff_reg
            + structure_loss
            + spectral_component_loss
            + spectral_dirichlet_loss
        )
        total_loss.backward()
        filter_optimizer.step()

        if epoch == 1 or epoch % align_log_interval == 0 or epoch == align_epochs:
            with torch.no_grad():
                source_coeff_now = _effective_filter(source_filter_logits, filter_temperature)
                target_coeff_now = _effective_filter(target_filter_logits, filter_temperature)
                source_real_now = _apply_monomial_once(
                    mono_prop,
                    source_x,
                    source_data.edge_index,
                    source_coeff_now,
                    lambda_max,
                )
                target_real_now = _apply_monomial_once(
                    mono_prop,
                    target_x,
                    target_data.edge_index,
                    target_coeff_now,
                    lambda_max,
                )
                metric_sample_size = int(_cfg_get(cfg, "metric_sample_size", 0))
                idx_s_log = sample_idx(source_real_now.size(0), metric_sample_size, device)
                idx_t_log = sample_idx(target_real_now.size(0), metric_sample_size, device)
                mmd_total, mmd_norm, mmd_angle = _compute_mmd_safe(
                    source_real_now[idx_s_log],
                    target_real_now[idx_t_log],
                    cfg,
                )
                source_sem_acc = float(
                    (
                        semantic_head(source_real_now)[source_train_mask].argmax(dim=1)
                        == source_y[source_train_mask]
                    )
                    .float()
                    .mean()
                    .item()
                )

                row = {
                    "epoch": float(epoch),
                    "regime": "mono_adv_aligned" if adversarial else "mono_aligned",
                    "probe_mmd": float(probe_mmd.detach().cpu().item()),
                    "semantic_loss": float(semantic_loss.detach().cpu().item()),
                    "filter_alignment_loss": float(filter_alignment_loss.detach().cpu().item()),
                    "structure_loss": float(structure_loss.detach().cpu().item()),
                    "spectral_component_loss": float(spectral_component_loss.detach().cpu().item()),
                    "spectral_dirichlet_loss": float(spectral_dirichlet_loss.detach().cpu().item()),
                    "source_semantic_acc": source_sem_acc,
                    "real_mmd2_total": float(mmd_total),
                    "real_mmd2_norm": float(mmd_norm),
                    "real_mmd2_angle": float(mmd_angle),
                    "total_loss": float(total_loss.detach().cpu().item()),
                    "adv_probe_mmd": float(adv_probe_mmd_val),
                    "adv_objective": float(adv_objective_val),
                    "adv_reg": float(adv_reg_val),
                    "alignment_metric": alignment_metric,
                }

                if adversarial and source_dist is not None and target_dist is not None:
                    source_diag = source_dist.diagnostics()
                    target_diag = target_dist.diagnostics()
                    row["source_adv_shift_l2"] = float(source_diag["shift_l2"])
                    row["source_adv_scale_mean"] = float(source_diag["scale_mean"])
                    row["target_adv_shift_l2"] = float(target_diag["shift_l2"])
                    row["target_adv_scale_mean"] = float(target_diag["scale_mean"])
                else:
                    row["source_adv_shift_l2"] = 0.0
                    row["source_adv_scale_mean"] = 0.0
                    row["target_adv_shift_l2"] = 0.0
                    row["target_adv_scale_mean"] = 0.0

                history_rows.append(row)

    with torch.no_grad():
        final_source_coeff = _effective_filter(source_filter_logits, filter_temperature).detach()
        final_target_coeff = _effective_filter(target_filter_logits, filter_temperature).detach()

    return {
        "regime": "mono_adv_aligned" if adversarial else "mono_aligned",
        "source_coeff": final_source_coeff,
        "target_coeff": final_target_coeff,
        "history": history_rows,
    }


def _evaluate_case_layers(
    *,
    case_to_layers: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]],
    source_y: torch.Tensor,
    target_y: torch.Tensor,
    source_train_mask: torch.Tensor,
    cfg,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    max_layers = int(_cfg_get(cfg, "max_layers", 10))
    num_trials = int(_cfg_get(cfg, "num_trials", 5))

    shift_rows = []
    transfer_rows = []

    for trial in range(num_trials):
        set_seed(int(_cfg_get(cfg, "seed", 0)) + 10000 + trial)
        metric_sample_size = int(_cfg_get(cfg, "metric_sample_size", 0))

        first_case = next(iter(case_to_layers.keys()))
        source_layers, target_layers = case_to_layers[first_case]
        n = (
            min(metric_sample_size, source_layers[0].size(0), target_layers[0].size(0))
            if metric_sample_size > 0
            else min(source_layers[0].size(0), target_layers[0].size(0))
        )
        idx_s = sample_idx(source_layers[0].size(0), n, source_layers[0].device)
        idx_t = sample_idx(target_layers[0].size(0), n, target_layers[0].device)

        for case, (src_layers, tgt_layers) in case_to_layers.items():
            for layer in range(1, max_layers + 1):
                source_feat = src_layers[layer].detach()
                target_feat = tgt_layers[layer].detach()

                mmd_total, mmd_norm, mmd_angle = _compute_mmd_safe(
                    source_feat[idx_s],
                    target_feat[idx_t],
                    cfg,
                )
                shift_rows.append(
                    {
                        "trial": float(trial),
                        "case": case,
                        "layer": float(layer),
                        "mmd2_total": float(mmd_total),
                        "mmd2_norm": float(mmd_norm),
                        "mmd2_angle": float(mmd_angle),
                    }
                )

                transfer_metrics = train_eval_transfer_once(
                    source_feat=source_feat,
                    target_feat=target_feat,
                    source_y=source_y,
                    target_y=target_y,
                    source_train_mask=source_train_mask,
                    cfg=cfg,
                    seed=int(_cfg_get(cfg, "seed", 0)) + 50000 + 1000 * layer + 100 * trial,
                )
                src_acc = float(transfer_metrics["source_acc"])
                tgt_acc = float(transfer_metrics["target_acc"])
                transfer_rows.append(
                    {
                        "trial": float(trial),
                        "case": case,
                        "layer": float(layer),
                        "source_micro_f1": src_acc,
                        "target_micro_f1": tgt_acc,
                        "delta_micro_f1": src_acc - tgt_acc,
                    }
                )

    shift_summary = _aggregate_case_layer(
        shift_rows,
        max_layers=max_layers,
        metric_keys=["mmd2_total", "mmd2_norm", "mmd2_angle"],
    )
    transfer_summary = _aggregate_case_layer(
        transfer_rows,
        max_layers=max_layers,
        metric_keys=["source_micro_f1", "target_micro_f1", "delta_micro_f1"],
    )
    return shift_rows, shift_summary, transfer_rows, transfer_summary


def _build_coeff_rows(nonadv_state: dict, adv_state: dict) -> list[dict]:
    rows = []
    for state in (nonadv_state, adv_state):
        regime = str(state["regime"])
        source_coeff = state["source_coeff"].detach().cpu().view(-1)
        target_coeff = state["target_coeff"].detach().cpu().view(-1)
        for idx, val in enumerate(source_coeff.tolist()):
            rows.append(
                {
                    "regime": regime,
                    "domain": "source",
                    "term": float(idx),
                    "coeff": float(val),
                }
            )
        for idx, val in enumerate(target_coeff.tolist()):
            rows.append(
                {
                    "regime": regime,
                    "domain": "target",
                    "term": float(idx),
                    "coeff": float(val),
                }
            )
    return rows


def run_case(config):
    set_seed(int(_cfg_get(config, "seed", 0)))
    source_data, target_data = load_pair(
        config.dataset,
        config.source,
        config.target,
        config.device,
        config.seed,
    )

    source_x = source_data.x.float()
    target_x = target_data.x.float()
    source_y, target_y = _remap_labels_contiguous(
        source_data.y.view(-1).long(),
        target_data.y.view(-1).long(),
    )

    w = _try_fit_joint_lda(
        source_x,
        target_x,
        source_y,
        target_y,
        _cfg_get(config, "lda_dim", 32),
        _cfg_get(config, "lda_eps", 1e-6),
    )
    if w is not None:
        source_x = source_x @ w
        target_x = target_x @ w

    source_train_mask = get_source_train_mask(source_data).to(source_x.device)

    nonadv_state = _train_monomial_aligner(
        source_data=source_data,
        target_data=target_data,
        source_x=source_x,
        target_x=target_x,
        source_y=source_y,
        target_y=target_y,
        source_train_mask=source_train_mask,
        cfg=config,
        adversarial=False,
        seed_offset=2000,
    )
    adv_state = _train_monomial_aligner(
        source_data=source_data,
        target_data=target_data,
        source_x=source_x,
        target_x=target_x,
        source_y=source_y,
        target_y=target_y,
        source_train_mask=source_train_mask,
        cfg=config,
        adversarial=True,
        seed_offset=4000,
    )

    max_layers = int(_cfg_get(config, "max_layers", 10))
    lambda_max = float(_cfg_get(config, "mono_lambda_max", 2.0))

    adj_prop = Propagation().to(source_x.device)
    mono_prop = MonoProp().to(source_x.device)

    case_to_layers = {
        "adjacency": (
            _propagate_adjacency_layers(source_x, source_data.edge_index, max_layers, adj_prop),
            _propagate_adjacency_layers(target_x, target_data.edge_index, max_layers, adj_prop),
        ),
        "mono_aligned": (
            _propagate_monomial_layers(
                source_x,
                source_data.edge_index,
                nonadv_state["source_coeff"],
                max_layers,
                lambda_max,
                mono_prop,
            ),
            _propagate_monomial_layers(
                target_x,
                target_data.edge_index,
                nonadv_state["target_coeff"],
                max_layers,
                lambda_max,
                mono_prop,
            ),
        ),
        "mono_adv_aligned": (
            _propagate_monomial_layers(
                source_x,
                source_data.edge_index,
                adv_state["source_coeff"],
                max_layers,
                lambda_max,
                mono_prop,
            ),
            _propagate_monomial_layers(
                target_x,
                target_data.edge_index,
                adv_state["target_coeff"],
                max_layers,
                lambda_max,
                mono_prop,
            ),
        ),
    }

    shift_rows, shift_summary, transfer_rows, transfer_summary = _evaluate_case_layers(
        case_to_layers=case_to_layers,
        source_y=source_y,
        target_y=target_y,
        source_train_mask=source_train_mask,
        cfg=config,
    )

    base_name = f"{config.source}_{config.target}"
    run_tag = str(_cfg_get(config, "run_tag", "")).strip()
    if run_tag:
        base_name = f"{base_name}__{run_tag}"
    out = Path(config.out_dir) / config.dataset / base_name
    out.mkdir(parents=True, exist_ok=True)

    coeff_rows = _build_coeff_rows(nonadv_state, adv_state)
    nonadv_history = nonadv_state["history"]
    adv_history = adv_state["history"]
    interpretability_rows = []
    if bool(_cfg_get(config, "compute_interpretability", True)):
        interpretability_rows = _induced_interpretability_rows(
            source_data=source_data,
            target_data=target_data,
            nonadv_state=nonadv_state,
            adv_state=adv_state,
            cfg=config,
        )

    save_rows(out / "case_shift_trials.csv", shift_rows)
    save_rows(out / "case_shift_summary.csv", shift_summary)
    save_rows(out / "case_transfer_trials.csv", transfer_rows)
    save_rows(out / "case_transfer_summary.csv", transfer_summary)
    save_rows(out / "filter_coefficients.csv", coeff_rows)
    save_rows(out / "mono_aligned_history.csv", nonadv_history)
    save_rows(out / "mono_adv_aligned_history.csv", adv_history)
    if interpretability_rows:
        save_rows(out / "interpretability_summary.csv", interpretability_rows)

    save_json(out / "motivation_config.json", to_plain_dict(config))
    save_json(out / "case_shift_summary.json", shift_summary)
    save_json(out / "case_transfer_summary.json", transfer_summary)
    if interpretability_rows:
        save_json(out / "interpretability_summary.json", interpretability_rows)

    transfer_plot = out / "case_transfer_accuracy.png"
    if _plot_case_transfer_accuracy(
        transfer_plot,
        transfer_summary,
        title=f"{config.dataset}: {config.source}->{config.target} (3-MLP transfer)",
    ):
        print(f"[saved] {transfer_plot}")

    shift_plot = out / "case_shift_components.png"
    if _plot_case_shift_components(
        shift_plot,
        shift_summary,
        title=f"{config.dataset}: {config.source}->{config.target} (shift tracking)",
    ):
        print(f"[saved] {shift_plot}")

    align_plot = out / "alignment_training_curves.png"
    if _plot_alignment_history(
        align_plot,
        nonadv_history,
        adv_history,
        title=f"{config.dataset}: {config.source}->{config.target} (alignment training)",
    ):
        print(f"[saved] {align_plot}")

    return shift_summary, transfer_summary, out, interpretability_rows, nonadv_history, adv_history


def _cfg_list(cfg, key: str, default):
    value = _cfg_get(cfg, key, default)
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except Exception:
        return [value]


def _find_case_row(rows: list[dict], case: str, layer: int):
    for row in rows:
        if str(row.get("case")) == str(case) and int(row.get("layer", -1)) == int(layer):
            return row
    return None


def _extract_eval_row(
    *,
    shift_summary: list[dict],
    transfer_summary: list[dict],
    interpretability_rows: list[dict],
    adv_history: list[dict],
    case: str,
    max_layers: int,
) -> dict:
    shift_row = _find_case_row(shift_summary, case, max_layers) or {}
    transfer_row = _find_case_row(transfer_summary, case, max_layers) or {}
    adj_row = _find_case_row(transfer_summary, "adjacency", max_layers) or {}
    interp_row = next((r for r in interpretability_rows if str(r.get("case")) == str(case)), {})
    probe_final = float(adv_history[-1]["probe_mmd"]) if adv_history else float("nan")

    out = {
        "case": str(case),
        "layer": float(max_layers),
        "target_micro_f1_mean": float(transfer_row.get("target_micro_f1_mean", float("nan"))),
        "source_micro_f1_mean": float(transfer_row.get("source_micro_f1_mean", float("nan"))),
        "delta_micro_f1_mean": float(transfer_row.get("delta_micro_f1_mean", float("nan"))),
        "mmd2_total_mean": float(shift_row.get("mmd2_total_mean", float("nan"))),
        "mmd2_norm_mean": float(shift_row.get("mmd2_norm_mean", float("nan"))),
        "mmd2_angle_mean": float(shift_row.get("mmd2_angle_mean", float("nan"))),
        "probe_mmd_final": float(probe_final),
        "degree_hist_l1": float(interp_row.get("degree_hist_l1", float("nan"))),
        "motif_l1": float(interp_row.get("motif_l1", float("nan"))),
    }
    if adj_row:
        out["target_micro_f1_delta_vs_adj"] = float(out["target_micro_f1_mean"] - float(adj_row["target_micro_f1_mean"]))
    else:
        out["target_micro_f1_delta_vs_adj"] = float("nan")
    return out


def _apply_overrides(cfg, overrides: dict):
    for key, value in overrides.items():
        setattr(cfg, key, value)


def _aggregate_profile_rows(rows: list[dict], metrics: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["profile"])].append(row)

    out = []
    for profile, prow in grouped.items():
        row = {
            "profile": profile,
            "n_runs": float(len(prow)),
        }
        for key in metrics:
            vals = [float(r[key]) for r in prow if key in r]
            if not vals:
                continue
            tens = torch.tensor(vals, dtype=torch.float32)
            row[f"{key}_mean"] = float(tens.mean().item())
            row[f"{key}_std"] = float(tens.std(unbiased=False).item())
        out.append(row)
    out.sort(key=lambda r: r.get("target_micro_f1_mean_mean", float("-inf")), reverse=True)
    return out


def _write_experiment_summary(
    out_root: Path,
    dataset: str,
    source: str,
    target: str,
    best_params: dict,
    hp_ranked_rows: list[dict],
    ablation_profile_rows: list[dict],
) -> None:
    lines = []
    lines.append(f"Monomial Filter Matching Experiment Summary")
    lines.append(f"Pair: {dataset} {source}->{target}")
    lines.append("")
    lines.append("Best HP configuration:")
    for key in sorted(best_params.keys()):
        lines.append(f"- {key}: {best_params[key]}")
    lines.append("")

    if hp_ranked_rows:
        best = hp_ranked_rows[0]
        lines.append("Best HP result:")
        lines.append(
            f"- score={best.get('score_mean', float('nan')):.4f}, "
            f"target_f1={best.get('target_micro_f1_mean', float('nan')):.4f}, "
            f"mmd2_total={best.get('mmd2_total_mean', float('nan')):.4f}, "
            f"probe_mmd={best.get('probe_mmd_final_mean', float('nan')):.4f}"
        )
        lines.append("")

    lines.append("Ablation ranking (by target_micro_f1):")
    for row in ablation_profile_rows:
        lines.append(
            f"- {row.get('profile')}: "
            f"target_f1={row.get('target_micro_f1_mean_mean', float('nan')):.4f}, "
            f"delta_vs_adj={row.get('target_micro_f1_delta_vs_adj_mean', float('nan')):.4f}, "
            f"mmd2_total={row.get('mmd2_total_mean_mean', float('nan')):.4f}, "
            f"degree_l1={row.get('degree_hist_l1_mean', float('nan')):.4f}, "
            f"motif_l1={row.get('motif_l1_mean', float('nan')):.4f}"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "- Lower probe_mmd/mmd2_total with higher target_f1 and lower degree_l1/motif_l1 "
        "indicates better filter pushforward alignment and induced-structure matching."
    )

    summary_path = out_root / "expt_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_hp_and_ablation(config):
    out_root = Path(config.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    dataset = str(_cfg_get(config, "search_dataset", "citation"))
    source = str(_cfg_get(config, "search_source", "ACMv9"))
    target = str(_cfg_get(config, "search_target", "DBLPv7"))
    max_layers = int(_cfg_get(config, "max_layers", 10))
    objective_case = str(_cfg_get(config, "objective_case", "mono_adv_aligned"))
    base_seed = int(_cfg_get(config, "seed", 0))

    hp_space = {
        "alignment_metric": _cfg_list(config, "hp_alignment_metrics", ["mmd", "sinkhorn"]),
        "probe_mmd_weight": _cfg_list(config, "hp_probe_mmd_weights", [1.0]),
        "source_semantic_weight": _cfg_list(config, "hp_source_semantic_weights", [1.0]),
        "filter_alignment_weight": _cfg_list(config, "hp_filter_alignment_weights", [0.1]),
        "structure_bpr_weight": _cfg_list(config, "hp_structure_bpr_weights", [0.0, 1e-4]),
        "spectral_component_weight": _cfg_list(config, "hp_spectral_component_weights", [0.0, 0.05]),
        "spectral_dirichlet_weight": _cfg_list(config, "hp_spectral_dirichlet_weights", [0.0, 0.05]),
    }
    hp_seed_offsets = [int(v) for v in _cfg_list(config, "hp_seed_offsets", [0])]
    hp_max_trials = int(_cfg_get(config, "hp_max_trials", 8))
    hp_mmd_penalty = float(_cfg_get(config, "hp_mmd_penalty", 0.2))
    hp_probe_penalty = float(_cfg_get(config, "hp_probe_penalty", 0.05))

    hp_keys = list(hp_space.keys())
    candidates = [dict(zip(hp_keys, vals)) for vals in product(*[hp_space[k] for k in hp_keys])]
    rng = random.Random(base_seed + 1729)
    rng.shuffle(candidates)
    target_trials = max(1, min(hp_max_trials, len(candidates)))
    if target_trials < len(candidates):
        by_metric = defaultdict(list)
        for cand in candidates:
            by_metric[str(cand.get("alignment_metric", "mmd")).lower()].append(cand)
        selected = []
        for metric in sorted(by_metric.keys()):
            if by_metric[metric]:
                selected.append(by_metric[metric].pop(0))
        remaining = [cand for group in by_metric.values() for cand in group]
        rng.shuffle(remaining)
        need = max(0, target_trials - len(selected))
        selected.extend(remaining[:need])
        candidates = selected[:target_trials]
    else:
        candidates = candidates[:target_trials]

    hp_trial_rows = []
    hp_ranked_rows = []
    for cid, candidate in enumerate(candidates):
        candidate_rows = []
        print(f"[hp] trial {cid + 1}/{len(candidates)} params={candidate}")
        for seed_offset in hp_seed_offsets:
            scfg = config.copy()
            scfg.dataset = dataset
            scfg.source = source
            scfg.target = target
            scfg.seed = base_seed + int(seed_offset)
            scfg.compute_interpretability = False
            scfg.run_tag = f"hp_{cid:03d}_seed{seed_offset}"
            _apply_overrides(scfg, candidate)
            shift_summary, transfer_summary, _, interpretability_rows, _, adv_history = run_case(scfg)
            eval_row = _extract_eval_row(
                shift_summary=shift_summary,
                transfer_summary=transfer_summary,
                interpretability_rows=interpretability_rows,
                adv_history=adv_history,
                case=objective_case,
                max_layers=max_layers,
            )
            score = (
                float(eval_row["target_micro_f1_mean"])
                - hp_mmd_penalty * float(eval_row["mmd2_total_mean"])
                - hp_probe_penalty * float(eval_row["probe_mmd_final"])
            )
            trial_row = {
                "trial_id": float(cid),
                "seed_offset": float(seed_offset),
                "score": float(score),
                **candidate,
                **eval_row,
            }
            hp_trial_rows.append(trial_row)
            candidate_rows.append(trial_row)

        candidate_scores = torch.tensor([float(r["score"]) for r in candidate_rows], dtype=torch.float32)
        candidate_target = torch.tensor([float(r["target_micro_f1_mean"]) for r in candidate_rows], dtype=torch.float32)
        candidate_mmd = torch.tensor([float(r["mmd2_total_mean"]) for r in candidate_rows], dtype=torch.float32)
        candidate_probe = torch.tensor([float(r["probe_mmd_final"]) for r in candidate_rows], dtype=torch.float32)
        hp_ranked_rows.append(
            {
                "trial_id": float(cid),
                **candidate,
                "score_mean": float(candidate_scores.mean().item()),
                "score_std": float(candidate_scores.std(unbiased=False).item()),
                "target_micro_f1_mean": float(candidate_target.mean().item()),
                "target_micro_f1_std": float(candidate_target.std(unbiased=False).item()),
                "mmd2_total_mean": float(candidate_mmd.mean().item()),
                "mmd2_total_std": float(candidate_mmd.std(unbiased=False).item()),
                "probe_mmd_final_mean": float(candidate_probe.mean().item()),
                "probe_mmd_final_std": float(candidate_probe.std(unbiased=False).item()),
            }
        )

    hp_ranked_rows.sort(key=lambda r: float(r.get("score_mean", float("-inf"))), reverse=True)
    save_rows(out_root / "hp_trials.csv", hp_trial_rows)
    save_rows(out_root / "hp_ranked.csv", hp_ranked_rows)

    if not hp_ranked_rows:
        raise RuntimeError("HP search produced no rows.")
    best_row = hp_ranked_rows[0]
    best_params = {k: best_row[k] for k in hp_keys}
    print(f"[hp] best params: {best_params}")

    ablation_seed_offsets = [int(v) for v in _cfg_list(config, "ablation_seed_offsets", [0])]
    interpret_profiles = set(
        str(v) for v in _cfg_list(
            config,
            "ablation_interpretability_profiles",
            ["best_full", "alignment_mmd", "alignment_sinkhorn"],
        )
    )
    best_alignment = str(best_params["alignment_metric"]).lower()
    alt_alignment = "sinkhorn" if best_alignment == "mmd" else "mmd"
    ablation_override_map = {
        "best_full": {},
        "alignment_mmd": {"alignment_metric": "mmd"},
        "alignment_sinkhorn": {"alignment_metric": "sinkhorn"},
        "no_probe_alignment": {"probe_mmd_weight": 0.0},
        "no_semantic": {"source_semantic_weight": 0.0},
        "no_structure_bpr": {"structure_bpr_weight": 0.0},
        "no_spectral": {"spectral_component_weight": 0.0, "spectral_dirichlet_weight": 0.0},
        "no_filter_match": {"filter_alignment_weight": 0.0},
    }
    requested_profiles = [str(v) for v in _cfg_list(config, "ablation_profiles", list(ablation_override_map.keys()))]
    ablation_profiles = [
        (name, ablation_override_map[name]) for name in requested_profiles if name in ablation_override_map
    ]
    if alt_alignment == best_alignment:
        ablation_profiles = [p for p in ablation_profiles if p[0] not in {"alignment_mmd", "alignment_sinkhorn"}]

    ablation_rows = []
    for profile_name, overrides in ablation_profiles:
        print(f"[ablation] profile={profile_name} overrides={overrides}")
        for seed_offset in ablation_seed_offsets:
            scfg = config.copy()
            scfg.dataset = dataset
            scfg.source = source
            scfg.target = target
            scfg.seed = base_seed + int(seed_offset)
            scfg.compute_interpretability = bool(profile_name in interpret_profiles)
            scfg.run_tag = f"ablation_{profile_name}_seed{seed_offset}"
            _apply_overrides(scfg, best_params)
            _apply_overrides(scfg, overrides)

            shift_summary, transfer_summary, _, interpretability_rows, _, adv_history = run_case(scfg)
            eval_row = _extract_eval_row(
                shift_summary=shift_summary,
                transfer_summary=transfer_summary,
                interpretability_rows=interpretability_rows,
                adv_history=adv_history,
                case=objective_case,
                max_layers=max_layers,
            )
            ablation_rows.append(
                {
                    "profile": profile_name,
                    "seed_offset": float(seed_offset),
                    **best_params,
                    **overrides,
                    **eval_row,
                }
            )

    save_rows(out_root / "ablation_runs.csv", ablation_rows)
    ablation_profile_rows = _aggregate_profile_rows(
        ablation_rows,
        metrics=[
            "target_micro_f1_mean",
            "target_micro_f1_delta_vs_adj",
            "mmd2_total_mean",
            "probe_mmd_final",
            "degree_hist_l1",
            "motif_l1",
        ],
    )
    save_rows(out_root / "ablation_profiles.csv", ablation_profile_rows)

    _write_experiment_summary(
        out_root=out_root,
        dataset=dataset,
        source=source,
        target=target,
        best_params=best_params,
        hp_ranked_rows=hp_ranked_rows,
        ablation_profile_rows=ablation_profile_rows,
    )

    save_json(out_root / "best_params.json", best_params)
    save_json(out_root / "hp_ranked.json", hp_ranked_rows)
    save_json(out_root / "ablation_profiles.json", ablation_profile_rows)

    print(f"[saved] {out_root / 'expt_summary.txt'}")


def run_all_scenarios(scenarios, config):
    out_root = Path(config.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    save_json(out_root / "motivation_config.json", to_plain_dict(config))
    save_json(out_root / "scenarios.json", scenarios)

    final_rows = []
    failed_rows = []

    all_shift_summary = []
    all_transfer_summary = []
    dataset_to_shift = {dataset: [] for dataset in scenarios.keys()}
    dataset_to_transfer = {dataset: [] for dataset in scenarios.keys()}

    max_layers = int(_cfg_get(config, "max_layers", 10))

    for dataset, pairs in scenarios.items():
        for source, target in pairs:
            print(f"Processing pair: {dataset} - {source} -> {target}")
            scfg = config.copy()
            scfg.dataset = dataset
            scfg.source = source
            scfg.target = target
            try:
                shift_summary, transfer_summary, out, interpretability_rows, _, _ = run_case(scfg)
                if shift_summary:
                    all_shift_summary.append(shift_summary)
                    dataset_to_shift[dataset].append(shift_summary)
                if transfer_summary:
                    all_transfer_summary.append(transfer_summary)
                    dataset_to_transfer[dataset].append(transfer_summary)
                interp_map = {
                    str(r["case"]): r for r in interpretability_rows
                }

                for case in sorted({str(r["case"]) for r in transfer_summary}):
                    transfer_row = next(
                        (
                            r
                            for r in transfer_summary
                            if str(r["case"]) == case and int(r["layer"]) == max_layers
                        ),
                        None,
                    )
                    shift_row = next(
                        (
                            r
                            for r in shift_summary
                            if str(r["case"]) == case and int(r["layer"]) == max_layers
                        ),
                        None,
                    )
                    if transfer_row is None or shift_row is None:
                        continue
                    interp_row = interp_map.get(case, {})
                    final_rows.append(
                        {
                            "dataset": dataset,
                            "source": source,
                            "target": target,
                            "case": case,
                            "final_layer": int(transfer_row["layer"]),
                            "target_micro_f1_mean": float(transfer_row["target_micro_f1_mean"]),
                            "source_micro_f1_mean": float(transfer_row["source_micro_f1_mean"]),
                            "delta_micro_f1_mean": float(transfer_row["delta_micro_f1_mean"]),
                            "mmd2_total_mean": float(shift_row["mmd2_total_mean"]),
                            "mmd2_norm_mean": float(shift_row["mmd2_norm_mean"]),
                            "mmd2_angle_mean": float(shift_row["mmd2_angle_mean"]),
                            "degree_hist_l1": float(interp_row.get("degree_hist_l1", 0.0)),
                            "motif_l1": float(interp_row.get("motif_l1", 0.0)),
                        }
                    )
                print(f"[saved] {out}")
            except Exception as exc:
                failed_rows.append(
                    {
                        "dataset": dataset,
                        "source": source,
                        "target": target,
                        "error": str(exc),
                    }
                )
                print(f"[failed] {dataset} {source}->{target}: {exc}")

    save_rows(out_root / "summary_final_layer.csv", final_rows)

    global_shift_avg = _average_case_layer_summaries(
        all_shift_summary,
        max_layers,
        SHIFT_CASE_SUMMARY_KEYS,
    )
    global_transfer_avg = _average_case_layer_summaries(
        all_transfer_summary,
        max_layers,
        TRANSFER_CASE_SUMMARY_KEYS,
    )
    save_rows(out_root / "global_case_shift_layers.csv", global_shift_avg)
    save_rows(out_root / "global_case_transfer_layers.csv", global_transfer_avg)

    global_transfer_plot = out_root / "global_case_transfer_accuracy.png"
    if _plot_case_transfer_accuracy(
        global_transfer_plot,
        global_transfer_avg,
        title="Global average transfer accuracy (3-MLP)",
    ):
        print(f"[saved] {global_transfer_plot}")

    global_shift_plot = out_root / "global_case_shift_components.png"
    if _plot_case_shift_components(
        global_shift_plot,
        global_shift_avg,
        title="Global average shift components",
    ):
        print(f"[saved] {global_shift_plot}")

    for dataset in scenarios.keys():
        ddir = out_root / dataset
        ddir.mkdir(parents=True, exist_ok=True)

        dataset_shift_avg = _average_case_layer_summaries(
            dataset_to_shift[dataset],
            max_layers,
            SHIFT_CASE_SUMMARY_KEYS,
        )
        dataset_transfer_avg = _average_case_layer_summaries(
            dataset_to_transfer[dataset],
            max_layers,
            TRANSFER_CASE_SUMMARY_KEYS,
        )
        save_rows(ddir / "dataset_case_shift_layers.csv", dataset_shift_avg)
        save_rows(ddir / "dataset_case_transfer_layers.csv", dataset_transfer_avg)

        dataset_transfer_plot = ddir / "dataset_case_transfer_accuracy.png"
        if _plot_case_transfer_accuracy(
            dataset_transfer_plot,
            dataset_transfer_avg,
            title=f"{dataset}: average transfer accuracy (3-MLP)",
        ):
            print(f"[saved] {dataset_transfer_plot}")

        dataset_shift_plot = ddir / "dataset_case_shift_components.png"
        if _plot_case_shift_components(
            dataset_shift_plot,
            dataset_shift_avg,
            title=f"{dataset}: average shift components",
        ):
            print(f"[saved] {dataset_shift_plot}")

    failed_path = out_root / "failed_scenarios.csv"
    if failed_rows:
        save_rows(
            failed_path,
            failed_rows,
            fieldnames=["dataset", "source", "target", "error"],
        )
    elif failed_path.exists():
        failed_path.unlink()


if __name__ == "__main__":
    config_path = Path("configs/ablation_configs/motivation.yaml")
    config = load_config(config_path)
    config.device = _resolve_device(config.device)
    config.out_dir = _timestamped_out_dir(config.out_dir)
    run_mode = str(_cfg_get(config, "run_mode", "hp_ablation")).lower()

    if run_mode == "full_scenarios":
        run_all_scenarios(_build_scenarios(config.transfer_settings), config)
    elif run_mode == "single_case":
        run_case(config)
    else:
        run_hp_and_ablation(config)
    print(f"Saved -> {config.out_dir}")
