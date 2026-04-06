"""Adversarial monomial-filter motivation experiment with layer-wise transfer evaluation."""

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

from Learn.Clean_SCGDA.models.__filters.mono import MonoProp
from Learn.Clean_SCGDA.utils.ablation_utils.common import load_pair, sample_idx
from Learn.Clean_SCGDA.utils.ablation_utils.motivation_reporting import save_json, save_rows, to_plain_dict
from Learn.Clean_SCGDA.utils.ablation_utils.propagation import Propagation
from Learn.Clean_SCGDA.utils.ablation_utils.transfer import get_source_train_mask, train_eval_transfer_once
from Learn.Clean_SCGDA.utils.config_utils import load_config
from Learn.Clean_SCGDA.utils.expt_utils import set_seed
from Learn.Clean_SCGDA.utils.filter_utils import mmd_rbf, median_bandwidth


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


def _remap_labels_contiguous(ys: torch.Tensor, yt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    classes = torch.unique(torch.cat([ys.view(-1), yt.view(-1)], dim=0), sorted=True)
    mapping = {int(c.item()): i for i, c in enumerate(classes)}

    def _remap(y):
        return torch.tensor([mapping[int(v.item())] for v in y.view(-1)], device=y.device, dtype=torch.long)

    return _remap(ys), _remap(yt)


class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.dropout = float(dropout)
        dims = [int(in_dim)]
        if num_layers == 1:
            dims.append(int(out_dim))
        else:
            dims.extend([int(hidden_dim)] * (int(num_layers) - 1))
            dims.append(int(out_dim))
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


def _mlp_project_features(
    source_x: torch.Tensor,
    target_x: torch.Tensor,
    source_y: torch.Tensor,
    source_train_mask: torch.Tensor,
    cfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    in_dim = int(source_x.size(1))
    out_dim = int(
        _cfg_get(
            cfg,
            "proj_dim",
            _cfg_get(cfg, "lda_dim", in_dim),
        )
    )
    if out_dim <= 0:
        out_dim = in_dim
    out_dim = min(out_dim, in_dim)
    if out_dim == in_dim and not bool(_cfg_get(cfg, "force_projection", False)):
        return source_x, target_x

    hidden_dim = int(_cfg_get(cfg, "proj_hidden_dim", max(32, min(256, in_dim))))
    num_layers = int(_cfg_get(cfg, "proj_layers", 2))
    dropout = float(_cfg_get(cfg, "proj_dropout", 0.0))
    epochs = int(_cfg_get(cfg, "proj_epochs", 120))
    lr = float(_cfg_get(cfg, "proj_lr", 1e-3))
    weight_decay = float(_cfg_get(cfg, "proj_weight_decay", 5e-4))

    num_classes = int(source_y.max().item()) + 1
    projector = MLPProjector(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(source_x.device)
    classifier = nn.Linear(out_dim, num_classes).to(source_x.device)
    optimizer = torch.optim.Adam(
        list(projector.parameters()) + list(classifier.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    for _ in range(max(1, epochs)):
        projector.train()
        classifier.train()
        optimizer.zero_grad()
        source_proj = projector(source_x)
        source_logits = classifier(source_proj)
        loss = F.cross_entropy(source_logits[source_train_mask], source_y[source_train_mask])
        loss.backward()
        optimizer.step()

    projector.eval()
    with torch.no_grad():
        source_proj = projector(source_x).detach()
        target_proj = projector(target_x).detach()
    return source_proj, target_proj


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
        for layer in range(0, int(max_layers) + 1):
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
        for layer in range(0, int(max_layers) + 1):
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


def _adjacency_reference_coeffs(num_terms: int) -> list[float]:
    ref = [0.0 for _ in range(max(1, int(num_terms)))]
    ref[1 if int(num_terms) > 1 else 0] = 1.0
    return ref


def _plot_monomial_basis_params_by_term(
    path: Path,
    nonadv_coeff_rows: list[dict],
    adv_coeff_rows: list[dict],
    title: str,
) -> bool:
    if not nonadv_coeff_rows and not adv_coeff_rows:
        return False

    target_nonadv_rows = [r for r in nonadv_coeff_rows if str(r.get("domain", "")) == "target"]
    target_adv_rows = [r for r in adv_coeff_rows if str(r.get("domain", "")) == "target"]
    if not target_nonadv_rows and not target_adv_rows:
        return False

    rows = []
    rows.extend(target_nonadv_rows)
    rows.extend(target_adv_rows)
    terms = sorted({int(r["term"]) for r in rows})
    if not terms:
        return False

    ref = _adjacency_reference_coeffs(max(terms) + 1)
    n_cols = len(terms)

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(3.8 * n_cols, 3.6),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    for col_idx, term in enumerate(terms):
        ax = axes[0][col_idx]
        nonadv_series = [r for r in target_nonadv_rows if int(r["term"]) == term]
        adv_series = [r for r in target_adv_rows if int(r["term"]) == term]
        nonadv_series.sort(key=lambda x: int(x["epoch"]))
        adv_series.sort(key=lambda x: int(x["epoch"]))

        if nonadv_series:
            ax.plot(
                [int(r["epoch"]) for r in nonadv_series],
                [float(r["coeff"]) for r in nonadv_series],
                linewidth=1.8,
                color="tab:orange",
                label="non-adversarial",
            )
        if adv_series:
            ax.plot(
                [int(r["epoch"]) for r in adv_series],
                [float(r["coeff"]) for r in adv_series],
                linewidth=1.8,
                color="tab:green",
                label="adversarial",
            )
        ref_val = ref[term] if term < len(ref) else 0.0
        ax.axhline(
            y=ref_val,
            linestyle="--",
            linewidth=1.3,
            color="tab:blue",
            alpha=0.8,
            label="fixed adjacency ref",
        )
        ax.set_title(f"T{term}")
        if col_idx == 0:
            ax.set_ylabel("target\ncoefficient")
        ax.set_xlabel("Alignment epoch")
        ax.grid(alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for handle, label in zip(handles, labels):
        if label in seen or not label:
            continue
        seen.add(label)
        uniq_handles.append(handle)
        uniq_labels.append(label)
    if uniq_handles:
        fig.legend(uniq_handles, uniq_labels, ncol=min(3, len(uniq_labels)), loc="upper center")
    fig.suptitle(
        f"{title}\nTarget monomial coeffs by term | reference={_adjacency_reference_coeffs(max(terms) + 1)}",
        fontsize=11,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
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
    probe_mmd_weight = float(_cfg_get(cfg, "probe_mmd_weight", 1.0))
    source_semantic_weight = float(_cfg_get(cfg, "source_semantic_weight", 1.0))
    filter_alignment_weight = float(_cfg_get(cfg, "filter_alignment_weight", 0.1))
    filter_coeff_reg_weight = float(_cfg_get(cfg, "filter_coeff_reg_weight", 0.0))
    adv_distribution_reg_weight = float(_cfg_get(cfg, "adv_distribution_reg_weight", 1e-3))
    adv_steps = max(1, int(_cfg_get(cfg, "adv_steps", 1)))

    history_rows = []
    coeff_history_rows = []

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
                adv_probe_mmd = _mmd_tensor(
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
        probe_mmd = _mmd_tensor(source_push[idx_s], target_push[idx_t], cfg)

        source_real = _apply_monomial_once(
            mono_prop,
            source_x,
            source_data.edge_index,
            source_coeff,
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

        total_loss = (
            probe_mmd_weight * probe_mmd
            + source_semantic_weight * semantic_loss
            + filter_alignment_weight * filter_alignment_loss
            + coeff_reg
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
                    "source_semantic_acc": source_sem_acc,
                    "real_mmd2_total": float(mmd_total),
                    "real_mmd2_norm": float(mmd_norm),
                    "real_mmd2_angle": float(mmd_angle),
                    "total_loss": float(total_loss.detach().cpu().item()),
                    "adv_probe_mmd": float(adv_probe_mmd_val),
                    "adv_objective": float(adv_objective_val),
                    "adv_reg": float(adv_reg_val),
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
                for term_idx, val in enumerate(source_coeff_now.tolist()):
                    coeff_history_rows.append(
                        {
                            "epoch": float(epoch),
                            "regime": "mono_adv_aligned" if adversarial else "mono_aligned",
                            "domain": "source",
                            "term": float(term_idx),
                            "coeff": float(val),
                        }
                    )
                for term_idx, val in enumerate(target_coeff_now.tolist()):
                    coeff_history_rows.append(
                        {
                            "epoch": float(epoch),
                            "regime": "mono_adv_aligned" if adversarial else "mono_aligned",
                            "domain": "target",
                            "term": float(term_idx),
                            "coeff": float(val),
                        }
                    )

    with torch.no_grad():
        final_source_coeff = _effective_filter(source_filter_logits, filter_temperature).detach()
        final_target_coeff = _effective_filter(target_filter_logits, filter_temperature).detach()

    return {
        "regime": "mono_adv_aligned" if adversarial else "mono_aligned",
        "source_coeff": final_source_coeff,
        "target_coeff": final_target_coeff,
        "history": history_rows,
        "coeff_history": coeff_history_rows,
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
            for layer in range(0, max_layers + 1):
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
                src_micro = float(transfer_metrics["source_micro_f1"])
                tgt_micro = float(transfer_metrics["target_micro_f1"])
                transfer_rows.append(
                    {
                        "trial": float(trial),
                        "case": case,
                        "layer": float(layer),
                        "source_micro_f1": src_micro,
                        "target_micro_f1": tgt_micro,
                        "delta_micro_f1": src_micro - tgt_micro,
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

    source_train_mask = get_source_train_mask(source_data).to(source_x.device)
    source_x, target_x = _mlp_project_features(
        source_x=source_x,
        target_x=target_x,
        source_y=source_y,
        source_train_mask=source_train_mask,
        cfg=config,
    )

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

    out = Path(config.out_dir) / config.dataset / f"{config.source}_{config.target}"
    out.mkdir(parents=True, exist_ok=True)

    coeff_rows = _build_coeff_rows(nonadv_state, adv_state)
    nonadv_history = nonadv_state["history"]
    adv_history = adv_state["history"]
    nonadv_coeff_history = nonadv_state["coeff_history"]
    adv_coeff_history = adv_state["coeff_history"]

    save_rows(out / "case_shift_trials.csv", shift_rows)
    save_rows(out / "case_shift_summary.csv", shift_summary)
    save_rows(out / "case_transfer_trials.csv", transfer_rows)
    save_rows(out / "case_transfer_summary.csv", transfer_summary)
    save_rows(out / "filter_coefficients.csv", coeff_rows)
    save_rows(out / "mono_aligned_history.csv", nonadv_history)
    save_rows(out / "mono_adv_aligned_history.csv", adv_history)
    save_rows(out / "mono_aligned_coeff_history.csv", nonadv_coeff_history)
    save_rows(out / "mono_adv_aligned_coeff_history.csv", adv_coeff_history)

    save_json(out / "motivation_config.json", to_plain_dict(config))
    save_json(out / "case_shift_summary.json", shift_summary)
    save_json(out / "case_transfer_summary.json", transfer_summary)

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

    adv_coeff_plot = out / "mono_basis_parameter.png"
    if _plot_monomial_basis_params_by_term(
        adv_coeff_plot,
        nonadv_coeff_history,
        adv_coeff_history,
        title=f"{config.dataset}: {config.source}->{config.target}",
    ):
        print(f"[saved] {adv_coeff_plot}")

    return shift_summary, transfer_summary, out


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
                shift_summary, transfer_summary, out = run_case(scfg)
                if shift_summary:
                    all_shift_summary.append(shift_summary)
                    dataset_to_shift[dataset].append(shift_summary)
                if transfer_summary:
                    all_transfer_summary.append(transfer_summary)
                    dataset_to_transfer[dataset].append(transfer_summary)

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

    run_all_scenarios(_build_scenarios(config.transfer_settings), config)
    print(f"Saved -> {config.out_dir}")
