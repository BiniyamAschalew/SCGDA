"""Motivation figure: propagation-induced shift and transferability degradation.

Produces a single PDF with three side-by-side subplots averaged over all transfer
scenarios:
  (a) MMD between source and target features A^k X across layers
  (b) KL divergence of PCA-2D projected features across layers
  (c) Transfer accuracy for adjacency / MMD-aligned / adversarial-aligned filters

Also writes a description_{id}.txt file with full reproducibility details.
"""

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import torch

from Learn.Clean_SCGDA.models.__filters.mono import MonoProp
from Learn.Clean_SCGDA.utils.ablation_utils.common import load_pair, sample_idx
from Learn.Clean_SCGDA.utils.ablation_utils.mono_filter_utils import (
    cfg_get,
    resolve_device,
    remap_labels_contiguous,
    joint_pca_projection,
    compute_mmd_decomposition,
    propagate_adjacency_layers,
    propagate_monomial_layers,
    train_monomial_aligner,
)
from Learn.Clean_SCGDA.utils.ablation_utils.motivation_reporting import save_json, save_rows, to_plain_dict
from Learn.Clean_SCGDA.utils.ablation_utils.propagation import Propagation
from Learn.Clean_SCGDA.utils.ablation_utils.transfer import get_source_train_mask, train_eval_transfer_once
from Learn.Clean_SCGDA.utils.config_utils import load_config
from Learn.Clean_SCGDA.utils.expt_utils import set_seed


# ---------------------------------------------------------------------------
# Default scenarios & config
# ---------------------------------------------------------------------------

DEFAULT_SCENARIOS = {
    "citation": [("ACMv9", "DBLPv7"), ("DBLPv7", "ACMv9"), ("ACMv9", "Citationv1")],
    "airport": [("USA", "BRAZIL"), ("BRAZIL", "EUROPE"), ("EUROPE", "USA")],
}


def _flatten_scenarios(raw: dict) -> dict:
    """Convert {dataset: {src: [tgts]}} to {dataset: [(src, tgt)]}."""
    out = {}
    for dataset, pairs in raw.items():
        if isinstance(pairs, dict):
            out[dataset] = [(s, t) for s, targets in pairs.items() for t in targets]
        else:
            out[dataset] = list(pairs)
    return out

DEFAULT_CFG_OVERRIDES = dict(
    seed=2026,
    device="cuda:0",
    max_layers=6,
    num_trials=3,
    metric_sample_size=2048,
    kernel_mul=2.0,
    kernel_num=5,
    fix_sigma=None,
    mono_degree=3,
    mono_lambda_max=2.0,
    mono_filter_temperature=1.0,
    mono_init="uniform",
    align_epochs=40,
    align_log_interval=10,
    align_sample_size=1024,
    align_lr=1e-2,
    align_weight_decay=0.0,
    alignment_metric="mmd",
    probe_mmd_weight=1.0,
    source_semantic_weight=1.0,
    filter_alignment_weight=0.1,
    adv_steps=1,
    adv_lr=3e-3,
    adv_weight_decay=0.0,
    adv_distribution_reg_weight=1e-3,
    mlp_hid_dim=64,
    mlp_layers=2,
    mlp_dropout=0.0,
    mlp_lr=1e-3,
    mlp_weight_decay=5e-4,
    mlp_epochs=150,
    pca_dim=32,
)


# ---------------------------------------------------------------------------
# PCA + KL divergence
# ---------------------------------------------------------------------------

def _pca_project(source: torch.Tensor, target: torch.Tensor, n_components: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Project source and target to `n_components` dims via joint PCA."""
    combined = torch.cat([source, target], dim=0).detach().cpu().float()
    mean = combined.mean(dim=0, keepdim=True)
    centered = combined - mean
    _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
    proj = Vt[:n_components].T  # (D, n_components)
    s_proj = (source.detach().cpu().float() - mean) @ proj
    t_proj = (target.detach().cpu().float() - mean) @ proj
    return s_proj, t_proj


def _marginal_kl(p: np.ndarray, q: np.ndarray, n_bins: int = 80) -> float:
    """Estimate 1-D KL(p||q) via histogram binning."""
    lo = min(p.min(), q.min())
    hi = max(p.max(), q.max())
    pad = max((hi - lo) * 0.05, 1e-6)
    lo -= pad; hi += pad
    p_hist = np.histogram(p, bins=n_bins, range=(lo, hi))[0].astype(np.float64) + 1e-10
    q_hist = np.histogram(q, bins=n_bins, range=(lo, hi))[0].astype(np.float64) + 1e-10
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return float((p_hist * np.log(p_hist / q_hist)).sum())


def _pca_marginal_kl_sum(source: torch.Tensor, target: torch.Tensor, n_components: int = 3, n_bins: int = 80) -> float:
    """Sum of marginal KL divergences along top PCA components.

    By the chain rule for KL divergence, sum_i KL(p_i || q_i) <= KL(p || q),
    making the sum of marginals a lower bound on the joint KL.  Using the top
    principal components ensures the most informative directions are measured.
    """
    s_proj, t_proj = _pca_project(source, target, n_components)
    s_np = s_proj.numpy()
    t_np = t_proj.numpy()
    total = 0.0
    for d in range(n_components):
        total += _marginal_kl(s_np[:, d], t_np[:, d], n_bins)
    return total


def compute_kl_per_layer(
    source_layers: list[torch.Tensor],
    target_layers: list[torch.Tensor],
    max_layers: int,
    n_components: int = 3,
) -> list[float]:
    """Sum-of-marginal KL across top PCA components for each layer 1..max_layers."""
    kls = []
    for layer in range(1, max_layers + 1):
        kls.append(_pca_marginal_kl_sum(source_layers[layer], target_layers[layer], n_components))
    return kls


# ---------------------------------------------------------------------------
# Single scenario evaluation
# ---------------------------------------------------------------------------

def run_scenario(cfg, dataset: str, source: str, target: str) -> dict:
    """Run a single source->target scenario and return per-layer metrics."""
    set_seed(int(cfg_get(cfg, "seed", 0)))
    device = str(cfg_get(cfg, "device", "cpu"))
    source_data, target_data = load_pair(dataset, source, target, device, int(cfg_get(cfg, "seed", 0)))

    source_x = source_data.x.float()
    target_x = target_data.x.float()
    source_y, target_y = remap_labels_contiguous(
        source_data.y.view(-1).long(), target_data.y.view(-1).long(),
    )

    pca_dim = int(cfg_get(cfg, "pca_dim", 32))
    source_x, target_x = joint_pca_projection(source_x, target_x, pca_dim)

    source_train_mask = get_source_train_mask(source_data).to(device)
    max_layers = int(cfg_get(cfg, "max_layers", 6))
    lambda_max = float(cfg_get(cfg, "mono_lambda_max", 2.0))
    num_trials = int(cfg_get(cfg, "num_trials", 3))

    # Train aligners
    nonadv = train_monomial_aligner(
        source_data=source_data, target_data=target_data,
        source_x=source_x, target_x=target_x,
        source_y=source_y, target_y=target_y,
        source_train_mask=source_train_mask,
        cfg=cfg, adversarial=False, seed_offset=2000,
    )
    adv = train_monomial_aligner(
        source_data=source_data, target_data=target_data,
        source_x=source_x, target_x=target_x,
        source_y=source_y, target_y=target_y,
        source_train_mask=source_train_mask,
        cfg=cfg, adversarial=True, seed_offset=4000,
    )

    # Propagate all cases
    adj_prop = Propagation().to(device)
    mono_prop = MonoProp().to(device)

    cases = {
        "adjacency": (
            propagate_adjacency_layers(source_x, source_data.edge_index, max_layers, adj_prop),
            propagate_adjacency_layers(target_x, target_data.edge_index, max_layers, adj_prop),
        ),
        "mono_aligned": (
            propagate_monomial_layers(source_x, source_data.edge_index, nonadv["source_coeff"], max_layers, lambda_max, mono_prop),
            propagate_monomial_layers(target_x, target_data.edge_index, nonadv["target_coeff"], max_layers, lambda_max, mono_prop),
        ),
        "mono_adv_aligned": (
            propagate_monomial_layers(source_x, source_data.edge_index, adv["source_coeff"], max_layers, lambda_max, mono_prop),
            propagate_monomial_layers(target_x, target_data.edge_index, adv["target_coeff"], max_layers, lambda_max, mono_prop),
        ),
    }

    # MMD per layer per case
    mmd_per_case = {}
    kl_per_case = {}
    transfer_per_case = {}

    metric_sample = int(cfg_get(cfg, "metric_sample_size", 0))

    for case_name, (s_layers, t_layers) in cases.items():
        mmd_layers = []
        for layer in range(1, max_layers + 1):
            sf = s_layers[layer].detach()
            tf = t_layers[layer].detach()
            ids = sample_idx(sf.size(0), metric_sample, sf.device)
            idt = sample_idx(tf.size(0), metric_sample, tf.device)
            total, _, _ = compute_mmd_decomposition(sf[ids], tf[idt], cfg)
            mmd_layers.append(total)
        mmd_per_case[case_name] = mmd_layers

        kl_per_case[case_name] = compute_kl_per_layer(s_layers, t_layers, max_layers)

        # Transfer evaluation over trials
        trial_accs = [[] for _ in range(max_layers)]
        for trial in range(num_trials):
            for layer in range(1, max_layers + 1):
                metrics = train_eval_transfer_once(
                    source_feat=s_layers[layer].detach(),
                    target_feat=t_layers[layer].detach(),
                    source_y=source_y, target_y=target_y,
                    source_train_mask=source_train_mask, cfg=cfg,
                    seed=int(cfg_get(cfg, "seed", 0)) + 50000 + 1000 * layer + 100 * trial,
                )
                trial_accs[layer - 1].append(float(metrics["target_acc"]))
        transfer_per_case[case_name] = {
            "mean": [float(np.mean(a)) for a in trial_accs],
            "std": [float(np.std(a)) for a in trial_accs],
        }

    return {
        "dataset": dataset, "source": source, "target": target,
        "max_layers": max_layers,
        "mmd": mmd_per_case,
        "kl": kl_per_case,
        "transfer": transfer_per_case,
    }


# ---------------------------------------------------------------------------
# Averaging across scenarios
# ---------------------------------------------------------------------------

def average_results(all_results: list[dict]) -> dict:
    """Average per-layer metrics across scenarios."""
    if not all_results:
        raise ValueError("No results to average.")
    max_layers = all_results[0]["max_layers"]
    case_names = list(all_results[0]["mmd"].keys())

    avg = {"max_layers": max_layers}
    for metric in ("mmd", "kl"):
        avg[metric] = {}
        for case in case_names:
            vals = np.array([r[metric][case] for r in all_results])  # (n_scenarios, max_layers)
            avg[metric][case] = {
                "mean": vals.mean(axis=0).tolist(),
                "std": vals.std(axis=0).tolist(),
            }

    avg["transfer"] = {}
    for case in case_names:
        means = np.array([r["transfer"][case]["mean"] for r in all_results])
        avg["transfer"][case] = {
            "mean": means.mean(axis=0).tolist(),
            "std": means.std(axis=0).tolist(),
        }
    return avg


# ---------------------------------------------------------------------------
# Paper-quality plot
# ---------------------------------------------------------------------------

CASE_STYLE = {
    "adjacency":        {"color": "#1f77b4", "marker": "o",  "label": r"$\widetilde{A}^k$ (adjacency)"},
    "mono_aligned":     {"color": "#ff7f0e", "marker": "s",  "label": "MMD-aligned filter"},
    "mono_adv_aligned": {"color": "#2ca02c", "marker": "^",  "label": "Adversarial-aligned filter"},
}


def plot_motivation_figure(avg: dict, out_path: Path) -> None:
    """Create a 1x3 PDF figure for the paper."""
    max_layers = avg["max_layers"]
    layers = list(range(1, max_layers + 1))

    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 11,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "lines.linewidth": 2.2,
        "lines.markersize": 7,
        "figure.dpi": 150,
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) MMD across layers
    ax = axes[0]
    for case, style in CASE_STYLE.items():
        y = np.array(avg["mmd"][case]["mean"])
        s = np.array(avg["mmd"][case]["std"])
        ax.plot(layers, y, marker=style["marker"], color=style["color"], label=style["label"])
        ax.fill_between(layers, y - s, y + s, alpha=0.15, color=style["color"])
    ax.set_xlabel("Propagation layer $k$")
    ax.set_ylabel("MMD$^2$")
    ax.set_title("Feature distribution shift")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best")
    ax.set_xticks(layers)

    # (b) KL divergence (PCA-2D)
    ax = axes[1]
    for case, style in CASE_STYLE.items():
        y = np.array(avg["kl"][case]["mean"])
        s = np.array(avg["kl"][case]["std"])
        ax.plot(layers, y, marker=style["marker"], color=style["color"], label=style["label"])
        ax.fill_between(layers, y - s, y + s, alpha=0.15, color=style["color"])
    ax.set_xlabel("Propagation layer $k$")
    ax.set_ylabel("KL divergence")
    ax.set_title("PCA distribution divergence")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best")
    ax.set_xticks(layers)

    # (c) Transfer accuracy
    ax = axes[2]
    for case, style in CASE_STYLE.items():
        y = np.array(avg["transfer"][case]["mean"])
        s = np.array(avg["transfer"][case]["std"])
        ax.plot(layers, y, marker=style["marker"], color=style["color"], label=style["label"])
        ax.fill_between(layers, y - s, y + s, alpha=0.15, color=style["color"])
    ax.set_xlabel("Propagation layer $k$")
    ax.set_ylabel("Target micro-F1")
    ax.set_title("Transfer accuracy")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best")
    ax.set_xticks(layers)

    fig.tight_layout(w_pad=2.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ---------------------------------------------------------------------------
# Description file
# ---------------------------------------------------------------------------

def write_description(path: Path, cfg, scenarios: dict, avg: dict, run_id: str) -> None:
    lines = [
        f"Motivation experiment: {run_id}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Scenarios:",
    ]
    for ds, pairs in scenarios.items():
        for s, t in pairs:
            lines.append(f"  {ds}: {s} -> {t}")

    lines += [
        "",
        "Settings:",
        f"  seed: {cfg_get(cfg, 'seed', 0)}",
        f"  max_layers: {cfg_get(cfg, 'max_layers', 6)}",
        f"  num_trials (transfer): {cfg_get(cfg, 'num_trials', 3)}",
        f"  mono_degree: {cfg_get(cfg, 'mono_degree', 3)}",
        f"  align_epochs: {cfg_get(cfg, 'align_epochs', 40)}",
        f"  alignment_metric: {cfg_get(cfg, 'alignment_metric', 'mmd')}",
        f"  pca_dim: {cfg_get(cfg, 'pca_dim', 32)}",
        f"  metric_sample_size: {cfg_get(cfg, 'metric_sample_size', 2048)}",
        f"  mlp_epochs: {cfg_get(cfg, 'mlp_epochs', 150)}",
        f"  mlp_hid_dim: {cfg_get(cfg, 'mlp_hid_dim', 64)}",
        f"  device: {cfg_get(cfg, 'device', 'cpu')}",
        "",
        "Cases compared:",
        "  adjacency: standard GCN-normalized propagation (A_tilde^k X)",
        "  mono_aligned: monomial filter aligned via MMD probe-pushforward",
        "  mono_adv_aligned: monomial filter aligned with adversarial probe distribution",
        "",
        "Subplots:",
        "  (a) MMD^2 between source and target features at each layer",
        "  (b) Sum of marginal KL divergences along top-3 PCA components of source/target features",
        "  (c) Target micro-F1 from MLP trained on source features only",
        "",
        "Averaged results (final layer):",
    ]

    max_layers = avg["max_layers"]
    for case in CASE_STYLE:
        mmd_final = avg["mmd"][case]["mean"][-1]
        kl_final = avg["kl"][case]["mean"][-1]
        tf_final = avg["transfer"][case]["mean"][-1]
        lines.append(f"  {case}: MMD={mmd_final:.4f}, KL={kl_final:.4f}, F1={tf_final:.4f}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config_path: str | None = None, scenarios: dict | None = None) -> None:
    if config_path and Path(config_path).exists():
        cfg = load_config(config_path)
    else:
        cfg = OmegaConf.create(DEFAULT_CFG_OVERRIDES)

    cfg.device = resolve_device(str(cfg_get(cfg, "device", "cpu")))

    if scenarios is None:
        raw = cfg_get(cfg, "transfer_settings", DEFAULT_SCENARIOS)
        raw = OmegaConf.to_container(raw, resolve=True) if OmegaConf.is_config(raw) else raw
        scenarios = _flatten_scenarios(raw)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = Path(str(cfg_get(cfg, "out_dir", "./__saved__/results/ablation/motivation03")))
    out_dir = base_out / f"motivation_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(out_dir / "config.json", to_plain_dict(cfg))
    scenarios = to_plain_dict(scenarios)
    save_json(out_dir / "scenarios.json", scenarios)

    all_results = []
    for dataset, pairs in scenarios.items():
        for source, target in pairs:
            print(f"\n=== {dataset}: {source} -> {target} ===")
            try:
                result = run_scenario(cfg, dataset, source, target)
                all_results.append(result)
                save_json(out_dir / f"{dataset}_{source}_{target}.json", result)
            except Exception as exc:
                print(f"[FAILED] {dataset} {source}->{target}: {exc}")

    if not all_results:
        print("No successful scenarios. Exiting.")
        return

    avg = average_results(all_results)
    save_json(out_dir / "averaged_results.json", avg)

    # Save raw data as CSV
    rows = []
    for case in CASE_STYLE:
        for i, layer in enumerate(range(1, avg["max_layers"] + 1)):
            rows.append({
                "case": case, "layer": layer,
                "mmd_mean": avg["mmd"][case]["mean"][i],
                "mmd_std": avg["mmd"][case]["std"][i],
                "kl_mean": avg["kl"][case]["mean"][i],
                "kl_std": avg["kl"][case]["std"][i],
                "transfer_mean": avg["transfer"][case]["mean"][i],
                "transfer_std": avg["transfer"][case]["std"][i],
            })
    save_rows(out_dir / "averaged_layers.csv", rows)

    # Paper-quality figure
    pdf_path = out_dir / f"motivation_{run_id}.pdf"
    plot_motivation_figure(avg, pdf_path)

    # Description file
    desc_path = out_dir / f"description_{run_id}.txt"
    write_description(desc_path, cfg, scenarios, avg, run_id)

    print(f"\nAll outputs -> {out_dir}")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/ablation_configs/motivation.yaml"
    run(config_path)
