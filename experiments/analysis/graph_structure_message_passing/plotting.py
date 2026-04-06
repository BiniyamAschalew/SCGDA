from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASE_ORDER = ("label_one_hot", "random_noise", "pca_projection")
CASE_LABELS = {
    "label_one_hot": "One-hot labels",
    "random_noise": "Random noise",
    "pca_projection": "PCA projection",
}

METRIC_ORDER = ("ari", "nmi", "aligned_accuracy")
METRIC_LABELS = {
    "ari": "ARI",
    "nmi": "NMI",
    "aligned_accuracy": "Rank-aligned accuracy",
}

PALETTE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
]

CASE_COLORS = {
    "label_one_hot": "#1f77b4",
    "random_noise": "#d62728",
    "pca_projection": "#2ca02c",
}


def _domain_colors(domains: list[str]):
    return {domain: PALETTE[idx % len(PALETTE)] for idx, domain in enumerate(domains)}


def _plot_mean_with_band(
    ax,
    x,
    mean_values,
    std_values,
    *,
    color: str,
    label: str,
):
    ax.plot(x, mean_values, color=color, linewidth=2.0, marker="o", markersize=3.8, label=label)
    if std_values is None:
        return
    lower = mean_values - std_values
    upper = mean_values + std_values
    ax.fill_between(x, lower, upper, color=color, alpha=0.16, linewidth=0)


def plot_domain_metric_grid(
    summary_df,
    *,
    out_dir: str | Path,
    dataset_name: str,
    domains: list[str],
):
    out_dir = Path(out_dir)
    colors = _domain_colors(domains)
    fig, axes = plt.subplots(
        len(CASE_ORDER),
        len(METRIC_ORDER),
        figsize=(13.4, 10.8),
        sharex=True,
        squeeze=False,
    )

    for row_idx, case_name in enumerate(CASE_ORDER):
        case_df = summary_df[summary_df["case"] == case_name]
        for col_idx, metric_name in enumerate(METRIC_ORDER):
            ax = axes[row_idx, col_idx]
            mean_col = f"{metric_name}_mean"
            std_col = f"{metric_name}_std"
            for domain in domains:
                domain_df = case_df[case_df["domain"] == domain].sort_values("step")
                if domain_df.empty:
                    continue
                _plot_mean_with_band(
                    ax,
                    domain_df["step"].to_numpy(),
                    domain_df[mean_col].to_numpy(),
                    domain_df[std_col].to_numpy(),
                    color=colors[domain],
                    label=domain,
                )

            ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.7)
            ax.grid(alpha=0.25)
            ax.set_title(f"{CASE_LABELS[case_name]}: {METRIC_LABELS[metric_name]}")
            if col_idx == 0:
                ax.set_ylabel("Score")
            if row_idx == len(CASE_ORDER) - 1:
                ax.set_xlabel("Propagation step")
            if row_idx == 0 and col_idx == len(METRIC_ORDER) - 1:
                ax.legend(frameon=False, fontsize=9)

    fig.suptitle(f"{dataset_name}: clustering quality over repeated message passing", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / f"{str(dataset_name).lower()}_domain_metric_grid.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_case_summary(
    summary_df,
    *,
    out_dir: str | Path,
    dataset_name: str,
):
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(1, len(METRIC_ORDER), figsize=(11.2, 4.2), sharex=True)
    if len(METRIC_ORDER) == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, METRIC_ORDER):
        mean_col = f"{metric_name}_mean"
        std_col = f"{metric_name}_std"
        for case_name in CASE_ORDER:
            case_df = summary_df[summary_df["case"] == case_name].sort_values("step")
            if case_df.empty:
                continue
            _plot_mean_with_band(
                ax,
                case_df["step"].to_numpy(),
                case_df[mean_col].to_numpy(),
                case_df[std_col].to_numpy(),
                color=CASE_COLORS[case_name],
                label=CASE_LABELS[case_name],
            )

        ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.grid(alpha=0.25)
        ax.set_title(METRIC_LABELS[metric_name])
        ax.set_xlabel("Propagation step")
        ax.set_ylabel("Mean score across domains")

    axes[-1].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{dataset_name}: mean across domains", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / f"{str(dataset_name).lower()}_case_summary.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path
