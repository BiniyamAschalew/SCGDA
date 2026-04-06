from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_PALETTE = {
    "gnn": "#1f77b4",
    "simgda": "#d62728",
    "gnn_oracle": "#2ca02c",
    "simgda_oracle": "#9467bd",
}

MODEL_LABELS = {
    "gnn": "GNN",
    "simgda": "SimGDA",
    "gnn_oracle": "GNN Oracle",
    "simgda_oracle": "SimGDA Oracle",
}

SNAPSHOT_SPECS = [
    ("mmd_after_linear1", "After Linear 1"),
    ("mmd_after_activation1", "After Activation 1"),
    ("mmd_after_mp1", "After MP1"),
    ("mmd_after_linear2", "After Linear 2"),
    ("mmd_after_activation2", "After Activation 2"),
    ("mmd_after_mp2", "After MP2"),
]


def _plot_with_band(ax, x, mean_values, std_values, *, color: str, label: str):
    ax.plot(x, mean_values, color=color, linewidth=2.0, label=label)
    if std_values is not None:
        lower = mean_values - std_values
        upper = mean_values + std_values
        ax.fill_between(x, lower, upper, color=color, alpha=0.15, linewidth=0)


def plot_snapshot_mmd(mean_df, *, out_dir: str | Path, std_df=None):
    out_dir = Path(out_dir)
    epochs = sorted(mean_df["epoch"].unique().tolist())
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.8), sharex=True)

    for ax, (column, title) in zip(axes.reshape(-1), SNAPSHOT_SPECS):
        for model_key, model_label in MODEL_LABELS.items():
            cur_mean = mean_df[mean_df["model_key"] == model_key].sort_values("epoch")
            if cur_mean.empty:
                continue
            cur_std = None
            if std_df is not None:
                cur_std = std_df[std_df["model_key"] == model_key].sort_values("epoch")
            _plot_with_band(
                ax,
                epochs,
                cur_mean[column].to_numpy(),
                None if cur_std is None else cur_std[column].fillna(0.0).to_numpy(),
                color=MODEL_PALETTE[model_key],
                label=model_label,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MMD")
        ax.grid(alpha=0.25)

    handles, labels = axes.reshape(-1)[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = out_dir / "snapshot_mmd_over_epochs.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_performance(mean_df, *, out_dir: str | Path, std_df=None):
    out_dir = Path(out_dir)
    epochs = sorted(mean_df["epoch"].unique().tolist())
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), sharex=True)

    source_ax, target_ax = axes

    for model_key, model_label in MODEL_LABELS.items():
        cur_mean = mean_df[mean_df["model_key"] == model_key].sort_values("epoch")
        if cur_mean.empty:
            continue
        cur_std = None
        if std_df is not None:
            cur_std = std_df[std_df["model_key"] == model_key].sort_values("epoch")
        _plot_with_band(
            source_ax,
            epochs,
            cur_mean["source_micro_f1"].to_numpy(),
            None if cur_std is None else cur_std["source_micro_f1"].fillna(0.0).to_numpy(),
            color=MODEL_PALETTE[model_key],
            label=model_label,
        )
    source_ax.set_title("Source Performance")
    source_ax.set_xlabel("Epoch")
    source_ax.set_ylabel("Micro-F1")
    source_ax.grid(alpha=0.25)
    source_ax.legend(frameon=False, fontsize=9)

    target_gap_ax = target_ax.twinx()
    perf_handles = []
    gap_handles = []
    for model_key, model_label in MODEL_LABELS.items():
        cur_mean = mean_df[mean_df["model_key"] == model_key].sort_values("epoch")
        if cur_mean.empty:
            continue
        cur_std = None
        if std_df is not None:
            cur_std = std_df[std_df["model_key"] == model_key].sort_values("epoch")

        perf_line = target_ax.plot(
            epochs,
            cur_mean["target_micro_f1"].to_numpy(),
            color=MODEL_PALETTE[model_key],
            linewidth=2.0,
            label=f"{model_label} target",
        )[0]
        perf_handles.append(perf_line)
        if cur_std is not None:
            perf_std = cur_std["target_micro_f1"].fillna(0.0).to_numpy()
            target_ax.fill_between(
                epochs,
                cur_mean["target_micro_f1"].to_numpy() - perf_std,
                cur_mean["target_micro_f1"].to_numpy() + perf_std,
                color=MODEL_PALETTE[model_key],
                alpha=0.10,
                linewidth=0,
            )

        gap_line = target_gap_ax.plot(
            epochs,
            cur_mean["dirichlet_energy_gap"].to_numpy(),
            color=MODEL_PALETTE[model_key],
            linewidth=1.8,
            linestyle="--",
            label=f"{model_label} energy gap",
        )[0]
        gap_handles.append(gap_line)
        if cur_std is not None:
            gap_std = cur_std["dirichlet_energy_gap"].fillna(0.0).to_numpy()
            target_gap_ax.fill_between(
                epochs,
                cur_mean["dirichlet_energy_gap"].to_numpy() - gap_std,
                cur_mean["dirichlet_energy_gap"].to_numpy() + gap_std,
                color=MODEL_PALETTE[model_key],
                alpha=0.06,
                linewidth=0,
            )

    target_ax.set_title("Target Performance + Energy Gap")
    target_ax.set_xlabel("Epoch")
    target_ax.set_ylabel("Target Micro-F1")
    target_gap_ax.set_ylabel("Dirichlet Gap (target - source)")
    target_gap_ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle=":", alpha=0.7)
    target_ax.grid(alpha=0.25)
    target_ax.legend(
        perf_handles + gap_handles,
        [line.get_label() for line in perf_handles + gap_handles],
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="upper center",
    )

    fig.tight_layout()
    out_path = out_dir / "performance_over_epochs.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_dirichlet_energy(mean_df, *, out_dir: str | Path, std_df=None):
    out_dir = Path(out_dir)
    epochs = sorted(mean_df["epoch"].unique().tolist())
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), sharex=True)

    specs = [
        ("source_dirichlet_energy", "Source Dirichlet Energy / Node"),
        ("target_dirichlet_energy", "Target Dirichlet Energy / Node"),
    ]
    for ax, (column, title) in zip(axes, specs):
        for model_key, model_label in MODEL_LABELS.items():
            cur_mean = mean_df[mean_df["model_key"] == model_key].sort_values("epoch")
            if cur_mean.empty:
                continue
            cur_std = None
            if std_df is not None:
                cur_std = std_df[std_df["model_key"] == model_key].sort_values("epoch")
            _plot_with_band(
                ax,
                epochs,
                cur_mean[column].to_numpy(),
                None if cur_std is None else cur_std[column].fillna(0.0).to_numpy(),
                color=MODEL_PALETTE[model_key],
                label=model_label,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Energy / node")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    out_path = out_dir / "dirichlet_energy_over_epochs.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path
