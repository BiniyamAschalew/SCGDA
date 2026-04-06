from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_PALETTE = {
    "layernorm_gnn": "#1f77b4",
    "layernorm_mlp": "#d62728",
}

MODEL_LABELS = {
    "layernorm_gnn": "LayerNorm GNN",
    "layernorm_mlp": "LayerNorm MLP",
}


def _plot_with_band(ax, x, mean_values, std_values, *, color: str, label: str):
    ax.plot(x, mean_values, color=color, linewidth=2.0, label=label)
    if std_values is not None:
        lower = mean_values - std_values
        upper = mean_values + std_values
        ax.fill_between(x, lower, upper, color=color, alpha=0.14, linewidth=0)


def _snapshot_palette(num_snapshots: int):
    cmap = plt.get_cmap("tab20")
    return [cmap(idx % cmap.N) for idx in range(int(num_snapshots))]


def plot_training_overview(mean_df, *, out_dir: str | Path, std_df=None):
    out_dir = Path(out_dir)
    epochs = sorted(mean_df["epoch"].unique().tolist())
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.8), sharex=True)

    specs = [
        ("source_micro_f1", "Source Micro-F1"),
        ("target_micro_f1", "Target Micro-F1"),
        ("mmd_final", "Final Embedding MMD"),
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
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Value")
    axes[0].legend(frameon=False, fontsize=9)

    fig.tight_layout()
    out_path = out_dir / "training_overview.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_snapshot_mmd(snapshot_mean_df, *, out_dir: str | Path, snapshot_std_df=None):
    out_dir = Path(out_dir)
    model_keys = [key for key in MODEL_LABELS if key in set(snapshot_mean_df["model_key"].unique())]
    if not model_keys:
        raise ValueError("snapshot_mean_df does not contain any known model keys.")

    epochs = sorted(snapshot_mean_df["epoch"].unique().tolist())
    fig, axes = plt.subplots(len(model_keys), 1, figsize=(16.0, 5.0 * len(model_keys)), sharex=True)
    axes = [axes] if len(model_keys) == 1 else list(axes)

    for ax, model_key in zip(axes, model_keys):
        view = snapshot_mean_df[snapshot_mean_df["model_key"] == model_key]
        snapshot_specs = (
            view[["snapshot", "snapshot_title", "snapshot_order"]]
            .drop_duplicates()
            .sort_values("snapshot_order", kind="stable")
        )
        colors = _snapshot_palette(len(snapshot_specs))

        for color, snapshot_row in zip(colors, snapshot_specs.itertuples(index=False)):
            snapshot_name = str(snapshot_row.snapshot)
            snapshot_title = str(snapshot_row.snapshot_title)
            cur_mean = view[view["snapshot"] == snapshot_name].sort_values("epoch")
            cur_std = None
            if snapshot_std_df is not None:
                cur_std = snapshot_std_df[
                    (snapshot_std_df["model_key"] == model_key) & (snapshot_std_df["snapshot"] == snapshot_name)
                ].sort_values("epoch")
            _plot_with_band(
                ax,
                epochs,
                cur_mean["mmd"].to_numpy(),
                None if cur_std is None else cur_std["mmd"].fillna(0.0).to_numpy(),
                color=color,
                label=snapshot_title,
            )

        ax.set_title(MODEL_LABELS[model_key])
        ax.set_ylabel("Source-Target MMD")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center")

    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    out_path = out_dir / "snapshot_mmd_over_epochs.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_final_snapshot_mmd(snapshot_mean_df, *, out_dir: str | Path, snapshot_std_df=None):
    out_dir = Path(out_dir)
    final_epoch = int(snapshot_mean_df["epoch"].max())
    model_keys = [key for key in MODEL_LABELS if key in set(snapshot_mean_df["model_key"].unique())]
    if not model_keys:
        raise ValueError("snapshot_mean_df does not contain any known model keys.")

    fig, axes = plt.subplots(1, len(model_keys), figsize=(7.2 * len(model_keys), 5.3), sharey=True)
    axes = [axes] if len(model_keys) == 1 else list(axes)

    for ax, model_key in zip(axes, model_keys):
        cur_mean = snapshot_mean_df[
            (snapshot_mean_df["model_key"] == model_key) & (snapshot_mean_df["epoch"] == final_epoch)
        ].sort_values("snapshot_order", kind="stable")
        cur_std = None
        if snapshot_std_df is not None:
            cur_std = snapshot_std_df[
                (snapshot_std_df["model_key"] == model_key) & (snapshot_std_df["epoch"] == final_epoch)
            ].sort_values("snapshot_order", kind="stable")

        x = list(range(len(cur_mean)))
        y = cur_mean["mmd"].to_numpy()
        ax.plot(
            x,
            y,
            color=MODEL_PALETTE[model_key],
            linewidth=2.2,
            marker="o",
            markersize=6,
        )
        if cur_std is not None:
            yerr = cur_std["mmd"].fillna(0.0).to_numpy()
            ax.fill_between(x, y - yerr, y + yerr, color=MODEL_PALETTE[model_key], alpha=0.15, linewidth=0)

        ax.set_title(MODEL_LABELS[model_key])
        ax.set_xticks(x)
        ax.set_xticklabels(cur_mean["snapshot_title"].astype(str).tolist(), rotation=35, ha="right")
        ax.set_xlabel("Snapshot")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("MMD at final epoch")
    fig.tight_layout()
    out_path = out_dir / "final_snapshot_mmd.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path
