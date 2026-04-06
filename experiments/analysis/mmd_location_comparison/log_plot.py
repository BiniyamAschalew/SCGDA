from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_output_dir(*, out_root: str, run_name: str) -> Path:
    out_dir = Path(out_root) / run_name
    suffix = 1
    while out_dir.exists():
        out_dir = Path(out_root) / f"{run_name}_{suffix}"
        suffix += 1
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _flatten_summary(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    summary = df.groupby(group_cols, as_index=False, sort=False)[metric_cols].agg(["mean", "std"])
    summary.columns = [
        "_".join([part for part in col if part]).rstrip("_")
        for col in summary.columns.to_flat_index()
    ]
    return summary


def plot_target_performance(summary_by_variant: pd.DataFrame, out_dir: Path):
    datasets = list(dict.fromkeys(summary_by_variant["dataset"].tolist()))
    variants = list(dict.fromkeys(summary_by_variant["variant_label"].tolist()))
    x = np.arange(len(datasets))
    width = 0.35 if len(variants) <= 2 else 0.8 / max(len(variants), 1)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for idx, variant in enumerate(variants):
        frame = summary_by_variant[summary_by_variant["variant_label"] == variant]
        means = [
            frame.loc[frame["dataset"] == dataset, "target_micro_f1_mean"].iloc[0]
            for dataset in datasets
        ]
        errs = [
            frame.loc[frame["dataset"] == dataset, "target_micro_f1_std"].fillna(0.0).iloc[0]
            for dataset in datasets
        ]
        offset = (idx - (len(variants) - 1) / 2.0) * width
        ax.bar(x + offset, means, width=width, yerr=errs, capsize=3, label=variant)

    ax.set_xticks(x, datasets)
    ax.set_ylabel("Target Micro-F1")
    ax.set_title("Target Performance by MMD Location")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "target_performance_by_variant.png", dpi=200)
    plt.close(fig)


def plot_shift_summary(summary_by_variant: pd.DataFrame, out_dir: Path):
    datasets = list(dict.fromkeys(summary_by_variant["dataset"].tolist()))
    variants = list(dict.fromkeys(summary_by_variant["variant_label"].tolist()))
    metrics = [
        ("encoder_val_mmd_mean", "Encoder Val MMD"),
        ("logit_val_mmd_mean", "Logit Val MMD"),
        ("encoder_val_w1_mean", "Encoder Val W1"),
        ("logit_val_w1_mean", "Logit Val W1"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    axes = axes.flatten()
    x = np.arange(len(datasets))
    width = 0.35 if len(variants) <= 2 else 0.8 / max(len(variants), 1)

    for ax, (column, title) in zip(axes, metrics):
        for idx, variant in enumerate(variants):
            frame = summary_by_variant[summary_by_variant["variant_label"] == variant]
            means = [frame.loc[frame["dataset"] == dataset, column].iloc[0] for dataset in datasets]
            offset = (idx - (len(variants) - 1) / 2.0) * width
            ax.bar(x + offset, means, width=width, label=variant)
        ax.set_xticks(x, datasets)
        ax.set_title(title)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    fig.savefig(out_dir / "shift_summary_by_variant.png", dpi=200)
    plt.close(fig)


def plot_target_heatmaps(summary_by_pair_variant: pd.DataFrame, out_dir: Path):
    for dataset, dataset_frame in summary_by_pair_variant.groupby("dataset", sort=False):
        variants = list(dict.fromkeys(dataset_frame["variant_label"].tolist()))
        fig, axes = plt.subplots(1, len(variants), figsize=(5.3 * len(variants), 4.5))
        axes = np.atleast_1d(axes).flatten()

        for ax, variant in zip(axes, variants):
            frame = dataset_frame[dataset_frame["variant_label"] == variant]
            pivot = frame.pivot(index="source", columns="target", values="target_micro_f1_mean")
            im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
            ax.set_yticks(range(len(pivot.index)), pivot.index)
            ax.set_title(f"{dataset}: {variant}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_target_heatmaps.png", dpi=200)
        plt.close(fig)


def plot_target_delta_heatmap(summary_by_pair_variant: pd.DataFrame, out_dir: Path):
    if summary_by_pair_variant["variant_key"].nunique() < 2:
        return

    wide = summary_by_pair_variant.pivot_table(
        index=["dataset", "source", "target"],
        columns="variant_key",
        values="target_micro_f1_mean",
    ).reset_index()
    if not {"encoder_mmd", "logit_mmd"}.issubset(wide.columns):
        return

    wide["target_delta_logit_minus_encoder"] = wide["logit_mmd"] - wide["encoder_mmd"]
    for dataset, frame in wide.groupby("dataset", sort=False):
        pivot = frame.pivot(index="source", columns="target", values="target_delta_logit_minus_encoder")
        fig, ax = plt.subplots(figsize=(5.4, 4.5))
        vmax = float(np.nanmax(np.abs(pivot.values))) if np.isfinite(pivot.values).any() else 1.0
        vmax = max(vmax, 1e-6)
        im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_title(f"{dataset}: Target Delta (Logit - Encoder)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_target_delta_heatmap.png", dpi=200)
        plt.close(fig)


def finalize_outputs(raw_df: pd.DataFrame, history_df: pd.DataFrame, out_dir: Path):
    metric_cols = [
        "source_micro_f1",
        "target_micro_f1",
        "source_target_gap",
        "encoder_val_mmd",
        "logit_val_mmd",
        "encoder_val_w1",
        "logit_val_w1",
        "final_ce_loss",
        "final_train_mmd",
        "train_time",
    ]

    summary_by_variant = _flatten_summary(
        raw_df,
        ["dataset", "variant_key", "variant_label"],
        metric_cols,
    )
    summary_by_pair_variant = _flatten_summary(
        raw_df,
        ["dataset", "source", "target", "variant_key", "variant_label"],
        metric_cols,
    )

    raw_df.to_csv(out_dir / "raw_results.csv", index=False)
    history_df.to_csv(out_dir / "training_history.csv", index=False)
    summary_by_variant.to_csv(out_dir / "summary_by_variant.csv", index=False)
    summary_by_pair_variant.to_csv(out_dir / "summary_by_pair_variant.csv", index=False)

    plot_target_performance(summary_by_variant, out_dir)
    plot_shift_summary(summary_by_variant, out_dir)
    plot_target_heatmaps(summary_by_pair_variant, out_dir)
    plot_target_delta_heatmap(summary_by_pair_variant, out_dir)

