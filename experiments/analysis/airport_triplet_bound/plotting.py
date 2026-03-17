from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CASE_ORDER = ["Source Only", "Source + MMD", "Oracle"]
CASE_COLORS = {
    "Source Only": "#4C78A8",
    "Source + MMD": "#F58518",
    "Oracle": "#54A24B",
}


def _ordered_frame(df: pd.DataFrame) -> pd.DataFrame:
    order = {label: idx for idx, label in enumerate(CASE_ORDER)}
    return df.sort_values("case_label", key=lambda s: s.map(order.get)).reset_index(drop=True)


def _annotate_heatmap(ax, data: np.ndarray, fmt: str = ".3f") -> None:
    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                format(float(data[row_idx, col_idx]), fmt),
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    if x.nunique() <= 1 or y.nunique() <= 1:
        return float("nan")
    return float(x.corr(y))


def plot_domain_performance(summary_by_case: pd.DataFrame, out_dir: Path, dataset_name: str) -> None:
    summary_by_case = _ordered_frame(summary_by_case)
    domains = ["source", "target", "reference"]
    labels = ["Source", "Target", "Reference"]
    x = np.arange(len(summary_by_case))
    width = 0.22

    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, (domain, label) in enumerate(zip(domains, labels)):
        ax.bar(
            x + (offset - 1) * width,
            summary_by_case[f"{domain}_micro_f1_mean"],
            width=width,
            yerr=summary_by_case[f"{domain}_micro_f1_std"].fillna(0.0),
            capsize=4,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(summary_by_case["case_label"])
    ax.set_ylabel("Validation micro-F1")
    ax.set_title(f"{dataset_name} Triplet Performance by Training Regime")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "domain_performance_by_case.png", dpi=200)
    plt.close(fig)


def plot_target_heatmap(summary_by_triplet_case: pd.DataFrame, out_dir: Path, dataset_name: str) -> None:
    pivot = summary_by_triplet_case.pivot(
        index="triplet_label",
        columns="case_label",
        values="target_micro_f1_mean",
    )
    pivot = pivot.reindex(columns=CASE_ORDER)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap="YlGn")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{dataset_name} Target micro-F1 by Ordered Triplet")
    _annotate_heatmap(ax, data)
    fig.colorbar(im, ax=ax, shrink=0.85, label="micro-F1")
    fig.tight_layout()
    fig.savefig(out_dir / "target_micro_f1_heatmap.png", dpi=200)
    plt.close(fig)


def plot_pairwise_mmd(summary_by_case: pd.DataFrame, out_dir: Path, dataset_name: str) -> None:
    summary_by_case = _ordered_frame(summary_by_case)
    columns = [
        ("val_mmd_source_target_mean", "S val - T val"),
        ("val_mmd_source_reference_mean", "S val - R val"),
        ("val_mmd_target_reference_mean", "T val - R val"),
    ]
    matrix = summary_by_case[[name for name, _ in columns]].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    im = ax.imshow(matrix, aspect="auto", cmap="magma_r")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([label for _, label in columns], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(summary_by_case)))
    ax.set_yticklabels(summary_by_case["case_label"])
    ax.set_title(f"{dataset_name} Pairwise Validation MMD")
    _annotate_heatmap(ax, matrix)
    fig.colorbar(im, ax=ax, shrink=0.85, label="MMD")
    fig.tight_layout()
    fig.savefig(out_dir / "pairwise_val_mmd_heatmap.png", dpi=200)
    plt.close(fig)


def plot_shift_summary(summary_by_case: pd.DataFrame, out_dir: Path, dataset_name: str) -> None:
    summary_by_case = _ordered_frame(summary_by_case)
    columns = [
        ("shift_mmd_source_train_source_val_mean", "S train - S val"),
        ("shift_mmd_source_train_target_val_mean", "S train - T val"),
        ("train_st_mmd_mean", "S train - T train"),
    ]
    x = np.arange(len(summary_by_case))
    width = 0.22

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for offset, (column, label) in enumerate(columns):
        ax.bar(
            x + (offset - 1) * width,
            summary_by_case[column],
            width=width,
            yerr=summary_by_case[column.replace("_mean", "_std")].fillna(0.0),
            capsize=4,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(summary_by_case["case_label"])
    ax.set_ylabel("Embedding MMD")
    ax.set_title(f"{dataset_name} Shift Terms by Training Regime")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "shift_terms_by_case.png", dpi=200)
    plt.close(fig)


def plot_component_summary(summary_by_case: pd.DataFrame, out_dir: Path, dataset_name: str) -> None:
    summary_by_case = _ordered_frame(summary_by_case)
    panels = [
        ("source_error_mean", "Source error"),
        ("shift_mmd_source_train_source_val_mean", "S train - S val"),
        ("shift_mmd_source_train_target_val_mean", "S train - T val"),
        ("global_grad_input_max_mean", "Sensitivity"),
        ("oracle_target_micro_f1_mean", "Oracle target micro-F1"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(15.2, 3.8))
    for ax, (column, title) in zip(axes, panels):
        ax.bar(
            summary_by_case["case_label"],
            summary_by_case[column],
            color=[CASE_COLORS[label] for label in summary_by_case["case_label"]],
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle(f"{dataset_name} Source Error, Shift, Sensitivity, and Oracle Components", y=1.04)
    fig.tight_layout()
    fig.savefig(out_dir / "component_summary_by_case.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lipschitz_scatter(raw_df: pd.DataFrame, out_dir: Path, dataset_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for case_label, frame in raw_df.groupby("case_label", sort=False):
        ax.scatter(
            frame["global_grad_input_max"],
            frame["target_micro_f1"],
            alpha=0.75,
            s=45,
            label=case_label,
            color=CASE_COLORS.get(case_label),
        )
    ax.set_xlabel("Max gradient proxy")
    ax.set_ylabel("Target validation micro-F1")
    ax.set_title(f"{dataset_name} Target Performance vs Sensitivity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "lipschitz_vs_target_micro_f1.png", dpi=200)
    plt.close(fig)


def plot_bound_proxy(
    raw_df: pd.DataFrame,
    out_dir: Path,
    dataset_name: str,
    *,
    column: str,
    filename: str,
    title: str,
) -> None:
    bound_df = raw_df[raw_df["case_key"] != "oracle"].copy()
    if bound_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for case_label, frame in bound_df.groupby("case_label", sort=False):
        ax.scatter(
            frame[column],
            frame["target_error"],
            alpha=0.75,
            s=45,
            label=case_label,
            color=CASE_COLORS.get(case_label),
        )
    lim_low = min(bound_df["target_error"].min(), bound_df[column].min())
    lim_high = max(bound_df["target_error"].max(), bound_df[column].max())
    ax.plot([lim_low, lim_high], [lim_low, lim_high], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlabel("Bound proxy RHS")
    ax.set_ylabel("Target error")
    ax.set_title(f"{dataset_name} {title}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=200)
    plt.close(fig)


def plot_component_influence(raw_df: pd.DataFrame, out_dir: Path, dataset_name: str) -> None:
    view = raw_df[raw_df["case_key"] != "oracle"].copy()
    if view.empty:
        return

    components = [
        ("source_error", "Source error"),
        ("shift_mmd_source_train_source_val", "S train - S val"),
        ("shift_mmd_source_train_target_val", "S train - T val"),
        ("global_grad_input_max", "Sensitivity"),
        ("oracle_target_micro_f1", "Oracle target micro-F1"),
        ("bound_proxy_rhs_train_target", "Bound RHS"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    for ax, (column, label) in zip(axes.flat, components):
        for case_label, frame in view.groupby("case_label", sort=False):
            ax.scatter(
                frame[column],
                frame["target_micro_f1"],
                alpha=0.75,
                s=35,
                label=case_label,
                color=CASE_COLORS.get(case_label),
            )
        corr = _safe_corr(view[column], view["target_micro_f1"])
        corr_label = "n/a" if np.isnan(corr) else f"{corr:.2f}"
        ax.set_title(f"{label}\nr={corr_label}")
        ax.set_xlabel(label)
        ax.set_ylabel("Target micro-F1")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.suptitle(f"{dataset_name} Component Influence on Observed Target Performance", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "component_influence_grid.png", dpi=200)
    plt.close(fig)


def plot_training_curves(history_df: pd.DataFrame, out_dir: Path, dataset_name: str) -> None:
    grouped = (
        history_df.groupby(["case_label", "epoch"], as_index=False)[["total_loss", "train_st_mmd"]]
        .mean()
        .sort_values(["case_label", "epoch"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for case_label, frame in grouped.groupby("case_label", sort=False):
        axes[0].plot(
            frame["epoch"],
            frame["total_loss"],
            label=case_label,
            linewidth=2.0,
            color=CASE_COLORS.get(case_label),
        )
        axes[1].plot(
            frame["epoch"],
            frame["train_st_mmd"],
            label=case_label,
            linewidth=2.0,
            color=CASE_COLORS.get(case_label),
        )

    axes[0].set_title("Average Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Average Source-Target Train MMD")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MMD")
    axes[1].legend(frameon=False)
    fig.suptitle(f"{dataset_name} Training Curves", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
