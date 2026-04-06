from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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


def compute_component_correlations(raw_df: pd.DataFrame) -> pd.DataFrame:
    outcomes = {
        "normal_target_micro_f1": "pearson_with_normal_target_micro_f1",
        "normalized_da_gain": "pearson_with_normalized_da_gain",
    }
    components = [
        "base_source_micro_f1",
        "base_target_micro_f1",
        "base_target_error",
        "base_domain_mmd",
        "base_smoothness_grad_max",
        "normal_source_micro_f1",
        "normal_source_error",
        "normal_domain_mmd",
        "normal_smoothness_grad_max",
        "oracle_target_micro_f1",
        "oracle_target_error",
        "oracle_joint_error",
        "bound_rhs_proxy",
    ]

    rows = []
    for model_name, frame in raw_df.groupby("model", sort=False):
        for outcome_col, corr_col in outcomes.items():
            valid = frame[[outcome_col] + components].copy()
            for component in components:
                cur = valid[[outcome_col, component]].dropna()
                if cur.empty or cur[component].nunique() <= 1 or cur[outcome_col].nunique() <= 1:
                    corr = float("nan")
                else:
                    corr = float(cur[component].corr(cur[outcome_col]))
                rows.append(
                    {
                        "model": model_name,
                        "outcome": outcome_col,
                        "component": component,
                        corr_col: corr,
                    }
                )
    return pd.DataFrame(rows)


def plot_performance_summary(summary_df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = range(len(summary_df))
    labels = summary_df["model"].tolist()

    axes[0].bar(x, summary_df["base_target_micro_f1_mean"], width=0.25, label="Base Target")
    axes[0].bar([i + 0.25 for i in x], summary_df["normal_target_micro_f1_mean"], width=0.25, label="Normal Target")
    axes[0].bar([i + 0.50 for i in x], summary_df["oracle_target_micro_f1_mean"], width=0.25, label="Oracle Target")
    axes[0].set_xticks([i + 0.25 for i in x], labels, rotation=20)
    axes[0].set_ylabel("Micro-F1")
    axes[0].set_title("Target Performance Regimes")
    axes[0].legend()

    axes[1].bar(x, summary_df["base_target_error_mean"], width=0.25, label="Base Target Error")
    axes[1].bar([i + 0.25 for i in x], summary_df["normal_target_error_mean"], width=0.25, label="Normal Target Error")
    axes[1].bar([i + 0.50 for i in x], summary_df["oracle_joint_error_mean"], width=0.25, label="Oracle Joint Error")
    axes[1].set_xticks([i + 0.25 for i in x], labels, rotation=20)
    axes[1].set_ylabel("Error")
    axes[1].set_title("Error Summary")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "performance_summary.png", dpi=200)
    plt.close(fig)


def plot_component_summary(summary_df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = range(len(summary_df))
    labels = summary_df["model"].tolist()

    axes[0].bar(x, summary_df["base_domain_mmd_mean"], width=0.35, color="#94a3b8", label="Base")
    axes[0].bar([i + 0.35 for i in x], summary_df["normal_domain_mmd_mean"], width=0.35, color="#3b82f6", label="Normal")
    axes[0].set_xticks([i + 0.175 for i in x], labels, rotation=20)
    axes[0].set_ylabel("MMD")
    axes[0].set_title("Domain Divergence")
    axes[0].legend()

    axes[1].bar(x, summary_df["base_smoothness_grad_max_mean"], width=0.35, color="#fdba74", label="Base")
    axes[1].bar([i + 0.35 for i in x], summary_df["normal_smoothness_grad_max_mean"], width=0.35, color="#f97316", label="Normal")
    axes[1].set_xticks([i + 0.175 for i in x], labels, rotation=20)
    axes[1].set_ylabel("Max Gradient Norm")
    axes[1].set_title("Smoothness Proxy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "component_summary.png", dpi=200)
    plt.close(fig)


def plot_target_component_scatter(raw_df: pd.DataFrame, out_dir: Path):
    components = [
        ("normal_source_micro_f1", "Source Micro-F1"),
        ("normal_domain_mmd", "Feature MMD"),
        ("normal_smoothness_grad_max", "Smoothness"),
        ("oracle_target_micro_f1", "Oracle Target Micro-F1"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for ax, (column, title) in zip(axes, components):
        for model_name, frame in raw_df.groupby("model", sort=False):
            ax.scatter(frame[column], frame["normal_target_micro_f1"], alpha=0.85, label=model_name)
        ax.set_xlabel(title)
        ax.set_ylabel("Normal Target Micro-F1")
        ax.set_title(f"Target vs {title}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "target_component_scatter.png", dpi=200)
    plt.close(fig)


def plot_gain_component_scatter(raw_df: pd.DataFrame, out_dir: Path):
    components = [
        ("base_target_micro_f1", "Base Target Micro-F1"),
        ("normal_domain_mmd", "Normal Feature MMD"),
        ("normal_smoothness_grad_max", "Normal Smoothness"),
        ("oracle_target_micro_f1", "Oracle Target Micro-F1"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for ax, (column, title) in zip(axes, components):
        for model_name, frame in raw_df.groupby("model", sort=False):
            ax.scatter(frame[column], frame["normalized_da_gain"], alpha=0.85, label=model_name)
        ax.set_xlabel(title)
        ax.set_ylabel("Normalized DA Gain")
        ax.set_title(f"Gain vs {title}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "gain_component_scatter.png", dpi=200)
    plt.close(fig)


def plot_bound_proxy(raw_df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for model_name, frame in raw_df.groupby("model", sort=False):
        ax.scatter(frame["bound_rhs_proxy"], frame["normal_target_error"], alpha=0.85, label=model_name)
    ax.set_xlabel("Source Error + MMD + Oracle Joint Error")
    ax.set_ylabel("Target Error")
    ax.set_title("Bound Proxy vs Target Error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "bound_proxy_scatter.png", dpi=200)
    plt.close(fig)


def plot_correlation_heatmap(correlation_df: pd.DataFrame, out_dir: Path, *, outcome: str, value_col: str, filename: str, title: str):
    pivot = correlation_df[correlation_df["outcome"] == outcome].pivot(
        index="model",
        columns="component",
        values=value_col,
    )
    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=200)
    plt.close(fig)


def finalize_outputs(raw_df: pd.DataFrame, out_dir: Path):
    metric_cols = [
        "base_source_micro_f1",
        "base_target_micro_f1",
        "base_target_error",
        "base_domain_mmd",
        "base_smoothness_grad_max",
        "base_train_time",
        "normal_source_micro_f1",
        "normal_source_error",
        "normal_target_micro_f1",
        "normal_target_error",
        "normal_domain_mmd",
        "normal_smoothness_grad_max",
        "oracle_target_micro_f1",
        "oracle_target_error",
        "oracle_joint_error",
        "bound_rhs_proxy",
        "bound_gap_proxy",
        "oracle_target_gap",
        "normalized_da_gain",
        "absolute_da_gain",
        "oracle_headroom",
        "normal_train_time",
        "oracle_train_time",
    ]

    summary_by_model = _flatten_summary(raw_df, ["model"], metric_cols)
    summary_by_pair_model = _flatten_summary(raw_df, ["dataset", "pair", "model"], metric_cols)
    correlation_df = compute_component_correlations(raw_df)

    raw_df.to_csv(out_dir / "raw_results.csv", index=False)
    summary_by_model.to_csv(out_dir / "summary_by_model.csv", index=False)
    summary_by_pair_model.to_csv(out_dir / "summary_by_pair_model.csv", index=False)
    correlation_df.to_csv(out_dir / "component_correlations.csv", index=False)

    plot_performance_summary(summary_by_model, out_dir)
    plot_component_summary(summary_by_model, out_dir)
    plot_target_component_scatter(raw_df, out_dir)
    plot_gain_component_scatter(raw_df, out_dir)
    plot_bound_proxy(raw_df, out_dir)
    plot_correlation_heatmap(
        correlation_df,
        out_dir,
        outcome="normal_target_micro_f1",
        value_col="pearson_with_normal_target_micro_f1",
        filename="target_component_correlation_heatmap.png",
        title="Correlation With Target Micro-F1",
    )
    plot_correlation_heatmap(
        correlation_df,
        out_dir,
        outcome="normalized_da_gain",
        value_col="pearson_with_normalized_da_gain",
        filename="gain_component_correlation_heatmap.png",
        title="Correlation With Normalized DA Gain",
    )
