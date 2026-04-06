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


def _standardized_effects(frame: pd.DataFrame) -> dict[str, float]:
    cols = ["source_micro_f1", "feature_mmd", "target_micro_f1"]
    cur = frame[cols].dropna().copy()
    if len(cur) < 3:
        return {
            "source_micro_f1_beta": float("nan"),
            "feature_mmd_beta": float("nan"),
            "r2": float("nan"),
        }

    stds = cur.std(ddof=0)
    if (stds == 0).any():
        return {
            "source_micro_f1_beta": float("nan"),
            "feature_mmd_beta": float("nan"),
            "r2": float("nan"),
        }

    z = (cur - cur.mean()) / stds
    x = z[["source_micro_f1", "feature_mmd"]].to_numpy()
    y = z["target_micro_f1"].to_numpy()
    x = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    resid = ((y - pred) ** 2).sum()
    total = ((y - y.mean()) ** 2).sum()
    r2 = float("nan") if total == 0 else 1.0 - float(resid / total)
    return {
        "source_micro_f1_beta": float(beta[1]),
        "feature_mmd_beta": float(beta[2]),
        "r2": r2,
    }


def compute_dataset_correlations(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, frame in raw_df.groupby("dataset", sort=False):
        for component in ["source_micro_f1", "source_error", "feature_mmd"]:
            cur = frame[[component, "target_micro_f1"]].dropna()
            if cur.empty or cur[component].nunique() <= 1 or cur["target_micro_f1"].nunique() <= 1:
                corr = float("nan")
            else:
                corr = float(cur[component].corr(cur["target_micro_f1"]))
            rows.append(
                {
                    "dataset": dataset,
                    "component": component,
                    "pearson_with_target_micro_f1": corr,
                }
            )

        effects = _standardized_effects(frame)
        rows.append(
            {
                "dataset": dataset,
                "component": "standardized_regression",
                "pearson_with_target_micro_f1": float("nan"),
                **effects,
            }
        )

    return pd.DataFrame(rows)


def build_pair_comparison_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, frame in raw_df.groupby("dataset", sort=False):
        domains = sorted(set(frame["source"]).union(set(frame["target"])))

        self_perf = (
            frame[["dataset", "seed", "source", "source_micro_f1"]]
            .drop_duplicates()
            .groupby(["dataset", "source"], as_index=False)["source_micro_f1"]
            .mean()
            .rename(columns={"source": "domain", "source_micro_f1": "self_micro_f1_mean"})
        )
        cross_perf = (
            frame.groupby(["dataset", "source", "target"], as_index=False)["target_micro_f1"]
            .mean()
            .rename(columns={"target_micro_f1": "cross_micro_f1_mean"})
        )

        for idx, domain_a in enumerate(domains):
            for domain_b in domains[idx + 1 :]:
                a_to_a = self_perf[
                    (self_perf["dataset"] == dataset) & (self_perf["domain"] == domain_a)
                ]["self_micro_f1_mean"].iloc[0]
                b_to_b = self_perf[
                    (self_perf["dataset"] == dataset) & (self_perf["domain"] == domain_b)
                ]["self_micro_f1_mean"].iloc[0]
                a_to_b = cross_perf[
                    (cross_perf["dataset"] == dataset)
                    & (cross_perf["source"] == domain_a)
                    & (cross_perf["target"] == domain_b)
                ]["cross_micro_f1_mean"].iloc[0]
                b_to_a = cross_perf[
                    (cross_perf["dataset"] == dataset)
                    & (cross_perf["source"] == domain_b)
                    & (cross_perf["target"] == domain_a)
                ]["cross_micro_f1_mean"].iloc[0]
                rows.append(
                    {
                        "dataset": dataset,
                        "domain_a": domain_a,
                        "domain_b": domain_b,
                        "a_to_a_micro_f1_mean": float(a_to_a),
                        "b_to_a_micro_f1_mean": float(b_to_a),
                        "a_to_b_micro_f1_mean": float(a_to_b),
                        "b_to_b_micro_f1_mean": float(b_to_b),
                    }
                )

    return pd.DataFrame(rows)


def plot_target_heatmaps(raw_df: pd.DataFrame, out_dir: Path):
    for dataset, frame in raw_df.groupby("dataset", sort=False):
        pivot = frame.pivot_table(
            index="source",
            columns="target",
            values="target_micro_f1",
            aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=(5.2, 4.5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_title(f"{dataset}: Target Performance")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_target_heatmap.png", dpi=200)
        plt.close(fig)


def plot_source_target_scatter(raw_df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for dataset, frame in raw_df.groupby("dataset", sort=False):
        ax.scatter(frame["source_micro_f1"], frame["target_micro_f1"], alpha=0.85, label=dataset)
    ax.set_xlabel("Source Self Performance")
    ax.set_ylabel("Cross-Domain Target Performance")
    ax.set_title("Target Performance vs Source Difficulty")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "source_vs_target_scatter.png", dpi=200)
    plt.close(fig)


def plot_mmd_target_scatter(raw_df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for dataset, frame in raw_df.groupby("dataset", sort=False):
        ax.scatter(frame["feature_mmd"], frame["target_micro_f1"], alpha=0.85, label=dataset)
    ax.set_xlabel("Feature MMD")
    ax.set_ylabel("Cross-Domain Target Performance")
    ax.set_title("Target Performance vs Shift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mmd_vs_target_scatter.png", dpi=200)
    plt.close(fig)


def plot_difficulty_shift_grid(raw_df: pd.DataFrame, out_dir: Path):
    datasets = list(raw_df["dataset"].drop_duplicates())
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.8), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        frame = raw_df[raw_df["dataset"] == dataset]
        scatter = ax.scatter(
            frame["source_micro_f1"],
            frame["feature_mmd"],
            c=frame["target_micro_f1"],
            cmap="viridis",
            s=70,
            alpha=0.9,
        )
        ax.set_xlabel("Source Self Performance")
        ax.set_ylabel("Feature MMD")
        ax.set_title(dataset)
    fig.colorbar(scatter, ax=axes, fraction=0.025, pad=0.03, label="Target Micro-F1")
    fig.savefig(out_dir / "difficulty_shift_grid.png", dpi=200)
    plt.close(fig)


def finalize_outputs(raw_df: pd.DataFrame, out_dir: Path):
    metric_cols = [
        "source_micro_f1",
        "source_error",
        "target_micro_f1",
        "target_error",
        "feature_mmd",
        "train_time",
    ]

    source_domain_summary = _flatten_summary(
        raw_df,
        ["dataset", "source"],
        ["source_micro_f1", "source_error", "train_time"],
    )
    pair_summary = _flatten_summary(
        raw_df,
        ["dataset", "source", "target"],
        metric_cols,
    )
    correlation_df = compute_dataset_correlations(raw_df)
    pair_comparison_table = build_pair_comparison_table(raw_df)

    raw_df.to_csv(out_dir / "raw_results.csv", index=False)
    source_domain_summary.to_csv(out_dir / "source_domain_summary.csv", index=False)
    pair_summary.to_csv(out_dir / "pair_summary.csv", index=False)
    correlation_df.to_csv(out_dir / "dataset_correlations.csv", index=False)
    pair_comparison_table.to_csv(out_dir / "pair_comparison_table.csv", index=False)

    plot_target_heatmaps(raw_df, out_dir)
    plot_source_target_scatter(raw_df, out_dir)
    plot_mmd_target_scatter(raw_df, out_dir)
    plot_difficulty_shift_grid(raw_df, out_dir)
