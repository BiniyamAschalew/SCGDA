from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DATASET_COLORS = {
    "airport": "#1f77b4",
    "blog": "#ff7f0e",
    "citation": "#2ca02c",
}

PREDICTOR_LABELS = {
    "source_micro_f1": "Source micro-F1",
    "target_entropy": "Target entropy",
    "latent_mmd": "Latent MMD",
    "raw_mmd": "Raw MMD",
    "latent_cmmd": "Latent cMMD",
    "raw_cmmd": "Raw cMMD",
    "target_grad_global_mean": "Target grad mean",
    "smooth_mmd": "Grad x latent MMD",
    "joint_oracle_target_micro_f1": "Joint oracle target F1",
    "target_only_target_micro_f1": "Target-only target F1",
}


def _dataset_color(name: str) -> str:
    return DATASET_COLORS.get(str(name).lower(), "#444444")


def _pretty_predictor(name: str) -> str:
    return PREDICTOR_LABELS.get(name, name.replace("_", " "))


def plot_correlation_bars(corr_df: pd.DataFrame, out_path: Path, regime: str, family: str) -> None:
    subset = corr_df[
        (corr_df["regime"] == regime)
        & (corr_df["family"] == family)
        & (corr_df["dataset"] == "all")
    ].copy()
    if subset.empty:
        return

    subset = subset.sort_values("spearman", ascending=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, max(3.5, 0.45 * len(subset))))
    colors = ["#c44e52" if val < 0 else "#4c72b0" for val in subset["spearman"]]
    ax.barh(subset["predictor"], subset["spearman"], color=colors, alpha=0.9)
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.set_xlabel("Spearman correlation with target micro-F1")
    ax.set_ylabel("")
    ax.set_title(f"{regime}: {family} predictors")
    ax.grid(axis="x", alpha=0.25)

    labels = [_pretty_predictor(name) for name in subset["predictor"]]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_predictor_scatter_grid(
    scenario_df: pd.DataFrame,
    out_path: Path,
    regime: str,
    predictors: list[str],
) -> None:
    target_col = f"{regime}_target_micro_f1"
    present = [col for col in predictors if f"{regime}_{col}" in scenario_df.columns or col in scenario_df.columns]
    if not present:
        return

    ncols = 2
    nrows = (len(present) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4.2 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, predictor in zip(axes, present):
        pred_col = predictor if predictor in scenario_df.columns else f"{regime}_{predictor}"
        subset = scenario_df[[pred_col, target_col, "dataset", "source", "target"]].dropna()
        for dataset, group in subset.groupby("dataset"):
            ax.scatter(
                group[pred_col],
                group[target_col],
                s=55,
                alpha=0.85,
                color=_dataset_color(dataset),
                label=dataset,
            )
            for _, row in group.iterrows():
                ax.annotate(
                    f"{row['source']}->{row['target']}",
                    (row[pred_col], row[target_col]),
                    fontsize=7,
                    alpha=0.7,
                    xytext=(3, 3),
                    textcoords="offset points",
                )
        ax.set_xlabel(_pretty_predictor(predictor))
        ax.set_ylabel("Target micro-F1")
        ax.set_title(_pretty_predictor(predictor))
        ax.grid(alpha=0.25)

    for ax in axes[len(present) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)))
    fig.suptitle(f"{regime}: target performance vs candidate predictors", y=0.99)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_cv_bars(cv_df: pd.DataFrame, out_path: Path, regime: str) -> None:
    if cv_df.empty or "regime" not in cv_df.columns:
        return
    subset = cv_df[cv_df["regime"] == regime].copy()
    if subset.empty:
        return

    summary = (
        subset.groupby("model_name", as_index=False)[["mae", "rmse"]]
        .mean()
        .sort_values("mae", ascending=True)
    )

    fig, ax = plt.subplots(1, 1, figsize=(8.5, max(4.0, 0.45 * len(summary))))
    ax.barh(summary["model_name"], summary["mae"], color="#4c72b0", alpha=0.9)
    ax.set_xlabel("Leave-one-dataset-out MAE on target error")
    ax.set_ylabel("")
    ax.set_title(f"{regime}: predictor model quality")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
