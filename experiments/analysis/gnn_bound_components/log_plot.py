from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split


TARGET_METRIC = "target_micro_f1"
CORRELATION_PREDICTORS = [
    "source_micro_f1",
    "source_random_micro_f1",
    "target_random_micro_f1",
    "source_lift",
    "target_self_micro_f1",
    "oracle_target_micro_f1",
    "mmd_shift",
    "w1_shift",
    "structural_mmd_shift",
    "structural_w1_shift",
    "smoothness_grad_max",
    "spectral_lipschitz_log10",
    "num_layers",
]
REGRESSION_NUMERIC_PREDICTORS = CORRELATION_PREDICTORS.copy()
SUMMARY_METRICS = [
    "source_micro_f1",
    "target_micro_f1",
    "target_self_micro_f1",
    "source_random_micro_f1",
    "target_random_micro_f1",
    "source_lift",
    "target_lift",
    "oracle_target_micro_f1",
    "mmd_shift",
    "w1_shift",
    "structural_mmd_shift",
    "structural_w1_shift",
    "smoothness_grad_max",
    "spectral_lipschitz_log10",
]

LASSO_REPEATS = 5
TEST_SIZE_FRACTION = 0.25
LASSO_ALPHAS = np.logspace(-4, 0, 40)
EPS = 1e-8


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


def _safe_log10(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.log10(np.maximum(arr, EPS))


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["pair_id"] = (
        enriched["dataset"].astype(str)
        + "::"
        + enriched["model"].astype(str)
        + "::L"
        + enriched["num_layers"].astype(int).astype(str)
        + "::"
        + enriched["source"].astype(str)
        + "->"
        + enriched["target"].astype(str)
    )
    enriched["source_lift"] = enriched["source_micro_f1"] - enriched["source_random_micro_f1"]
    enriched["target_lift"] = enriched["target_self_micro_f1"] - enriched["target_random_micro_f1"]
    enriched["oracle_target_lift"] = enriched["oracle_target_micro_f1"] - enriched["target_random_micro_f1"]
    enriched["source_error"] = 1.0 - enriched["source_micro_f1"]
    enriched["target_error"] = 1.0 - enriched["target_micro_f1"]
    enriched["oracle_target_error"] = 1.0 - enriched["oracle_target_micro_f1"]
    enriched["spectral_lipschitz_log10"] = _safe_log10(
        np.power(10.0, np.asarray(enriched["spectral_lipschitz_log10"], dtype=float))
    )
    return enriched


def _build_pair_mean_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "model", "num_layers", "source", "target"]
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    if "seed" in numeric_cols:
        numeric_cols.remove("seed")

    pair_mean_df = raw_df.groupby(group_cols, as_index=False, sort=False)[numeric_cols].mean()
    seed_counts = raw_df.groupby(group_cols, as_index=False, sort=False).size()
    seed_counts = seed_counts.rename(columns={"size": "seed_count"})
    pair_mean_df = pair_mean_df.merge(seed_counts, on=group_cols, how="left")
    return _add_derived_columns(pair_mean_df)


def compute_target_correlations(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_name, frame in raw_df.groupby("dataset", sort=False):
        for predictor in CORRELATION_PREDICTORS:
            cur = frame[[predictor, TARGET_METRIC]].dropna()
            if len(cur) < 2 or cur[predictor].nunique() <= 1 or cur[TARGET_METRIC].nunique() <= 1:
                corr = float("nan")
            else:
                corr = float(cur[predictor].corr(cur[TARGET_METRIC]))
            rows.append(
                {
                    "dataset": dataset_name,
                    "predictor": predictor,
                    "n_rows": int(len(cur)),
                    f"pearson_with_{TARGET_METRIC}": corr,
                    f"abs_pearson_with_{TARGET_METRIC}": abs(corr) if pd.notna(corr) else float("nan"),
                }
            )

    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        return corr_df

    corr_df["correlation_rank"] = corr_df.groupby("dataset")[f"abs_pearson_with_{TARGET_METRIC}"].rank(
        method="dense",
        ascending=False,
    )
    return corr_df.sort_values(["dataset", "correlation_rank", "predictor"], kind="stable").reset_index(drop=True)


def _prepare_regression_frame(frame: pd.DataFrame):
    numeric_predictors = [name for name in REGRESSION_NUMERIC_PREDICTORS if name in frame.columns]
    keep_cols = ["dataset", "model", "seed", "source", "target", "pair_id", TARGET_METRIC] + numeric_predictors
    cur = frame[keep_cols].dropna().copy()
    if cur.empty:
        return cur, []

    model_dummies = pd.get_dummies(cur["model"], prefix="model", drop_first=True, dtype=float)
    cur = pd.concat([cur, model_dummies], axis=1)
    predictor_cols = numeric_predictors + model_dummies.columns.tolist()
    return cur, predictor_cols


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    total = float(((y_true - y_true.mean()) ** 2).sum())
    if total <= 0:
        return float("nan")
    resid = float(((y_true - y_pred) ** 2).sum())
    return 1.0 - resid / total


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).mean())


def _fit_standardized_lasso(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    predictors: list[str],
    target: str,
    random_state: int,
):
    if not predictors:
        return None

    x_train = train_df[predictors].to_numpy(dtype=float)
    y_train = train_df[target].to_numpy(dtype=float)
    x_test = test_df[predictors].to_numpy(dtype=float)
    y_test = test_df[target].to_numpy(dtype=float)

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0, ddof=0)
    x_scale = np.where(x_std > 0, x_std, 1.0)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std(ddof=0))
    if y_std <= EPS:
        return None

    x_train_z = (x_train - x_mean) / x_scale
    x_test_z = (x_test - x_mean) / x_scale
    y_train_z = (y_train - y_mean) / y_std

    cv = min(5, len(train_df))
    if cv < 2:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model = LassoCV(
            alphas=LASSO_ALPHAS,
            cv=cv,
            random_state=random_state,
            max_iter=20000,
        ).fit(x_train_z, y_train_z)

    train_pred = model.predict(x_train_z) * y_std + y_mean
    test_pred = model.predict(x_test_z) * y_std + y_mean

    return {
        "model": model,
        "train_true": y_train,
        "test_true": y_test,
        "train_pred": train_pred,
        "test_pred": test_pred,
    }


def _fit_target_regression_repeats(frame: pd.DataFrame, dataset_name: str):
    cur, predictor_cols = _prepare_regression_frame(frame)
    if len(cur) < 8 or not predictor_cols:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    test_size = max(1, int(round(len(cur) * TEST_SIZE_FRACTION)))
    test_size = min(test_size, len(cur) - 5)
    if test_size < 1:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    split_rows = []
    coefficient_rows = []
    prediction_rows = []
    importance_rows = []

    for repeat in range(LASSO_REPEATS):
        random_state = 2025 + repeat
        train_df, test_df = train_test_split(
            cur,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

        base_fit = _fit_standardized_lasso(
            train_df,
            test_df,
            predictors=predictor_cols,
            target=TARGET_METRIC,
            random_state=random_state,
        )
        if base_fit is None:
            continue

        base_train_r2 = _r2_score(base_fit["train_true"], base_fit["train_pred"])
        base_test_r2 = _r2_score(base_fit["test_true"], base_fit["test_pred"])
        base_train_mae = _mae(base_fit["train_true"], base_fit["train_pred"])
        base_test_mae = _mae(base_fit["test_true"], base_fit["test_pred"])

        split_rows.append(
            {
                "dataset": dataset_name,
                "repeat": repeat,
                "random_state": random_state,
                "alpha": float(base_fit["model"].alpha_),
                "n_rows": int(len(cur)),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "train_r2": base_train_r2,
                "test_r2": base_test_r2,
                "train_mae": base_train_mae,
                "test_mae": base_test_mae,
            }
        )

        for predictor, coef in zip(predictor_cols, base_fit["model"].coef_):
            coefficient_rows.append(
                {
                    "dataset": dataset_name,
                    "repeat": repeat,
                    "random_state": random_state,
                    "predictor": predictor,
                    "coefficient": float(coef),
                    "selected": int(abs(float(coef)) > 1e-10),
                    "alpha": float(base_fit["model"].alpha_),
                }
            )

        test_df = test_df.reset_index(drop=True)
        for idx in range(len(test_df)):
            prediction_rows.append(
                {
                    "dataset": dataset_name,
                    "repeat": repeat,
                    "random_state": random_state,
                    "pair_id": test_df.loc[idx, "pair_id"],
                    "source": test_df.loc[idx, "source"],
                    "target": test_df.loc[idx, "target"],
                    "model": test_df.loc[idx, "model"],
                    "num_layers": int(test_df.loc[idx, "num_layers"]),
                    f"{TARGET_METRIC}_true": float(base_fit["test_true"][idx]),
                    f"{TARGET_METRIC}_pred": float(base_fit["test_pred"][idx]),
                }
            )

        for dropped_predictor in predictor_cols:
            reduced_predictors = [name for name in predictor_cols if name != dropped_predictor]
            reduced_fit = _fit_standardized_lasso(
                train_df,
                test_df,
                predictors=reduced_predictors,
                target=TARGET_METRIC,
                random_state=random_state,
            )
            if reduced_fit is None:
                continue

            reduced_test_r2 = _r2_score(reduced_fit["test_true"], reduced_fit["test_pred"])
            reduced_test_mae = _mae(reduced_fit["test_true"], reduced_fit["test_pred"])
            importance_rows.append(
                {
                    "dataset": dataset_name,
                    "repeat": repeat,
                    "random_state": random_state,
                    "predictor": dropped_predictor,
                    "base_test_r2": base_test_r2,
                    "reduced_test_r2": reduced_test_r2,
                    "test_r2_drop": base_test_r2 - reduced_test_r2,
                    "base_test_mae": base_test_mae,
                    "reduced_test_mae": reduced_test_mae,
                    "test_mae_increase": reduced_test_mae - base_test_mae,
                }
            )

    return (
        pd.DataFrame(split_rows),
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(importance_rows),
    )


def build_target_regression_outputs(raw_df: pd.DataFrame):
    split_frames = []
    coefficient_frames = []
    prediction_frames = []
    importance_frames = []

    for dataset_name, frame in raw_df.groupby("dataset", sort=False):
        split_df, coefficient_df, prediction_df, importance_df = _fit_target_regression_repeats(frame, dataset_name)
        if not split_df.empty:
            split_frames.append(split_df)
        if not coefficient_df.empty:
            coefficient_frames.append(coefficient_df)
        if not prediction_df.empty:
            prediction_frames.append(prediction_df)
        if not importance_df.empty:
            importance_frames.append(importance_df)

    split_results = pd.concat(split_frames, ignore_index=True) if split_frames else pd.DataFrame()
    coefficient_results = pd.concat(coefficient_frames, ignore_index=True) if coefficient_frames else pd.DataFrame()
    prediction_results = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    importance_results = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()

    if split_results.empty:
        split_summary = pd.DataFrame()
    else:
        split_summary = split_results.groupby("dataset", as_index=False, sort=False).agg(
            repeats=("repeat", "count"),
            alpha_mean=("alpha", "mean"),
            alpha_std=("alpha", "std"),
            train_r2_mean=("train_r2", "mean"),
            train_r2_std=("train_r2", "std"),
            test_r2_mean=("test_r2", "mean"),
            test_r2_std=("test_r2", "std"),
            train_mae_mean=("train_mae", "mean"),
            train_mae_std=("train_mae", "std"),
            test_mae_mean=("test_mae", "mean"),
            test_mae_std=("test_mae", "std"),
            n_rows_mean=("n_rows", "mean"),
            n_train_mean=("n_train", "mean"),
            n_test_mean=("n_test", "mean"),
        )

    if coefficient_results.empty:
        coefficient_summary = pd.DataFrame()
    else:
        coefficient_summary = coefficient_results.groupby(
            ["dataset", "predictor"],
            as_index=False,
            sort=False,
        ).agg(
            coefficient_mean=("coefficient", "mean"),
            coefficient_std=("coefficient", "std"),
            selection_freq=("selected", "mean"),
        )
        coefficient_summary["abs_coefficient_mean"] = coefficient_summary["coefficient_mean"].abs()

    if importance_results.empty:
        importance_summary = pd.DataFrame()
    else:
        importance_summary = importance_results.groupby(
            ["dataset", "predictor"],
            as_index=False,
            sort=False,
        ).agg(
            test_r2_drop_mean=("test_r2_drop", "mean"),
            test_r2_drop_std=("test_r2_drop", "std"),
            test_mae_increase_mean=("test_mae_increase", "mean"),
            test_mae_increase_std=("test_mae_increase", "std"),
        )

    return (
        split_results,
        split_summary,
        coefficient_results,
        coefficient_summary,
        prediction_results,
        importance_results,
        importance_summary,
    )


def build_predictive_signal_summary(
    correlation_df: pd.DataFrame,
    coefficient_summary: pd.DataFrame,
    importance_summary: pd.DataFrame,
) -> pd.DataFrame:
    frames = [df for df in [correlation_df, coefficient_summary, importance_summary] if not df.empty]
    if not frames:
        return pd.DataFrame()

    summary = frames[0].copy()
    for frame in frames[1:]:
        summary = summary.merge(
            frame,
            on=["dataset", "predictor"],
            how="outer",
        )
    if summary.empty:
        return summary

    abs_corr_col = f"abs_pearson_with_{TARGET_METRIC}"
    if abs_corr_col in summary.columns:
        summary["abs_corr"] = summary[abs_corr_col]
        summary["corr_rank"] = summary.groupby("dataset")["abs_corr"].rank(method="dense", ascending=False)
    if "selection_freq" in summary.columns:
        summary["selection_rank"] = summary.groupby("dataset")["selection_freq"].rank(
            method="dense",
            ascending=False,
        )
    if "test_r2_drop_mean" in summary.columns:
        summary["importance_rank"] = summary.groupby("dataset")["test_r2_drop_mean"].rank(
            method="dense",
            ascending=False,
        )
    sort_cols = ["dataset", "predictor"]
    if "corr_rank" in summary.columns:
        sort_cols = ["dataset", "corr_rank", "predictor"]
    return summary.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def plot_target_correlations(correlation_df: pd.DataFrame, out_dir: Path):
    if correlation_df.empty:
        return

    datasets = list(dict.fromkeys(correlation_df["dataset"].tolist()))
    ncols = 2
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.8 * nrows))
    axes = np.atleast_1d(axes).flatten()
    corr_col = f"pearson_with_{TARGET_METRIC}"

    for ax, dataset_name in zip(axes, datasets):
        frame = correlation_df[correlation_df["dataset"] == dataset_name].sort_values(
            f"abs_pearson_with_{TARGET_METRIC}",
            ascending=True,
        )
        ax.barh(frame["predictor"], frame[corr_col])
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_title(dataset_name)
        ax.set_xlabel(f"Pearson Correlation with {TARGET_METRIC}")

    for ax in axes[len(datasets):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "target_performance_correlations.png", dpi=200)
    plt.close(fig)


def plot_model_layer_summary(model_layer_summary: pd.DataFrame, out_dir: Path):
    if model_layer_summary.empty:
        return

    datasets = list(dict.fromkeys(model_layer_summary["dataset"].tolist()))
    ncols = 2
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, dataset_name in zip(axes, datasets):
        frame = model_layer_summary[model_layer_summary["dataset"] == dataset_name].copy()
        for model_name, subframe in frame.groupby("model", sort=False):
            subframe = subframe.sort_values("num_layers")
            ax.errorbar(
                subframe["num_layers"],
                subframe["target_micro_f1_mean"],
                yerr=subframe["target_micro_f1_std"].fillna(0.0),
                marker="o",
                linewidth=1.5,
                capsize=3,
                label=model_name,
            )
        ax.set_title(dataset_name)
        ax.set_xlabel("Num Layers")
        ax.set_ylabel("Target Micro-F1")
        ax.set_xticks(sorted(frame["num_layers"].astype(int).unique().tolist()))

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout()
    fig.savefig(out_dir / "target_performance_by_model_layer.png", dpi=200)
    plt.close(fig)


def plot_regression_coefficients(coefficient_summary: pd.DataFrame, out_dir: Path):
    if coefficient_summary.empty:
        return

    datasets = list(dict.fromkeys(coefficient_summary["dataset"].tolist()))
    ncols = 2
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.8 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, dataset_name in zip(axes, datasets):
        frame = coefficient_summary[coefficient_summary["dataset"] == dataset_name].sort_values(
            "abs_coefficient_mean",
            ascending=True,
        )
        ax.barh(frame["predictor"], frame["coefficient_mean"])
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_title(dataset_name)
        ax.set_xlabel("Mean Standardized L1 Coefficient")

    for ax in axes[len(datasets):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "target_regression_coefficients.png", dpi=200)
    plt.close(fig)


def plot_regression_importance(importance_summary: pd.DataFrame, out_dir: Path):
    if importance_summary.empty:
        return

    datasets = list(dict.fromkeys(importance_summary["dataset"].tolist()))
    ncols = 2
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.8 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, dataset_name in zip(axes, datasets):
        frame = importance_summary[importance_summary["dataset"] == dataset_name].sort_values(
            "test_r2_drop_mean",
            ascending=True,
        )
        ax.barh(frame["predictor"], frame["test_r2_drop_mean"])
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_title(dataset_name)
        ax.set_xlabel("Mean Test R^2 Drop")

    for ax in axes[len(datasets):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "target_regression_importance.png", dpi=200)
    plt.close(fig)


def plot_prediction_scatter(prediction_df: pd.DataFrame, out_dir: Path):
    if prediction_df.empty:
        return

    datasets = list(dict.fromkeys(prediction_df["dataset"].tolist()))
    ncols = 2
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    true_col = f"{TARGET_METRIC}_true"
    pred_col = f"{TARGET_METRIC}_pred"
    for ax, dataset_name in zip(axes, datasets):
        frame = prediction_df[prediction_df["dataset"] == dataset_name]
        ax.scatter(frame[true_col], frame[pred_col], alpha=0.8)
        line_min = min(frame[true_col].min(), frame[pred_col].min())
        line_max = max(frame[true_col].max(), frame[pred_col].max())
        ax.plot([line_min, line_max], [line_min, line_max], linestyle="--", linewidth=1.0, color="black")
        ax.set_title(dataset_name)
        ax.set_xlabel("True Target Micro-F1")
        ax.set_ylabel("Predicted Target Micro-F1")

    for ax in axes[len(datasets):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "target_regression_test_predictions.png", dpi=200)
    plt.close(fig)


def finalize_outputs(raw_df: pd.DataFrame, out_dir: Path):
    raw_df = _add_derived_columns(raw_df)
    pair_mean_df = _build_pair_mean_frame(raw_df)

    pair_average_summary = _flatten_summary(
        raw_df,
        ["dataset", "model", "num_layers", "source", "target"],
        SUMMARY_METRICS,
    )
    model_layer_summary = _flatten_summary(
        raw_df,
        ["dataset", "model", "num_layers"],
        SUMMARY_METRICS,
    )
    correlation_df = compute_target_correlations(raw_df)
    (
        split_results,
        split_summary,
        coefficient_results,
        coefficient_summary,
        prediction_results,
        importance_results,
        importance_summary,
    ) = build_target_regression_outputs(raw_df)
    predictive_signal_summary = build_predictive_signal_summary(
        correlation_df,
        coefficient_summary,
        importance_summary,
    )

    raw_df.to_csv(out_dir / "all_bound_components.csv", index=False)
    pair_mean_df.to_csv(out_dir / "pair_means_by_model_layer.csv", index=False)
    pair_average_summary.to_csv(out_dir / "pair_average_summary.csv", index=False)
    model_layer_summary.to_csv(out_dir / "model_layer_summary.csv", index=False)
    correlation_df.to_csv(out_dir / "target_performance_correlations.csv", index=False)
    split_results.to_csv(out_dir / "target_regression_split_results.csv", index=False)
    split_summary.to_csv(out_dir / "target_regression_split_summary.csv", index=False)
    coefficient_results.to_csv(out_dir / "target_regression_coefficients_by_repeat.csv", index=False)
    coefficient_summary.to_csv(out_dir / "target_regression_coefficients.csv", index=False)
    prediction_results.to_csv(out_dir / "target_regression_predictions.csv", index=False)
    importance_results.to_csv(out_dir / "target_regression_importance_by_repeat.csv", index=False)
    importance_summary.to_csv(out_dir / "target_regression_importance.csv", index=False)
    predictive_signal_summary.to_csv(out_dir / "predictive_signal_summary.csv", index=False)

    plot_target_correlations(correlation_df, out_dir)
    plot_model_layer_summary(model_layer_summary, out_dir)
    plot_regression_coefficients(coefficient_summary, out_dir)
    plot_regression_importance(importance_summary, out_dir)
    plot_prediction_scatter(prediction_results, out_dir)
