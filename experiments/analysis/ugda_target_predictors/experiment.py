from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import math
import os
from pathlib import Path
import sys
import types
import warnings

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)

ROOT_DIR = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _bootstrap_repo_alias() -> None:
    learn_root = ROOT_DIR.parent
    learn_pkg = sys.modules.setdefault("Learn", types.ModuleType("Learn"))
    learn_pkg.__path__ = [str(learn_root)]

    clean_pkg = sys.modules.get("Learn.Clean_SCGDA")
    if clean_pkg is None:
        clean_pkg = types.ModuleType("Learn.Clean_SCGDA")
        sys.modules["Learn.Clean_SCGDA"] = clean_pkg
        setattr(learn_pkg, "Clean_SCGDA", clean_pkg)
    clean_pkg.__path__ = [str(ROOT_DIR)]


_bootstrap_repo_alias()

from plotting import plot_correlation_bars, plot_cv_bars, plot_predictor_scatter_grid

from Learn.Clean_SCGDA.models.build_model import build_model
from Learn.Clean_SCGDA.utils.ablation_utils.common import load_pair
from Learn.Clean_SCGDA.utils.config_utils import build_config, load_config
from Learn.Clean_SCGDA.utils.expt_utils import set_seed
from Learn.Clean_SCGDA.utils.filter_utils import conditional_mmd, mmd_rbf
from Learn.Clean_SCGDA.utils.train_utils.metrics import BaseMetric


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


ALL_REGIMES = ("source_only", "ugda", "joint_oracle", "target_only")
MAIN_REGIMES = ("source_only", "ugda")

LABEL_FREE_PREDICTORS = [
    "source_micro_f1",
    "target_entropy",
    "latent_mmd",
    "raw_mmd",
    "target_grad_global_mean",
    "smooth_mmd",
]

LABEL_AWARE_PREDICTORS = [
    "joint_oracle_target_micro_f1",
    "target_only_target_micro_f1",
    "latent_cmmd",
    "raw_cmmd",
    "latent_prototype_gap",
    "label_js",
]


@contextlib.contextmanager
def _suppress_output():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _resolve_device(device: str) -> str:
    return "cpu" if str(device).startswith("cuda") and not torch.cuda.is_available() else str(device)


def _domains_for_dataset(dataset: str) -> list[str]:
    cfg = load_config(ROOT_DIR / f"configs/data_configs/{dataset}.yaml")
    return list(cfg["domains"])


def _build_model_config(
    *,
    dataset: str,
    source: str,
    target: str,
    seed: int,
    device: str,
    epochs: int,
    mmd_weight: float,
):
    config_setup = {"data": dataset, "expt": "default", "model": "simgda"}
    update_config = {
        "expt": {
            "source": source,
            "target": target,
            "device": device,
            "seed": int(seed),
            "verbose": 0,
            "wandb_enabled": False,
            "project": "ugda_target_predictors",
        },
        "model": {
            "epochs": int(epochs),
            "use_mask": True,
            "mmd_weight": float(mmd_weight),
        },
    }
    with _suppress_output():
        config = build_config(
            config_setup,
            update_config=update_config,
            borrow="gnn",
            use_tuned=2,
        )
    config["model"]["epochs"] = int(epochs)
    config["model"]["mmd_weight"] = float(mmd_weight)
    config["model"]["use_mask"] = True
    return config


def _mask_features(data, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = data.val_mask.bool()
    return features[mask], data.y[mask]


def _micro_f1(metrics: dict) -> float:
    return float(metrics.get("micro_f1", float("nan")))


def _entropy_stats(logits: torch.Tensor) -> tuple[float, float]:
    probs = logits.detach().exp()
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
    top2 = torch.topk(probs, k=min(2, probs.size(1)), dim=1).values
    if top2.size(1) == 1:
        margin = top2[:, 0]
    else:
        margin = top2[:, 0] - top2[:, 1]
    return float(entropy.mean().item()), float(margin.mean().item())


def _sample_indices(n: int, max_samples: int, seed: int) -> torch.Tensor:
    if max_samples <= 0 or n <= max_samples:
        return torch.arange(n)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randperm(n, generator=g)[:max_samples]


def _subsample_features(
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    source_y: torch.Tensor,
    target_y: torch.Tensor,
    *,
    max_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    src_idx = _sample_indices(int(source_feat.size(0)), max_samples, seed)
    tgt_idx = _sample_indices(int(target_feat.size(0)), max_samples, seed + 13)
    return source_feat[src_idx], target_feat[tgt_idx], source_y[src_idx], target_y[tgt_idx]


def _prototype_gap(source_feat: torch.Tensor, target_feat: torch.Tensor, source_y: torch.Tensor, target_y: torch.Tensor) -> tuple[float, float]:
    classes = torch.unique(torch.cat([source_y, target_y], dim=0))
    gaps = []
    scales = []
    for cls in classes:
        src_mask = source_y == cls
        tgt_mask = target_y == cls
        if int(src_mask.sum().item()) == 0 or int(tgt_mask.sum().item()) == 0:
            continue
        src_center = source_feat[src_mask].mean(dim=0)
        tgt_center = target_feat[tgt_mask].mean(dim=0)
        gap = torch.norm(src_center - tgt_center, p=2)

        src_spread = torch.norm(source_feat[src_mask] - src_center, dim=1).mean()
        tgt_spread = torch.norm(target_feat[tgt_mask] - tgt_center, dim=1).mean()
        scale = 0.5 * (src_spread + tgt_spread)

        gaps.append(gap)
        scales.append(scale)

    if not gaps:
        return 0.0, 0.0

    gap_tensor = torch.stack(gaps)
    scale_tensor = torch.stack(scales).clamp_min(1e-8)
    return float(gap_tensor.mean().item()), float((gap_tensor / scale_tensor).mean().item())


def _js_divergence(labels_a: torch.Tensor, labels_b: torch.Tensor) -> float:
    classes = torch.unique(torch.cat([labels_a, labels_b], dim=0), sorted=True)
    pa = []
    pb = []
    for cls in classes.tolist():
        cls_t = torch.tensor(cls, device=labels_a.device)
        pa.append(float((labels_a == cls_t).float().mean().item()))
        pb.append(float((labels_b == cls_t).float().mean().item()))
    pa = np.asarray(pa, dtype=np.float64)
    pb = np.asarray(pb, dtype=np.float64)
    m = 0.5 * (pa + pb)
    eps = 1e-12
    pa = np.clip(pa, eps, 1.0)
    pb = np.clip(pb, eps, 1.0)
    m = np.clip(m, eps, 1.0)
    return float(0.5 * np.sum(pa * np.log(pa / m)) + 0.5 * np.sum(pb * np.log(pb / m)))


def _extract_latent(model, data) -> tuple[torch.Tensor, torch.Tensor]:
    model.simgda.eval()
    with torch.no_grad():
        latent = model.simgda.feat_bottleneck(data.x, data.edge_index)
    return _mask_features(data, latent)


def _masked_raw(data) -> tuple[torch.Tensor, torch.Tensor]:
    return _mask_features(data, data.x.float())


def _smoothness_proxy(model, data, *, max_nodes: int, seed: int) -> tuple[float, float]:
    val_idx = data.val_mask.bool().nonzero(as_tuple=False).view(-1)
    if int(val_idx.numel()) == 0:
        return float("nan"), float("nan")

    if 0 < max_nodes < int(val_idx.numel()):
        sample = _sample_indices(int(val_idx.numel()), max_nodes, seed).to(val_idx.device)
        val_idx = val_idx[sample]

    x = data.x.detach().clone().requires_grad_(True)
    logits = model.simgda(x, data.edge_index)
    pred = logits.detach().argmax(dim=1)

    grad_vals = []
    for step, idx in enumerate(val_idx.tolist()):
        score = logits[idx, pred[idx]]
        grad = torch.autograd.grad(
            score,
            x,
            retain_graph=step < (len(val_idx) - 1),
            create_graph=False,
        )[0]
        grad_vals.append(float(grad.norm(dim=1).max().detach().cpu().item()))

    if not grad_vals:
        return float("nan"), float("nan")
    grad_arr = np.asarray(grad_vals, dtype=np.float64)
    return float(grad_arr.mean()), float(grad_arr.max())


def _regime_train_args(regime: str, source_data, target_data) -> tuple[object, object, bool, float]:
    if regime == "source_only":
        return source_data, source_data, False, 0.0
    if regime == "ugda":
        return source_data, target_data, False, 0.1
    if regime == "joint_oracle":
        return source_data, target_data, True, 0.0
    if regime == "target_only":
        return target_data, target_data, False, 0.0
    raise ValueError(f"Unknown regime: {regime}")


def _fit_and_measure(
    *,
    dataset: str,
    source: str,
    target: str,
    seed: int,
    device: str,
    epochs: int,
    regime: str,
    source_data,
    target_data,
    metric_sample_size: int,
    grad_nodes: int,
) -> dict:
    train_source, train_target, oracle, mmd_weight = _regime_train_args(regime, source_data, target_data)
    config = _build_model_config(
        dataset=dataset,
        source=source,
        target=target,
        seed=seed,
        device=device,
        epochs=epochs,
        mmd_weight=mmd_weight,
    )
    config["model"]["in_dim"] = int(source_data.x.size(1))
    all_labels = torch.unique(torch.cat([source_data.y.view(-1), target_data.y.view(-1)], dim=0))
    config["model"]["num_classes"] = int(all_labels.numel())

    set_seed(int(seed))
    with _suppress_output():
        model = build_model(config, from_pygda=False)
        start = pd.Timestamp.now()
        model.fit(train_source, train_target, use_mask=True, oracle=oracle)
        end = pd.Timestamp.now()

    metrics = BaseMetric(config)
    source_logits, source_labels = model.predict(source_data, use_mask=True)
    target_logits, target_labels = model.predict(target_data, use_mask=True)
    source_metrics = metrics(source_logits, source_labels)
    target_metrics = metrics(target_logits, target_labels)

    source_latent, source_y = _extract_latent(model, source_data)
    target_latent, target_y = _extract_latent(model, target_data)
    source_raw, _ = _masked_raw(source_data)
    target_raw, _ = _masked_raw(target_data)

    sub_seed = int(seed) + 1009 * (ALL_REGIMES.index(regime) + 1)
    s_lat, t_lat, ys_lat, yt_lat = _subsample_features(
        source_latent.detach().cpu(),
        target_latent.detach().cpu(),
        source_y.detach().cpu(),
        target_y.detach().cpu(),
        max_samples=metric_sample_size,
        seed=sub_seed,
    )
    s_raw, t_raw, ys_raw, yt_raw = _subsample_features(
        source_raw.detach().cpu(),
        target_raw.detach().cpu(),
        source_y.detach().cpu(),
        target_y.detach().cpu(),
        max_samples=metric_sample_size,
        seed=sub_seed + 37,
    )

    latent_mmd = float(mmd_rbf(s_lat, t_lat).item())
    latent_cmmd = float(conditional_mmd(s_lat, t_lat, ys_lat, yt_lat).item())
    raw_mmd = float(mmd_rbf(s_raw, t_raw).item())
    raw_cmmd = float(conditional_mmd(s_raw, t_raw, ys_raw, yt_raw).item())
    latent_gap, latent_gap_norm = _prototype_gap(s_lat, t_lat, ys_lat, yt_lat)
    raw_gap, raw_gap_norm = _prototype_gap(s_raw, t_raw, ys_raw, yt_raw)

    target_entropy, target_margin = _entropy_stats(target_logits)
    source_entropy, source_margin = _entropy_stats(source_logits)
    grad_mean, grad_max = _smoothness_proxy(
        model,
        target_data,
        max_nodes=grad_nodes,
        seed=sub_seed + 91,
    )

    row = {
        "dataset": dataset,
        "source": source,
        "target": target,
        "seed": int(seed),
        "regime": regime,
        "train_time_sec": float((end - start).total_seconds()),
        "source_micro_f1": _micro_f1(source_metrics),
        "source_macro_f1": float(source_metrics.get("macro_f1", float("nan"))),
        "target_micro_f1": _micro_f1(target_metrics),
        "target_macro_f1": float(target_metrics.get("macro_f1", float("nan"))),
        "generalization_gap": _micro_f1(source_metrics) - _micro_f1(target_metrics),
        "source_entropy": source_entropy,
        "source_margin": source_margin,
        "target_entropy": target_entropy,
        "target_margin": target_margin,
        "raw_mmd": raw_mmd,
        "raw_cmmd": raw_cmmd,
        "latent_mmd": latent_mmd,
        "latent_cmmd": latent_cmmd,
        "raw_prototype_gap": raw_gap,
        "raw_prototype_gap_norm": raw_gap_norm,
        "latent_prototype_gap": latent_gap,
        "latent_prototype_gap_norm": latent_gap_norm,
        "label_js": _js_divergence(ys_lat, yt_lat),
        "target_grad_global_mean": grad_mean,
        "target_grad_global_max": grad_max,
    }
    row["smooth_mmd"] = row["target_grad_global_mean"] * row["latent_mmd"]
    row["source_error"] = 1.0 - row["source_micro_f1"]
    row["target_error"] = 1.0 - row["target_micro_f1"]

    del model
    _cleanup()
    return row


def run_experiment(args: argparse.Namespace) -> Path:
    device = _resolve_device(args.device)
    out_root = Path(args.out_root)
    run_name = args.run_name or f"run_{pd.Timestamp.now().strftime('%m%d_%H%M%S')}"
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "datasets": list(args.datasets),
        "seeds": [int(seed) for seed in args.seeds],
        "device": device,
        "epochs": int(args.epochs),
        "metric_sample_size": int(args.metric_sample_size),
        "grad_nodes": int(args.grad_nodes),
        "regimes": list(args.regimes),
        "borrow_tuned_from": "gnn",
        "model": "simgda",
    }
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    rows = []
    total = 0
    for dataset in args.datasets:
        domains = _domains_for_dataset(dataset)
        total += len(args.seeds) * len(args.regimes) * len(domains) * (len(domains) - 1)

    step = 0
    for dataset in args.datasets:
        domains = _domains_for_dataset(dataset)
        for seed in args.seeds:
            for source in domains:
                for target in domains:
                    if source == target:
                        continue
                    source_data, target_data = load_pair(dataset, source, target, device, int(seed))
                    for regime in args.regimes:
                        step += 1
                        row = _fit_and_measure(
                            dataset=dataset,
                            source=source,
                            target=target,
                            seed=int(seed),
                            device=device,
                            epochs=int(args.epochs),
                            regime=regime,
                            source_data=source_data,
                            target_data=target_data,
                            metric_sample_size=int(args.metric_sample_size),
                            grad_nodes=int(args.grad_nodes),
                        )
                        rows.append(row)
                        print(
                            f"[{step}/{total}] {dataset} {source}->{target} seed={seed} regime={regime} "
                            f"src={row['source_micro_f1']:.4f} tgt={row['target_micro_f1']:.4f} "
                            f"mmd={row['latent_mmd']:.4f} cmmd={row['latent_cmmd']:.4f} "
                            f"ent={row['target_entropy']:.4f}"
                        )

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(out_dir / "raw_runs.csv", index=False)

    group_cols = ["dataset", "source", "target", "regime"]
    numeric_cols = [col for col in raw_df.columns if col not in group_cols + ["seed"]]
    scenario_df = raw_df.groupby(group_cols, as_index=False)[numeric_cols].mean()
    scenario_df.to_csv(out_dir / "scenario_means.csv", index=False)

    scenario_wide = scenario_df.pivot(index=["dataset", "source", "target"], columns="regime")
    scenario_wide.columns = [f"{regime}_{metric}" for metric, regime in scenario_wide.columns]
    scenario_wide = scenario_wide.reset_index()
    for regime in ALL_REGIMES:
        target_col = f"{regime}_target_micro_f1"
        source_col = f"{regime}_source_micro_f1"
        if target_col in scenario_wide.columns:
            scenario_wide[f"{regime}_target_error"] = 1.0 - scenario_wide[target_col]
        if source_col in scenario_wide.columns:
            scenario_wide[f"{regime}_source_error"] = 1.0 - scenario_wide[source_col]
    scenario_wide.to_csv(out_dir / "scenario_summary_wide.csv", index=False)

    corr_rows = []
    for regime in MAIN_REGIMES:
        target_col = f"{regime}_target_micro_f1"
        for family, predictors in (("label_free", LABEL_FREE_PREDICTORS), ("label_aware", LABEL_AWARE_PREDICTORS)):
            for predictor in predictors:
                pred_col = predictor if predictor in scenario_wide.columns else f"{regime}_{predictor}"
                if pred_col not in scenario_wide.columns:
                    continue
                for dataset_name, df_part in [("all", scenario_wide)] + [
                    (name, group) for name, group in scenario_wide.groupby("dataset")
                ]:
                    subset = df_part[[pred_col, target_col]].dropna()
                    if len(subset) < 2:
                        spearman = float("nan")
                        pearson = float("nan")
                    else:
                        spearman = float(subset[pred_col].corr(subset[target_col], method="spearman"))
                        pearson = float(subset[pred_col].corr(subset[target_col], method="pearson"))
                    corr_rows.append(
                        {
                            "regime": regime,
                            "family": family,
                            "dataset": dataset_name,
                            "predictor": predictor,
                            "predictor_col": pred_col,
                            "spearman": spearman,
                            "pearson": pearson,
                            "n": int(len(subset)),
                        }
                    )
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(out_dir / "predictor_correlations.csv", index=False)

    cv_spec = {
        "source_only": {
            "source_only": ["source_only_source_error"],
            "mmd_only": ["source_only_latent_mmd"],
            "uncertainty_only": ["source_only_target_entropy"],
            "source_plus_mmd": ["source_only_source_error", "source_only_latent_mmd"],
            "source_plus_uncertainty": ["source_only_source_error", "source_only_target_entropy"],
            "source_plus_smooth_mmd": ["source_only_source_error", "source_only_smooth_mmd"],
            "source_plus_joint": ["source_only_source_error", "joint_oracle_target_error"],
            "bound_mmd": ["source_only_source_error", "source_only_latent_mmd", "joint_oracle_target_error"],
            "bound_smooth_mmd": ["source_only_source_error", "source_only_smooth_mmd", "joint_oracle_target_error"],
            "bound_cmmd": ["source_only_source_error", "source_only_latent_cmmd", "joint_oracle_target_error"],
        },
        "ugda": {
            "source_only": ["ugda_source_error"],
            "mmd_only": ["ugda_latent_mmd"],
            "uncertainty_only": ["ugda_target_entropy"],
            "source_plus_mmd": ["ugda_source_error", "ugda_latent_mmd"],
            "source_plus_uncertainty": ["ugda_source_error", "ugda_target_entropy"],
            "source_plus_smooth_mmd": ["ugda_source_error", "ugda_smooth_mmd"],
            "source_plus_joint": ["ugda_source_error", "joint_oracle_target_error"],
            "bound_mmd": ["ugda_source_error", "ugda_latent_mmd", "joint_oracle_target_error"],
            "bound_smooth_mmd": ["ugda_source_error", "ugda_smooth_mmd", "joint_oracle_target_error"],
            "bound_cmmd": ["ugda_source_error", "ugda_latent_cmmd", "joint_oracle_target_error"],
        },
    }

    cv_rows = []
    for regime in MAIN_REGIMES:
        target_col = f"{regime}_target_error"
        for model_name, feature_cols in cv_spec[regime].items():
            cols = feature_cols + [target_col, "dataset", "source", "target"]
            subset = scenario_wide[cols].dropna()
            if subset.empty:
                continue
            for holdout, test_df in subset.groupby("dataset"):
                train_df = subset[subset["dataset"] != holdout]
                if len(train_df) < len(feature_cols) + 1 or test_df.empty:
                    continue
                x_train = train_df[feature_cols].to_numpy(dtype=np.float64)
                y_train = train_df[target_col].to_numpy(dtype=np.float64)
                x_test = test_df[feature_cols].to_numpy(dtype=np.float64)
                y_test = test_df[target_col].to_numpy(dtype=np.float64)

                x_train_aug = np.concatenate([np.ones((x_train.shape[0], 1)), x_train], axis=1)
                x_test_aug = np.concatenate([np.ones((x_test.shape[0], 1)), x_test], axis=1)
                coef, *_ = np.linalg.lstsq(x_train_aug, y_train, rcond=None)
                pred = np.clip(x_test_aug @ coef, 0.0, 1.0)
                mae = np.abs(pred - y_test).mean()
                rmse = math.sqrt(float(np.mean((pred - y_test) ** 2)))
                denom = float(np.sum((y_test - y_test.mean()) ** 2))
                r2 = float("nan") if denom <= 1e-12 else float(1.0 - np.sum((pred - y_test) ** 2) / denom)

                for idx, (_, row) in enumerate(test_df.iterrows()):
                    cv_rows.append(
                        {
                            "regime": regime,
                            "model_name": model_name,
                            "holdout_dataset": holdout,
                            "dataset": row["dataset"],
                            "source": row["source"],
                            "target": row["target"],
                            "mae": float(mae),
                            "rmse": float(rmse),
                            "r2": r2,
                            "true_target_error": float(y_test[idx]),
                            "pred_target_error": float(pred[idx]),
                        }
                    )
    cv_df = pd.DataFrame(
        cv_rows,
        columns=[
            "regime",
            "model_name",
            "holdout_dataset",
            "dataset",
            "source",
            "target",
            "mae",
            "rmse",
            "r2",
            "true_target_error",
            "pred_target_error",
        ],
    )
    cv_df.to_csv(out_dir / "predictor_cv.csv", index=False)

    for regime in MAIN_REGIMES:
        plot_correlation_bars(corr_df, out_dir / f"{regime}_label_free_correlations.png", regime, "label_free")
        plot_correlation_bars(corr_df, out_dir / f"{regime}_label_aware_correlations.png", regime, "label_aware")
        plot_predictor_scatter_grid(
            scenario_wide,
            out_dir / f"{regime}_scatter_grid.png",
            regime,
            predictors=["source_micro_f1", "target_entropy", "latent_mmd", "latent_cmmd"],
        )
        plot_cv_bars(cv_df, out_dir / f"{regime}_cv_mae.png", regime)

    report_lines = []
    for regime in MAIN_REGIMES:
        report_lines.append(f"## {regime}")
        for family in ("label_free", "label_aware"):
            subset = corr_df[
                (corr_df["regime"] == regime)
                & (corr_df["family"] == family)
                & (corr_df["dataset"] == "all")
            ].copy()
            if not subset.empty:
                top_idx = subset["spearman"].abs().sort_values(ascending=False).index[:3]
                top = subset.loc[top_idx]
                report_lines.append(f"{family}:")
                for _, row in top.iterrows():
                    report_lines.append(
                        f"- {row['predictor']}: spearman={row['spearman']:.3f}, pearson={row['pearson']:.3f}"
                    )
        subset_cv = cv_df[cv_df["regime"] == regime] if "regime" in cv_df.columns else pd.DataFrame()
        if not subset_cv.empty:
            summary = subset_cv.groupby("model_name", as_index=False)[["mae", "rmse"]].mean().sort_values("mae")
            best = summary.iloc[0]
            report_lines.append(
                f"best_cv_model: {best['model_name']} (mae={best['mae']:.3f}, rmse={best['rmse']:.3f})"
            )
        report_lines.append("")
    (out_dir / "report.md").write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

    print(f"results_dir={out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predictors of UGDA target performance.")
    parser.add_argument("--datasets", nargs="+", default=["airport", "blog", "citation"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2025, 2026])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--metric-sample-size", type=int, default=1024)
    parser.add_argument("--grad-nodes", type=int, default=32)
    parser.add_argument("--out-root", default="./__saved__/analysis/ugda_target_predictors")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--regimes", nargs="+", default=list(ALL_REGIMES), choices=list(ALL_REGIMES))
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
