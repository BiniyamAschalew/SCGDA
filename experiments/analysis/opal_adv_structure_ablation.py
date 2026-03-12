"""Ablation: adversarial probe learning for OPAL structural matching.

Runs two regimes:
1) non_adversarial_probe: learnable probe enabled but no adversarial updates
2) adversarial_probe: learnable probe enabled with adversarial updates

Saves:
- per-run metrics
- per-epoch parameter traces
- summary table
- evolution plots
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.build_dataset import build_dataset
from models.build_model import build_model
from utils.config_utils import build_config
from utils.expt_utils import set_seed
from utils.train_utils.metrics import BaseMetric


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run_once(
    *,
    dataset: str,
    source: str,
    target: str,
    device: str,
    seed: int,
    epochs: int,
    probe_adv_steps: int,
    probe_adv_lr: float,
    probe_reg_weight: float,
    distance: str,
    distance_sampling_num: int,
) -> tuple[dict, pd.DataFrame]:
    set_seed(seed)

    config_setup = {"data": dataset, "expt": "default", "model": "opal"}
    update_config = {
        "expt": {
            "source": source,
            "target": target,
            "device": device,
            "seed": seed,
            "wandb_enabled": False,
            "verbose": 0,
        },
        "model": {
            "epochs": epochs,
            "learnable_probe": True,
            "probe_adv_steps": int(probe_adv_steps),
            "probe_adv_lr": float(probe_adv_lr),
            "probe_reg_weight": float(probe_reg_weight),
            "distance": distance,
            "distance_sampling_num": int(distance_sampling_num),
        },
    }
    cfg = build_config(config_setup, update_config)

    source_dataset, target_dataset = build_dataset(cfg)
    source_data = source_dataset[0].to(device)
    target_data = target_dataset[0].to(device)

    cfg["model"]["in_dim"] = int(source_data.x.shape[1])
    cfg["model"]["num_classes"] = int(source_data.y.unique().numel())

    model = build_model(cfg)
    model.fit(source_data, target_data)

    metric = BaseMetric(cfg)
    target_logits, target_labels = model.predict(target_data, source=False)
    target_result = metric(target_logits, target_labels)
    source_logits, source_labels = model.predict(source_data, source=True)
    source_result = metric(source_logits, source_labels)

    metrics_row = {
        "seed": seed,
        "source_micro_f1": float(source_result.get("micro_f1", float("nan"))),
        "source_macro_f1": float(source_result.get("macro_f1", float("nan"))),
        "target_micro_f1": float(target_result.get("micro_f1", float("nan"))),
        "target_macro_f1": float(target_result.get("macro_f1", float("nan"))),
    }

    trace_df = pd.DataFrame(getattr(model, "param_trace", []))
    return metrics_row, trace_df


def _plot_group(df: pd.DataFrame, columns: list[str], title: str, out_path: Path) -> None:
    if df.empty or not columns:
        return
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for col in cols:
        ax.plot(df["epoch"], df[col], linewidth=2.0, label=col)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_evolution(trace_df: pd.DataFrame, out_dir: Path) -> None:
    if trace_df.empty:
        return

    # average over repeats for each regime+epoch
    num_cols = [c for c in trace_df.columns if c not in {"regime", "repeat"}]
    agg = (
        trace_df.groupby(["regime", "epoch"], as_index=False)[num_cols]
        .mean(numeric_only=True)
        .sort_values(["regime", "epoch"])
    )

    for regime in sorted(agg["regime"].unique()):
        df = agg[agg["regime"] == regime].copy()
        s_cols = [c for c in df.columns if c.startswith("s_param_")]
        t_cols = [c for c in df.columns if c.startswith("t_param_")]
        probe_cols = [c for c in ["probe_mean_l2", "probe_std_mean", "probe_std_min", "probe_std_max"] if c in df.columns]
        layer_cols = [c for c in df.columns if c.startswith("conv_") or c == "cls_param_l2"]

        _plot_group(df, s_cols, f"{regime}: Source Filter Coefficients", out_dir / f"{regime}_s_params.png")
        _plot_group(df, t_cols, f"{regime}: Target Filter Coefficients", out_dir / f"{regime}_t_params.png")
        _plot_group(df, probe_cols, f"{regime}: Probe Generator Stats", out_dir / f"{regime}_probe_stats.png")
        _plot_group(df, layer_cols, f"{regime}: Layer Parameter Norms", out_dir / f"{regime}_layer_norms.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="OPAL adversarial structure-matching ablation.")
    parser.add_argument("--dataset", type=str, default="citation")
    parser.add_argument("--source", type=str, default="ACMv9")
    parser.add_argument("--target", type=str, default="DBLPv7")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distance", type=str, default="mmd", choices=["mmd", "sinkhorn"])
    parser.add_argument("--distance-sampling-num", type=int, default=1000)
    parser.add_argument("--probe-adv-steps", type=int, default=1)
    parser.add_argument("--probe-adv-lr", type=float, default=3e-3)
    parser.add_argument("--probe-reg-weight", type=float, default=1e-3)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./__saved__/analysis/opal_adv_structure_ablation",
    )
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    stamp = time.strftime("%m%d_%H%M%S")
    out_dir = out_root / f"{args.dataset}_{args.source}_{args.target}_{stamp}"
    _ensure_dir(out_dir)

    regimes = [
        ("non_adversarial_probe", 0),
        ("adversarial_probe", int(args.probe_adv_steps)),
    ]

    metric_rows = []
    trace_rows = []
    for regime_name, probe_adv_steps in regimes:
        for repeat in range(args.repeats):
            cur_seed = int(args.seed + repeat)
            metrics_row, trace_df = _run_once(
                dataset=args.dataset,
                source=args.source,
                target=args.target,
                device=args.device,
                seed=cur_seed,
                epochs=args.epochs,
                probe_adv_steps=probe_adv_steps,
                probe_adv_lr=args.probe_adv_lr,
                probe_reg_weight=args.probe_reg_weight,
                distance=args.distance,
                distance_sampling_num=args.distance_sampling_num,
            )
            metrics_row["regime"] = regime_name
            metrics_row["repeat"] = repeat
            metric_rows.append(metrics_row)

            if not trace_df.empty:
                trace_df["regime"] = regime_name
                trace_df["repeat"] = repeat
                trace_rows.append(trace_df)

            print(
                f"[{regime_name}] repeat={repeat} seed={cur_seed} "
                f"target_micro_f1={metrics_row['target_micro_f1']:.4f}"
            )

    metrics_df = pd.DataFrame(metric_rows)
    trace_df = pd.concat(trace_rows, ignore_index=True) if trace_rows else pd.DataFrame()

    summary_df = (
        metrics_df.groupby("regime", as_index=False)[
            ["source_micro_f1", "source_macro_f1", "target_micro_f1", "target_macro_f1"]
        ]
        .agg(["mean", "std"])
    )
    summary_df.columns = ["_".join(c).strip("_") for c in summary_df.columns.to_flat_index()]

    metrics_df.to_csv(out_dir / "metrics_per_run.csv", index=False)
    if not trace_df.empty:
        trace_df.to_csv(out_dir / "param_trace.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    _plot_evolution(trace_df, out_dir)

    print(f"Saved ablation artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
