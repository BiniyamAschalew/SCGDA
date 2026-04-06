from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from Learn.Clean_SCGDA.experiments.motivation.prob_form.plot import plot_embedding_evolution, plot_raw_features, plot_training_curves
from Learn.Clean_SCGDA.experiments.motivation.prob_form.synthetic_data import generate_synthetic_da_data, save_synthetic_data
from Learn.Clean_SCGDA.experiments.motivation.prob_form.train import TrainingConfig, combine_histories, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic MMD domain-adaptation analysis with embedding evolution plots."
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--projection-seed", type=int, default=17)

    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--n-per-class", type=int, default=400)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--class-sep", type=float, default=4.0)
    parser.add_argument("--source-std", type=float, default=0.7)
    parser.add_argument("--target-std-scale", type=float, default=1.15)
    parser.add_argument("--rotation-mix", type=float, default=0.45)
    parser.add_argument("--target-shift-scale", type=float, default=1.2)
    parser.add_argument("--class-shift-scale", type=float, default=0.35)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--snapshot-stride", type=int, default=10)
    parser.add_argument("--lambda-mmd", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def resolve_output_dir(output_dir: str | None) -> Path:
    if output_dir:
        path = Path(output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("prob_form/results") / f"run_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_history_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_snapshots(base_dir: Path, regime: str, snapshots: dict[int, dict[str, object]]) -> None:
    regime_dir = base_dir / "snapshots" / regime
    regime_dir.mkdir(parents=True, exist_ok=True)
    for epoch, payload in snapshots.items():
        path = regime_dir / f"epoch_{epoch:04d}.npz"
        npz_payload = {k: v for k, v in payload.items()}
        np.savez_compressed(path, **npz_payload)


def main() -> None:
    args = parse_args()
    out_dir = resolve_output_dir(args.output_dir)

    synthetic = generate_synthetic_da_data(
        n_classes=args.n_classes,
        n_per_class=args.n_per_class,
        feature_dim=args.feature_dim,
        class_sep=args.class_sep,
        source_std=args.source_std,
        target_std_scale=args.target_std_scale,
        rotation_mix=args.rotation_mix,
        target_shift_scale=args.target_shift_scale,
        class_shift_scale=args.class_shift_scale,
        seed=args.seed,
    )
    save_synthetic_data(out_dir / "synthetic_data.npz", synthetic)

    cfg = TrainingConfig(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        snapshot_stride=args.snapshot_stride,
        seed=args.seed,
        device=args.device,
    )

    baseline = run_training(regime="baseline", data=synthetic, cfg=cfg, lambda_mmd=0.0)
    mmd = run_training(regime="mmd", data=synthetic, cfg=cfg, lambda_mmd=args.lambda_mmd)

    rows = combine_histories([baseline, mmd])
    save_history_csv(out_dir / "metrics.csv", rows)

    save_snapshots(out_dir, baseline.regime, baseline.snapshots)
    save_snapshots(out_dir, mmd.regime, mmd.snapshots)

    histories_by_regime = {
        baseline.regime: baseline.history,
        mmd.regime: mmd.history,
    }
    snapshots_by_regime = {
        baseline.regime: baseline.snapshots,
        mmd.regime: mmd.snapshots,
    }
    snapshot_epochs = sorted(set(baseline.snapshots.keys()) & set(mmd.snapshots.keys()))

    plot_raw_features(
        source_x=synthetic.source_x,
        target_x=synthetic.target_x,
        source_y=synthetic.source_y,
        target_y=synthetic.target_y,
        projection_seed=args.projection_seed,
        output_path=out_dir / "raw_features_fixed_projection.png",
    )
    plot_embedding_evolution(
        snapshots_by_regime=snapshots_by_regime,
        histories_by_regime=histories_by_regime,
        source_y=synthetic.source_y,
        target_y=synthetic.target_y,
        snapshot_epochs=snapshot_epochs,
        projection_seed=args.projection_seed,
        output_path=out_dir / "embedding_evolution_fixed_projection.png",
    )
    plot_training_curves(
        histories_by_regime=histories_by_regime,
        output_path=out_dir / "training_curves.png",
    )

    config = {
        "seed": args.seed,
        "projection_seed": args.projection_seed,
        "n_classes": args.n_classes,
        "n_per_class": args.n_per_class,
        "feature_dim": args.feature_dim,
        "class_sep": args.class_sep,
        "source_std": args.source_std,
        "target_std_scale": args.target_std_scale,
        "rotation_mix": args.rotation_mix,
        "target_shift_scale": args.target_shift_scale,
        "class_shift_scale": args.class_shift_scale,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "snapshot_stride": args.snapshot_stride,
        "lambda_mmd": args.lambda_mmd,
        "device": args.device,
    }
    with (out_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved analysis artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
