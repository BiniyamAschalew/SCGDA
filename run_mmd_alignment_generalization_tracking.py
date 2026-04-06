from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from Learn.Clean_SCGDA.experiments.analysis.mmd_alignment_generalization_tracking.experiment import run_experiment


TRANSFER_SETTINGS = {
    "airport": [
        ("USA", "BRAZIL"),
        ("EUROPE", "USA"),
    ],
    "blog": [
        ("Blog1", "Blog2"),
    ],
    "citation": [
        ("ACMv9", "DBLPv7"),
        ("Citationv1", "ACMv9"),
    ],
}


def _flatten_scenarios(transfer_settings: dict) -> list[tuple[str, str, str]]:
    scenarios = []
    for dataset, pairs in transfer_settings.items():
        if isinstance(pairs, dict):
            for source, targets in pairs.items():
                if isinstance(targets, list):
                    for target in targets:
                        scenarios.append((dataset, source, target))
                else:
                    scenarios.append((dataset, source, targets))
        else:
            for source, target in pairs:
                scenarios.append((dataset, source, target))
    return scenarios


def main():
    parser = argparse.ArgumentParser(
        description="Compare source-only, MMD-aligned, oracle, and oracle+MMD two-layer GNNs."
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--num-seeds", type=int, default=7)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hid-dim", type=int, default=64)
    parser.add_argument("--dropout-ratio", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--mmd-weight", type=float, default=0.1)
    parser.add_argument("--mmd-sampling-num", type=int, default=256)
    parser.add_argument("--mmd-times", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument(
        "--out-root",
        type=str,
        default="./__saved__/analysis/mmd_alignment_generalization_tracking",
    )
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    seeds = tuple(range(int(args.seed_start), int(args.seed_start) + int(args.num_seeds)))

    if args.dataset is not None or args.source is not None or args.target is not None:
        if not (args.dataset and args.source and args.target):
            raise ValueError("When overriding from CLI, --dataset, --source, and --target must all be provided.")
        scenarios = [(args.dataset, args.source, args.target)]
    else:
        scenarios = _flatten_scenarios(TRANSFER_SETTINGS)

    run_root = Path(args.out_root) / f"run_{time.strftime('%m%d_%H%M%S')}_seeds{seeds[0]}to{seeds[-1]}"
    run_root.mkdir(parents=True, exist_ok=False)
    with open(run_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "scenarios": [{"dataset": d, "source": s, "target": t} for d, s, t in scenarios],
                "seeds": list(seeds),
                "epochs": int(args.epochs),
                "hid_dim": int(args.hid_dim),
                "dropout_ratio": float(args.dropout_ratio),
                "activation": str(args.activation),
                "mmd_weight": float(args.mmd_weight),
                "mmd_sampling_num": int(args.mmd_sampling_num),
                "mmd_times": int(args.mmd_times),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "device": device,
            },
            f,
            indent=2,
        )

    result_dirs = []
    for dataset, source, target in scenarios:
        print(f"Running scenario: {dataset} {source} -> {target}")
        out_dir = run_experiment(
            dataset=dataset,
            source=source,
            target=target,
            seeds=seeds,
            epochs=args.epochs,
            hid_dim=args.hid_dim,
            dropout_ratio=args.dropout_ratio,
            activation=args.activation,
            mmd_weight=args.mmd_weight,
            mmd_sampling_num=args.mmd_sampling_num,
            mmd_times=args.mmd_times,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            verbose=args.verbose,
            out_root=args.out_root,
            run_root=run_root,
        )
        result_dirs.append((dataset, source, target, Path(out_dir)))

    print("\nCompleted runs:")
    for dataset, source, target, out_dir in result_dirs:
        print(f"- {dataset} {source}->{target}: {out_dir}")


if __name__ == "__main__":
    main()
