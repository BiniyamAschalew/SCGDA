from __future__ import annotations

import argparse
from pathlib import Path

import torch

from  experiments.analysis.airport_source_only_gnn_layers.experiment import run_experiment


TRANSFER_SETTINGS = {
    "airport": [
        ("USA", "BRAZIL"),
    ],
    "blog": [
        ("Blog2", "Blog1"),
    ],
    "citation": [
        ("ACMv9", "DBLPv7"),
    ]
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
        description="Run source-only GNN layerwise snapshot experiments for transfer scenarios."
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument(
        "--out-root",
        type=str,
        default="./__saved__/analysis/airport_source_only_gnn_layers",
    )
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.dataset is not None or args.source is not None or args.target is not None:
        if not (args.dataset and args.source and args.target):
            raise ValueError("When overriding from CLI, --dataset, --source, and --target must all be provided.")
        scenarios = [(args.dataset, args.source, args.target)]
    else:
        scenarios = _flatten_scenarios(TRANSFER_SETTINGS)

    result_dirs = []
    for dataset, source, target in scenarios:
        print(f"Running scenario: {dataset} {source} -> {target}")
        out_dir = run_experiment(
            dataset=dataset,
            source=source,
            target=target,
            seed=args.seed,
            epochs=args.epochs,
            device=device,
            verbose=args.verbose,
            out_root=args.out_root,
        )
        result_dirs.append((dataset, source, target, Path(out_dir)))

    print("\nCompleted runs:")
    for dataset, source, target, out_dir in result_dirs:
        print(f"- {dataset} {source}->{target}: {out_dir}")


if __name__ == "__main__":
    main()
