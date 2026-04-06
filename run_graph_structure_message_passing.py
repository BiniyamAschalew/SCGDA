from __future__ import annotations

import argparse
from pathlib import Path

import torch

from  experiments.analysis.graph_structure_message_passing.experiment import (
    DEFAULT_DATASETS,
    run_experiment,
)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze repeated message passing with label, random, and PCA feature initializations."
    )
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--out-root",
        type=str,
        default="./__saved__/analysis/graph_structure_message_passing",
    )
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_key = str(args.dataset).lower()
    datasets = DEFAULT_DATASETS if dataset_key == "all" else (dataset_key,)

    if args.domain is not None and len(datasets) != 1:
        raise ValueError("--domain can only be used together with a single --dataset.")

    out_dir = run_experiment(
        datasets=datasets,
        max_layers=args.layers,
        seed=args.seed,
        num_seeds=args.num_seeds,
        device=device,
        out_root=args.out_root,
        domain=args.domain,
    )
    print(f"Saved outputs to {Path(out_dir)}")


if __name__ == "__main__":
    main()
