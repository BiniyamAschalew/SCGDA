from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from  experiments.analysis.multi_model_domain_generalization_tracking.experiment import (
    MODEL_VARIANTS,
    build_variant_dataset_rollup,
    run_multi_seed_variant_experiment,
)
from  experiments.analysis.plain_gnn_cross_domain.log_plot import make_output_dir


EXPERIMENT_SETTINGS = {
    "airport": {
        "sources": ["BRAZIL", "EUROPE", "USA"],
        "eval_domains": ["BRAZIL", "EUROPE", "USA"],
        "epochs": 200,
        "num_layers": 3,
        "hid_dim": 64,
        "dropout_ratio": 0.1,
    },
    "blog": {
        "sources": ["Blog1", "Blog2"],
        "eval_domains": ["Blog1", "Blog2"],
        "epochs": 200,
        "num_layers": 3,
        "hid_dim": 64,
        "dropout_ratio": 0.1,
    },
    "citation": {
        "sources": ["ACMv9", "Citationv1", "DBLPv7"],
        "eval_domains": ["ACMv9", "Citationv1", "DBLPv7"],
        "epochs": 200,
        "num_layers": 3,
        "hid_dim": 64,
        "dropout_ratio": 0.1,
    },
    "twitch": {
        "sources": ["EN", "DE"],
        "eval_domains": ["EN", "DE"],
        "epochs": 200,
        "num_layers": 3,
        "hid_dim": 64,
        "dropout_ratio": 0.1,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple source-trained alignment models during domain generalization tracking."
    )
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--hid-dim", type=int, default=None)
    parser.add_argument("--dropout-ratio", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument(
        "--out-root",
        type=str,
        default="./__saved__/analysis/multi_model_domain_generalization_tracking",
    )
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    seeds = [int(args.seed) + idx for idx in range(int(args.num_seeds))]

    requested_dataset = str(args.dataset).lower()
    if requested_dataset == "all":
        dataset_keys = list(EXPERIMENT_SETTINGS.keys())
        run_root = make_output_dir(
            out_root=str(Path(args.out_root)),
            run_name=f"run_{time.strftime('%m%d_%H%M%S')}_seeds{seeds[0]}to{seeds[-1]}",
        )
    else:
        if requested_dataset not in EXPERIMENT_SETTINGS:
            raise ValueError(
                f"Unknown dataset setting '{args.dataset}'. Available: {sorted(EXPERIMENT_SETTINGS) + ['all']}"
            )
        dataset_keys = [requested_dataset]
        dataset_out_root = Path(args.out_root) / requested_dataset
        run_root = make_output_dir(
            out_root=str(dataset_out_root),
            run_name=f"run_{time.strftime('%m%d_%H%M%S')}_seeds{seeds[0]}to{seeds[-1]}",
        )

    requested_model = str(args.model).lower()
    if requested_model == "all":
        model_variants = list(MODEL_VARIANTS.keys())
    else:
        if requested_model not in MODEL_VARIANTS:
            raise ValueError(
                f"Unknown model variant '{args.model}'. Available: {sorted(MODEL_VARIANTS) + ['all']}"
            )
        model_variants = [requested_model]

    result_dirs = []
    for dataset_key in dataset_keys:
        settings = dict(EXPERIMENT_SETTINGS[dataset_key])
        if args.source is not None and len(dataset_keys) > 1:
            raise ValueError("--source can only be used when running a single dataset.")

        sources = [args.source] if args.source is not None else list(settings["sources"])
        epochs = args.epochs if args.epochs is not None else int(settings["epochs"])
        num_layers = args.num_layers if args.num_layers is not None else int(settings["num_layers"])
        hid_dim = args.hid_dim if args.hid_dim is not None else int(settings["hid_dim"])
        dropout_ratio = args.dropout_ratio if args.dropout_ratio is not None else float(settings["dropout_ratio"])
        eval_domains = list(settings["eval_domains"])
        dataset_root = run_root / dataset_key if len(dataset_keys) > 1 else run_root

        for model_variant in model_variants:
            model_root = dataset_root / model_variant
            case_dirs = []
            for source in sources:
                print(
                    f"Running scenario: dataset={dataset_key} model={model_variant} "
                    f"source={source} eval={eval_domains} seeds={seeds}"
                )
                target_domains = [domain for domain in eval_domains if domain != source]
                case_dir = model_root / f"{source}_to_{'-'.join(target_domains) if target_domains else 'none'}"
                out_dir = run_multi_seed_variant_experiment(
                    dataset=dataset_key,
                    source=source,
                    eval_domains=eval_domains,
                    variant_name=model_variant,
                    seeds=seeds,
                    device=device,
                    epochs=epochs,
                    verbose=args.verbose,
                    num_layers=num_layers,
                    hid_dim=hid_dim,
                    dropout_ratio=dropout_ratio,
                    out_dir=str(case_dir),
                )
                case_dirs.append(Path(out_dir))
                result_dirs.append((dataset_key, model_variant, source, Path(out_dir)))

            build_variant_dataset_rollup(
                dataset=dataset_key,
                dataset_out_dir=model_root,
                variant_name=model_variant,
                case_dirs=case_dirs,
            )

    print("\nCompleted runs:")
    print(f"- shared run root: {run_root}")
    for dataset_key, model_variant, source, out_dir in result_dirs:
        print(f"- {dataset_key} model={model_variant} source={source}: {out_dir}")


if __name__ == "__main__":
    main()
