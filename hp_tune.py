"""
Simplified HP tuning script. Designed to be invoked per-seed from run_hptune.sh.

Usage:
    python hp_tune2.py --config hp2 --space-dir __hps__/space/dgsda --seed 200

When --use-tuned is passed, non-search params are loaded from the best.yaml
of a previous tuning run for each (source, target) pair. Only the params
in the search space are varied; everything else comes from the tuned config.
"""

import argparse
import os
import time
from itertools import product
from pathlib import Path

import pandas as pd
import yaml

from utils.config_utils import build_config
from run import run

TUNED_DIR = Path("__hps__/tuned")


def load_search_space(space_dir: Path, dataset: str) -> dict:
    path = space_dir / f"{dataset.lower()}.yaml"
    with path.open() as f:
        space = yaml.safe_load(f) or {}
    model_space = space.get("model", {})
    return {k: (v if isinstance(v, list) else [v]) for k, v in model_space.items()}


def load_tuned_config(model: str, dataset: str, source: str, target: str) -> dict:
    path = TUNED_DIR / model / dataset.lower() / f"{source}_{target}" / "best.yaml"
    if not path.exists():
        print(f"  [warn] No tuned config at {path}, using defaults")
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def iter_transfer_pairs(transfer_settings: dict):
    for dataset, settings in transfer_settings.items():
        for source, targets in settings.items():
            for target in (targets if isinstance(targets, list) else [targets]):
                yield dataset, source, target


def main():
    parser = argparse.ArgumentParser(description="Simplified HP tuning")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--space-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--use-tuned", action="store_true",
                        help="Load non-search params from best.yaml of previous tuning")
    args = parser.parse_args()

    # Load experiment config
    config_path = Path(f"./configs/expt_configs/{args.config}.yaml")
    if not config_path.exists():
        config_path = Path(args.config)
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    models = cfg["models"]
    transfer_settings = cfg["transfer_settings"]
    output_dir = Path(cfg.get("output_dir", "./__saved__/results/hp_tune"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.get("device", "cuda:0")
    wandb_enabled = cfg.get("wandb", False)
    wandb_project = cfg.get("wandb_project", "SCGDA_HPTune")
    verbose = cfg.get("verbose", 0)
    metrics = cfg.get("metrics", ["micro_f1", "macro_f1"])
    max_combos = cfg.get("max_combos", 0)
    space_dir = Path(args.space_dir.strip())
    run_id = args.run_id.strip() or time.strftime("%m%d_%H%M%S")

    result_path = output_dir / f"results_{run_id}_seed{args.seed}.csv"

    # Resume: load already-completed experiment keys
    completed = set()
    if result_path.exists():
        try:
            df = pd.read_csv(result_path)
            for _, row in df.iterrows():
                completed.add((row.get("dataset"), row.get("source"),
                               row.get("target"), row.get("model"), row.get("hp_id")))
            print(f"[seed {args.seed}] Resuming: {len(completed)} completed, will skip")
        except Exception:
            pass

    # Build search combos per dataset
    combo_cache = {}
    for dataset in transfer_settings:
        space = load_search_space(space_dir, dataset)
        keys = sorted(space)
        vals = [space[k] for k in keys]
        combos = []
        for i, combo in enumerate(product(*vals)):
            if max_combos and i >= max_combos:
                break
            combos.append((i, dict(zip(keys, combo))))
        combo_cache[dataset] = combos

    # Run trials
    all_rows = []
    total = sum(
        len(combo_cache[d]) * len(models)
        for d, _, _ in iter_transfer_pairs(transfer_settings)
    )
    n = 0

    for dataset, source, target in iter_transfer_pairs(transfer_settings):
        combos = combo_cache[dataset]

        # Load tuned base config for this pair if requested
        tuned_base = {}
        if args.use_tuned and len(models) > 0:
            tuned_base = load_tuned_config(models[0], dataset, source, target)

        for model in models:
            if args.use_tuned:
                tuned_base = load_tuned_config(model, dataset, source, target)

            for hp_id, hp_params in combos:
                n += 1
                key = (dataset, source, target, model, hp_id)
                if key in completed:
                    continue

                print(f"[seed {args.seed}] {n}/{total}: {model} hp={hp_id} "
                      f"{dataset} {source}->{target}")

                # Build model params: start from tuned base, override with search params
                model_overrides = {}
                if args.use_tuned and tuned_base:
                    model_overrides.update(tuned_base)
                model_overrides.update(hp_params)

                config_setup = {
                    "data": dataset,
                    "expt": "default",
                    "model": model,
                }
                update_config = {
                    "expt": {
                        "source": source,
                        "target": target,
                        "device": device,
                        "wandb_enabled": wandb_enabled,
                        "project": wandb_project,
                        "seed": args.seed,
                        "verbose": verbose,
                        "metrics": metrics,
                    },
                    "model": model_overrides,
                }

                run_config = build_config(config_setup, update_config)

                try:
                    result = run(run_config)
                except Exception as e:
                    result = {"status": "error", "error": str(e)}
                    for m in metrics:
                        result[m] = float("nan")

                row = {
                    "run_id": run_id,
                    "dataset": dataset,
                    "source": source,
                    "target": target,
                    "model": model,
                    "seed": args.seed,
                    "hp_id": hp_id,
                    "status": result.get("status", "ok"),
                    "train_time": result.get("train_time", 0),
                }
                for m in metrics:
                    row[m] = result.get(m)
                for k, v in hp_params.items():
                    row[k] = v

                all_rows.append(row)

                # Append incrementally
                df_row = pd.DataFrame([row])
                df_row.to_csv(result_path, mode="a",
                              header=not result_path.exists() or os.path.getsize(result_path) == 0,
                              index=False)

    print(f"[seed {args.seed}] Done. {len(all_rows)} trials. Results: {result_path}")


if __name__ == "__main__":
    main()
