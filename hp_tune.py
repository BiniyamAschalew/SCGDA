"""
Code for running hp search.
"""

import argparse
import csv
import fcntl
import multiprocessing as mp
import os
import time
from itertools import product
from pathlib import Path

import yaml

from utils.config_utils import build_config
from run import run


def _resolve_config_path(config_value: str) -> Path:
    path = Path(config_value)
    if path.suffix != ".yaml":
        path = Path("./configs/expt_configs") / f"{config_value}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


def load_search_space(space_dir: Path, dataset: str) -> dict:
    path = space_dir / f"{dataset.lower()}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Search space file not found: {path}")

    with path.open("r") as f:
        space = yaml.safe_load(f) or {}

    model_space = space.get("model", {})
    if not model_space:
        raise ValueError(
            f"Search space file must have a top-level 'model' key: {path}"
        )

    normalized = {}
    for key, value in model_space.items():
        normalized[key] = value if isinstance(value, list) else [value]

    return normalized


def build_combos(space: dict, max_combos: int) -> list:
    keys = sorted(space.keys())
    values = [space[key] for key in keys]
    combos = []

    for idx, combo in enumerate(product(*values)):
        if max_combos and idx >= max_combos:
            break
        combos.append((idx, dict(zip(keys, combo))))

    return combos


def iter_transfer_pairs(transfer_settings: dict):
    for dataset, settings in transfer_settings.items():
        if not isinstance(settings, dict):
            raise ValueError(
                "transfer_settings must map dataset to {source: target(s)}"
            )

        for source, targets in settings.items():
            if isinstance(targets, (list, tuple)):
                for target in targets:
                    yield dataset, source, target
            else:
                yield dataset, source, targets


def unique_list(items: list) -> list:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def append_row_csv(path: Path, row: dict, columns: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {col: row.get(col, "") for col in columns}

    with path.open("a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            needs_header = f.tell() == 0
            writer = csv.DictWriter(f, fieldnames=columns)
            if needs_header:
                writer.writeheader()
            writer.writerow(payload)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _build_error_result(config: dict, error: Exception, stage: str) -> dict:
    expt = config.get("expt", {}) if isinstance(config, dict) else {}
    model = config.get("model", {}) if isinstance(config, dict) else {}
    return {
        "status": "error",
        "error_type": "exception",
        "error_stage": stage,
        "error": str(error),
        "source": expt.get("source"),
        "target": expt.get("target"),
        "model": model.get("name"),
        "seed": expt.get("seed"),
        "train_time": 0.0,
        "device": expt.get("device"),
    }


def _run_worker(config: dict, queue: mp.Queue) -> None:
    try:
        result = run(config)
    except Exception as exc:
        result = _build_error_result(config, exc, "run")
    queue.put(result)


def run_trial(config: dict, isolate_trials: bool) -> dict:
    if not isolate_trials:
        return run(config)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run_worker, args=(config, queue))
    proc.start()
    proc.join()

    try:
        result = queue.get_nowait()
    except Exception as exc:
        result = _build_error_result(config, exc, "subprocess")
    finally:
        queue.close()
        queue.join_thread()

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
    )
    parser.add_argument("--config", type=str)
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--space-dir", type=str)

    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    with config_path.open("r") as f:
        config = yaml.safe_load(f) or {}

    seed = args.seed if args.seed is not None else config.get("seed", 200)
    models = config["models"]
    transfer_settings = config["transfer_settings"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    search_space_dir = Path(args.space_dir.strip())
    device = config.get("device", "cuda:0")
    wandb_enabled = bool(config.get("wandb", False))
    wandb_project = config.get("wandb_project", "SCGDA_HPTune")
    verbose = int(config.get("verbose", 1))
    metrics = config.get("metrics") or ["micro_f1", "macro_f1"]
    max_combos = int(config.get("max_combos", 0))
    prefix = config.get("prefix")
    isolate_trials = bool(config.get("isolate_trials", True))

    run_id = args.run_id.strip() or time.strftime("%m%d_%H%M%S")
    result_path = output_dir / f"results_{run_id}_seed{seed}.csv"

    # Load already completed experiments for resume capability
    completed_keys = set()
    if result_path.exists():
        try:
            import pandas as pd
            existing = pd.read_csv(result_path)
            for _, row in existing.iterrows():
                key = (row.get("dataset"), row.get("source"), row.get("target"), 
                       row.get("model"), row.get("hp_id"))
                completed_keys.add(key)
            print(f"[Seed {seed}] Found {len(completed_keys)} completed experiments, will skip them")
        except Exception as e:
            print(f"[Seed {seed}] Warning: Could not read existing results: {e}")

    combo_cache = {}
    hp_keys = set()
    for dataset in transfer_settings.keys():
        space = load_search_space(search_space_dir, dataset)
        hp_keys.update(space.keys())
        combo_cache[dataset] = build_combos(space, max_combos)

    hp_keys_sorted = sorted(hp_keys)
    columns = unique_list(
        [
            "run_id",
            "dataset",
            "source",
            "target",
            "model",
            "seed",
            "hp_id",
        ]
        + hp_keys_sorted
        + metrics
        + ["status", "train_time", "error_type", "error_stage", "error", "device"]
    )

    # Count total experiments for progress tracking
    total_experiments = sum(
        len(combo_cache[dataset]) * len(models)
        for dataset in transfer_settings.keys()
        for _ in iter_transfer_pairs({dataset: transfer_settings[dataset]})
    )
    current_exp = 0
    skipped_exp = 0

    for dataset, source, target in iter_transfer_pairs(transfer_settings):
        combos = combo_cache[dataset]
        for model in models:
            model_name = f"{prefix}{model}" if prefix else model
            for hp_id, hp_params in combos:
                current_exp += 1
                
                # Skip already completed experiments
                exp_key = (dataset, source, target, model_name, hp_id)
                if exp_key in completed_keys:
                    skipped_exp += 1
                    if verbose:
                        print(f"[Seed {seed}] Skipping {current_exp}/{total_experiments}: already completed")
                    continue
                
                print(f"[Seed {seed}] Running {current_exp}/{total_experiments}: {model_name} hp_id={hp_id} on {dataset} ({source}->{target})")
                
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
                        "seed": seed,
                        "verbose": verbose,
                        "metrics": metrics,
                    },
                    "model": hp_params,
                }

                run_config = build_config(config_setup, update_config)
                result = run_trial(run_config, isolate_trials)

                row = dict(result)
                row.update(
                    {
                        "run_id": run_id,
                        "dataset": dataset,
                        "source": source,
                        "target": target,
                        "model": model_name,
                        "seed": seed,
                        "hp_id": hp_id,
                    }
                )

                for key in hp_keys_sorted:
                    row[key] = hp_params.get(key)

                append_row_csv(result_path, row, columns)

    print(f"[Seed {seed}] HP tuning complete! Ran {current_exp - skipped_exp} experiments, skipped {skipped_exp}")
    print(f"[Seed {seed}] Results saved to {result_path}")


if __name__ == "__main__":
    main()
