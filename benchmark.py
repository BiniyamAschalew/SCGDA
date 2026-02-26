"""
Code for running deep evaluation benchmarks.
"""

import argparse
import glob
import os
import time
from typing import Dict, Iterator, Tuple

import pandas as pd
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=200,
        help="random seed for the benchmark run",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="experiment config name in configs/expt_configs (with or without .yaml)",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="directory to save results (created by run_benchmark.sh)",
    )
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="aggregate seed CSV files in run_dir and print summary tables",
    )
    return parser.parse_args()


def resolve_config_path(config_name: str) -> str:
    if config_name.endswith(".yaml"):
        return config_name
    return f"./configs/expt_configs/{config_name}.yaml"


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def iter_transfer_pairs(transfer_settings: dict) -> Iterator[Tuple[str, str, str]]:
    for dataset, settings in transfer_settings.items():
        if not isinstance(settings, dict):
            raise ValueError(
                f"transfer_settings['{dataset}'] must be a dict of source->target(s)"
            )
        for source, targets in settings.items():
            if isinstance(targets, list):
                for target in targets:
                    yield dataset, source, target
            else:
                yield dataset, source, targets


def parse_model_spec(raw_spec):
    if isinstance(raw_spec, bool):
        return {"use_tuned": int(raw_spec), "borrow": None, "epochs": None}
    if isinstance(raw_spec, int):
        return {"use_tuned": raw_spec, "borrow": None, "epochs": None}
    if isinstance(raw_spec, float) and raw_spec.is_integer():
        return {"use_tuned": int(raw_spec), "borrow": None, "epochs": None}
    if not isinstance(raw_spec, dict):
        raise ValueError(
            "model_hps entries must be int or dict "
            "(e.g., 2 or {use_tuned: 2, borrow: dgsda})"
        )

    use_tuned = raw_spec.get("use_tuned", raw_spec.get("hp", 0))
    if isinstance(use_tuned, bool):
        use_tuned = int(use_tuned)
    use_tuned = int(use_tuned)

    borrow = raw_spec.get("borrow")
    if borrow is None and raw_spec.get("borrow_dgsda", False):
        borrow = "dgsda"

    epochs = raw_spec.get("epochs")
    if epochs is not None:
        epochs = int(epochs)

    return {"use_tuned": use_tuned, "borrow": borrow, "epochs": epochs}


def normalize_model_specs(model_hps: dict) -> Dict[str, dict]:
    specs = {}
    for model, raw_spec in model_hps.items():
        specs[model] = parse_model_spec(raw_spec)
    return specs


def prepare_result_row(
    result: dict,
    dataset: str,
    source: str,
    target: str,
    model_name: str,
    seed: int,
    use_tuned: int,
    borrow: str,
):
    row = dict(result)
    row.setdefault("status", "ok")
    row["dataset"] = dataset
    row["source"] = source
    row["target"] = target
    row["model"] = model_name
    row["seed"] = seed
    row["cur_time"] = time.strftime("%d%H%M%S")
    row["hp_setting"] = use_tuned
    row["borrow"] = borrow if borrow is not None else ""

    for col in ("micro_f1", "macro_f1", "train_time"):
        if col not in row:
            row[col] = float("nan")
    return row


def print_seed_tables(df: pd.DataFrame, seed: int):
    if df.empty:
        print(f"[Seed {seed}] No results to summarize.")
        return

    cols = [
        "dataset",
        "source",
        "target",
        "model",
        "status",
        "micro_f1",
        "macro_f1",
        "train_time",
        "hp_setting",
        "borrow",
    ]
    present_cols = [c for c in cols if c in df.columns]
    table_df = df[present_cols].copy()
    for metric in ("micro_f1", "macro_f1", "train_time"):
        if metric in table_df.columns:
            table_df[metric] = pd.to_numeric(table_df[metric], errors="coerce").round(4)

    table_df = table_df.sort_values(["dataset", "source", "target", "model"])

    print(f"\n[Seed {seed}] Completed experiments table:")
    print(table_df.to_string(index=False))

    summary = build_model_summary(df)
    print(f"\n[Seed {seed}] Model summary:")
    print(summary.to_string(index=False))


def build_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for metric in ("micro_f1", "macro_f1", "train_time"):
        work[metric] = pd.to_numeric(work.get(metric), errors="coerce")
    work["ok_flag"] = (work.get("status", "") == "ok").astype(float)

    summary = (
        work.groupby("model", as_index=False)
        .agg(
            n=("status", "size"),
            ok_rate=("ok_flag", "mean"),
            micro_mean=("micro_f1", "mean"),
            micro_std=("micro_f1", "std"),
            macro_mean=("macro_f1", "mean"),
            macro_std=("macro_f1", "std"),
            time_mean=("train_time", "mean"),
        )
        .sort_values("model")
    )

    summary["ok_rate"] = (summary["ok_rate"] * 100.0).round(1)
    for col in ("micro_mean", "micro_std", "macro_mean", "macro_std", "time_mean"):
        summary[col] = summary[col].round(4)
    summary = summary.fillna(float("nan"))
    return summary


def aggregate_seed_results(run_dir: str, note: str):
    pattern = os.path.join(run_dir, f"benchmark_{note}_seed*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No seed files found in {run_dir} with pattern {pattern}")

    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    merged_path = os.path.join(run_dir, f"benchmark_{note}_all_seeds.csv")
    df.to_csv(merged_path, index=False)

    pair_summary = df.copy()
    pair_summary["micro_f1"] = pd.to_numeric(pair_summary.get("micro_f1"), errors="coerce")
    pair_summary["macro_f1"] = pd.to_numeric(pair_summary.get("macro_f1"), errors="coerce")
    pair_summary["train_time"] = pd.to_numeric(pair_summary.get("train_time"), errors="coerce")
    pair_summary["ok_flag"] = (pair_summary.get("status", "") == "ok").astype(float)

    pair_summary = (
        pair_summary.groupby(["dataset", "source", "target", "model"], as_index=False)
        .agg(
            rows=("status", "size"),
            ok_rate=("ok_flag", "mean"),
            micro_mean=("micro_f1", "mean"),
            micro_std=("micro_f1", "std"),
            macro_mean=("macro_f1", "mean"),
            macro_std=("macro_f1", "std"),
            time_mean=("train_time", "mean"),
        )
        .sort_values(["dataset", "source", "target", "model"])
    )
    pair_summary["ok_rate"] = (pair_summary["ok_rate"] * 100.0).round(1)
    for col in ("micro_mean", "micro_std", "macro_mean", "macro_std", "time_mean"):
        pair_summary[col] = pair_summary[col].round(4)

    pair_summary_path = os.path.join(run_dir, f"benchmark_{note}_pair_summary.csv")
    pair_summary.to_csv(pair_summary_path, index=False)

    model_summary = build_model_summary(df)
    model_summary_path = os.path.join(run_dir, f"benchmark_{note}_model_summary.csv")
    model_summary.to_csv(model_summary_path, index=False)

    print("\n[Aggregate] Pair summary table:")
    print(pair_summary.to_string(index=False))
    print("\n[Aggregate] Model summary table:")
    print(model_summary.to_string(index=False))
    print(
        f"\n[Aggregate] Wrote: {merged_path}, {pair_summary_path}, {model_summary_path}"
    )


def run_seed_benchmark(seed: int, config: dict, save_dir: str):
    from run import run
    from utils.config_utils import build_config

    model_specs = normalize_model_specs(config["model_hps"])
    transfer_pairs = list(iter_transfer_pairs(config["transfer_settings"]))

    device = config["device"]
    wandb_enabled = bool(config.get("wandb", False))
    verbose = int(config.get("verbose", 0))
    prefix = config.get("prefix")
    note = config["bench_note"]
    default_epochs = int(config.get("epochs", 200))
    wandb_project = config.get("wandb_project", "SCGDA_Benchmark")

    result_dir = os.path.join(save_dir, f"benchmark_{note}_seed{seed}.csv")
    print(f"[Seed {seed}] Starting benchmark, saving to: {result_dir}")

    combined_df = pd.DataFrame()

    total_experiments = len(transfer_pairs) * len(model_specs)
    current_exp = 0

    for dataset, source, target in transfer_pairs:
        for model, spec in model_specs.items():
            current_exp += 1
            print(
                f"[Seed {seed}] Running {current_exp}/{total_experiments}: "
                f"{model} on {dataset} ({source}->{target}) "
                f"[use_tuned={spec['use_tuned']}, borrow={spec['borrow']}]"
            )

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
                },
                "model": {
                    "epochs": spec["epochs"] if spec["epochs"] is not None else default_epochs
                },
            }

            try:
                run_config = build_config(
                    config_setup,
                    update_config,
                    borrow=spec["borrow"],
                    use_tuned=spec["use_tuned"],
                )
                result = run(run_config)
            except Exception as error:
                print(
                    f"[Seed {seed}] ERROR in {model} on {dataset} ({source}->{target}): {error}"
                )
                result = {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "micro_f1": float("nan"),
                    "macro_f1": float("nan"),
                    "train_time": float("nan"),
                }

            model_name = f"{prefix}{model}" if prefix else model
            row = prepare_result_row(
                result=result,
                dataset=dataset,
                source=source,
                target=target,
                model_name=model_name,
                seed=seed,
                use_tuned=spec["use_tuned"],
                borrow=spec["borrow"],
            )

            combined_df = pd.concat([combined_df, pd.DataFrame([row])], ignore_index=True)
            combined_df.to_csv(result_dir, index=False)
            print(
                f"[Seed {seed}] Saved ({current_exp}/{total_experiments}) to {result_dir}"
            )

    print_seed_tables(combined_df, seed)
    print(
        f"[Seed {seed}] Benchmark complete with {total_experiments} experiments. "
        f"Saved to {result_dir}"
    )


def main():
    args = parse_args()
    config_path = resolve_config_path(args.config)
    config = load_yaml(config_path)

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if args.run_dir:
        save_dir = args.run_dir
        os.makedirs(save_dir, exist_ok=True)
    else:
        cur_time = time.strftime("%m%d_%H%M%S")
        save_dir = os.path.join(output_dir, f"run_{cur_time}")
        os.makedirs(save_dir, exist_ok=True)

    if args.aggregate_only:
        aggregate_seed_results(save_dir, config["bench_note"])
        return

    run_seed_benchmark(args.seed, config, save_dir)


if __name__ == "__main__":
    main()
