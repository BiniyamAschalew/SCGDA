#!/usr/bin/env python3
"""
Aggregate benchmark results across multiple run folders.

For each selected run:
1) read benchmark rows (prefers *_all_seeds.csv, falls back to *_seed*.csv)
2) compute per-run scenario micro_f1 mean

Then aggregate across runs and output a single table with:
dataset, source, target, scenario(source->target), model, num_runs, micro_mean, micro_std
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate benchmark runs by transfer scenario.")
    parser.add_argument(
        "--bench-dir",
        type=str,
        required=True,
        help="Benchmark folder containing run_* directories, e.g. __saved__/results/benchmark/bench5",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional explicit run directory names to include, e.g. run_0226_153938 run_0227_101010",
    )
    parser.add_argument(
        "--run-glob",
        type=str,
        default="run_*",
        help="Glob for run directories when --runs is not provided.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model filter.",
    )
    parser.add_argument(
        "--include-non-ok",
        action="store_true",
        help="Include rows with status != ok (default excludes them).",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=4,
        help="Decimal places for output metrics.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output CSV path. Default: <bench-dir>/scenario_micro_f1_across_runs.csv",
    )
    return parser.parse_args()


def resolve_run_dirs(bench_dir: Path, runs: List[str] | None, run_glob: str) -> List[Path]:
    if runs:
        run_dirs = [bench_dir / name for name in runs]
    else:
        run_dirs = sorted([p for p in bench_dir.glob(run_glob) if p.is_dir()])

    missing = [str(p) for p in run_dirs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Run directories not found: {missing}")
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {bench_dir} with pattern {run_glob}")
    return run_dirs


def load_run_results(run_dir: Path) -> pd.DataFrame:
    all_seeds_files = sorted(glob.glob(str(run_dir / "benchmark_*_all_seeds.csv")))
    if all_seeds_files:
        frames = [pd.read_csv(path) for path in all_seeds_files]
        df = pd.concat(frames, ignore_index=True)
        return df

    seed_files = sorted(glob.glob(str(run_dir / "benchmark_*_seed*.csv")))
    if seed_files:
        frames = [pd.read_csv(path) for path in seed_files]
        df = pd.concat(frames, ignore_index=True)
        return df

    raise FileNotFoundError(
        f"No benchmark CSV files found in {run_dir} "
        f"(expected benchmark_*_all_seeds.csv or benchmark_*_seed*.csv)"
    )


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["source", "target", "micro_f1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    if "dataset" not in work.columns:
        work["dataset"] = "unknown"
    if "model" not in work.columns:
        work["model"] = "unknown"
    if "status" not in work.columns:
        work["status"] = "ok"
    return work


def aggregate_runs(
    run_dirs: List[Path],
    model_filter: str | None,
    include_non_ok: bool,
    round_digits: int,
) -> tuple[pd.DataFrame, List[str], List[str]]:
    per_run_frames: List[pd.DataFrame] = []
    used_runs: List[str] = []
    skipped_runs: List[str] = []

    for run_dir in run_dirs:
        try:
            raw_df = load_run_results(run_dir)
        except FileNotFoundError:
            skipped_runs.append(run_dir.name)
            continue

        run_df = ensure_columns(raw_df)
        run_df["micro_f1"] = pd.to_numeric(run_df["micro_f1"], errors="coerce")
        run_df = run_df.dropna(subset=["micro_f1"])

        if not include_non_ok:
            run_df = run_df[run_df["status"] == "ok"]
        if model_filter is not None:
            run_df = run_df[run_df["model"] == model_filter]

        if run_df.empty:
            skipped_runs.append(run_dir.name)
            continue

        group_cols = ["dataset", "source", "target", "model"]
        per_run = (
            run_df.groupby(group_cols, as_index=False)
            .agg(run_micro_mean=("micro_f1", "mean"))
        )
        per_run["run"] = run_dir.name
        per_run_frames.append(per_run)
        used_runs.append(run_dir.name)

    if not per_run_frames:
        raise ValueError("No valid rows found after filtering.")

    stacked = pd.concat(per_run_frames, ignore_index=True)
    final = (
        stacked.groupby(["dataset", "source", "target", "model"], as_index=False)
        .agg(
            num_runs=("run", "nunique"),
            micro_mean=("run_micro_mean", "mean"),
            micro_std=("run_micro_mean", "std"),
        )
    )
    final["micro_std"] = final["micro_std"].fillna(0.0)
    final["scenario"] = final["source"].astype(str) + "->" + final["target"].astype(str)
    final["micro_mean"] = final["micro_mean"].round(round_digits)
    final["micro_std"] = final["micro_std"].round(round_digits)
    final = final.sort_values(["dataset", "source", "target", "model"]).reset_index(drop=True)
    return final, used_runs, skipped_runs


def print_table(df: pd.DataFrame) -> None:
    display_cols = [
        "dataset",
        "scenario",
        "model",
        "num_runs",
        "micro_mean",
        "micro_std",
    ]
    available = [c for c in display_cols if c in df.columns]
    print(df[available].to_string(index=False))


def main():
    args = parse_args()

    bench_dir = Path(args.bench_dir).resolve()
    run_dirs = resolve_run_dirs(bench_dir, args.runs, args.run_glob)

    summary, used_runs, skipped_runs = aggregate_runs(
        run_dirs=run_dirs,
        model_filter=args.model,
        include_non_ok=args.include_non_ok,
        round_digits=args.round,
    )

    output_path = (
        Path(args.output)
        if args.output is not None
        else bench_dir / "scenario_micro_f1_across_runs.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print(f"Selected runs: {[p.name for p in run_dirs]}")
    print(f"Used runs: {used_runs}")
    if skipped_runs:
        print(f"Skipped runs (missing/empty after filtering): {sorted(set(skipped_runs))}")
    print(f"Wrote summary: {output_path}")
    print("")
    print_table(summary)


if __name__ == "__main__":
    main()
