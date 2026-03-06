"""Build benchmark tables from one or more hp_tune run IDs."""

import argparse
import time
from pathlib import Path

import pandas as pd


def parse_list(value: str) -> list:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def load_run_results(results_dir: Path, run_id: str) -> pd.DataFrame:
    run_dir = results_dir / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    csv_files = sorted(run_dir.glob(f"results_{run_id}_seed*.csv"))
    if not csv_files:
        csv_files = sorted(run_dir.glob("results_*_seed*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No seed result files found in {run_dir}")

    df = pd.concat([pd.read_csv(path) for path in csv_files], ignore_index=True)
    df["run_id"] = run_id
    return df


def ensure_config_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "hp_id" not in out.columns and "config_id" in out.columns:
        out["hp_id"] = out["config_id"]
    if "config_id" not in out.columns and "hp_id" in out.columns:
        out["config_id"] = out["hp_id"]
    if "hp_id" not in out.columns:
        raise KeyError("Neither 'hp_id' nor 'config_id' exists in results.")
    return out


def build_best_table(df: pd.DataFrame, metric: str, metrics: list) -> pd.DataFrame:
    work = df.copy()
    if metric not in work.columns:
        raise KeyError(f"Metric '{metric}' not found in results columns.")

    work = work[work.get("status", "ok") == "ok"].copy()
    for col in metrics:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work[pd.notna(work[metric])]

    if work.empty:
        raise ValueError("No successful rows found after filtering.")

    agg_metrics = [col for col in metrics if col in work.columns]
    grouped = (
        work.groupby(
            ["run_id", "model", "dataset", "source", "target", "hp_id", "config_id"],
            dropna=False,
            as_index=False,
        )
        .agg(
            **{col: (col, "mean") for col in agg_metrics},
            seeds=("seed", "nunique"),
        )
    )

    best = (
        grouped.sort_values([metric, "seeds"], ascending=[False, False])
        .groupby(["model", "dataset", "source", "target"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    return best.sort_values(["dataset", "source", "target", "model"]).reset_index(drop=True)


def scenario_label(row: pd.Series) -> str:
    return f"{row['dataset']}:{row['source']}->{row['target']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-ids", type=str, help="Comma-separated run IDs")
    parser.add_argument("--results-dir", type=str, default="./__saved__/results/hp_tune")
    parser.add_argument("--metric", type=str, default="micro_f1")
    parser.add_argument("--metrics", type=str, default="micro_f1,macro_f1")
    parser.add_argument("--output-dir", type=str, default="./__saved__/results/hp_tune")
    parser.add_argument("--name", type=str, default="")
    args = parser.parse_args()

    run_ids = "0228_195734,0301_183501,0301_190316"
    args.run_ids = run_ids

    print(f"\n Using run IDs: {args.run_ids}")

    run_ids = parse_list(args.run_ids)
    if not run_ids:
        raise ValueError("At least one run ID is required.")

    metrics = parse_list(args.metrics)
    if args.metric not in metrics:
        metrics.append(args.metric)

    results_dir = Path(args.results_dir)
    all_df = pd.concat([load_run_results(results_dir, run_id) for run_id in run_ids], ignore_index=True)
    all_df = ensure_config_id(all_df)

    best = build_best_table(all_df, metric=args.metric, metrics=metrics)
    best["scenario"] = best.apply(scenario_label, axis=1)

    pivot = (
        best.pivot_table(
            index=["dataset", "source", "target", "scenario"],
            columns="model",
            values=args.metric,
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["dataset", "source", "target"])
    )

    model_summary = (
        best.groupby("model", as_index=False)[[col for col in metrics if col in best.columns]]
        .mean()
        .sort_values(args.metric, ascending=False)
    )

    name = args.name.strip() or f"bench_{time.strftime('%m%d_%H%M%S')}"
    out_dir = Path(args.output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_path = out_dir / "best_config_per_scenario.csv"
    pivot_path = out_dir / f"benchmark_{args.metric}.csv"
    summary_path = out_dir / "model_summary.csv"

    best.to_csv(best_path, index=False)
    pivot.to_csv(pivot_path, index=False)
    model_summary.to_csv(summary_path, index=False)

    print(f"Saved: {best_path}")
    print(f"Saved: {pivot_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
