"""
Process hp tuning results and save best configs per model/scenario.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml
from utils.config_utils import build_config


META_COLS = {
    "run_id",
    "dataset",
    "source",
    "target",
    "model",
    "base_model",
    "seed",
    "hp_id",
    "config_id",
    "use_imported",
    "imported_config",
    "tuned_params",
    "status",
    "train_time",
    "error_type",
    "error_stage",
    "error",
    "device",
}


def _parse_list(value: str) -> list:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_time_id(df: pd.DataFrame, explicit: str) -> str:
    if explicit:
        return explicit
    if "run_id" in df.columns:
        unique = df["run_id"].dropna().unique()
        if len(unique) == 1:
            return str(unique[0])
    return time.strftime("%m%d_%H%M%S")


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_builtin(value):
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    return value


def _parse_tuned_params(value, fallback: list) -> list:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return list(fallback)
    text = str(value).strip()
    if not text:
        return list(fallback)
    params = [token.strip() for token in text.split("|") if token.strip()]
    return params if params else list(fallback)


def _load_imported_payload(sample_row: pd.Series) -> dict:
    model = str(sample_row.get("base_model", sample_row["model"])).lower()
    dataset = str(sample_row["dataset"]).lower()
    source = str(sample_row["source"])
    target = str(sample_row["target"])

    candidate = sample_row.get("imported_config")
    if candidate and str(candidate).strip():
        path = Path(str(candidate).strip())
    else:
        path = Path("__hps__/imported") / model / dataset / f"{source}_{target}" / "imported.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Imported config not found: {path}")

    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _build_full_model_payload(sample_row: pd.Series, best_row: pd.Series, hp_cols: list) -> dict:
    model = str(sample_row.get("base_model", sample_row["model"])).lower()
    dataset = str(sample_row["dataset"])
    source = str(sample_row["source"])
    target = str(sample_row["target"])

    try:
        base_cfg = build_config(
            {"data": dataset, "expt": "default", "model": model},
            {"expt": {"source": source, "target": target, "verbose": 0}},
        )
        payload = dict(base_cfg.get("model", {}))
    except Exception as exc:
        print(f"[Warning] Failed to load base config for {model} ({dataset}:{source}->{target}): {exc}")
        payload = {}

    if _as_bool(sample_row.get("use_imported", 0)):
        try:
            payload.update(_load_imported_payload(sample_row))
        except Exception as exc:
            print(f"[Warning] Failed to load imported config for {model} ({source}->{target}): {exc}")

    for key in hp_cols:
        value = best_row.get(key)
        if pd.notna(value):
            payload[key] = value

    payload["tuned_params"] = _parse_tuned_params(sample_row.get("tuned_params"), hp_cols)
    return _to_builtin(payload)


def _append_performance_csv(path: Path, time_id: str, metrics_dict: dict) -> None:
    """Append performance metrics to a CSV file, creating it if it doesn't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    row_data = {"time_id": time_id, **metrics_dict}
    new_row = pd.DataFrame([row_data])
    
    if path.exists():
        existing = pd.read_csv(path)
        # Check if this time_id already exists
        if time_id in existing["time_id"].values:
            # Update existing row
            existing.loc[existing["time_id"] == time_id, list(metrics_dict.keys())] = list(metrics_dict.values())
            existing.to_csv(path, index=False)
        else:
            # Append new row
            combined = pd.concat([existing, new_row], ignore_index=True)
            combined.to_csv(path, index=False)
    else:
        new_row.to_csv(path, index=False)


def _collect_csv_files_from_run_dir(run_dir: Path) -> list:
    if not run_dir.exists() or not run_dir.is_dir():
        return []
    seed_files = sorted(run_dir.glob("results_*_seed*.csv"))
    if seed_files:
        return seed_files
    return sorted(run_dir.glob("results_*.csv"))


def _find_latest_run_dir(results_dir: Path) -> Path | None:
    if not results_dir.exists() or not results_dir.is_dir():
        return None
    run_dirs = [path for path in results_dir.iterdir() if path.is_dir()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0]


def _load_from_csv_files(csv_files: list) -> pd.DataFrame:
    if not csv_files:
        raise FileNotFoundError("No result files found.")
    dfs = [pd.read_csv(path) for path in csv_files]
    return pd.concat(dfs, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Path to results CSV file, run-id directory, or parent results directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="__saved__/results/hp_tune",
        help="Directory to search for result files if --results is not specified",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID to filter result files (e.g., '0124_231819')",
    )
    parser.add_argument("--metric", type=str, default="micro_f1")
    parser.add_argument("--metrics", type=str, default="micro_f1,macro_f1")
    parser.add_argument("--output-root", type=str, default="__hps__/tuned")
    parser.add_argument("--time-id", type=str, default="")

    args = parser.parse_args()

    # Load results from file(s)
    if args.results:
        results_path = Path(args.results)
        if results_path.is_dir():
            # If run-id is provided and {results}/{run-id} exists, treat that as the run folder.
            run_dir = results_path / args.run_id if args.run_id and (results_path / args.run_id).is_dir() else results_path
            csv_files = _collect_csv_files_from_run_dir(run_dir)
            if not csv_files:
                # Backward compatibility: old flat layout under the provided directory.
                csv_files = sorted(results_path.glob("results_*_seed*.csv"))
                if not csv_files:
                    csv_files = sorted(results_path.glob("results_*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No result files found in: {results_path}")
            df = _load_from_csv_files(csv_files)
            print(f"Loaded {len(csv_files)} result files from {run_dir if run_dir.exists() else results_path}")
        else:
            if not results_path.exists():
                raise FileNotFoundError(f"Results file not found: {results_path}")
            df = pd.read_csv(results_path)
    else:
        # Search in results_dir for a run-id subfolder first.
        results_dir = Path(args.results_dir)
        csv_files = []
        loaded_from = results_dir

        if args.run_id:
            run_dir = results_dir / args.run_id
            csv_files = _collect_csv_files_from_run_dir(run_dir)
            if csv_files:
                loaded_from = run_dir
            else:
                # Backward compatibility for old flat layout.
                csv_files = sorted(results_dir.glob(f"results_{args.run_id}_seed*.csv"))
                if not csv_files:
                    csv_files = sorted(results_dir.glob(f"results_{args.run_id}.csv"))
        else:
            latest_run_dir = _find_latest_run_dir(results_dir)
            if latest_run_dir is not None:
                csv_files = _collect_csv_files_from_run_dir(latest_run_dir)
                if csv_files:
                    loaded_from = latest_run_dir

            if not csv_files:
                # Backward compatibility for old flat layout.
                seed_files = sorted(
                    results_dir.glob("results_*_seed*.csv"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if seed_files:
                    latest = seed_files[0].name
                    latest_run_id = latest.replace("results_", "").split("_seed")[0]
                    csv_files = sorted(results_dir.glob(f"results_{latest_run_id}_seed*.csv"))
                else:
                    old_files = sorted(
                        results_dir.glob("results_*.csv"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if old_files:
                        csv_files = [old_files[0]]
        
        if not csv_files:
            raise FileNotFoundError(f"No result files found in: {results_dir}")
        
        df = _load_from_csv_files(csv_files)
        print(f"Loaded {len(csv_files)} result files from {loaded_from}: {[f.name for f in csv_files]}")

    if df.empty:
        raise ValueError("Results file(s) are empty")

    if "hp_id" not in df.columns and "config_id" in df.columns:
        df["hp_id"] = df["config_id"]

    metric = args.metric
    metrics = set(_parse_list(args.metrics))
    metrics.add(metric)

    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found in results columns.")

    time_id = _resolve_time_id(df, args.time_id)

    df = df[df.get("status", "ok") == "ok"].copy()
    df = df[pd.notna(df[metric])]

    if df.empty:
        raise ValueError("No successful rows found after filtering.")

    hp_cols = [c for c in df.columns if c not in META_COLS and c not in metrics]

    group_cols = ["model", "dataset", "source", "target", "hp_id"] + hp_cols
    grouped = df.groupby(group_cols, dropna=False)[metric].mean().reset_index()

    # Also compute mean for all metrics for reporting
    all_metrics_grouped = df.groupby(group_cols, dropna=False)[list(metrics)].mean().reset_index()

    best_rows = (
        grouped.sort_values(metric, ascending=False)
        .groupby(["model", "dataset", "source", "target"], as_index=False)
        .head(1)
    )

    output_root = Path(args.output_root)

    for _, row in best_rows.iterrows():
        sample_mask = (
            (df["model"] == row["model"]) &
            (df["dataset"] == row["dataset"]) &
            (df["source"] == row["source"]) &
            (df["target"] == row["target"]) &
            (df["hp_id"] == row["hp_id"])
        )
        if not sample_mask.any():
            continue
        sample_row = df.loc[sample_mask].iloc[0]

        model = str(row["model"]).lower()
        dataset = str(row["dataset"]).lower()
        source = str(row["source"])
        target = str(row["target"])

        model_payload = _build_full_model_payload(sample_row, row, hp_cols)

        out_dir = output_root / model / dataset / f"{source}_{target}"
        _write_yaml(out_dir / f"{time_id}.yaml", model_payload)
        _write_yaml(out_dir / "best.yaml", model_payload)

        # Get metric values for this best config
        mask = (
            (all_metrics_grouped["model"] == row["model"]) &
            (all_metrics_grouped["dataset"] == row["dataset"]) &
            (all_metrics_grouped["source"] == row["source"]) &
            (all_metrics_grouped["target"] == row["target"]) &
            (all_metrics_grouped["hp_id"] == row["hp_id"])
        )
        metrics_row = all_metrics_grouped.loc[mask]
        
        if not metrics_row.empty:
            metrics_dict = {m: float(metrics_row[m].iloc[0]) for m in metrics if m in metrics_row.columns}
            _append_performance_csv(out_dir / "performance.csv", time_id, metrics_dict)

    print(f"Saved best configs to {output_root}")


if __name__ == "__main__":
    main()
