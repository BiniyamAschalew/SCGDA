import argparse
import csv
import gc
from tqdm import tqdm

import math
import time
from itertools import product
from pathlib import Path

import torch
from omegaconf import OmegaConf

from utils.config_utils import build_config
from run import run


def load_space(space_dir: Path, file_name: str) -> dict:
    path = space_dir / f"{file_name.lower()}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"HP space file not found: {path}")

    space = OmegaConf.load(path)
    if space is None or "model" not in space:
        raise ValueError(f"HP space file must have a top-level 'model' key: {path}")

    model_space = OmegaConf.to_container(space, resolve=True).get("model", {})
    if not model_space:
        raise ValueError(f"No model hyperparameters found in: {path}")

    normalized = {}
    for key, value in model_space.items():
        normalized[key] = value if isinstance(value, list) else [value]

    return normalized


def grid_from_space(model_space: dict):
    keys = list(model_space.keys())
    values = [model_space[key] for key in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def list_datasets(config_dir: Path) -> list:
    data_dir = config_dir / "data_configs"
    return sorted(p.stem for p in data_dir.glob("*.yaml"))


def parse_pairs(raw: str) -> list:
    if not raw:
        return []

    pairs = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "->" in item:
            src, tgt = item.split("->", 1)
        elif "_" in item:
            src, tgt = item.split("_", 1)
        else:
            raise ValueError(
                f"Invalid pair format '{item}'. Use SRC->TGT or SRC_TGT."
            )
        pairs.append((src, tgt))
    return pairs


def write_csv(path: Path, rows: list):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tune_for_pair(
    base_config,
    model_space: dict,
    seeds: list,
    metric: str,
    out_dir: Path,
    max_combos: int,
    write_best: bool,
):
    rows = []
    summary = []

    best_score = None
    best_params = None

    for config_id, params in enumerate(grid_from_space(model_space)):

        if base_config.expt.verbose > 0:
            print(f"Testing config {config_id}: {params}")
        if max_combos and config_id >= max_combos:
            break

        scores = []
        num_oom = 0
        for seed in seeds:
            cfg = OmegaConf.merge(
                base_config,
                {"model": params, "expt": {"seed": seed}},
            )
            result = run(OmegaConf.to_container(cfg, resolve=True))
            if base_config.expt.verbose > 0:
                print(f"  Seed {seed} result: {result}")

            row = dict(result)
            row.update(params)
            row["config_id"] = config_id
            rows.append(row)

            if result.get("status") == "oom":
                num_oom += 1
                continue

            if metric not in result:
                raise KeyError(f"Metric '{metric}' not found in run result.")

            score = float(result[metric])
            if math.isnan(score):
                continue
            scores.append(score)

            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        if scores:
            avg_score = sum(scores) / len(scores)
        else:
            avg_score = float("nan")
        summary.append(
            {
                "config_id": config_id,
                "avg_score": avg_score,
                "metric": metric,
                "num_success": len(scores),
                "num_oom": num_oom,
                "status": "ok" if scores else "oom",
                **params,
            }
        )

        if scores and (best_score is None or avg_score > best_score):
            best_score = avg_score
            best_params = params

    write_csv(out_dir / "results.csv", rows)
    write_csv(out_dir / "summary.csv", summary)

    if best_params is None:
        return

    best_model_config = OmegaConf.merge(base_config.model, best_params)
    OmegaConf.save(best_model_config, out_dir / "best.yaml")

    if write_best:
        OmegaConf.save(best_model_config, out_dir / "default.yaml")

    summary_payload = {
        "metric": metric,
        "score": best_score,
        "seeds": seeds,
        "params": best_params,
    }
    OmegaConf.save(OmegaConf.create(summary_payload), out_dir / "summary.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model name (e.g., gnn)")
    parser.add_argument("--expt", default="default", help="Experiment config name")
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset list (default: all in configs/data_configs)",
    )
    parser.add_argument(
        "--pairs",
        default="",
        help="Comma-separated pairs: SRC->TGT or SRC_TGT (optional)",
    )
    parser.add_argument("--metric", default="micro_f1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-r", "--repeat", type=int, default=3)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=200,
        help="Override epochs (omit to use config)",
    )
    parser.add_argument("-v", "--verbose", type=int, default=0)
    parser.add_argument("--max-combos", type=int, default=0)
    parser.add_argument("-w", "--write-best", action="store_true")
    parser.add_argument("--space-dir", default="hps/space")
    parser.add_argument("--config-dir", default="configs")

    args = parser.parse_args()

    seeds = [args.seed + i for i in range(args.repeat)]

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = list_datasets(Path(args.config_dir))

    pair_filter = parse_pairs(args.pairs)

    for dataset in tqdm(datasets, desc="Datasets"):
        model_space = load_space(Path(args.space_dir), dataset)
        base_config = OmegaConf.create(
            build_config({"data": dataset, "model": args.model, "expt": args.expt})
        )
        if args.device:
            base_config.expt.device = args.device
        if args.epochs is not None:
            base_config.expt.epochs = args.epochs
        base_config.expt.verbose = args.verbose

        domains = base_config.data.domains
        all_pairs = [(s, t) for s in domains for t in domains if s != t]

        if pair_filter:
            all_pairs = [(s, t) for s, t in all_pairs if (s, t) in pair_filter]
            if not all_pairs:
                print(
                    f"No valid pairs found for dataset '{dataset}' with filter {pair_filter}"
                )
                continue

        tuned_root = Path(base_config.expt.tuned_config_path)

        for source, target in tqdm(all_pairs, desc="Domain Pairs", leave=False):
            pair_config = OmegaConf.merge(
                base_config,
                {"expt": {"source": source, "target": target}},
            )

            out_dir = tuned_root / args.model.lower() / dataset / f"{source}_{target}"
            out_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"== Tuning {args.model} on {dataset}: {source} -> {target} =="
            )
            start = time.time()
            tune_for_pair(
                pair_config,
                model_space,
                seeds,
                args.metric,
                out_dir,
                args.max_combos,
                args.write_best,
            )
            elapsed = time.time() - start
            print(f"== Done in {elapsed:.1f}s, results saved to {out_dir} ==")


if __name__ == "__main__":
    main()
