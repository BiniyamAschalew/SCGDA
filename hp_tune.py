"""Hyperparameter tuning runner."""

import argparse
import csv
import fcntl
import random
import time
from itertools import product
from pathlib import Path

import yaml


PROGRESS_FIELDS = ["seed", "start_time", "completed", "total"]
SUPPORTED_IMPORTED_MODELS = {
    "a2gnn",
    "adagcn",
    "dane",
    "dgsda",
    "grade",
    "jhgda",
    "kbl",
    "pairalign",
    "specreg",
    "strurw",
    "tdss",
    "udagcn",
}
CUDA_MEM_FIELDS = [
    "cuda_device_index",
    "cuda_allocated_mb",
    "cuda_reserved_mb",
    "cuda_max_allocated_mb",
    "cuda_max_reserved_mb",
]


def resolve_config_path(config_value: str) -> Path:
    path = Path(config_value)
    if path.suffix != ".yaml":
        path = Path("./configs/expt_configs") / f"{config_value}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


def resolve_existing_path(path_value: str, base_dir: Path | None = None) -> Path:
    raw = Path(str(path_value).strip())
    candidates = [raw]
    if base_dir is not None:
        candidates.append(base_dir / raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    rendered = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Path not found. Tried: {rendered}")


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_search_space(space_path: Path, dataset: str) -> dict:
    path = space_path if space_path.is_file() else space_path / f"{dataset.lower()}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Search space file not found: {path}")

    space = load_yaml(path).get("model", {})
    if not isinstance(space, dict) or not space:
        raise ValueError(f"Missing or invalid top-level 'model' section in {path}")

    return {k: v if isinstance(v, list) else [v] for k, v in space.items()}


def build_combos(space: dict, max_combos: int):
    keys = sorted(space.keys())
    values = [space[k] for k in keys]

    total = 1
    for vals in values:
        total *= len(vals)

    selected = None
    if max_combos and max_combos < total:
        rng = random.Random(0)
        selected = set(rng.sample(range(total), max_combos))

    combos = []
    for idx, combo in enumerate(product(*values)):
        if selected is not None and idx not in selected:
            continue
        combos.append((idx, dict(zip(keys, combo))))
        if selected is None and max_combos and len(combos) >= max_combos:
            break

    return combos, total


def iter_transfer_pairs(transfer_settings: dict):
    for dataset, mapping in transfer_settings.items():
        if not isinstance(mapping, dict):
            raise ValueError("transfer_settings must map dataset -> {source: target(s)}")
        for source, targets in mapping.items():
            if isinstance(targets, (list, tuple)):
                for target in targets:
                    yield dataset, source, target
            else:
                yield dataset, source, targets


def validate_transfer_pairs(pairs: list) -> None:
    invalid = [(d, s, t) for d, s, t in pairs if s == t]
    if invalid:
        sample = ", ".join([f"{d}:{s}->{t}" for d, s, t in invalid[:5]])
        raise ValueError(f"source==target is not allowed in transfer settings. Examples: {sample}")


def validate_imported_models(models: list) -> None:
    missing = [m for m in models if m.lower() not in SUPPORTED_IMPORTED_MODELS]
    if missing:
        raise ValueError(
            f"from_pygda path currently supports {sorted(SUPPORTED_IMPORTED_MODELS)}, "
            f"but got unsupported models: {missing}"
        )


def validate_imported_files(models: list, pairs: list) -> None:
    missing = []
    for dataset, source, target in pairs:
        for model in models:
            path = Path("./__hps__/imported") / model.lower() / dataset.lower() / f"{source}_{target}" / "imported.yaml"
            if not path.exists():
                missing.append(str(path))
    if missing:
        sample = "\n".join(missing[:12])
        suffix = "" if len(missing) <= 12 else f"\n... (+{len(missing)-12} more)"
        raise FileNotFoundError(f"Missing imported configs:\n{sample}{suffix}")


def combo_is_valid(model: str, hp_params: dict) -> bool:
    model = model.lower()
    if model == "dgsda":
        num_layers = hp_params.get("num_layers")
        if num_layers is None:
            return True
        try:
            return int(num_layers) == 2
        except Exception:
            return False
    return True


def normalize_optional_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def normalize_hp_id(value):
    text = str(value).strip()
    if not text:
        return text
    try:
        return int(float(text))
    except Exception:
        return value


def load_completed_keys(result_path: Path, seed: int) -> set:
    if not result_path.exists():
        return set()

    with result_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    completed = {
        (
            row.get("dataset"),
            row.get("source"),
            row.get("target"),
            row.get("model"),
            normalize_hp_id(row.get("hp_id")),
            normalize_optional_text(row.get("borrow_model")),
        )
        for row in rows
    }
    print(f"[Seed {seed}] Found {len(completed)} completed experiments, will skip them")
    return completed


def append_row_csv(path: Path, row: dict, columns: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    payload = {col: row.get(col, "") for col in columns}
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if needs_header:
            writer.writeheader()
        writer.writerow(payload)


def update_progress(path: Path, seed: int, start_time: str, completed: int, total: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = {
        "seed": str(seed),
        "start_time": start_time,
        "completed": str(completed),
        "total": str(total),
    }

    with path.open("a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            rows = []
            if f.read(1):
                f.seek(0)
                rows = list(csv.DictReader(f))

            by_seed = {row.get("seed", ""): row for row in rows}
            by_seed[str(seed)] = current
            merged = [by_seed[s] for s in sorted(by_seed.keys(), key=lambda x: int(x) if x.isdigit() else x)]

            f.seek(0)
            f.truncate()
            writer = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS)
            writer.writeheader()
            writer.writerows(merged)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_imported_config(model: str, dataset: str, source: str, target: str):
    path = Path("./__hps__/imported") / model.lower() / dataset.lower() / f"{source}_{target}" / "imported.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Imported config not found: {path}")
    return load_yaml(path), str(path)


def validate_borrowed_files(borrow_model: str, pairs: list) -> None:
    missing = []
    for dataset, source, target in pairs:
        path = Path("./__hps__/tuned") / borrow_model.lower() / dataset.lower() / f"{source}_{target}" / "best.yaml"
        if not path.exists():
            missing.append(str(path))
    if missing:
        sample = "\n".join(missing[:12])
        suffix = "" if len(missing) <= 12 else f"\n... (+{len(missing)-12} more)"
        raise FileNotFoundError(f"Missing borrowed tuned configs:\n{sample}{suffix}")


def load_borrowed_config(borrow_model: str, dataset: str, source: str, target: str):
    path = Path("./__hps__/tuned") / borrow_model.lower() / dataset.lower() / f"{source}_{target}" / "best.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Borrowed tuned config not found: {path}")
    return load_yaml(path), str(path)


def sanitize_model_payload(payload: dict | None) -> dict:
    cleaned = dict(payload or {})
    cleaned.pop("name", None)
    cleaned.pop("tuned_params", None)
    cleaned.pop("gnn", None)
    return cleaned


def build_error_result(config: dict, error: Exception, stage: str) -> dict:
    expt = config.get("expt", {})
    model = config.get("model", {})
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--space-dir", type=str)
    parser.add_argument("--use-imported", action="store_true")
    parser.add_argument("--from-pygda", action="store_true")
    parser.add_argument("--borrow-model", type=str)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    config = load_yaml(config_path)
    seed = args.seed if args.seed is not None else int(config.get("seed", 200))
    models = config.get("models") or []
    transfer_settings = config.get("transfer_settings") or {}
    if not models:
        raise ValueError("No models specified in config")
    if not transfer_settings:
        raise ValueError("No transfer_settings specified in config")

    search_space_value = args.space_dir or config.get("search_space_dir") or config.get("search_space")
    if not search_space_value:
        raise ValueError("Search space path is required (--space-dir or config.search_space_dir)")
    search_space_path = resolve_existing_path(str(search_space_value), base_dir=config_path.parent)

    use_imported = args.use_imported or as_bool(config.get("use_imported", False))
    from_pygda = args.from_pygda or as_bool(config.get("from_pygda", False))
    borrow_model = normalize_optional_text(
        args.borrow_model or config.get("borrow_model") or config.get("borrow")
    ) or None
    models = [m.lower() for m in models]
    transfer_pairs = list(iter_transfer_pairs(transfer_settings))
    validate_transfer_pairs(transfer_pairs)

    if from_pygda:
        validate_imported_models(models)
    if use_imported:
        validate_imported_files(models, transfer_pairs)
    if borrow_model:
        validate_borrowed_files(borrow_model, transfer_pairs)

    output_dir = Path(config["output_dir"])
    run_id = (args.run_id or "").strip() or time.strftime("%m%d_%H%M%S")
    run_dir = output_dir / run_id
    result_path = run_dir / f"results_{run_id}_seed{seed}.csv"
    progress_path = run_dir / "progress.csv"

    device = config.get("device", "cuda:0")
    verbose = int(config.get("verbose", 1))
    wandb_enabled = bool(config.get("wandb", False))
    wandb_project = config.get("wandb_project", "SCGDA_HPTune")
    prefix = config.get("prefix")
    metrics = config.get("metrics") or ["micro_f1", "macro_f1"]
    max_combos = int(config.get("max_combos", 0))

    completed_keys = load_completed_keys(result_path, seed)
    combo_cache = {}
    hp_keys = set()
    for dataset in transfer_settings.keys():
        space = load_search_space(search_space_path, dataset)
        combos, total = build_combos(space, max_combos)
        combo_cache[dataset] = {
            "combos": combos,
            "total": total,
            "tuned_params": sorted(space.keys()),
        }
        hp_keys.update(space.keys())
        if max_combos and total > max_combos:
            print(f"[Seed {seed}] {dataset}: using {len(combos)}/{total} sampled HP combinations")

    hp_keys = sorted(hp_keys)
    columns = list(
        dict.fromkeys(
            [
                "run_id",
                "dataset",
                "source",
                "target",
                "model",
                "base_model",
                "seed",
                "config_id",
                "hp_id",
                "use_imported",
                "from_pygda",
                "imported_config",
                "borrow_model",
                "borrowed_config",
                "tuned_params",
            ]
            + hp_keys
            + metrics
            + ["status", "train_time", "error_type", "error_stage", "error", "device"]
            + CUDA_MEM_FIELDS
        )
    )

    skipped_invalid = 0
    trials = []
    for dataset, source, target in transfer_pairs:
        for model in models:
            model_name = f"{prefix}{model}" if prefix else model
            for hp_id, hp_params in combo_cache[dataset]["combos"]:
                if not combo_is_valid(model, hp_params):
                    skipped_invalid += 1
                    continue
                trials.append(
                    (
                        dataset,
                        source,
                        target,
                        model,
                        model_name,
                        hp_id,
                        hp_params,
                        combo_cache[dataset]["tuned_params"],
                    )
                )

    if skipped_invalid:
        print(f"[Seed {seed}] Skipped {skipped_invalid} invalid model/HP combinations")

    if not trials:
        raise ValueError("No valid trials generated. Check models, transfer_settings, and search space.")

    if args.dry_run:
        print(
            f"[Seed {seed}] Dry run OK: total_trials={len(trials)}, models={len(models)}, "
            f"pairs={len(transfer_pairs)}, use_imported={int(use_imported)}, "
            f"from_pygda={int(from_pygda)}, borrow_model={borrow_model or ''}"
        )
        return

    from Learn.Clean_SCGDA.run import run as run_experiment
    from Learn.Clean_SCGDA.utils.config_utils import build_config

    process_start = time.strftime("%Y-%m-%d %H:%M:%S")
    processed = 0
    skipped = 0
    total_trials = len(trials)
    update_progress(progress_path, seed, process_start, completed=0, total=total_trials)

    for idx, (dataset, source, target, model, model_name, hp_id, hp_params, tuned_params) in enumerate(trials, start=1):
        key = (dataset, source, target, model_name, hp_id, borrow_model or "")
        if key in completed_keys:
            skipped += 1
            update_progress(progress_path, seed, process_start, completed=processed + skipped, total=total_trials)
            if verbose:
                print(f"[Seed {seed}] Skipping {idx}/{total_trials}: already completed")
            continue

        borrow_note = f", borrow={borrow_model}" if borrow_model else ""
        print(
            f"[Seed {seed}] Running {idx}/{total_trials}: {model_name} hp_id={hp_id} "
            f"on {dataset} ({source}->{target}){borrow_note}"
        )

        imported_path = ""
        borrowed_path = ""
        merged_model_params = {}
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
            "model": merged_model_params,
        }

        try:
            if use_imported:
                imported_cfg, imported_path = load_imported_config(model=model, dataset=dataset, source=source, target=target)
                merged_model_params.update(sanitize_model_payload(imported_cfg))
            if borrow_model:
                borrowed_cfg, borrowed_path = load_borrowed_config(
                    borrow_model=borrow_model,
                    dataset=dataset,
                    source=source,
                    target=target,
                )
                merged_model_params.update(sanitize_model_payload(borrowed_cfg))

            merged_model_params.update(hp_params)
            update_config["model"] = merged_model_params

            run_config = build_config(
                {"data": dataset, "expt": "default", "model": model},
                update_config,
            )
            result = run_experiment(run_config, from_pygda=from_pygda)
        except Exception as exc:
            result = build_error_result(
                {"expt": update_config["expt"], "model": {"name": model_name}},
                exc,
                "build_or_run",
            )

        row = {
            **dict(result),
            "run_id": run_id,
            "dataset": dataset,
            "source": source,
            "target": target,
            "model": model_name,
            "base_model": model,
            "seed": seed,
            "config_id": hp_id,
            "hp_id": hp_id,
            "use_imported": int(use_imported),
            "from_pygda": int(from_pygda),
            "imported_config": imported_path,
            "borrow_model": borrow_model or "",
            "borrowed_config": borrowed_path,
            "tuned_params": "|".join(tuned_params),
        }
        for k in hp_keys:
            row[k] = hp_params.get(k)

        append_row_csv(result_path, row, columns)
        processed += 1
        update_progress(progress_path, seed, process_start, completed=processed + skipped, total=total_trials)

    print(f"[Seed {seed}] HP tuning complete! Ran {processed} experiments, skipped {skipped}")
    print(f"[Seed {seed}] Results saved to {result_path}")


if __name__ == "__main__":
    main()
