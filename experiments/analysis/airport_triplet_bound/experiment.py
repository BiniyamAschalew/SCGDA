from __future__ import annotations

import os

from .config import ExperimentConfig
from .run import run_experiment


def _env_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _env_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


# Edit these the same way the repo-level experiment.py is configured.
DATASETS = _env_tuple("TRIPLET_DATASETS", ("citation",))
SEEDS = _env_int_tuple("TRIPLET_SEEDS", (0, 1, 2))
DEVICE = os.environ.get("TRIPLET_DEVICE", "cuda:0")
EPOCHS = int(os.environ.get("TRIPLET_EPOCHS", "200"))

# Keep these at 0 for the full exact metric pass.
METRIC_SAMPLE_SIZE = int(os.environ.get("TRIPLET_METRIC_SAMPLE_SIZE", "0"))
GRAD_NODES_CAP = int(os.environ.get("TRIPLET_GRAD_NODES_CAP", "0"))

USE_PAIR_TUNED_HPARAMS = os.environ.get("TRIPLET_DISABLE_TUNED", "0") != "1"
REUSE_SAVED_MASKS = os.environ.get("TRIPLET_REUSE_SAVED_MASKS", "0") == "1"
OUT_ROOT = os.environ.get("TRIPLET_OUT_ROOT", "./__saved__/analysis/triplet_bound")


def main() -> None:
    for dataset in DATASETS:
        run_name = (
            f"{dataset.lower()}_seed{''.join(str(seed) for seed in SEEDS)}"
            f"_e{EPOCHS}_{DEVICE.replace(':', '')}_exact"
        )
        cfg = ExperimentConfig(
            dataset=dataset,
            seeds=SEEDS,
            device=DEVICE,
            epochs=EPOCHS,
            metric_sample_size=METRIC_SAMPLE_SIZE,
            grad_nodes_cap=GRAD_NODES_CAP,
            use_pair_tuned_hparams=USE_PAIR_TUNED_HPARAMS,
            resample_masks=not REUSE_SAVED_MASKS,
            out_root=OUT_ROOT,
            run_name=run_name,
            verbose=1,
        )
        out_dir = run_experiment(cfg)
        print(f"{dataset}: results_dir={out_dir}")


if __name__ == "__main__":
    main()
