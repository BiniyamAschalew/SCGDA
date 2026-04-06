from __future__ import annotations

import os
import time

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


DATASETS = _env_tuple("MMD_LOC_DATASETS", ("citation", "airport", "blog"))
SEEDS = _env_int_tuple("MMD_LOC_SEEDS", (0, 1, 2))
DEVICE = os.environ.get("MMD_LOC_DEVICE", "cuda:0")
EPOCHS = int(os.environ.get("MMD_LOC_EPOCHS", "200"))
OUT_ROOT = os.environ.get("MMD_LOC_OUT_ROOT", "./__saved__/analysis/mmd_location_comparison")


def main() -> None:
    run_name = (
        f"mmd_loc_seed{''.join(str(seed) for seed in SEEDS)}"
        f"_e{EPOCHS}_{DEVICE.replace(':', '')}_{time.strftime('%m%d_%H%M%S')}"
    )
    run_experiment(
        datasets=DATASETS,
        seeds=SEEDS,
        device=DEVICE,
        epochs=EPOCHS,
        out_root=OUT_ROOT,
        run_name=run_name,
        verbose=1,
    )


if __name__ == "__main__":
    main()
