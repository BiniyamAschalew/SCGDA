from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SyntheticDomainData:
    source_x: np.ndarray
    source_y: np.ndarray
    target_x: np.ndarray
    target_y: np.ndarray


def _orthonormal_matrix(dim: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def generate_synthetic_da_data(
    *,
    n_classes: int = 3,
    n_per_class: int = 400,
    feature_dim: int = 16,
    class_sep: float = 4.0,
    source_std: float = 0.7,
    target_std_scale: float = 1.15,
    rotation_mix: float = 0.45,
    target_shift_scale: float = 1.2,
    class_shift_scale: float = 0.35,
    seed: int = 7,
) -> SyntheticDomainData:
    rng = np.random.default_rng(seed)

    class_means = rng.normal(size=(n_classes, feature_dim))
    class_means /= np.linalg.norm(class_means, axis=1, keepdims=True) + 1e-12
    class_means *= class_sep

    rotation = _orthonormal_matrix(feature_dim, rng)
    transform = (1.0 - rotation_mix) * np.eye(feature_dim) + rotation_mix * rotation
    global_shift = rng.normal(scale=target_shift_scale, size=(feature_dim,))
    class_shifts = rng.normal(scale=class_shift_scale, size=(n_classes, feature_dim))

    source_x_all: list[np.ndarray] = []
    source_y_all: list[np.ndarray] = []
    target_x_all: list[np.ndarray] = []
    target_y_all: list[np.ndarray] = []

    for cls in range(n_classes):
        src = rng.normal(
            loc=class_means[cls],
            scale=source_std,
            size=(n_per_class, feature_dim),
        )
        tgt_base = rng.normal(
            loc=class_means[cls],
            scale=source_std * target_std_scale,
            size=(n_per_class, feature_dim),
        )
        tgt = tgt_base @ transform + global_shift + class_shifts[cls]

        source_x_all.append(src)
        source_y_all.append(np.full(n_per_class, cls, dtype=np.int64))
        target_x_all.append(tgt)
        target_y_all.append(np.full(n_per_class, cls, dtype=np.int64))

    source_x = np.concatenate(source_x_all, axis=0).astype(np.float32)
    source_y = np.concatenate(source_y_all, axis=0).astype(np.int64)
    target_x = np.concatenate(target_x_all, axis=0).astype(np.float32)
    target_y = np.concatenate(target_y_all, axis=0).astype(np.int64)

    src_perm = rng.permutation(source_x.shape[0])
    tgt_perm = rng.permutation(target_x.shape[0])

    return SyntheticDomainData(
        source_x=source_x[src_perm],
        source_y=source_y[src_perm],
        target_x=target_x[tgt_perm],
        target_y=target_y[tgt_perm],
    )


def save_synthetic_data(path: str | Path, data: SyntheticDomainData) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        source_x=data.source_x,
        source_y=data.source_y,
        target_x=data.target_x,
        target_y=data.target_y,
    )


def load_synthetic_data(path: str | Path) -> SyntheticDomainData:
    arr = np.load(path)
    return SyntheticDomainData(
        source_x=arr["source_x"],
        source_y=arr["source_y"],
        target_x=arr["target_x"],
        target_y=arr["target_y"],
    )

