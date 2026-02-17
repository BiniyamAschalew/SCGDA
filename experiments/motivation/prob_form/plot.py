from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class FixedRandomProjector:
    matrix: np.ndarray

    @classmethod
    def create(cls, in_dim: int, out_dim: int = 2, seed: int = 0) -> "FixedRandomProjector":
        if in_dim == out_dim:
            return cls(matrix=np.eye(in_dim, dtype=np.float32))
        rng = np.random.default_rng(seed)
        mat = rng.normal(size=(in_dim, out_dim)).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=0, keepdims=True) + 1e-8
        return cls(matrix=mat)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x @ self.matrix


def _history_to_epoch_map(history: Sequence[Mapping[str, float]], key: str) -> dict[int, float]:
    return {int(row["epoch"]): float(row[key]) for row in history}


def plot_embedding_evolution(
    *,
    snapshots_by_regime: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    histories_by_regime: Mapping[str, Sequence[Mapping[str, float]]],
    source_y: np.ndarray,
    target_y: np.ndarray,
    snapshot_epochs: Sequence[int],
    projection_seed: int,
    output_path: str | Path,
    regime_order: Sequence[str] = ("baseline", "mmd"),
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_regime = regime_order[0]
    first_epoch = snapshot_epochs[0]
    in_dim = snapshots_by_regime[first_regime][first_epoch]["source_latent"].shape[1]
    projector = FixedRandomProjector.create(in_dim=in_dim, seed=projection_seed)

    projected: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    all_points: list[np.ndarray] = []
    for regime in regime_order:
        projected[regime] = {}
        for epoch in snapshot_epochs:
            src = projector.transform(snapshots_by_regime[regime][epoch]["source_latent"])
            tgt = projector.transform(snapshots_by_regime[regime][epoch]["target_latent"])
            projected[regime][epoch] = {"source": src, "target": tgt}
            all_points.extend([src, tgt])

    stacked = np.concatenate(all_points, axis=0)
    x_min, y_min = stacked.min(axis=0)
    x_max, y_max = stacked.max(axis=0)
    x_pad = (x_max - x_min) * 0.06 + 1e-4
    y_pad = (y_max - y_min) * 0.06 + 1e-4

    n_rows = len(regime_order)
    n_cols = len(snapshot_epochs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)

    classes = np.unique(np.concatenate([source_y, target_y]))
    cmap = plt.get_cmap("tab10")

    for r, regime in enumerate(regime_order):
        mmd_map = _history_to_epoch_map(histories_by_regime[regime], "latent_mmd")
        cmmd_map = _history_to_epoch_map(histories_by_regime[regime], "conditional_latent_mmd")
        tgt_acc_map = _history_to_epoch_map(histories_by_regime[regime], "target_acc")
        src_acc_map = _history_to_epoch_map(histories_by_regime[regime], "source_acc")
        for c, epoch in enumerate(snapshot_epochs):
            ax = axes[r][c]
            src = projected[regime][epoch]["source"]
            tgt = projected[regime][epoch]["target"]

            for cls in classes:
                color = cmap(int(cls) % 10)
                src_mask = source_y == cls
                tgt_mask = target_y == cls
                ax.scatter(
                    src[src_mask, 0],
                    src[src_mask, 1],
                    s=10,
                    alpha=0.35,
                    c=[color],
                    marker="o",
                )
                ax.scatter(
                    tgt[tgt_mask, 0],
                    tgt[tgt_mask, 1],
                    s=14,
                    alpha=0.85,
                    c=[color],
                    marker="x",
                    linewidths=0.8,
                )

            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_xticks([])
            ax.set_yticks([])
            title = (
                f"{regime} | epoch {epoch}\n"
                f"MMD={mmd_map[epoch]:.3f}, cMMD={cmmd_map[epoch]:.3f}, "
                f"src acc={src_acc_map[epoch]:.3f}, tgt acc={tgt_acc_map[epoch]:.3f}"
            )
            ax.set_title(title, fontsize=9)

    fig.suptitle(
        f"Embedding Evolution (fixed random projection, seed={projection_seed})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_training_curves(
    *,
    histories_by_regime: Mapping[str, Sequence[Mapping[str, float]]],
    output_path: str | Path,
    regime_order: Sequence[str] = ("baseline", "mmd"),
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(22, 3.8))
    curve_defs = [
        ("total_loss", "Total Loss"),
        ("latent_mmd", "Latent MMD"),
        ("conditional_latent_mmd", "Conditional Latent MMD"),
        ("source_acc", "Source Accuracy"),
        ("target_acc", "Target Accuracy"),
    ]

    for regime in regime_order:
        history = histories_by_regime[regime]
        epochs = [int(row["epoch"]) for row in history]
        for ax, (key, title) in zip(axes, curve_defs):
            vals = [float(row[key]) for row in history]
            ax.plot(epochs, vals, label=regime, linewidth=1.8)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.25)

    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_raw_features(
    *,
    source_x: np.ndarray,
    target_x: np.ndarray,
    source_y: np.ndarray,
    target_y: np.ndarray,
    projection_seed: int,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    projector = FixedRandomProjector.create(in_dim=source_x.shape[1], seed=projection_seed)
    src = projector.transform(source_x)
    tgt = projector.transform(target_x)

    classes = np.unique(np.concatenate([source_y, target_y]))
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    for cls in classes:
        color = cmap(int(cls) % 10)
        src_mask = source_y == cls
        tgt_mask = target_y == cls
        ax.scatter(src[src_mask, 0], src[src_mask, 1], s=10, alpha=0.35, c=[color], marker="o")
        ax.scatter(
            tgt[tgt_mask, 0],
            tgt[tgt_mask, 1],
            s=15,
            alpha=0.85,
            c=[color],
            marker="x",
            linewidths=0.8,
        )

    ax.set_title(f"Raw Features (fixed random projection, seed={projection_seed})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
