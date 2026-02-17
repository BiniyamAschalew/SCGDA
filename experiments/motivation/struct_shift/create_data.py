"""Create synthetic attributed graphs for the structural-shift study."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np


def _build_block_matrix(num_classes: int, edge_probs: Sequence[float]) -> np.ndarray:
    probs = np.asarray(edge_probs, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("edge_probs must be a 1-D sequence.")
    if probs.size == 0:
        raise ValueError("edge_probs must contain at least one value.")

    k = num_classes
    block = np.full((k, k), np.mean(probs))

    if probs.size == k:
        np.fill_diagonal(block, probs)
    elif probs.size == k * (k - 1) // 2:
        tri = np.triu_indices(k, k=1)
        block[tri] = probs
        block[(tri[1], tri[0])] = probs
    elif probs.size == k * k:
        block = probs.reshape((k, k))
    else:
        raise ValueError(
            "Unsupported edge_probs size. Use len = num_classes, num_classes*(num_classes-1)//2, or num_classes*num_classes."
        )

    return np.clip(block, 0.0, 1.0)


def generate_synthetic_graph_data(
    *,
    num_nodes: int = 100,
    num_edges: int = 500,
    num_node_features: int = 16,
    num_classes: int = 3,
    edge_probs: list[float] | tuple[float, ...] = (0.05, 0.01, 0.02),
    feat_mean: float = 0.0,
    feat_std: float = 1.0,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return features, dense adjacency and class labels for one graph."""

    if num_nodes <= 1:
        raise ValueError("num_nodes must be greater than 1")

    rng = np.random.default_rng(seed)

    class_sizes = np.full(num_classes, num_nodes // num_classes, dtype=np.int64)
    class_sizes[: num_nodes % num_classes] += 1

    node_classes = np.empty(num_nodes, dtype=np.int64)
    start = 0
    for cls, size in enumerate(class_sizes):
        node_classes[start : start + size] = cls
        start += size
    rng.shuffle(node_classes)

    centers = rng.normal(size=(num_classes, num_node_features)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
    centers *= np.sqrt(num_node_features)

    feat = centers[node_classes] + rng.normal(
        loc=feat_mean,
        scale=feat_std,
        size=(num_nodes, num_node_features),
    ).astype(np.float32)

    block = _build_block_matrix(num_classes, edge_probs)
    upper = np.triu_indices(num_nodes, k=1)

    max_edges = upper[0].size
    if num_edges > max_edges:
        num_edges = max_edges

    if num_edges == 0:
        block = np.zeros_like(block)
    else:
        p_upper = block[node_classes[:, None], node_classes[None, :]][upper]
        expected = float(p_upper.sum())
        if expected > 0:
            block = np.clip(block * (num_edges / expected), 0.0, 1.0)

    p = block[node_classes[:, None], node_classes[None, :]]
    rnd = rng.random(size=p.shape)
    adj = (rnd < p) & np.triu(np.ones_like(p, dtype=bool), k=1)
    adj = (adj.astype(np.float32) + adj.astype(np.float32).T)

    return feat, adj, node_classes


def save_shift_pair(
    output_dir: str | Path,
    src_args: dict,
    tgt_args: dict,
) -> tuple[Path, Path]:
    """Generate and save source/target npz files and return their paths."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src_feat, src_adj, src_labels = generate_synthetic_graph_data(**src_args)
    tgt_feat, tgt_adj, tgt_labels = generate_synthetic_graph_data(**tgt_args)

    source_path = output_dir / "source.npz"
    target_path = output_dir / "target.npz"

    np.savez_compressed(source_path, feat=src_feat, adj=src_adj, labels=src_labels)
    np.savez_compressed(target_path, feat=tgt_feat, adj=tgt_adj, labels=tgt_labels)

    return source_path, target_path


def main() -> None:
    timestamp = time.strftime("%m%d_%H%M%S")
    save_dir = Path(f"/home/bini/codes/GDA/KDD/SCGDA/__saved__/synth_data/csbm_{timestamp}")

    shared = dict(
        seed=42,
        num_nodes=100,
        num_node_features=16,
        num_classes=2,
        feat_mean=0.0,
        feat_std=1.0,
    )

    src_args = {
        **shared,
        "num_edges": 1000,
        "edge_probs": [0.05, 0.01, 0.01, 0.03],
    }
    tgt_args = {
        **shared,
        "num_edges": 500,
        "edge_probs": [0.03, 0.02, 0.01, 0.04],
    }

    save_shift_pair(save_dir, src_args=src_args, tgt_args=tgt_args)
    print(f"Saved synthetic graphs to: {save_dir}")


if __name__ == "__main__":
    main()
