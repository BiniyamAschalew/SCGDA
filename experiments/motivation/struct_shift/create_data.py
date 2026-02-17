"""Create synthetic graph data for structural shift experiments."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np


def _build_block_matrix(num_classes: int, edge_probs: Sequence[float]) -> np.ndarray:
    probs = np.asarray(edge_probs, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("edge_probs must be a 1-D sequence of probabilities.")
    if probs.size == 0:
        raise ValueError("edge_probs must contain at least one value.")

    k = num_classes
    block = np.full((k, k), np.mean(probs))

    if probs.size == k:
        np.fill_diagonal(block, probs)
        return np.clip(block, 0.0, 1.0)

    if probs.size == k * (k - 1) // 2:
        triu_idx = np.triu_indices(k, k=1)
        block[triu_idx] = probs
        block[(triu_idx[1], triu_idx[0])] = probs
        return np.clip(block, 0.0, 1.0)

    if probs.size == k * k:
        return np.clip(probs.reshape((k, k)), 0.0, 1.0)

    raise ValueError(
        "Unsupported edge_probs size. Use len = num_classes, "
        "num_classes*(num_classes-1)//2, or num_classes*num_classes."
    )


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
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one attributed SBM-like graph (features, adjacency)."""

    if num_nodes <= 1:
        raise ValueError("num_nodes must be greater than 1.")
    if num_node_features <= 0:
        raise ValueError("num_node_features must be positive.")
    if num_edges < 0:
        raise ValueError("num_edges must be non-negative.")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

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
    feature_noise = rng.normal(
        loc=feat_mean, scale=feat_std, size=(num_nodes, num_node_features)
    )
    feat = centers[node_classes].astype(np.float32) + feature_noise.astype(np.float32)

    block = _build_block_matrix(num_classes, edge_probs)
    block = np.clip(block, 0.0, 1.0)

    tri = np.triu_indices(num_nodes, k=1)
    max_edges = tri[0].shape[0]
    if num_edges > max_edges:
        num_edges = max_edges

    if num_edges == 0:
        block = np.zeros_like(block)
    elif num_edges > 0:
        p_upper = block[node_classes[:, None], node_classes[None, :]][tri]
        expected = float(p_upper.sum())
        if expected > 0:
            block = np.clip(block * (num_edges / expected), 0.0, 1.0)

    p_matrix = block[node_classes[:, None], node_classes[None, :]]
    rand = rng.random(size=p_matrix.shape)
    upper = (rand < p_matrix) & np.triu(np.ones_like(p_matrix, dtype=bool), k=1)
    adj = upper.astype(np.float32)
    return feat, adj + adj.T


if __name__ == "__main__":
    time_stamp = time.strftime("%m%d_%H%M%S")
    save_dir = Path(
        f"/home/bini/codes/GDA/KDD/SCGDA/__saved__/synth_data/csbm_{time_stamp}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    shared_args = dict(
        seed=7,
        num_nodes=100,
        num_edges=500,
        num_node_features=16,
        num_classes=3,
        feat_mean=0.0,
        feat_std=1.0,
    )

    src_args = {
        **shared_args,
        "edge_probs": [0.05, 0.01, 0.02],
    }
    tgt_args = {
        **shared_args,
        "edge_probs": [0.01, 0.05, 0.02],
    }

    src_feat, src_adj = generate_synthetic_graph_data(**src_args)
    tgt_feat, tgt_adj = generate_synthetic_graph_data(**tgt_args)

    np.savez_compressed(save_dir / "source.npz", feat=src_feat, adj=src_adj)
    np.savez_compressed(save_dir / "target.npz", feat=tgt_feat, adj=tgt_adj)
