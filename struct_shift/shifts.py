"""Structural shift generators for source-target graph pairs."""

from __future__ import annotations

import numpy as np


def balanced_labels(num_nodes: int, num_classes: int, seed: int) -> np.ndarray:
    """Return shuffled near-balanced class labels."""
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    rng = np.random.default_rng(seed)
    counts = np.full(num_classes, num_nodes // num_classes, dtype=np.int64)
    counts[: num_nodes % num_classes] += 1
    labels = np.empty(num_nodes, dtype=np.int64)
    start = 0
    for cls, count in enumerate(counts):
        labels[start : start + count] = cls
        start += count
    rng.shuffle(labels)
    return labels


def sample_features(labels: np.ndarray, num_features: int, feat_std: float, seed: int) -> np.ndarray:
    """Return class-conditional features shared across source and target."""
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    centers = rng.normal(size=(num_classes, num_features)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
    centers *= np.sqrt(float(num_features))
    noise = rng.normal(scale=feat_std, size=(labels.shape[0], num_features)).astype(np.float32)
    return centers[labels] + noise


def sample_csbm_adjacency(
    labels: np.ndarray,
    prob_matrix: np.ndarray,
    seed: int,
    match_upper_edges: int | None = None,
) -> np.ndarray:
    """Sample a symmetric dense adjacency from label-dependent edge probabilities."""
    rng = np.random.default_rng(seed)
    n = labels.shape[0]
    triu = np.triu_indices(n, k=1)
    p = prob_matrix[labels[:, None], labels[None, :]][triu]
    if match_upper_edges is not None:
        expected = float(p.sum())
        if expected > 0:
            p = np.clip(p * (float(match_upper_edges) / expected), 0.0, 1.0)
    draw = rng.random(p.shape[0]) < p
    adj = np.zeros((n, n), dtype=np.float32)
    adj[triu] = draw.astype(np.float32)
    return adj + adj.T


def homophily_heterophily_shift(
    labels: np.ndarray,
    seed: int,
    p_homo: float = 0.12,
    p_hetero: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source homophilic and target heterophilic CSBM adjacencies."""
    num_classes = int(labels.max()) + 1
    src_prob = np.full((num_classes, num_classes), p_hetero, dtype=np.float32)
    tgt_prob = np.full((num_classes, num_classes), p_homo, dtype=np.float32)
    np.fill_diagonal(src_prob, p_homo)
    np.fill_diagonal(tgt_prob, p_hetero)
    source = sample_csbm_adjacency(labels, src_prob, seed=seed)
    src_edges = int(source[np.triu_indices(source.shape[0], k=1)].sum())
    target = sample_csbm_adjacency(labels, tgt_prob, seed=seed + 1, match_upper_edges=src_edges)
    return source, target


def normalize_adj_np(adj: np.ndarray, add_self_loop: bool = True, eps: float = 1e-12) -> np.ndarray:
    """Return symmetric-normalized dense adjacency in numpy."""
    out = adj.astype(np.float64, copy=True)
    if add_self_loop:
        out += np.eye(out.shape[0], dtype=out.dtype)
    deg = np.clip(out.sum(axis=1), eps, None)
    inv_sqrt = np.power(deg, -0.5)
    return inv_sqrt[:, None] * out * inv_sqrt[None, :]


def polynomial_shift(adj: np.ndarray, power: int = 2) -> np.ndarray:
    """Return a binary adjacency induced by polynomial structure A^power."""
    if power < 2:
        raise ValueError("power must be >= 2")
    n = adj.shape[0]
    triu = np.triu_indices(n, k=1)
    base_edges = int(adj[triu].sum())
    if base_edges <= 0:
        return np.zeros_like(adj, dtype=np.float32)

    score = np.linalg.matrix_power(normalize_adj_np(adj, add_self_loop=True), power)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)
    vals = score[triu]

    top_k = min(base_edges, vals.shape[0])
    if top_k == vals.shape[0]:
        keep_idx = np.arange(vals.shape[0], dtype=np.int64)
    else:
        keep_idx = np.argpartition(vals, vals.shape[0] - top_k)[-top_k:]

    new_adj = np.zeros((n, n), dtype=np.float32)
    upper = np.zeros(vals.shape[0], dtype=np.float32)
    upper[keep_idx] = 1.0
    new_adj[triu] = upper
    return new_adj + new_adj.T


def polynomial_structure_shift(
    labels: np.ndarray,
    seed: int,
    p_homo: float = 0.12,
    p_hetero: float = 0.02,
    power: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source CSBM adjacency and target polynomially transformed adjacency."""
    num_classes = int(labels.max()) + 1
    src_prob = np.full((num_classes, num_classes), p_hetero, dtype=np.float32)
    np.fill_diagonal(src_prob, p_homo)
    source = sample_csbm_adjacency(labels, src_prob, seed=seed)
    target = polynomial_shift(source, power=power)
    return source, target


def build_shift_pair(
    shift_type: str,
    labels: np.ndarray,
    seed: int,
    p_homo: float,
    p_hetero: float,
    poly_power: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch a structural shift generator by name."""
    key = shift_type.lower().strip()
    if key in {"hom_het", "csbm", "homophily_heterophily"}:
        return homophily_heterophily_shift(labels, seed=seed, p_homo=p_homo, p_hetero=p_hetero)
    if key in {"poly", "a_to_a2", "polynomial"}:
        return polynomial_structure_shift(labels, seed=seed, p_homo=p_homo, p_hetero=p_hetero, power=poly_power)
    raise ValueError(f"Unknown shift_type: {shift_type}")
