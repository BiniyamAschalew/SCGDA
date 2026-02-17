"""Utility helpers for structural-shift experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set numpy and torch random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_test_split(num_nodes: int, train_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return shuffled train and test node indices."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if num_nodes < 2:
        raise ValueError("num_nodes must be at least 2")
    n_train = int(np.floor(train_ratio * num_nodes))
    n_train = int(np.clip(n_train, 1, num_nodes - 1))
    rng = np.random.default_rng(seed)
    idx = np.arange(num_nodes, dtype=np.int64)
    rng.shuffle(idx)
    return idx[:n_train], idx[n_train:]


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Return classification accuracy as a python float."""
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def to_tensor(array: np.ndarray, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Convert numpy array to torch tensor on a target device."""
    return torch.as_tensor(array, dtype=dtype, device=device)


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
