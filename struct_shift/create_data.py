"""Create synthetic source-target graph pairs with structural shift only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    from .shifts import balanced_labels, build_shift_pair, sample_features
    from .utils import ensure_dir, train_test_split
except ImportError:
    from  struct_shift.shifts import balanced_labels, build_shift_pair, sample_features
    from  struct_shift.utils import ensure_dir, train_test_split


def parse_args() -> argparse.Namespace:
    """Parse command-line options for synthetic data generation."""
    parser = argparse.ArgumentParser(description="Create synthetic graph domain-shift data.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"/home/bini/codes/GDA/KDD/SCGDA/./__saved__/synth_data/csbm_{time.strftime('%m%d_%H%M%S')}",
        help="Directory where source.npz and target.npz are saved.",
    )
    parser.add_argument("--shift-type", type=str, default="hom_het", choices=["hom_het", "poly"])
    parser.add_argument("--num-nodes", type=int, default=200)
    parser.add_argument("--num-features", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--feat-std", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--p-homo", type=float, default=0.12)
    parser.add_argument("--p-hetero", type=float, default=0.02)
    parser.add_argument("--poly-power", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def create_pair(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build source and target domains with shared features and labels."""
    labels = balanced_labels(args.num_nodes, args.num_classes, seed=args.seed)
    feat = sample_features(labels, args.num_features, args.feat_std, seed=args.seed + 1)
    source_adj, target_adj = build_shift_pair(
        shift_type=args.shift_type,
        labels=labels,
        seed=args.seed + 2,
        p_homo=args.p_homo,
        p_hetero=args.p_hetero,
        poly_power=args.poly_power,
    )
    train_idx, test_idx = train_test_split(args.num_nodes, args.train_ratio, seed=args.seed + 3)

    source = {
        "feat": feat.astype(np.float32),
        "adj": source_adj.astype(np.float32),
        "labels": labels.astype(np.int64),
        "train_idx": train_idx.astype(np.int64),
        "test_idx": test_idx.astype(np.int64),
    }
    target = {
        "feat": feat.astype(np.float32),
        "adj": target_adj.astype(np.float32),
        "labels": labels.astype(np.int64),
        "train_idx": train_idx.astype(np.int64),
        "test_idx": test_idx.astype(np.int64),
    }
    return source, target


def save_pair(output_dir: str | Path, source: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> tuple[Path, Path]:
    """Save source and target dictionaries into compressed npz files."""
    out = ensure_dir(output_dir)
    source_path = out / "source.npz"
    target_path = out / "target.npz"
    np.savez_compressed(source_path, **source)
    np.savez_compressed(target_path, **target)
    return source_path, target_path


def main() -> None:
    """Generate and save one structural-shift dataset pair."""
    args = parse_args()
    source, target = create_pair(args)
    source_path, target_path = save_pair(args.output_dir, source, target)
    meta_path = Path(args.output_dir) / "meta.json"
    meta_path.write_text(json.dumps(vars(args), indent=2))
    print(f"Saved source: {source_path}")
    print(f"Saved target: {target_path}")
    print(f"Saved meta: {meta_path}")


if __name__ == "__main__":
    main()
