"""Train GNN and SpectralGNN alignment models and save trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import strftime

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .models import GNN, SpectralGNN
    from .utils import accuracy, ensure_dir, set_seed, to_tensor, train_test_split
except ImportError:
    from models import GNN, SpectralGNN
    from utils import accuracy, ensure_dir, set_seed, to_tensor, train_test_split


MODEL_ALIASES = {"weight_align": "gnn", "filter_align": "spectral_gnn"}
VALID_MODELS = {"gnn", "spectral_gnn"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model training."""
    parser = argparse.ArgumentParser(description="Run structural-shift alignment experiments.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/bini/codes/GDA/KDD/SCGDA/./__saved__/synth_data/csbm_0217_233042",
        help="Directory containing source.npz and target.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"/home/bini/codes/GDA/KDD/SCGDA/./__saved__/struct_shift/run_{strftime('%m%d_%H%M%S')}",
        help="Directory where run artifacts are written.",
    )
    parser.add_argument("--models", type=str, default="gnn,spectral_gnn", help="Comma-separated model names.")
    parser.add_argument("--modes", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--filter-order", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mmd-weight", type=float, default=1.0)
    parser.add_argument("--align-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def parse_models(raw: str) -> list[str]:
    """Parse and validate requested model names."""
    models: list[str] = []
    for token in raw.split(","):
        key = token.strip().lower()
        if not key:
            continue
        key = MODEL_ALIASES.get(key, key)
        if key not in VALID_MODELS:
            raise ValueError(f"Unknown model: {token.strip()}")
        if key not in models:
            models.append(key)
    if not models:
        raise ValueError("At least one model is required.")
    return models


def load_pair(data_dir: str | Path) -> tuple[np.lib.npyio.NpzFile, np.lib.npyio.NpzFile]:
    """Load source and target npz files from a directory."""
    root = Path(data_dir)
    source_path = root / "source.npz"
    target_path = root / "target.npz"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source file: {source_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Missing target file: {target_path}")
    return np.load(source_path), np.load(target_path)


def resolve_splits(
    source_npz: np.lib.npyio.NpzFile,
    target_npz: np.lib.npyio.NpzFile,
    train_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train/test indices for source and target domains."""
    if "train_idx" in source_npz.files and "test_idx" in source_npz.files:
        src_train = source_npz["train_idx"].astype(np.int64)
        src_test = source_npz["test_idx"].astype(np.int64)
    else:
        src_train, src_test = train_test_split(source_npz["feat"].shape[0], train_ratio, seed=seed)

    if "train_idx" in target_npz.files and "test_idx" in target_npz.files:
        tgt_train = target_npz["train_idx"].astype(np.int64)
        tgt_test = target_npz["test_idx"].astype(np.int64)
    elif target_npz["feat"].shape[0] == source_npz["feat"].shape[0]:
        tgt_train, tgt_test = src_train.copy(), src_test.copy()
    else:
        tgt_train, tgt_test = train_test_split(target_npz["feat"].shape[0], train_ratio, seed=seed + 1)
    return src_train, src_test, tgt_train, tgt_test


def build_model(name: str, *, input_dim: int, embed_dim: int, num_classes: int, filter_order: int, dropout: float) -> torch.nn.Module:
    """Construct one requested model with its default alignment target."""
    if name == "gnn":
        return GNN(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
            align_on="weight",
            add_self_loop=True,
        )
    if name == "spectral_gnn":
        return SpectralGNN(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_classes=num_classes,
            filter_order=filter_order,
            dropout=dropout,
            align_on="filter",
            add_self_loop=True,
        )
    raise ValueError(f"Unknown model: {name}")


class Trainer:
    """Train one model and save history and embedding snapshots."""

    def __init__(
        self,
        *,
        name: str,
        model: torch.nn.Module,
        epochs: int,
        snapshot_every: int,
        lr: float,
        mmd_weight: float,
        align_weight: float,
        device: str,
        output_dir: Path,
    ) -> None:
        """Initialize trainer state and optimizer."""
        self.name = name
        self.model = model.to(device)
        self.epochs = int(epochs)
        self.snapshot_every = max(1, int(snapshot_every))
        self.mmd_weight = float(mmd_weight)
        self.align_weight = float(align_weight)
        self.device = torch.device(device)
        self.output_dir = output_dir
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _to_device(
        self,
        source_feat: np.ndarray,
        source_adj: np.ndarray,
        source_labels: np.ndarray,
        source_train_idx: np.ndarray,
        source_test_idx: np.ndarray,
        target_feat: np.ndarray,
        target_adj: np.ndarray,
        target_labels: np.ndarray,
        target_train_idx: np.ndarray,
        target_test_idx: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """Move arrays for one run to the configured torch device."""
        return {
            "xs": to_tensor(source_feat, dtype=torch.float32, device=self.device),
            "as_": to_tensor(source_adj, dtype=torch.float32, device=self.device),
            "ys": to_tensor(source_labels, dtype=torch.long, device=self.device),
            "s_train": to_tensor(source_train_idx, dtype=torch.long, device=self.device),
            "s_test": to_tensor(source_test_idx, dtype=torch.long, device=self.device),
            "xt": to_tensor(target_feat, dtype=torch.float32, device=self.device),
            "at_": to_tensor(target_adj, dtype=torch.float32, device=self.device),
            "yt": to_tensor(target_labels, dtype=torch.long, device=self.device),
            "t_train": to_tensor(target_train_idx, dtype=torch.long, device=self.device),
            "t_test": to_tensor(target_test_idx, dtype=torch.long, device=self.device),
        }

    def _evaluate(self, out: dict[str, torch.Tensor], data: dict[str, torch.Tensor]) -> dict[str, float]:
        """Compute scalar losses and accuracies from one forward pass."""
        src_logits = out["source_logits"]
        tgt_logits = out["target_logits"]
        src_embed = out["source_embed"]
        tgt_embed = out["target_embed"]

        cls = float(F.cross_entropy(src_logits[data["s_train"]], data["ys"][data["s_train"]]).item())
        mmd_train = float(
            self.model.embedding_alignment_loss(src_embed[data["s_train"]], tgt_embed[data["t_train"]]).item()
        )
        mmd_test = float(
            self.model.embedding_alignment_loss(src_embed[data["s_test"]], tgt_embed[data["t_test"]]).item()
        )
        align = float(self.model.parameter_alignment_loss().item())
        total = cls + self.mmd_weight * mmd_train + self.align_weight * align
        return {
            "cls_loss": cls,
            "mmd_train": mmd_train,
            "mmd_test": mmd_test,
            "alignment_loss": align,
            "total_loss": total,
            "source_train_acc": accuracy(src_logits[data["s_train"]], data["ys"][data["s_train"]]),
            "source_test_acc": accuracy(src_logits[data["s_test"]], data["ys"][data["s_test"]]),
            "target_train_acc": accuracy(tgt_logits[data["t_train"]], data["yt"][data["t_train"]]),
            "target_test_acc": accuracy(tgt_logits[data["t_test"]], data["yt"][data["t_test"]]),
            "filter_gap": float(self.model.filter_gap().item()),
            "weight_gap": float(self.model.weight_gap().item()),
        }

    def train(
        self,
        source_feat: np.ndarray,
        source_adj: np.ndarray,
        source_labels: np.ndarray,
        source_train_idx: np.ndarray,
        source_test_idx: np.ndarray,
        target_feat: np.ndarray,
        target_adj: np.ndarray,
        target_labels: np.ndarray,
        target_train_idx: np.ndarray,
        target_test_idx: np.ndarray,
    ) -> dict[str, str]:
        """Run epochs and persist history and embeddings."""
        data = self._to_device(
            source_feat,
            source_adj,
            source_labels,
            source_train_idx,
            source_test_idx,
            target_feat,
            target_adj,
            target_labels,
            target_train_idx,
            target_test_idx,
        )

        history: dict[str, list[float]] = {
            "epoch": [],
            "cls_loss": [],
            "mmd_train": [],
            "mmd_test": [],
            "alignment_loss": [],
            "total_loss": [],
            "source_train_acc": [],
            "source_test_acc": [],
            "target_train_acc": [],
            "target_test_acc": [],
            "filter_gap": [],
            "weight_gap": [],
        }
        snapshots: dict[str, list[np.ndarray | float | int]] = {
            "epochs": [],
            "source_train": [],
            "source_test": [],
            "target_train": [],
            "target_test": [],
            "source_filter_weights": [],
            "target_filter_weights": [],
            "mmd_train": [],
            "mmd_test": [],
            "alignment_loss": [],
        }

        for epoch in range(self.epochs + 1):
            if epoch > 0:
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                out = self.model(data["xs"], data["as_"], data["xt"], data["at_"])
                cls_loss = F.cross_entropy(out["source_logits"][data["s_train"]], data["ys"][data["s_train"]])
                mmd_loss = self.model.embedding_alignment_loss(
                    out["source_embed"][data["s_train"]], out["target_embed"][data["t_train"]]
                )
                align_loss = self.model.parameter_alignment_loss()
                total_loss = cls_loss + self.mmd_weight * mmd_loss + self.align_weight * align_loss
                total_loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                out = self.model(data["xs"], data["as_"], data["xt"], data["at_"])
                metrics = self._evaluate(out, data)

                history["epoch"].append(epoch)
                for key in metrics:
                    history[key].append(metrics[key])

                if epoch == 0 or epoch % self.snapshot_every == 0:
                    src_w, tgt_w = self.model.filter_weights()
                    snapshots["epochs"].append(epoch)
                    snapshots["source_train"].append(out["source_embed"][data["s_train"]].cpu().numpy())
                    snapshots["source_test"].append(out["source_embed"][data["s_test"]].cpu().numpy())
                    snapshots["target_train"].append(out["target_embed"][data["t_train"]].cpu().numpy())
                    snapshots["target_test"].append(out["target_embed"][data["t_test"]].cpu().numpy())
                    snapshots["source_filter_weights"].append(src_w.detach().cpu().numpy())
                    snapshots["target_filter_weights"].append(tgt_w.detach().cpu().numpy())
                    snapshots["mmd_train"].append(metrics["mmd_train"])
                    snapshots["mmd_test"].append(metrics["mmd_test"])
                    snapshots["alignment_loss"].append(metrics["alignment_loss"])

        history_path = self.output_dir / f"{self.name}_history.npz"
        embeddings_path = self.output_dir / f"{self.name}_embeddings.npz"
        np.savez_compressed(
            history_path,
            epoch=np.asarray(history["epoch"], dtype=np.int64),
            cls_loss=np.asarray(history["cls_loss"], dtype=np.float32),
            mmd_train=np.asarray(history["mmd_train"], dtype=np.float32),
            mmd_test=np.asarray(history["mmd_test"], dtype=np.float32),
            alignment_loss=np.asarray(history["alignment_loss"], dtype=np.float32),
            total_loss=np.asarray(history["total_loss"], dtype=np.float32),
            source_train_acc=np.asarray(history["source_train_acc"], dtype=np.float32),
            source_test_acc=np.asarray(history["source_test_acc"], dtype=np.float32),
            target_train_acc=np.asarray(history["target_train_acc"], dtype=np.float32),
            target_test_acc=np.asarray(history["target_test_acc"], dtype=np.float32),
            filter_gap=np.asarray(history["filter_gap"], dtype=np.float32),
            weight_gap=np.asarray(history["weight_gap"], dtype=np.float32),
        )
        np.savez_compressed(
            embeddings_path,
            epochs=np.asarray(snapshots["epochs"], dtype=np.int64),
            source_train=np.asarray(snapshots["source_train"], dtype=np.float32),
            source_test=np.asarray(snapshots["source_test"], dtype=np.float32),
            target_train=np.asarray(snapshots["target_train"], dtype=np.float32),
            target_test=np.asarray(snapshots["target_test"], dtype=np.float32),
            source_train_labels=source_labels[source_train_idx],
            source_test_labels=source_labels[source_test_idx],
            target_train_labels=target_labels[target_train_idx],
            target_test_labels=target_labels[target_test_idx],
            source_train_idx=source_train_idx,
            source_test_idx=source_test_idx,
            target_train_idx=target_train_idx,
            target_test_idx=target_test_idx,
            source_filter_weights=np.asarray(snapshots["source_filter_weights"], dtype=np.float32),
            target_filter_weights=np.asarray(snapshots["target_filter_weights"], dtype=np.float32),
            mmd_train=np.asarray(snapshots["mmd_train"], dtype=np.float32),
            mmd_test=np.asarray(snapshots["mmd_test"], dtype=np.float32),
            alignment_loss=np.asarray(snapshots["alignment_loss"], dtype=np.float32),
        )
        return {"name": self.name, "history": str(history_path), "embeddings": str(embeddings_path)}


def main() -> None:
    """Run requested models and save one summary file."""
    args = parse_args()
    requested = args.models if not args.modes else args.modes
    models = parse_models(requested)

    source_npz, target_npz = load_pair(args.data_dir)
    source_train_idx, source_test_idx, target_train_idx, target_test_idx = resolve_splits(
        source_npz,
        target_npz,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    source_feat = source_npz["feat"].astype(np.float32)
    source_adj = source_npz["adj"].astype(np.float32)
    source_labels = source_npz["labels"].astype(np.int64)
    target_feat = target_npz["feat"].astype(np.float32)
    target_adj = target_npz["adj"].astype(np.float32)
    target_labels = target_npz["labels"].astype(np.int64)
    if source_feat.shape[1] != target_feat.shape[1]:
        raise ValueError("Source and target feature dimensions must match.")

    output_dir = ensure_dir(args.output_dir)
    num_classes = int(max(source_labels.max(), target_labels.max()) + 1)

    results: list[dict[str, str]] = []
    for i, name in enumerate(models):
        set_seed(args.seed + i)
        model = build_model(
            name=name,
            input_dim=int(source_feat.shape[1]),
            embed_dim=args.embed_dim,
            num_classes=num_classes,
            filter_order=args.filter_order,
            dropout=args.dropout,
        )
        trainer = Trainer(
            name=name,
            model=model,
            epochs=args.epochs,
            snapshot_every=args.snapshot_every,
            lr=args.lr,
            mmd_weight=args.mmd_weight,
            align_weight=args.align_weight,
            device=args.device,
            output_dir=output_dir,
        )
        result = trainer.train(
            source_feat=source_feat,
            source_adj=source_adj,
            source_labels=source_labels,
            source_train_idx=source_train_idx,
            source_test_idx=source_test_idx,
            target_feat=target_feat,
            target_adj=target_adj,
            target_labels=target_labels,
            target_train_idx=target_train_idx,
            target_test_idx=target_test_idx,
        )
        results.append(result)
        print(f"{name}: history={result['history']}")
        print(f"{name}: embeddings={result['embeddings']}")

    summary = {"args": vars(args), "models": models, "results": results}
    summary_path = Path(output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved run summary: {summary_path}")
    print(f"Saved run directory: {output_dir}")


if __name__ == "__main__":
    main()
