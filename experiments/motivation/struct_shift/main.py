"""Train MLP and GNN for structural-shift synthetic graphs and save embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import strftime

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .models import GNN, MLP
except ImportError:  # when running as a script
    from models import GNN, MLP


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


class DomainShiftTrainer:
    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
        *,
        epochs: int,
        snapshot_every: int,
        lr: float,
        mmd_weight: float,
        seed: int,
        device: str,
        output_dir: Path,
    ) -> None:
        self.name = name
        self.model = model.to(device)
        self.epochs = epochs
        self.snapshot_every = max(1, snapshot_every)
        self.mmd_weight = mmd_weight
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device(device)
        self.output_dir = output_dir

        _set_seed(seed)

    @property
    def _needs_adj(self) -> bool:
        return isinstance(self.model, GNN)

    def train(
        self,
        source_feat: np.ndarray,
        source_adj: np.ndarray,
        source_labels: np.ndarray,
        target_feat: np.ndarray,
        target_adj: np.ndarray,
        target_labels: np.ndarray,
    ) -> dict[str, str]:
        xs = torch.as_tensor(source_feat, dtype=torch.float32, device=self.device)
        ys = torch.as_tensor(source_labels, dtype=torch.long, device=self.device)
        xt = torch.as_tensor(target_feat, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(target_labels, dtype=torch.long, device=self.device)

        as_ = torch.as_tensor(source_adj, dtype=torch.float32, device=self.device) if self._needs_adj else None
        at_ = torch.as_tensor(target_adj, dtype=torch.float32, device=self.device) if self._needs_adj else None

        history: dict[str, list[float]] = {
            "epoch": [],
            "cls_loss": [],
            "mmd_loss": [],
            "total_loss": [],
            "source_acc": [],
            "target_acc": [],
        }

        snapshot_epochs: list[int] = []
        source_snaps: list[np.ndarray] = []
        target_snaps: list[np.ndarray] = []
        mmd_snaps: list[float] = []

        for epoch in range(self.epochs + 1):
            if epoch > 0:
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                logits_s, z_s = self.model(xs, as_)
                _, z_t = self.model(xt, at_)

                cls_loss = F.cross_entropy(logits_s, ys)
                mmd_loss = self.model.mmd_loss(z_s, z_t)
                total_loss = cls_loss + self.mmd_weight * mmd_loss

                total_loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                logits_s, z_s = self.model(xs, as_)
                logits_t, z_t = self.model(xt, at_)

                cls_val = float(F.cross_entropy(logits_s, ys).item())
                mmd_val = float(self.model.mmd_loss(z_s, z_t).item())
                total = cls_val + self.mmd_weight * mmd_val

                history["epoch"].append(epoch)
                history["cls_loss"].append(cls_val)
                history["mmd_loss"].append(mmd_val)
                history["total_loss"].append(total)
                history["source_acc"].append(_accuracy(logits_s, ys))
                history["target_acc"].append(_accuracy(logits_t, yt))

                if epoch == 0 or epoch % self.snapshot_every == 0:
                    snapshot_epochs.append(epoch)
                    source_snaps.append(z_s.detach().cpu().numpy())
                    target_snaps.append(z_t.detach().cpu().numpy())
                    mmd_snaps.append(mmd_val)

        embedding_path = self.output_dir / f"{self.name}_embeddings.npz"
        history_path = self.output_dir / f"{self.name}_history.npz"

        np.savez_compressed(
            embedding_path,
            epochs=np.array(snapshot_epochs, dtype=np.int64),
            source=np.asarray(source_snaps, dtype=np.float32),
            target=np.asarray(target_snaps, dtype=np.float32),
            source_labels=source_labels.astype(np.int64),
            target_labels=target_labels.astype(np.int64),
            mmd=np.asarray(mmd_snaps, dtype=np.float32),
        )

        np.savez_compressed(
            history_path,
            epoch=np.array(history["epoch"], dtype=np.int64),
            cls_loss=np.array(history["cls_loss"], dtype=np.float32),
            mmd_loss=np.array(history["mmd_loss"], dtype=np.float32),
            total_loss=np.array(history["total_loss"], dtype=np.float32),
            source_acc=np.array(history["source_acc"], dtype=np.float32),
            target_acc=np.array(history["target_acc"], dtype=np.float32),
        )

        return {
            "name": self.name,
            "embeddings": str(embedding_path),
            "history": str(history_path),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structural-shift experiment")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing source.npz and target.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"__saved__/struct_shift/run_{strftime('%m%d_%H%M%S')}",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--output-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--mmd-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    source_path = data_dir / "source.npz"
    target_path = data_dir / "target.npz"

    if not source_path.exists():
        raise FileNotFoundError(f"Missing source file: {source_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Missing target file: {target_path}")

    source_npz = np.load(source_path)
    target_npz = np.load(target_path)

    source_feat = source_npz["feat"]
    source_adj = source_npz["adj"]
    source_labels = source_npz["labels"]

    target_feat = target_npz["feat"]
    target_adj = target_npz["adj"]
    target_labels = target_npz["labels"]

    num_classes = int(max(source_labels.max(), target_labels.max()) + 1)
    model_kwargs = {
        "input_dim": int(source_feat.shape[1]),
        "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim,
        "num_classes": num_classes,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, str]] = []
    results.append(
        DomainShiftTrainer(
            name="mlp",
            model=MLP(**model_kwargs),
            epochs=args.epochs,
            snapshot_every=args.snapshot_every,
            lr=args.lr,
            mmd_weight=args.mmd_weight,
            seed=args.seed,
            device=args.device,
            output_dir=output_dir,
        ).train(source_feat, source_adj, source_labels, target_feat, target_adj, target_labels)
    )

    results.append(
        DomainShiftTrainer(
            name="gnn",
            model=GNN(**model_kwargs, add_self_loop=True),
            epochs=args.epochs,
            snapshot_every=args.snapshot_every,
            lr=args.lr,
            mmd_weight=args.mmd_weight,
            seed=args.seed + 1,
            device=args.device,
            output_dir=output_dir,
        ).train(source_feat, source_adj, source_labels, target_feat, target_adj, target_labels)
    )

    print(f"Saved embeddings in: {output_dir}")
    print(f"MLP embeddings: {results[0]['embeddings']}")
    print(f"GNN embeddings: {results[1]['embeddings']}")


if __name__ == "__main__":
    main()
