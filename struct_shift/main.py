"""Train MLP and GNN for structural-shift synthetic graphs and save embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import strftime

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .models import GNN, MLP, SpectralGNN
except ImportError:  # when running as a script
    from models import GNN, MLP, SpectralGNN


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _train_test_split(num_nodes: int, train_ratio: float, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if num_nodes < 2:
        raise ValueError("Need at least 2 nodes to create train/test split")

    n_train = int(np.floor(num_nodes * train_ratio))
    n_train = int(np.clip(n_train, 1, num_nodes - 1))

    rng = np.random.default_rng(seed)
    idx = np.arange(num_nodes)
    rng.shuffle(idx)

    return idx[:n_train], idx[n_train:]


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
        self._needs_adj = bool(getattr(self.model, "needs_adjacency", False))

        _set_seed(seed)

    @staticmethod
    def _to_tensor(data: np.ndarray | None, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
        if data is None:
            return None
        return torch.as_tensor(data, dtype=dtype, device=device)

    def _forward(self, x: torch.Tensor, adj: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, adj)

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
        xs = torch.as_tensor(source_feat, dtype=torch.float32, device=self.device)
        ys = torch.as_tensor(source_labels, dtype=torch.long, device=self.device)
        xt = torch.as_tensor(target_feat, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(target_labels, dtype=torch.long, device=self.device)

        as_ = self._to_tensor(source_adj, torch.float32, self.device) if self._needs_adj else None
        at_ = self._to_tensor(target_adj, torch.float32, self.device) if self._needs_adj else None

        s_train_idx = torch.as_tensor(source_train_idx, device=self.device, dtype=torch.long)
        s_test_idx = torch.as_tensor(source_test_idx, device=self.device, dtype=torch.long)
        t_train_idx = torch.as_tensor(target_train_idx, device=self.device, dtype=torch.long)
        t_test_idx = torch.as_tensor(target_test_idx, device=self.device, dtype=torch.long)

        history: dict[str, list[float]] = {
            "epoch": [],
            "cls_loss": [],
            "mmd_train": [],
            "mmd_test": [],
            "total_loss": [],
            "source_train_acc": [],
            "source_test_acc": [],
            "target_train_acc": [],
            "target_test_acc": [],
        }

        snapshot_epochs: list[int] = []
        source_train_snaps: list[np.ndarray] = []
        source_test_snaps: list[np.ndarray] = []
        target_train_snaps: list[np.ndarray] = []
        target_test_snaps: list[np.ndarray] = []
        mmd_train_snaps: list[float] = []
        mmd_test_snaps: list[float] = []
        spectral_weight_snaps: list[np.ndarray] = []

        for epoch in range(self.epochs + 1):
            if epoch > 0:
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                logits_s, z_s = self._forward(xs, as_)
                logits_t, z_t = self._forward(xt, at_)

                cls_loss = F.cross_entropy(logits_s[s_train_idx], ys[s_train_idx])
                mmd_train = self.model.mmd_loss(z_s[s_train_idx], z_t[t_train_idx])
                total_loss = cls_loss + self.mmd_weight * mmd_train

                total_loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                logits_s, z_s = self._forward(xs, as_)
                logits_t, z_t = self._forward(xt, at_)

                logits_s_train = logits_s[s_train_idx]
                logits_s_test = logits_s[s_test_idx]
                logits_t_train = logits_t[t_train_idx]
                logits_t_test = logits_t[t_test_idx]

                z_s_train = z_s[s_train_idx]
                z_s_test = z_s[s_test_idx]
                z_t_train = z_t[t_train_idx]
                z_t_test = z_t[t_test_idx]

                cls_val = float(F.cross_entropy(logits_s_train, ys[s_train_idx]).item())
                mmd_train_val = float(self.model.mmd_loss(z_s_train, z_t_train).item())
                mmd_test_val = float(self.model.mmd_loss(z_s_test, z_t_test).item())
                total = cls_val + self.mmd_weight * mmd_train_val

                history["epoch"].append(epoch)
                history["cls_loss"].append(cls_val)
                history["mmd_train"].append(mmd_train_val)
                history["mmd_test"].append(mmd_test_val)
                history["total_loss"].append(total)
                history["source_train_acc"].append(_accuracy(logits_s_train, ys[s_train_idx]))
                history["source_test_acc"].append(_accuracy(logits_s_test, ys[s_test_idx]))
                history["target_train_acc"].append(_accuracy(logits_t_train, yt[t_train_idx]))
                history["target_test_acc"].append(_accuracy(logits_t_test, yt[t_test_idx]))

                if epoch == 0 or epoch % self.snapshot_every == 0:
                    snapshot_epochs.append(epoch)
                    source_train_snaps.append(z_s_train.detach().cpu().numpy())
                    source_test_snaps.append(z_s_test.detach().cpu().numpy())
                    target_train_snaps.append(z_t_train.detach().cpu().numpy())
                    target_test_snaps.append(z_t_test.detach().cpu().numpy())
                    mmd_train_snaps.append(mmd_train_val)
                    mmd_test_snaps.append(mmd_test_val)
                    if hasattr(self.model, "spectral_logits"):
                        spectral_weight_snaps.append(
                            F.softmax(self.model.spectral_logits.detach(), dim=0).cpu().numpy()
                        )

        embedding_path = self.output_dir / f"{self.name}_embeddings.npz"
        history_path = self.output_dir / f"{self.name}_history.npz"
        embedding_payload: dict[str, np.ndarray | str] = {
            "epochs": np.array(snapshot_epochs, dtype=np.int64),
            "source_train": np.asarray(source_train_snaps, dtype=np.float32),
            "source_test": np.asarray(source_test_snaps, dtype=np.float32),
            "target_train": np.asarray(target_train_snaps, dtype=np.float32),
            "target_test": np.asarray(target_test_snaps, dtype=np.float32),
            "source_train_labels": source_labels[source_train_idx],
            "source_test_labels": source_labels[source_test_idx],
            "target_train_labels": target_labels[target_train_idx],
            "target_test_labels": target_labels[target_test_idx],
            "source_train_idx": source_train_idx,
            "source_test_idx": source_test_idx,
            "target_train_idx": target_train_idx,
            "target_test_idx": target_test_idx,
            "mmd_train": np.asarray(mmd_train_snaps, dtype=np.float32),
            "mmd_test": np.asarray(mmd_test_snaps, dtype=np.float32),
        }
        if spectral_weight_snaps:
            embedding_payload["spectral_layer_weights"] = np.asarray(spectral_weight_snaps, dtype=np.float32)

        np.savez_compressed(embedding_path, **embedding_payload)

        np.savez_compressed(
            history_path,
            epoch=np.array(history["epoch"], dtype=np.int64),
            cls_loss=np.array(history["cls_loss"], dtype=np.float32),
            mmd_train=np.array(history["mmd_train"], dtype=np.float32),
            mmd_test=np.array(history["mmd_test"], dtype=np.float32),
            total_loss=np.array(history["total_loss"], dtype=np.float32),
            source_train_acc=np.array(history["source_train_acc"], dtype=np.float32),
            source_test_acc=np.array(history["source_test_acc"], dtype=np.float32),
            target_train_acc=np.array(history["target_train_acc"], dtype=np.float32),
            target_test_acc=np.array(history["target_test_acc"], dtype=np.float32),
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
        default="/home/bini/codes/GDA/KDD/SCGDA/__saved__/synth_data/csbm_0217_221652",
        help="Directory containing source.npz and target.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"/home/bini/codes/GDA/KDD/SCGDA/__saved__/struct_shift/run_{strftime('%m%d_%H%M%S')}",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.7)
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

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("train-ratio must be in (0, 1)")

    source_npz = np.load(source_path)
    target_npz = np.load(target_path)

    source_feat = source_npz["feat"]
    source_adj = source_npz["adj"]
    source_labels = source_npz["labels"]

    target_feat = target_npz["feat"]
    target_adj = target_npz["adj"]
    target_labels = target_npz["labels"]

    source_train_idx, source_test_idx = _train_test_split(
        num_nodes=source_feat.shape[0],
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    target_train_idx, target_test_idx = _train_test_split(
        num_nodes=target_feat.shape[0],
        train_ratio=args.train_ratio,
        seed=args.seed + 1,
    )

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
    model_specs = [
        ("mlp", MLP(**model_kwargs)),
        ("gnn", GNN(**model_kwargs, add_self_loop=True)),
        ("spectral_gnn", SpectralGNN(**{**model_kwargs, "num_layers": 3})),
    ]

    for i, (name, model) in enumerate(model_specs):
        results.append(
            DomainShiftTrainer(
                name=name,
                model=model,
                epochs=args.epochs,
                snapshot_every=args.snapshot_every,
                lr=args.lr,
                mmd_weight=args.mmd_weight,
                seed=args.seed + i,
                device=args.device,
                output_dir=output_dir,
            ).train(
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
        )

    for item in results:
        print(f"{item['name']} embeddings: {item['embeddings']}")
    print(f"Saved embeddings in: {output_dir}")


if __name__ == "__main__":
    main()
