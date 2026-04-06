from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.transforms import OneHotDegree

torch.set_num_threads(4)

from Learn.Clean_SCGDA.data.build_dataset import get_dataset, get_max_degree
from Learn.Clean_SCGDA.experiments.analysis.mmd_alignment_generalization_tracking.plotting import (
    MODEL_LABELS,
    plot_dirichlet_energy,
    plot_performance,
    plot_snapshot_mmd,
)
from Learn.Clean_SCGDA.experiments.analysis.plain_gnn_cross_domain.log_plot import make_output_dir
from Learn.Clean_SCGDA.utils.config_utils import load_config
from Learn.Clean_SCGDA.utils.expt_utils import set_seed
from Learn.Clean_SCGDA.utils.train_utils.metrics import BaseMetric
from Learn.Clean_SCGDA.utils.train_utils.mmd import MMD


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message="The `pre_transform` argument differs from the one used in the pre-processed version of this dataset.*",
    category=UserWarning,
)


MODEL_SPECS = (
    {"key": "gnn", "label": "GNN", "oracle": False, "use_mmd": False},
    {"key": "simgda", "label": "SimGDA", "oracle": False, "use_mmd": True},
    {"key": "gnn_oracle", "label": "GNN Oracle", "oracle": True, "use_mmd": False},
    {"key": "simgda_oracle", "label": "SimGDA Oracle", "oracle": True, "use_mmd": True},
)

SNAPSHOT_COLUMNS = [
    "after_linear1",
    "after_activation1",
    "after_mp1",
    "after_linear2",
    "after_activation2",
    "after_mp2",
]

SNAPSHOT_TO_METRIC = {name: f"mmd_{name}" for name in SNAPSHOT_COLUMNS}


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _metric_config() -> dict:
    return {"expt": {"metrics": ["micro_f1"], "verbose": 0}}


def _safe_mask(data, mask_name: str) -> torch.Tensor:
    mask = getattr(data, mask_name, None)
    if mask is None:
        return torch.ones_like(data.y, dtype=torch.bool, device=data.y.device)
    return mask.bool()


def _masked_micro_f1(metric: BaseMetric, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum().item()) <= 0:
        return float("nan")
    return float(metric(logits[mask], labels[mask])["micro_f1"])


def _weighted_nll_loss(
    source_logits: torch.Tensor,
    source_labels: torch.Tensor,
    source_mask: torch.Tensor,
    target_logits: torch.Tensor,
    target_labels: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    oracle: bool,
) -> torch.Tensor:
    device = source_logits.device
    source_count = int(source_mask.sum().item())
    target_count = int(target_mask.sum().item()) if oracle else 0
    total_count = source_count + target_count

    if total_count <= 0:
        return torch.zeros((), device=device, dtype=source_logits.dtype)

    loss = torch.zeros((), device=device, dtype=source_logits.dtype)
    if source_count > 0:
        loss = loss + F.nll_loss(source_logits[source_mask], source_labels[source_mask]) * source_count
    if oracle and target_count > 0:
        loss = loss + F.nll_loss(target_logits[target_mask], target_labels[target_mask]) * target_count
    return loss / float(total_count)


def _safe_mmd(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    *,
    sampling_num: int,
    times: int,
) -> float:
    if int(features_a.size(0)) <= 0 or int(features_b.size(0)) <= 0:
        return float("nan")
    return float(
        MMD(features_a, features_b, sampling_num=sampling_num, times=times).detach().cpu().item()
    )


def _dirichlet_energy(data, features: torch.Tensor) -> float:
    edge_index = data.edge_index
    if edge_index.numel() == 0:
        return 0.0

    row, col = edge_index[0], edge_index[1]
    diff = features[row] - features[col]
    edge_energy = diff.pow(2).sum(dim=1)
    edge_weight = getattr(data, "edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device=features.device, dtype=features.dtype)
        edge_energy = edge_weight * edge_energy
    return float(edge_energy.sum().item() / max(1, int(features.size(0))))


def _load_pair_graphs(dataset_name: str, *, source: str, target: str, device: str):
    data_cfg = load_config(f"./configs/data_configs/{dataset_name}.yaml")
    dataset_config = {
        "data": {
            "name": data_cfg["name"],
            "root": data_cfg["root"],
            "domains": list(data_cfg["domains"]),
            "raw_file_names": list(data_cfg["raw_file_names"]),
            "processed_file_names": list(data_cfg["processed_file_names"]),
            "train_split": float(data_cfg["train_split"]),
            "val_split": float(data_cfg["val_split"]),
            "test_split": float(data_cfg["test_split"]),
            "metric": data_cfg["metric"],
        },
        "model": {"name": "gnn"},
        "expt": {"verbose": 0},
    }

    domains = list(dataset_config["data"]["domains"])
    if source not in domains or target not in domains:
        raise ValueError(f"Invalid transfer pair {source}->{target} for dataset {dataset_name}.")

    max_degree = 0
    if dataset_config["data"]["name"].lower() == "airport":
        max_degree = int(get_max_degree(dataset_config))

    graphs = {}
    for domain in (source, target):
        dataset = get_dataset(domain, dataset_config)
        if dataset_config["data"]["name"].lower() == "airport":
            dataset.transform = OneHotDegree(max_degree)
        data = dataset[0].to(device)
        if data.edge_index is not None:
            data.edge_index = data.edge_index.contiguous()
        graphs[domain] = data

    return graphs, {
        "dataset": dataset_name,
        "source": source,
        "target": target,
        "num_features": int(graphs[source].x.size(1)),
        "num_classes": int(torch.unique(torch.cat([graphs[source].y, graphs[target].y])).numel()),
        "max_degree": max_degree,
    }


class SnapshotTwoLayerGNN(nn.Module):
    """Two-layer linear-activation-propagation encoder with explicit snapshot taps."""

    def __init__(self, *, in_dim: int, hid_dim: int, num_classes: int, dropout: float, activation: str):
        super(SnapshotTwoLayerGNN, self).__init__()

        self.dropout = float(dropout)
        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.cls = nn.Linear(hid_dim, num_classes)

        if activation.lower() == "relu":
            self.act = nn.ReLU()
        elif activation.lower() == "elu":
            self.act = nn.ELU()
        elif activation.lower() == "leaky_relu":
            self.act = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    @staticmethod
    def _propagate_once(x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None):
        if edge_index.numel() == 0:
            return x

        src, dst = edge_index[0], edge_index[1]
        if edge_weight is None:
            msg = x[src]
        else:
            msg = x[src] * edge_weight.view(-1, 1)

        out = x.new_zeros((x.size(0), x.size(1)))
        out.index_add_(0, dst, msg)
        return out

    def _normalized_graph(self, x, edge_index, edge_weight=None):
        norm_edge_index, norm_edge_weight = gcn_norm(
            edge_index,
            edge_weight,
            x.size(0),
            improved=False,
            add_self_loops=True,
            dtype=x.dtype,
        )
        if not isinstance(norm_edge_index, torch.Tensor):
            raise TypeError("SnapshotTwoLayerGNN expects COO edge_index tensors.")
        return norm_edge_index, norm_edge_weight

    def encode_with_snapshots(self, x, edge_index, edge_weight=None):
        norm_edge_index, norm_edge_weight = self._normalized_graph(x, edge_index, edge_weight=edge_weight)

        after_linear1 = self.lin1(x)
        after_activation1 = self.act(after_linear1)
        after_mp1 = self._propagate_once(after_activation1, norm_edge_index, norm_edge_weight)

        hidden = F.dropout(after_mp1, p=self.dropout, training=self.training)
        after_linear2 = self.lin2(hidden)
        after_activation2 = self.act(after_linear2)
        after_mp2 = self._propagate_once(after_activation2, norm_edge_index, norm_edge_weight)

        snapshots = {
            "after_linear1": after_linear1,
            "after_activation1": after_activation1,
            "after_mp1": after_mp1,
            "after_linear2": after_linear2,
            "after_activation2": after_activation2,
            "after_mp2": after_mp2,
        }
        return after_mp2, snapshots

    def feat_bottleneck(self, x, edge_index, edge_weight=None, *, return_snapshots: bool = False):
        features, snapshots = self.encode_with_snapshots(x, edge_index, edge_weight=edge_weight)
        if return_snapshots:
            return features, snapshots
        return features

    def feat_classifier(self, features):
        logits = self.cls(F.dropout(features, p=self.dropout, training=self.training))
        return F.log_softmax(logits, dim=1)

    def forward(self, x, edge_index, edge_weight=None, *, return_snapshots: bool = False):
        features, snapshots = self.encode_with_snapshots(x, edge_index, edge_weight=edge_weight)
        logits = self.feat_classifier(features)
        if return_snapshots:
            return logits, features, snapshots
        return logits


@dataclass
class ExperimentConfig:
    dataset: str
    source: str
    target: str
    seeds: tuple[int, ...]
    epochs: int
    hid_dim: int
    dropout_ratio: float
    activation: str
    mmd_weight: float
    mmd_sampling_num: int
    mmd_times: int
    lr: float
    weight_decay: float
    device: str
    verbose: int


def _evaluate_epoch(
    model: SnapshotTwoLayerGNN,
    source_data,
    target_data,
    *,
    metric: BaseMetric,
    mmd_sampling_num: int,
    mmd_times: int,
):
    model.eval()
    with torch.no_grad():
        source_logits, source_features, source_snapshots = model(
            source_data.x,
            source_data.edge_index,
            edge_weight=getattr(source_data, "edge_weight", None),
            return_snapshots=True,
        )
        target_logits, target_features, target_snapshots = model(
            target_data.x,
            target_data.edge_index,
            edge_weight=getattr(target_data, "edge_weight", None),
            return_snapshots=True,
        )

    row = {
        "source_micro_f1": _masked_micro_f1(metric, source_logits, source_data.y, _safe_mask(source_data, "val_mask")),
        "target_micro_f1": _masked_micro_f1(metric, target_logits, target_data.y, _safe_mask(target_data, "val_mask")),
        "source_dirichlet_energy": _dirichlet_energy(source_data, source_features),
        "target_dirichlet_energy": _dirichlet_energy(target_data, target_features),
        "mmd_final": _safe_mmd(
            source_features,
            target_features,
            sampling_num=mmd_sampling_num,
            times=mmd_times,
        ),
    }
    row["dirichlet_energy_gap"] = (
        float(row["target_dirichlet_energy"]) - float(row["source_dirichlet_energy"])
    )

    for snapshot_name in SNAPSHOT_COLUMNS:
        row[SNAPSHOT_TO_METRIC[snapshot_name]] = _safe_mmd(
            source_snapshots[snapshot_name],
            target_snapshots[snapshot_name],
            sampling_num=mmd_sampling_num,
            times=mmd_times,
        )

    return row


def _train_single_model(
    *,
    model_key: str,
    model_label: str,
    oracle: bool,
    use_mmd: bool,
    graphs: dict[str, object],
    config: ExperimentConfig,
    seed: int,
):
    set_seed(seed)
    source_data = graphs[config.source]
    target_data = graphs[config.target]

    model = SnapshotTwoLayerGNN(
        in_dim=int(source_data.x.size(1)),
        hid_dim=int(config.hid_dim),
        num_classes=int(torch.unique(torch.cat([source_data.y, target_data.y])).numel()),
        dropout=float(config.dropout_ratio),
        activation=config.activation,
    ).to(config.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    metric = BaseMetric(_metric_config())

    source_train_mask = _safe_mask(source_data, "train_mask")
    target_train_mask = _safe_mask(target_data, "train_mask")
    rows = []

    for epoch in range(config.epochs):
        model.train()

        source_logits, source_features, _ = model(
            source_data.x,
            source_data.edge_index,
            edge_weight=getattr(source_data, "edge_weight", None),
            return_snapshots=True,
        )
        target_logits, target_features, _ = model(
            target_data.x,
            target_data.edge_index,
            edge_weight=getattr(target_data, "edge_weight", None),
            return_snapshots=True,
        )

        loss = _weighted_nll_loss(
            source_logits,
            source_data.y,
            source_train_mask,
            target_logits,
            target_data.y,
            target_train_mask,
            oracle=oracle,
        )
        mmd_loss_value = 0.0
        if use_mmd:
            mmd_loss = MMD(
                source_features[source_train_mask],
                target_features[target_train_mask],
                sampling_num=config.mmd_sampling_num,
                times=config.mmd_times,
            )
            mmd_loss_value = float(mmd_loss.detach().cpu().item())
            loss = loss + float(config.mmd_weight) * mmd_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        diagnostics = _evaluate_epoch(
            model,
            source_data,
            target_data,
            metric=metric,
            mmd_sampling_num=config.mmd_sampling_num,
            mmd_times=config.mmd_times,
        )
        rows.append(
            {
                "model_key": model_key,
                "model_label": model_label,
                "seed": int(seed),
                "epoch": int(epoch + 1),
                "loss": float(loss.detach().cpu().item()),
                "train_mmd_loss": float(mmd_loss_value),
                **diagnostics,
            }
        )

        if config.verbose >= 2:
            print(
                f"[{model_label}] epoch={epoch+1:03d} "
                f"src={diagnostics['source_micro_f1']:.4f} "
                f"tgt={diagnostics['target_micro_f1']:.4f} "
                f"mmd_final={diagnostics['mmd_final']:.4f}"
            )

    return pd.DataFrame(rows)


def _aggregate_histories(history_df: pd.DataFrame):
    group_cols = ["model_key", "model_label", "epoch"]
    metric_cols = [col for col in history_df.columns if col not in group_cols + ["seed"]]

    mean_df = history_df.groupby(group_cols, as_index=False, sort=False)[metric_cols].mean()
    std_df = history_df.groupby(group_cols, as_index=False, sort=False)[metric_cols].std().fillna(0.0)
    return mean_df, std_df


def _build_summary_text(
    *,
    dataset: str,
    source: str,
    target: str,
    mean_df: pd.DataFrame,
    std_df: pd.DataFrame,
    config: ExperimentConfig,
):
    lines = [
        "MMD Alignment vs Target Generalization",
        f"Dataset: {dataset}",
        f"Transfer: {source} -> {target}",
        f"Seeds: {list(config.seeds)}",
        f"Epochs: {config.epochs}",
        f"Backbone: explicit 2-layer linear -> activation -> message-passing encoder",
        f"MMD setup: weight={config.mmd_weight}, sampling_num={config.mmd_sampling_num}, times={config.mmd_times}",
        "",
        "Final epoch mean +/- std:",
    ]

    final_epoch = int(mean_df["epoch"].max())
    for model_key, model_label in MODEL_LABELS.items():
        cur_mean = mean_df[(mean_df["model_key"] == model_key) & (mean_df["epoch"] == final_epoch)]
        if cur_mean.empty:
            continue
        cur_std = std_df[(std_df["model_key"] == model_key) & (std_df["epoch"] == final_epoch)]
        row_mean = cur_mean.iloc[0]
        row_std = cur_std.iloc[0] if not cur_std.empty else pd.Series(dtype=float)
        lines.append(
            f"- {model_label}: "
            f"source={float(row_mean['source_micro_f1']):.4f} +/- {float(row_std.get('source_micro_f1', 0.0)):.4f}, "
            f"target={float(row_mean['target_micro_f1']):.4f} +/- {float(row_std.get('target_micro_f1', 0.0)):.4f}, "
            f"final MMD={float(row_mean['mmd_after_mp2']):.4f} +/- {float(row_std.get('mmd_after_mp2', 0.0)):.4f}"
        )

    lines.append("")
    lines.append("Tracked snapshot MMDs are measured on full-graph source/target representations in eval mode.")
    lines.append("Dirichlet energy is measured on the final embedding right before the classifier and normalized by node count.")
    return "\n".join(lines) + "\n"


def run_experiment(
    *,
    dataset: str,
    source: str,
    target: str,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    epochs: int = 200,
    hid_dim: int = 64,
    dropout_ratio: float = 0.1,
    activation: str = "relu",
    mmd_weight: float = 0.1,
    mmd_sampling_num: int = 256,
    mmd_times: int = 2,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    device: str = "cpu",
    verbose: int = 0,
    out_root: str = "./__saved__/analysis/mmd_alignment_generalization_tracking",
    run_root: str | Path | None = None,
):
    config = ExperimentConfig(
        dataset=dataset,
        source=source,
        target=target,
        seeds=tuple(int(seed) for seed in seeds),
        epochs=int(epochs),
        hid_dim=int(hid_dim),
        dropout_ratio=float(dropout_ratio),
        activation=str(activation),
        mmd_weight=float(mmd_weight),
        mmd_sampling_num=int(mmd_sampling_num),
        mmd_times=int(mmd_times),
        lr=float(lr),
        weight_decay=float(weight_decay),
        device=str(device),
        verbose=int(verbose),
    )

    if run_root is None:
        run_dir = make_output_dir(
            out_root=out_root,
            run_name=f"mmd_alignment_generalization_{dataset}_{source}_to_{target}_seeds{config.seeds[0]}to{config.seeds[-1]}_{time.strftime('%m%d_%H%M%S')}",
        )
    else:
        run_dir = Path(run_root) / dataset / f"{source}_to_{target}"
        run_dir.mkdir(parents=True, exist_ok=True)

    graphs, metadata = _load_pair_graphs(
        dataset,
        source=source,
        target=target,
        device=config.device,
    )

    history_frames = []
    start_time = time.time()
    for seed in config.seeds:
        if config.verbose >= 1:
            print(f"[seed {seed}] {dataset} {source}->{target}")
        for spec in MODEL_SPECS:
            history_frames.append(
                _train_single_model(
                    model_key=spec["key"],
                    model_label=spec["label"],
                    oracle=bool(spec["oracle"]),
                    use_mmd=bool(spec["use_mmd"]),
                    graphs=graphs,
                    config=config,
                    seed=int(seed),
                )
            )
        _cleanup_cuda()

    history_df = pd.concat(history_frames, ignore_index=True)
    mean_df, std_df = _aggregate_histories(history_df)

    history_df.to_csv(run_dir / "histories_all_seeds.csv", index=False)
    mean_df.to_csv(run_dir / "histories_mean.csv", index=False)
    std_df.to_csv(run_dir / "histories_std.csv", index=False)

    plot_snapshot_mmd(mean_df, out_dir=run_dir, std_df=std_df)
    plot_performance(mean_df, out_dir=run_dir, std_df=std_df)
    plot_dirichlet_energy(mean_df, out_dir=run_dir, std_df=std_df)

    summary_text = _build_summary_text(
        dataset=dataset,
        source=source,
        target=target,
        mean_df=mean_df,
        std_df=std_df,
        config=config,
    )
    (run_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    metadata_payload = {
        "dataset": dataset,
        "source": source,
        "target": target,
        "seeds": list(config.seeds),
        "epochs": config.epochs,
        "hid_dim": config.hid_dim,
        "dropout_ratio": config.dropout_ratio,
        "activation": config.activation,
        "mmd_weight": config.mmd_weight,
        "mmd_sampling_num": config.mmd_sampling_num,
        "mmd_times": config.mmd_times,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "device": config.device,
        "num_features": metadata["num_features"],
        "num_classes": metadata["num_classes"],
        "train_time_sec": float(time.time() - start_time),
        "artifacts": {
            "histories_all_seeds": "histories_all_seeds.csv",
            "histories_mean": "histories_mean.csv",
            "histories_std": "histories_std.csv",
            "snapshot_mmd_plot": "snapshot_mmd_over_epochs.png",
            "performance_plot": "performance_over_epochs.png",
            "dirichlet_energy_plot": "dirichlet_energy_over_epochs.png",
            "summary": "summary.txt",
        },
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    if verbose >= 1:
        print(f"[saved] {run_dir}")

    return run_dir
