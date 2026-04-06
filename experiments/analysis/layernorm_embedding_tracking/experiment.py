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

from  data.build_dataset import get_dataset, get_max_degree
from  experiments.analysis.layernorm_embedding_tracking.plotting import (
    MODEL_LABELS,
    plot_final_snapshot_mmd,
    plot_snapshot_mmd,
    plot_training_overview,
)
from  experiments.analysis.plain_gnn_cross_domain.log_plot import make_output_dir
from  utils.config_utils import load_config
from  utils.expt_utils import set_seed
from  utils.train_utils.metrics import BaseMetric
from  utils.train_utils.mmd import MMD


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message="The `pre_transform` argument differs from the one used in the pre-processed version of this dataset.*",
    category=UserWarning,
)


MODEL_SPECS = (
    {"key": "layernorm_gnn", "label": "LayerNorm GNN", "kind": "gnn"},
    {"key": "layernorm_mlp", "label": "LayerNorm MLP", "kind": "mlp"},
)


def _build_snapshot_specs(model_kind: str, num_layers: int) -> list[tuple[str, str]]:
    specs = [("raw_input", "Raw Input")]
    for layer_idx in range(1, int(num_layers) + 1):
        specs.append((f"after_linear{layer_idx}", f"Linear {layer_idx}"))
        specs.append((f"after_norm{layer_idx}", f"LayerNorm {layer_idx}"))
        specs.append((f"after_activation{layer_idx}", f"Activation {layer_idx}"))
        if model_kind == "gnn":
            specs.append((f"after_mp{layer_idx}", f"MP {layer_idx}"))
    return specs


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


def _feature_mean_norm(features: torch.Tensor) -> float:
    if int(features.size(0)) <= 0:
        return 0.0
    return float(torch.linalg.vector_norm(features, ord=2, dim=1).mean().detach().cpu().item())


def _centroid_gap(features_a: torch.Tensor, features_b: torch.Tensor) -> float:
    if int(features_a.size(0)) <= 0 or int(features_b.size(0)) <= 0:
        return float("nan")
    return float(torch.linalg.vector_norm(features_a.mean(0) - features_b.mean(0), ord=2).detach().cpu().item())


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


def _build_activation(activation: str):
    activation = str(activation).lower()
    if activation == "relu":
        return nn.ReLU()
    if activation == "elu":
        return nn.ELU()
    if activation == "leaky_relu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported activation: {activation}")


class LayerNormSnapshotGNN(nn.Module):
    def __init__(self, *, in_dim: int, hid_dim: int, num_classes: int, num_layers: int, dropout: float, activation: str):
        super(LayerNormSnapshotGNN, self).__init__()

        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.act = _build_activation(activation)

        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.lins.append(nn.Linear(in_dim, hid_dim))
        self.norms.append(nn.LayerNorm(hid_dim))
        for _ in range(self.num_layers - 1):
            self.lins.append(nn.Linear(hid_dim, hid_dim))
            self.norms.append(nn.LayerNorm(hid_dim))

        self.cls = nn.Linear(hid_dim, num_classes)

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
            raise TypeError("LayerNormSnapshotGNN expects COO edge_index tensors.")
        return norm_edge_index, norm_edge_weight

    def encode_with_snapshots(self, x, edge_index, edge_weight=None):
        norm_edge_index, norm_edge_weight = self._normalized_graph(x, edge_index, edge_weight=edge_weight)

        hidden = x
        snapshots = {"raw_input": x}
        for layer_idx, (lin, norm) in enumerate(zip(self.lins, self.norms), start=1):
            after_linear = lin(hidden)
            after_norm = norm(after_linear)
            after_activation = self.act(after_norm)
            after_mp = self._propagate_once(after_activation, norm_edge_index, norm_edge_weight)

            snapshots[f"after_linear{layer_idx}"] = after_linear
            snapshots[f"after_norm{layer_idx}"] = after_norm
            snapshots[f"after_activation{layer_idx}"] = after_activation
            snapshots[f"after_mp{layer_idx}"] = after_mp

            hidden = after_mp
            if layer_idx < self.num_layers:
                hidden = F.dropout(hidden, p=self.dropout, training=self.training)

        return hidden, snapshots

    def feat_classifier(self, features):
        logits = self.cls(F.dropout(features, p=self.dropout, training=self.training))
        return F.log_softmax(logits, dim=1)

    def forward(self, x, edge_index, edge_weight=None, *, return_snapshots: bool = False):
        features, snapshots = self.encode_with_snapshots(x, edge_index, edge_weight=edge_weight)
        logits = self.feat_classifier(features)
        if return_snapshots:
            return logits, features, snapshots
        return logits


class LayerNormSnapshotMLP(nn.Module):
    def __init__(self, *, in_dim: int, hid_dim: int, num_classes: int, num_layers: int, dropout: float, activation: str):
        super(LayerNormSnapshotMLP, self).__init__()

        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.act = _build_activation(activation)

        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.lins.append(nn.Linear(in_dim, hid_dim))
        self.norms.append(nn.LayerNorm(hid_dim))
        for _ in range(self.num_layers - 1):
            self.lins.append(nn.Linear(hid_dim, hid_dim))
            self.norms.append(nn.LayerNorm(hid_dim))

        self.cls = nn.Linear(hid_dim, num_classes)

    def encode_with_snapshots(self, x, edge_index=None, edge_weight=None):
        hidden = x
        snapshots = {"raw_input": x}
        for layer_idx, (lin, norm) in enumerate(zip(self.lins, self.norms), start=1):
            after_linear = lin(hidden)
            after_norm = norm(after_linear)
            after_activation = self.act(after_norm)

            snapshots[f"after_linear{layer_idx}"] = after_linear
            snapshots[f"after_norm{layer_idx}"] = after_norm
            snapshots[f"after_activation{layer_idx}"] = after_activation

            hidden = after_activation
            if layer_idx < self.num_layers:
                hidden = F.dropout(hidden, p=self.dropout, training=self.training)

        return hidden, snapshots

    def feat_classifier(self, features):
        logits = self.cls(F.dropout(features, p=self.dropout, training=self.training))
        return F.log_softmax(logits, dim=1)

    def forward(self, x, edge_index=None, edge_weight=None, *, return_snapshots: bool = False):
        features, snapshots = self.encode_with_snapshots(x, edge_index=edge_index, edge_weight=edge_weight)
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
    num_layers: int
    hid_dim: int
    dropout_ratio: float
    activation: str
    mmd_sampling_num: int
    mmd_times: int
    lr: float
    weight_decay: float
    device: str
    verbose: int


def _build_model(*, model_kind: str, in_dim: int, hid_dim: int, num_classes: int, num_layers: int, dropout: float, activation: str):
    common_kwargs = {
        "in_dim": int(in_dim),
        "hid_dim": int(hid_dim),
        "num_classes": int(num_classes),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "activation": str(activation),
    }
    if model_kind == "gnn":
        return LayerNormSnapshotGNN(**common_kwargs)
    if model_kind == "mlp":
        return LayerNormSnapshotMLP(**common_kwargs)
    raise ValueError(f"Unknown model_kind '{model_kind}'.")


def _evaluate_epoch(
    model: nn.Module,
    source_data,
    target_data,
    *,
    model_key: str,
    model_label: str,
    model_kind: str,
    config: ExperimentConfig,
    seed: int,
    metric: BaseMetric,
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
            sampling_num=config.mmd_sampling_num,
            times=config.mmd_times,
        ),
    }
    row["dirichlet_energy_gap"] = (
        float(row["target_dirichlet_energy"]) - float(row["source_dirichlet_energy"])
    )

    snapshot_rows = []
    for snapshot_order, (snapshot_name, snapshot_title) in enumerate(
        _build_snapshot_specs(model_kind, config.num_layers)
    ):
        source_snapshot = source_snapshots[snapshot_name]
        target_snapshot = target_snapshots[snapshot_name]
        snapshot_rows.append(
            {
                "model_key": model_key,
                "model_label": model_label,
                "seed": int(seed),
                "snapshot": snapshot_name,
                "snapshot_title": snapshot_title,
                "snapshot_order": int(snapshot_order),
                "mmd": _safe_mmd(
                    source_snapshot,
                    target_snapshot,
                    sampling_num=config.mmd_sampling_num,
                    times=config.mmd_times,
                ),
                "centroid_gap": _centroid_gap(source_snapshot, target_snapshot),
                "source_mean_norm": _feature_mean_norm(source_snapshot),
                "target_mean_norm": _feature_mean_norm(target_snapshot),
            }
        )

    return row, snapshot_rows


def _train_single_model(
    *,
    model_key: str,
    model_label: str,
    model_kind: str,
    graphs: dict[str, object],
    config: ExperimentConfig,
    seed: int,
):
    set_seed(seed)
    source_data = graphs[config.source]
    target_data = graphs[config.target]

    model = _build_model(
        model_kind=model_kind,
        in_dim=int(source_data.x.size(1)),
        hid_dim=int(config.hid_dim),
        num_classes=int(torch.unique(torch.cat([source_data.y, target_data.y])).numel()),
        num_layers=int(config.num_layers),
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
    history_rows = []
    snapshot_rows = []

    for epoch in range(config.epochs):
        model.train()

        source_logits = model(
            source_data.x,
            source_data.edge_index,
            edge_weight=getattr(source_data, "edge_weight", None),
        )
        if int(source_train_mask.sum().item()) <= 0:
            loss = source_logits.sum() * 0.0
        else:
            loss = F.nll_loss(source_logits[source_train_mask], source_data.y[source_train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        diagnostics, epoch_snapshots = _evaluate_epoch(
            model,
            source_data,
            target_data,
            model_key=model_key,
            model_label=model_label,
            model_kind=model_kind,
            config=config,
            seed=seed,
            metric=metric,
        )
        history_rows.append(
            {
                "model_key": model_key,
                "model_label": model_label,
                "seed": int(seed),
                "epoch": int(epoch + 1),
                "loss": float(loss.detach().cpu().item()),
                **diagnostics,
            }
        )
        for snapshot_row in epoch_snapshots:
            snapshot_rows.append(
                {
                    **snapshot_row,
                    "epoch": int(epoch + 1),
                }
            )

        if config.verbose >= 2:
            print(
                f"[{model_label}] epoch={epoch+1:03d} "
                f"src={diagnostics['source_micro_f1']:.4f} "
                f"tgt={diagnostics['target_micro_f1']:.4f} "
                f"mmd_final={diagnostics['mmd_final']:.4f}"
            )

    return pd.DataFrame(history_rows), pd.DataFrame(snapshot_rows)


def _aggregate_histories(history_df: pd.DataFrame):
    group_cols = ["model_key", "model_label", "epoch"]
    metric_cols = [col for col in history_df.columns if col not in group_cols + ["seed"]]

    mean_df = history_df.groupby(group_cols, as_index=False, sort=False)[metric_cols].mean()
    std_df = history_df.groupby(group_cols, as_index=False, sort=False)[metric_cols].std().fillna(0.0)
    return mean_df, std_df


def _aggregate_snapshot_histories(snapshot_df: pd.DataFrame):
    group_cols = ["model_key", "model_label", "snapshot", "snapshot_title", "snapshot_order", "epoch"]
    metric_cols = [col for col in snapshot_df.columns if col not in group_cols + ["seed"]]

    mean_df = snapshot_df.groupby(group_cols, as_index=False, sort=False)[metric_cols].mean()
    std_df = snapshot_df.groupby(group_cols, as_index=False, sort=False)[metric_cols].std().fillna(0.0)
    return mean_df, std_df


def _build_summary_text(
    *,
    dataset: str,
    source: str,
    target: str,
    mean_df: pd.DataFrame,
    std_df: pd.DataFrame,
    snapshot_mean_df: pd.DataFrame,
    snapshot_std_df: pd.DataFrame,
    config: ExperimentConfig,
):
    lines = [
        "LayerNorm Embedding Tracking",
        f"Dataset: {dataset}",
        f"Transfer: {source} -> {target}",
        f"Seeds: {list(config.seeds)}",
        f"Epochs: {config.epochs}",
        f"Depth: {config.num_layers} hidden blocks",
        f"Hidden dim: {config.hid_dim}",
        f"Activation: {config.activation}",
        "",
        "Setup:",
        "- Both models use LayerNorm after every linear layer.",
        "- The GNN applies message passing after each normalized/activated block.",
        "- The MLP baseline matches the depth and width but omits message passing entirely.",
        "- No explicit alignment loss is used; MMD is tracked only as a diagnostic.",
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
            f"final MMD={float(row_mean['mmd_final']):.4f} +/- {float(row_std.get('mmd_final', 0.0)):.4f}"
        )

        lines.append(f"  {model_label} final layerwise MMD:")
        model_snapshot_mean = snapshot_mean_df[
            (snapshot_mean_df["model_key"] == model_key) & (snapshot_mean_df["epoch"] == final_epoch)
        ].sort_values("snapshot_order", kind="stable")
        model_snapshot_std = snapshot_std_df[
            (snapshot_std_df["model_key"] == model_key) & (snapshot_std_df["epoch"] == final_epoch)
        ].sort_values("snapshot_order", kind="stable")
        std_map = {
            str(row.snapshot): float(row.mmd)
            for row in model_snapshot_std.itertuples(index=False)
        }
        for row in model_snapshot_mean.itertuples(index=False):
            lines.append(
                f"    - {row.snapshot_title}: {float(row.mmd):.4f} +/- {std_map.get(str(row.snapshot), 0.0):.4f}"
            )

    lines.append("")
    lines.append("Snapshot MMDs are computed on full-graph source/target representations in eval mode.")
    lines.append("Dirichlet energy is measured on the final embedding and normalized by node count.")
    return "\n".join(lines) + "\n"


def run_experiment(
    *,
    dataset: str,
    source: str,
    target: str,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    epochs: int = 300,
    num_layers: int = 3,
    hid_dim: int = 64,
    dropout_ratio: float = 0.1,
    activation: str = "relu",
    mmd_sampling_num: int = 256,
    mmd_times: int = 2,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    device: str = "cpu",
    verbose: int = 0,
    out_root: str = "./__saved__/analysis/layernorm_embedding_tracking",
    run_root: str | Path | None = None,
):
    config = ExperimentConfig(
        dataset=dataset,
        source=source,
        target=target,
        seeds=tuple(int(seed) for seed in seeds),
        epochs=int(epochs),
        num_layers=int(num_layers),
        hid_dim=int(hid_dim),
        dropout_ratio=float(dropout_ratio),
        activation=str(activation),
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
            run_name=(
                f"layernorm_embedding_tracking_{dataset}_{source}_to_{target}_"
                f"layers{config.num_layers}_seeds{config.seeds[0]}to{config.seeds[-1]}_{time.strftime('%m%d_%H%M%S')}"
            ),
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
    snapshot_frames = []
    start_time = time.time()
    for seed in config.seeds:
        if config.verbose >= 1:
            print(f"[seed {seed}] {dataset} {source}->{target}")
        for spec in MODEL_SPECS:
            history_df, snapshot_df = _train_single_model(
                model_key=spec["key"],
                model_label=spec["label"],
                model_kind=spec["kind"],
                graphs=graphs,
                config=config,
                seed=int(seed),
            )
            history_frames.append(history_df)
            snapshot_frames.append(snapshot_df)
        _cleanup_cuda()

    history_df = pd.concat(history_frames, ignore_index=True)
    snapshot_df = pd.concat(snapshot_frames, ignore_index=True)
    mean_df, std_df = _aggregate_histories(history_df)
    snapshot_mean_df, snapshot_std_df = _aggregate_snapshot_histories(snapshot_df)

    history_df.to_csv(run_dir / "histories_all_seeds.csv", index=False)
    mean_df.to_csv(run_dir / "histories_mean.csv", index=False)
    std_df.to_csv(run_dir / "histories_std.csv", index=False)
    snapshot_df.to_csv(run_dir / "snapshot_histories_all_seeds.csv", index=False)
    snapshot_mean_df.to_csv(run_dir / "snapshot_histories_mean.csv", index=False)
    snapshot_std_df.to_csv(run_dir / "snapshot_histories_std.csv", index=False)

    plot_training_overview(mean_df, out_dir=run_dir, std_df=std_df)
    plot_snapshot_mmd(snapshot_mean_df, out_dir=run_dir, snapshot_std_df=snapshot_std_df)
    plot_final_snapshot_mmd(snapshot_mean_df, out_dir=run_dir, snapshot_std_df=snapshot_std_df)

    summary_text = _build_summary_text(
        dataset=dataset,
        source=source,
        target=target,
        mean_df=mean_df,
        std_df=std_df,
        snapshot_mean_df=snapshot_mean_df,
        snapshot_std_df=snapshot_std_df,
        config=config,
    )
    (run_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    metadata_payload = {
        "dataset": dataset,
        "source": source,
        "target": target,
        "seeds": list(config.seeds),
        "epochs": config.epochs,
        "num_layers": config.num_layers,
        "hid_dim": config.hid_dim,
        "dropout_ratio": config.dropout_ratio,
        "activation": config.activation,
        "mmd_sampling_num": config.mmd_sampling_num,
        "mmd_times": config.mmd_times,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "device": config.device,
        "num_features": metadata["num_features"],
        "num_classes": metadata["num_classes"],
        "train_time_sec": float(time.time() - start_time),
        "models": {
            "layernorm_gnn": {
                "label": "LayerNorm GNN",
                "snapshots": [name for name, _ in _build_snapshot_specs("gnn", config.num_layers)],
            },
            "layernorm_mlp": {
                "label": "LayerNorm MLP",
                "snapshots": [name for name, _ in _build_snapshot_specs("mlp", config.num_layers)],
            },
        },
        "artifacts": {
            "histories_all_seeds": "histories_all_seeds.csv",
            "histories_mean": "histories_mean.csv",
            "histories_std": "histories_std.csv",
            "snapshot_histories_all_seeds": "snapshot_histories_all_seeds.csv",
            "snapshot_histories_mean": "snapshot_histories_mean.csv",
            "snapshot_histories_std": "snapshot_histories_std.csv",
            "training_overview": "training_overview.png",
            "snapshot_mmd_over_epochs": "snapshot_mmd_over_epochs.png",
            "final_snapshot_mmd": "final_snapshot_mmd.png",
            "summary": "summary.txt",
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    print(f"[saved] {run_dir / 'training_overview.png'}")
    print(f"[saved] {run_dir / 'snapshot_mmd_over_epochs.png'}")
    print(f"[saved] {run_dir / 'final_snapshot_mmd.png'}")
    print(f"[saved] {run_dir / 'snapshot_histories_mean.csv'}")
    print(f"[saved] {run_dir / 'summary.txt'}")
    print(f"results_dir={run_dir}")
    return run_dir
