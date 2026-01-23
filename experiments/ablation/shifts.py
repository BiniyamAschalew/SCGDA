import copy
import os
from typing import Dict

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from data.build_dataset import build_dataset
from models.baselines.gnn.gnn_base import GNNBase
from models.baselines.mlp.mlp_base import MLPBase
from utils.expt_utils import set_seed


_BACKBONE_MAP = {
    "mlp": MLPBase,
    "simmlp": MLPBase,  # supervised training; alignment losses are not applied here
    "gnn": GNNBase,
}


def _build_backbone(config: dict) -> torch.nn.Module:
    model_name = config["model"]["name"].lower()
    if model_name not in _BACKBONE_MAP:
        raise ValueError(
            f"Unsupported model '{model_name}' for shift embeddings. "
            f"Supported: {sorted(_BACKBONE_MAP.keys())}"
        )
    return _BACKBONE_MAP[model_name](config)


def _forward_logits(model: torch.nn.Module, data: Data) -> torch.Tensor:
    edge_weight = getattr(data, "edge_weight", None)
    batch = getattr(data, "batch", None)
    try:
        return model(data.x, data.edge_index, edge_weight, batch)
    except TypeError:
        return model(data.x, data.edge_index)


def _feat_bottleneck(model: torch.nn.Module, data: Data) -> torch.Tensor:
    edge_weight = getattr(data, "edge_weight", None)
    batch = getattr(data, "batch", None)
    try:
        return model.feat_bottleneck(data.x, data.edge_index, edge_weight, batch)
    except TypeError:
        return model.feat_bottleneck(data.x)


def _train_supervised(
    model: torch.nn.Module,
    data: Data,
    train_mask: torch.Tensor,
    config: dict,
) -> torch.nn.Module:
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["model"]["lr"],
        weight_decay=config["model"]["weight_decay"],
    )

    epochs = config["expt"]["epochs"]
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = _forward_logits(model, data)
        loss = F.nll_loss(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

    return model


def _subgraph_from_mask(data: Data, mask: torch.Tensor) -> Data:
    subset = mask.nonzero(as_tuple=False).view(-1)
    edge_weight = getattr(data, "edge_weight", None)
    edge_index, edge_weight = subgraph(
        subset,
        data.edge_index,
        edge_weight,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )
    sub = Data(
        x=data.x[subset],
        y=data.y[subset],
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=subset.numel(),
    )
    return sub


def _concat_graphs(source_data: Data, target_data: Data) -> Data:
    offset = source_data.num_nodes
    x = torch.cat([source_data.x, target_data.x], dim=0)
    y = torch.cat([source_data.y, target_data.y], dim=0)
    edge_index = torch.cat(
        [source_data.edge_index, target_data.edge_index + offset], dim=1
    )

    source_w = getattr(source_data, "edge_weight", None)
    target_w = getattr(target_data, "edge_weight", None)
    edge_weight = None
    if source_w is not None or target_w is not None:
        if source_w is None:
            source_w = torch.ones(
                source_data.edge_index.size(1), device=target_data.edge_index.device
            )
        if target_w is None:
            target_w = torch.ones(
                target_data.edge_index.size(1), device=source_data.edge_index.device
            )
        edge_weight = torch.cat([source_w, target_w], dim=0)

    combined = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=x.size(0),
    )
    return combined


def _save_embeddings(embeddings: Dict[str, torch.Tensor], base_dir: str) -> None:
    os.makedirs(base_dir, exist_ok=True)
    for split_name, tensor in embeddings.items():
        path = os.path.join(base_dir, f"{split_name}.pt")
        torch.save(tensor.detach().cpu(), path)


def _load_embeddings(base_dir: str) -> Dict[str, torch.Tensor]:
    splits = ["source_train", "source_val", "target_train", "target_val"]
    out = {}
    for split_name in splits:
        path = os.path.join(base_dir, f"{split_name}.pt")
        out[split_name] = torch.load(path, map_location="cpu")
    return out


def _embedding_dir(config: dict, save_root: str) -> str:
    model_name = config["model"]["name"].lower()
    source = config["expt"]["source"]
    target = config["expt"]["target"]
    return os.path.join(save_root, model_name, f"{source}_{target}")


def train_and_save_shift_embeddings(config: dict, save_root: str = "__saved__/embeddings") -> str:
    """Train source-only and oracle models, then save split embeddings."""
    set_seed(config["expt"]["seed"])

    source_dataset, target_dataset = build_dataset(config)
    source_full = source_dataset[0]
    target_full = target_dataset[0]

    config["model"]["in_dim"] = source_full.x.shape[1]
    config["model"]["num_classes"] = len(source_full.y.unique())

    device = config["expt"]["device"]
    source_full = source_full.to(device)
    target_full = target_full.to(device)

    source_train = _subgraph_from_mask(source_full, source_full.train_mask)
    target_train = _subgraph_from_mask(target_full, target_full.train_mask)

    source_train = source_train.to(device)
    target_train = target_train.to(device)

    model = _build_backbone(config).to(device)
    oracle_model = copy.deepcopy(model).to(device)

    train_mask = torch.ones(source_train.num_nodes, dtype=torch.bool, device=device)
    _train_supervised(model, source_train, train_mask, config)

    oracle_train = _concat_graphs(source_train, target_train).to(device)
    oracle_mask = torch.ones(oracle_train.num_nodes, dtype=torch.bool, device=device)
    _train_supervised(oracle_model, oracle_train, oracle_mask, config)

    model.eval()
    oracle_model.eval()
    with torch.no_grad():
        source_feats = _feat_bottleneck(model, source_full)
        target_feats = _feat_bottleneck(model, target_full)
        oracle_source_feats = _feat_bottleneck(oracle_model, source_full)
        oracle_target_feats = _feat_bottleneck(oracle_model, target_full)

    base_dir = _embedding_dir(config, save_root)
    _save_embeddings(
        {
            "source_train": source_feats[source_full.train_mask],
            "source_val": source_feats[source_full.val_mask],
            "target_train": target_feats[target_full.train_mask],
            "target_val": target_feats[target_full.val_mask],
        },
        os.path.join(base_dir, "original"),
    )
    _save_embeddings(
        {
            "source_train": oracle_source_feats[source_full.train_mask],
            "source_val": oracle_source_feats[source_full.val_mask],
            "target_train": oracle_target_feats[target_full.train_mask],
            "target_val": oracle_target_feats[target_full.val_mask],
        },
        os.path.join(base_dir, "oracle"),
    )

    return base_dir


def load_shift_embeddings(config: dict, save_root: str = "__saved__/embeddings") -> Dict[str, Dict[str, torch.Tensor]]:
    """Load embeddings saved by train_and_save_shift_embeddings."""
    base_dir = _embedding_dir(config, save_root)
    return {
        "original": _load_embeddings(os.path.join(base_dir, "original")),
        "oracle": _load_embeddings(os.path.join(base_dir, "oracle")),
    }
