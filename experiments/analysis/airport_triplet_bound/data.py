from __future__ import annotations

import copy
import warnings

import torch
from torch_geometric.transforms import OneHotDegree

from data.build_dataset import get_dataset, get_max_degree

from .config import ExperimentConfig, TripletSpec, dataset_config


warnings.filterwarnings(
    "ignore",
    message="The `pre_transform` argument differs from the one used in the pre-processed version of this dataset.*",
    category=UserWarning,
)


def _domain_seed(base_seed: int, domain_idx: int) -> int:
    return int(base_seed + 10007 * domain_idx)


def _clone_graph(data, device: str):
    graph = copy.deepcopy(data)
    if graph.edge_index is not None:
        graph.edge_index = graph.edge_index.contiguous()
    return graph.to(device)


def _resample_masks(data, *, train_split: float, val_split: float, seed: int):
    num_nodes = int(data.num_nodes)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(num_nodes, generator=generator)

    train_size = int(num_nodes * train_split)
    val_size = int(num_nodes * val_split)
    if train_size + val_size > num_nodes:
        val_size = max(0, num_nodes - train_size)

    train_idx = perm[:train_size]
    val_idx = perm[train_size : train_size + val_size]
    test_idx = perm[train_size + val_size :]

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
    return data


def _dataset_domains(cfg: dict) -> list[str]:
    return list(cfg["data"]["domains"])


def _needs_degree_features(cfg: dict) -> bool:
    return cfg["data"]["name"].lower() == "airport"


def load_dataset_graphs(exp_cfg: ExperimentConfig, seed: int) -> tuple[dict[str, object], dict[str, int]]:
    cfg = dataset_config(exp_cfg.dataset, exp_cfg.train_split, exp_cfg.val_split)
    domains = _dataset_domains(cfg)
    max_degree = 0
    if _needs_degree_features(cfg):
        max_degree = int(get_max_degree(cfg))

    graphs = {}
    for domain_idx, domain in enumerate(domains):
        dataset = get_dataset(domain, cfg)
        if _needs_degree_features(cfg):
            dataset.transform = OneHotDegree(max_degree)

        graph = _clone_graph(dataset[0], exp_cfg.resolved_device())
        if exp_cfg.resample_masks:
            graph = _resample_masks(
                graph,
                train_split=exp_cfg.train_split,
                val_split=exp_cfg.val_split,
                seed=_domain_seed(seed, domain_idx),
            ).to(exp_cfg.resolved_device())
        graphs[domain] = graph

    num_features = int(next(iter(graphs.values())).x.size(1))
    for domain, graph in graphs.items():
        if int(graph.x.size(1)) != num_features:
            raise ValueError(
                f"Feature dimension mismatch for dataset '{exp_cfg.dataset}': "
                f"{domain} has {graph.x.size(1)} features, expected {num_features}."
            )

    metadata = {
        "dataset": cfg["data"]["name"],
        "max_degree": max_degree,
        "num_features": num_features,
        "num_classes": int(torch.unique(torch.cat([graphs[d].y.cpu() for d in domains])).numel()),
    }
    return graphs, metadata


def triplet_graphs(graphs: dict[str, object], triplet: TripletSpec) -> dict[str, object]:
    return {
        "source": graphs[triplet.source],
        "target": graphs[triplet.target],
        "reference": graphs[triplet.reference],
    }
