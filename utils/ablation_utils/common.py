from pathlib import Path
import csv

import torch

from data.build_dataset import build_dataset
from utils.config_utils import build_config


def load_pair(dataset: str, source: str, target: str, device: str, seed: int):
    config = build_config(
        {"data": dataset, "expt": "default", "model": "test"},
        {
            "expt": {
                "source": source,
                "target": target,
                "device": device,
                "seed": seed,
                "verbose": 0,
                "wandb_enabled": False,
            }
        },
    )
    source_dataset, target_dataset = build_dataset(config)
    source_data = source_dataset[0].to(device)
    target_data = target_dataset[0].to(device)

    if source_data.edge_index is not None:
        source_data.edge_index = source_data.edge_index.contiguous()
    if target_data.edge_index is not None:
        target_data.edge_index = target_data.edge_index.contiguous()

    if source_data.x is None or target_data.x is None:
        raise ValueError("Missing node features.")
    if source_data.edge_index is None or target_data.edge_index is None:
        raise ValueError("Missing edge_index.")
    if source_data.y is None or target_data.y is None:
        raise ValueError("Missing node labels.")
    if int(source_data.x.size(1)) != int(target_data.x.size(1)):
        raise ValueError(
            f"Feature dim mismatch: source={source_data.x.size(1)}, target={target_data.x.size(1)}"
        )
    source_data.y = source_data.y.view(-1).long()
    target_data.y = target_data.y.view(-1).long()
    return source_data, target_data


def sample_idx(n: int, max_samples: int, device: torch.device) -> torch.Tensor:
    if max_samples <= 0 or n <= max_samples:
        return torch.arange(n, device=device)
    return torch.randperm(n, device=device)[:max_samples]


def as_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def save_table(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError("fieldnames must be provided when rows is empty.")
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_average_rows(all_scenario_rows: list[list[dict]], max_layers: int, metric_keys: list[str]) -> list[dict]:
    avg_rows = []
    if not all_scenario_rows:
        return avg_rows
    for layer in range(max_layers + 1):
        layer_rows = [rows[layer] for rows in all_scenario_rows if len(rows) > layer]
        if not layer_rows:
            continue
        n = float(len(layer_rows))
        row = {"layer": layer, "n_scenarios": int(n)}
        for key in metric_keys:
            row[key] = sum(float(r[key]) for r in layer_rows) / n
        avg_rows.append(row)
    return avg_rows


def project_pair_features( # this is a random projection of the features to a lower dimension, used for MMD computation
    source_x: torch.Tensor,
    target_x: torch.Tensor,
    out_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    in_dim = int(source_x.size(1))
    if out_dim <= 0 or out_dim >= in_dim:
        return source_x, target_x

    g = torch.Generator()
    g.manual_seed(int(seed))
    proj_cpu = torch.randn((in_dim, out_dim), generator=g, dtype=torch.float32)
    proj = proj_cpu.to(device=source_x.device, dtype=source_x.dtype) / (in_dim ** 0.5)
    return source_x @ proj, target_x @ proj
