import json
from pathlib import Path

import torch

from Learn.Clean_SCGDA.utils.ablation_utils.common import save_table
from Learn.Clean_SCGDA.utils.ablation_utils.motivation_plots import plot_average_quad


MMD_SUMMARY_KEYS = [
    "mmd2_total_mean",
    "mmd2_total_std",
    "mmd2_norm_mean",
    "mmd2_norm_std",
    "mmd2_angle_mean",
    "mmd2_angle_std",
]

TRANSFER_SUMMARY_KEYS = [
    "source_micro_f1_mean",
    "source_micro_f1_std",
    "target_micro_f1_mean",
    "target_micro_f1_std",
    "oracle_target_micro_f1_mean",
    "oracle_target_micro_f1_std",
    "delta_micro_f1_mean",
    "delta_micro_f1_std",
    "delta_oracle_micro_f1_mean",
    "delta_oracle_micro_f1_std",
    "majority_source_label_target_f1_mean",
    "majority_source_label_target_f1_std",
]

DISTORTION_SUMMARY_KEYS = [
    "source_distortion_mean",
    "source_distortion_std",
    "target_distortion_mean",
    "target_distortion_std",
]


def to_plain_dict(cfg):
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    return cfg if isinstance(cfg, dict) else dict(cfg)


def save_rows(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> bool:
    if not rows:
        return False
    save_table(path, rows, fieldnames=fieldnames)
    print(f"[saved] {path}")
    return True


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2))
    print(f"[saved] {path}")


def aggregate_by_layer(rows: list[dict], max_layers: int, metric_keys: list[str]) -> list[dict]:
    out = []
    for layer in range(1, int(max_layers) + 1):
        layer_rows = [r for r in rows if int(r["layer"]) == layer]
        if not layer_rows:
            continue
        row = {"layer": float(layer), "n_trials": float(len(layer_rows))}
        for key in metric_keys:
            vals = torch.tensor([float(r[key]) for r in layer_rows], dtype=torch.float32)
            row[f"{key}_mean"] = float(vals.mean().item())
            row[f"{key}_std"] = float(vals.std(unbiased=False).item())
        out.append(row)
    return out


def average_summary_rows(all_summaries: list[list[dict]], max_layers: int, metric_keys: list[str]) -> list[dict]:
    out = []
    for layer in range(1, int(max_layers) + 1):
        layer_rows = [r for rows in all_summaries for r in rows if int(r["layer"]) == layer]
        if not layer_rows:
            continue
        row = {"layer": float(layer), "n_scenarios": float(len(layer_rows))}
        for key in metric_keys:
            vals = torch.tensor([float(r[key]) for r in layer_rows], dtype=torch.float32)
            row[key] = float(vals.mean().item())
        out.append(row)
    return out


def save_case_artifacts(
    out_dir: Path,
    cfg,
    mmd_rows: list[dict],
    mmd_summary: list[dict],
    transfer_rows: list[dict],
    transfer_summary: list[dict],
    distortion_summary: list[dict],
    title: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rows(out_dir / "mmd_decomposition_trials.csv", mmd_rows)
    save_rows(out_dir / "mmd_decomposition_summary.csv", mmd_summary)
    save_rows(out_dir / "transferability_trials.csv", transfer_rows)
    save_rows(out_dir / "transferability_summary.csv", transfer_summary)
    save_rows(out_dir / "distortion_summary.csv", distortion_summary)
    save_json(out_dir / "mmd_decomposition_summary.json", mmd_summary)
    save_json(out_dir / "transferability_summary.json", transfer_summary)
    save_json(out_dir / "distortion_summary.json", distortion_summary)
    save_json(out_dir / "mmd_decomposition_config.json", to_plain_dict(cfg))
    quad_path = out_dir / "mmd_transfer_delta_distortion_quad.png"
    if plot_average_quad(quad_path, mmd_summary, transfer_summary, distortion_summary, title):
        print(f"[saved] {quad_path}")


def save_average_bundle(
    out_dir: Path,
    prefix: str,
    mmd_rows: list[dict],
    transfer_rows: list[dict],
    distortion_rows: list[dict],
    title: str,
) -> None:
    save_rows(out_dir / f"{prefix}_mmd_layers.csv", mmd_rows)
    save_rows(out_dir / f"{prefix}_transferability_layers.csv", transfer_rows)
    save_rows(out_dir / f"{prefix}_distortion_layers.csv", distortion_rows)
    quad_path = out_dir / f"{prefix}_mmd_transfer_delta_distortion_quad.png"
    if plot_average_quad(quad_path, mmd_rows, transfer_rows, distortion_rows, title):
        print(f"[saved] {quad_path}")
