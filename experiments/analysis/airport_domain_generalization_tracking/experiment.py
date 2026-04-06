from __future__ import annotations

import gc
import json
import numpy as np
from pathlib import Path
import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.transforms import OneHotDegree

from Learn.Clean_SCGDA.data.build_dataset import get_dataset, get_max_degree
from Learn.Clean_SCGDA.experiments.analysis.airport_domain_generalization_tracking.plotting import (
    plot_target_quality_vs_performance,
    plot_training_overview,
)
from Learn.Clean_SCGDA.experiments.analysis.plain_gnn_cross_domain.log_plot import make_output_dir
from Learn.Clean_SCGDA.models.build_model import build_model
from Learn.Clean_SCGDA.utils.config_utils import build_config, load_config
from Learn.Clean_SCGDA.utils.expt_utils import set_seed
from Learn.Clean_SCGDA.utils.train_utils.mmd import MMD


DEFAULT_NUM_LAYERS = 2
DEFAULT_HID_DIM = 64
DEFAULT_DROPOUT = 0.1


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _build_experiment_config(
    *,
    dataset: str,
    source: str,
    seed: int,
    device: str,
    epochs: int,
    verbose: int,
    num_layers: int,
    hid_dim: int,
    dropout_ratio: float,
):
    config_setup = {
        "model": "source_only_gnn",
        "data": dataset,
        "expt": "default",
    }
    update_config = {
        "expt": {
            "source": source,
            "target": source,
            "device": device,
            "seed": seed,
            "wandb_enabled": False,
            "project": "airport_domain_generalization_tracking",
            "verbose": verbose,
            "metrics": ["micro_f1"],
        },
        "model": {
            "epochs": epochs,
            "num_layers": num_layers,
            "hid_dim": hid_dim,
            "dropout_ratio": float(dropout_ratio),
            "use_mask": True,
        },
    }
    config = build_config(config_setup, update_config=update_config, use_tuned=0)
    config["model"]["epochs"] = epochs
    config["model"]["num_layers"] = num_layers
    config["model"]["hid_dim"] = hid_dim
    config["model"]["dropout_ratio"] = float(dropout_ratio)
    config["model"]["use_mask"] = True
    config["expt"]["metrics"] = ["micro_f1"]
    config["expt"]["epochs"] = epochs
    return config


def _load_dataset_graphs(dataset_name: str, config: dict, *, device: str):
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
        "model": {"name": config["model"]["name"]},
        "expt": {"verbose": 0},
    }

    graphs = {}
    max_degree = 0
    if dataset_config["data"]["name"].lower() == "airport":
        max_degree = int(get_max_degree(dataset_config))

    for domain in dataset_config["data"]["domains"]:
        dataset = get_dataset(domain, dataset_config)
        if dataset_config["data"]["name"].lower() == "airport":
            dataset.transform = OneHotDegree(max_degree)
        data = dataset[0].to(device)
        if data.edge_index is not None:
            data.edge_index = data.edge_index.contiguous()
        graphs[domain] = data

    num_features = int(next(iter(graphs.values())).x.size(1))
    num_classes = int(torch.unique(torch.cat([graph.y for graph in graphs.values()])).numel())
    return graphs, {
        "domains": list(dataset_config["data"]["domains"]),
        "num_features": num_features,
        "num_classes": num_classes,
        "max_degree": max_degree,
    }


def _full_graph_micro_f1(model, logits, labels) -> float:
    return float(model.metrics(logits, labels).get("micro_f1", 0.0))


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


def _adjacency_propagation(data, features: torch.Tensor) -> torch.Tensor:
    if data.edge_index.numel() == 0:
        return features

    edge_index = data.edge_index
    edge_weight = getattr(data, "edge_weight", None)
    if edge_weight is None:
        edge_weight = torch.ones(
            edge_index.size(1),
            device=features.device,
            dtype=features.dtype,
        )
    else:
        edge_weight = edge_weight.to(device=features.device, dtype=features.dtype)

    adj = torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        size=(features.size(0), features.size(0)),
        device=features.device,
        dtype=features.dtype,
    ).coalesce()
    return torch.sparse.mm(adj, features)


def _consistency_norm(data, features: torch.Tensor) -> float:
    diff = _adjacency_propagation(data, features) - features
    return float(torch.linalg.norm(diff, ord="fro").item() / max(1, features.size(0)))


def _domain_forward(backbone, data):
    backbone.eval()
    with torch.no_grad():
        features = backbone.feat_bottleneck(data.x, data.edge_index)
        logits = backbone.feat_classifier(features, data.edge_index)
        logits = F.log_softmax(logits, dim=1)
    return features, logits


def _self_split_mmd(features: torch.Tensor, *, seed: int) -> float:
    num_nodes = int(features.size(0))
    if num_nodes < 2:
        return 0.0

    rng = np.random.default_rng(seed)
    perm = torch.as_tensor(rng.permutation(num_nodes), device=features.device, dtype=torch.long)
    split = max(1, num_nodes // 2)
    left_idx = perm[:split]
    right_idx = perm[split:]
    if int(right_idx.numel()) == 0:
        return 0.0
    return float(MMD(features[left_idx], features[right_idx]).detach().cpu().item())


def _epoch_domain_metrics(
    model,
    graphs: dict[str, object],
    *,
    source_domain: str,
    eval_domains: list[str],
    epoch_index: int,
    seed: int,
):
    backbone = model.gnn
    features_by_domain = {}
    row = {}

    for domain in eval_domains:
        data = graphs[domain]
        features, logits = _domain_forward(backbone, data)
        features_by_domain[domain] = features
        row[f"micro_f1_{domain}"] = _full_graph_micro_f1(model, logits, data.y)
        row[f"dirichlet_energy_{domain}"] = _dirichlet_energy(data, features)
        row[f"consistency_norm_{domain}"] = _consistency_norm(data, features)

    source_features = features_by_domain[source_domain]
    for domain in eval_domains:
        if domain == source_domain:
            row[f"mmd_{source_domain}_to_{domain}"] = _self_split_mmd(
                source_features,
                seed=int(seed) + int(epoch_index) * 9973 + 17,
            )
        else:
            row[f"mmd_{source_domain}_to_{domain}"] = float(
                MMD(source_features, features_by_domain[domain]).detach().cpu().item()
            )

    return row


def _train_source_and_track(
    model,
    source_data,
    graphs: dict[str, object],
    *,
    source_domain: str,
    eval_domains: list[str],
    use_mask: bool,
    seed: int,
):
    source_loader = model.get_loader(source_data)
    model.gnn = model.init_model(**model.kwargs)

    optimizer = torch.optim.Adam(
        model.gnn.parameters(),
        lr=model.lr,
        weight_decay=model.weight_decay,
    )

    rows = []
    start_time = time.time()

    from tqdm import tqdm

    for epoch in tqdm(range(model.epoch), desc=f"Training {source_domain}"):
        epoch_loss = 0.0
        epoch_source_logits = None
        epoch_source_labels = None

        for sampled_source_data in source_loader:
            sampled_source_data = sampled_source_data.to(model.device)
            model.gnn.train()
            loss, source_logits = model.forward_model(
                sampled_source_data,
                use_mask=use_mask,
                oracle=False,
            )
            epoch_loss += float(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_source_logits, train_source_labels, train_source_mask = model.mask_logits_and_labels(
                source_logits,
                sampled_source_data,
                use_mask=use_mask,
                mask_name="train_mask",
            )
            if int(train_source_mask.sum().item()) > 0:
                if epoch_source_logits is None:
                    epoch_source_logits = train_source_logits
                    epoch_source_labels = train_source_labels
                else:
                    epoch_source_logits = torch.cat((epoch_source_logits, train_source_logits))
                    epoch_source_labels = torch.cat((epoch_source_labels, train_source_labels))

        if epoch_source_logits is None or epoch_source_labels is None:
            raise RuntimeError("No source training nodes were available for loss/metric computation.")

        epoch_row = {
            "epoch": int(epoch + 1),
            "loss": float(epoch_loss),
            "train_micro_f1_source": float(model.metrics(epoch_source_logits, epoch_source_labels)["micro_f1"]),
        }
        epoch_row.update(
            _epoch_domain_metrics(
                model,
                graphs,
                source_domain=source_domain,
                eval_domains=eval_domains,
                epoch_index=epoch,
                seed=seed,
            )
        )
        rows.append(epoch_row)

    model.train_time = time.time() - start_time
    model.finish()
    return pd.DataFrame(rows)


def _add_derived_metrics(history_df: pd.DataFrame, *, eval_domains: list[str]) -> pd.DataFrame:
    out = history_df.copy()
    for domain in eval_domains:
        out[f"graph_signal_denoising_{domain}"] = (
            out[f"consistency_norm_{domain}"] + out[f"dirichlet_energy_{domain}"]
        )
        out[f"dirichlet_energy_delta_{domain}"] = (
            out[f"dirichlet_energy_{domain}"].diff().fillna(0.0)
        )
    return out


def _aggregate_histories(history_frames: list[pd.DataFrame]):
    merged = pd.concat(history_frames, ignore_index=True)
    numeric_cols = [
        col for col in merged.columns
        if col not in {"seed"} and pd.api.types.is_numeric_dtype(merged[col])
    ]
    mean_df = merged.groupby("epoch", as_index=False, sort=True)[numeric_cols].mean()
    std_df = merged.groupby("epoch", as_index=False, sort=True)[numeric_cols].std(ddof=0).fillna(0.0)
    return merged, mean_df, std_df


def _slot_mapping(source_domain: str, eval_domains: list[str]):
    target_domains = [domain for domain in eval_domains if domain != source_domain]
    domain_to_slot = {source_domain: "source"}
    slot_order = ["source"]
    for idx, domain in enumerate(target_domains, start=1):
        slot_name = f"target{idx}"
        domain_to_slot[domain] = slot_name
        slot_order.append(slot_name)
    return domain_to_slot, slot_order


def _map_case_history_to_slots(
    history_df: pd.DataFrame,
    *,
    case_name: str,
    source_domain: str,
    eval_domains: list[str],
):
    domain_to_slot, slot_order = _slot_mapping(source_domain, eval_domains)
    out = pd.DataFrame({"epoch": history_df["epoch"]})
    if "seed" in history_df.columns:
        out["seed"] = history_df["seed"]
    out["case"] = case_name
    out["source_domain"] = source_domain

    for scalar_col in ["loss", "train_micro_f1_source"]:
        if scalar_col in history_df.columns:
            out[scalar_col] = history_df[scalar_col]

    metric_prefixes = [
        "micro_f1",
        "dirichlet_energy",
        "dirichlet_energy_delta",
        "consistency_norm",
        "graph_signal_denoising",
    ]
    for domain, slot_name in domain_to_slot.items():
        for prefix in metric_prefixes:
            src_col = f"{prefix}_{domain}"
            if src_col in history_df.columns:
                out[f"{prefix}_{slot_name}"] = history_df[src_col]
        src_mmd_col = f"mmd_{source_domain}_to_{domain}"
        if src_mmd_col in history_df.columns:
            out[f"mmd_source_to_{slot_name}"] = history_df[src_mmd_col]

    return out, slot_order


def _safe_corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x.std(ddof=0), 0.0) or np.allclose(y.std(ddof=0), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _zscore(values: pd.Series) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    std = arr.std(ddof=0)
    if np.allclose(std, 0.0):
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def _build_target_indicator_table(dataset_all_df: pd.DataFrame, *, target_slots: list[str]):
    metric_names = [
        "mmd",
        "energy_t",
        "energy_delta_t",
        "consistency_t",
        "denoise_t",
        "energy_ratio",
        "consistency_ratio",
        "denoise_ratio",
    ]
    metric_labels = {
        "mmd": "MMD To Source",
        "energy_t": "Target Dirichlet Energy",
        "energy_delta_t": "Target Dirichlet Energy Delta",
        "consistency_t": "Target Consistency",
        "denoise_t": "Target Graph Signal Denoising",
        "energy_ratio": "Energy Ratio Target/Source",
        "consistency_ratio": "Consistency Ratio Target/Source",
        "denoise_ratio": "Denoising Ratio Target/Source",
    }

    target_rows = []
    for slot_name in target_slots:
        required = [
            f"micro_f1_{slot_name}",
            f"mmd_source_to_{slot_name}",
            f"dirichlet_energy_{slot_name}",
            f"dirichlet_energy_delta_{slot_name}",
            f"consistency_norm_{slot_name}",
            f"graph_signal_denoising_{slot_name}",
            "dirichlet_energy_source",
            "consistency_norm_source",
            "graph_signal_denoising_source",
        ]
        if any(col not in dataset_all_df.columns for col in required):
            continue

        slot_df = dataset_all_df[
            [
                "case",
                "seed",
                "epoch",
                f"micro_f1_{slot_name}",
                f"mmd_source_to_{slot_name}",
                f"dirichlet_energy_{slot_name}",
                f"dirichlet_energy_delta_{slot_name}",
                f"consistency_norm_{slot_name}",
                f"graph_signal_denoising_{slot_name}",
                "dirichlet_energy_source",
                "consistency_norm_source",
                "graph_signal_denoising_source",
            ]
        ].copy()
        slot_df["target_slot"] = slot_name
        slot_df["target_f1"] = slot_df[f"micro_f1_{slot_name}"]
        slot_df["mmd"] = slot_df[f"mmd_source_to_{slot_name}"]
        slot_df["energy_t"] = slot_df[f"dirichlet_energy_{slot_name}"]
        slot_df["energy_delta_t"] = slot_df[f"dirichlet_energy_delta_{slot_name}"]
        slot_df["consistency_t"] = slot_df[f"consistency_norm_{slot_name}"]
        slot_df["denoise_t"] = slot_df[f"graph_signal_denoising_{slot_name}"]
        slot_df["energy_ratio"] = slot_df["energy_t"] / (slot_df["dirichlet_energy_source"] + 1e-8)
        slot_df["consistency_ratio"] = slot_df["consistency_t"] / (slot_df["consistency_norm_source"] + 1e-8)
        slot_df["denoise_ratio"] = slot_df["denoise_t"] / (slot_df["graph_signal_denoising_source"] + 1e-8)
        target_rows.append(slot_df)

    if not target_rows:
        return pd.DataFrame(columns=[
            "property",
            "label",
            "corr_all",
            "corr_post_peak",
            "corr_future_drop_10",
            "corr_future_drop_20",
            "indicator_score",
            "rank",
        ])

    pooled_df = pd.concat(target_rows, ignore_index=True)
    z_parts = []
    for (_, _, _), traj_df in pooled_df.groupby(["case", "seed", "target_slot"], sort=False):
        traj_df = traj_df.sort_values("epoch").copy()
        traj_df["target_f1_z"] = _zscore(traj_df["target_f1"])
        for metric_name in metric_names:
            traj_df[f"{metric_name}_z"] = _zscore(traj_df[metric_name])

        peak_pos = int(traj_df["target_f1"].to_numpy().argmax())
        traj_df["post_peak"] = False
        traj_df.iloc[peak_pos:, traj_df.columns.get_loc("post_peak")] = True
        traj_df["future_drop_10"] = traj_df["target_f1"].shift(-10) - traj_df["target_f1"]
        traj_df["future_drop_20"] = traj_df["target_f1"].shift(-20) - traj_df["target_f1"]
        z_parts.append(traj_df)

    pooled_z_df = pd.concat(z_parts, ignore_index=True)
    table_rows = []
    for metric_name in metric_names:
        metric_z = pooled_z_df[f"{metric_name}_z"]
        table_rows.append(
            {
                "property": metric_name,
                "label": metric_labels[metric_name],
                "corr_all": _safe_corr(metric_z, pooled_z_df["target_f1_z"]),
                "corr_post_peak": _safe_corr(
                    metric_z[pooled_z_df["post_peak"]],
                    pooled_z_df.loc[pooled_z_df["post_peak"], "target_f1_z"],
                ),
                "corr_future_drop_10": _safe_corr(
                    metric_z[pooled_z_df["future_drop_10"].notna()],
                    pooled_z_df.loc[pooled_z_df["future_drop_10"].notna(), "future_drop_10"],
                ),
                "corr_future_drop_20": _safe_corr(
                    metric_z[pooled_z_df["future_drop_20"].notna()],
                    pooled_z_df.loc[pooled_z_df["future_drop_20"].notna(), "future_drop_20"],
                ),
            }
        )

    table_df = pd.DataFrame(table_rows)
    table_df["indicator_score"] = table_df[
        ["corr_all", "corr_post_peak", "corr_future_drop_10", "corr_future_drop_20"]
    ].abs().mean(axis=1)
    table_df = table_df.sort_values("indicator_score", ascending=False, na_position="last").reset_index(drop=True)
    table_df["rank"] = np.arange(1, len(table_df) + 1)
    return table_df


def _write_dataset_rollup_summary(
    out_dir: Path,
    *,
    dataset_name: str,
    case_dirs: list[Path],
    mean_df: pd.DataFrame,
    std_df: pd.DataFrame,
    slot_order: list[str],
    predictor_df: pd.DataFrame,
):
    final_row = mean_df.iloc[-1]
    final_std_row = std_df.iloc[-1]

    def _fmt(metric_name: str):
        if metric_name not in final_row:
            return "n/a"
        return f"{float(final_row[metric_name]):.4f} +/- {float(final_std_row.get(metric_name, 0.0)):.4f}"

    lines = [
        "Dataset-Level Domain Generalization Summary",
        "",
        f"Dataset: {dataset_name}",
        f"Scenarios averaged: {len(case_dirs)}",
        f"Scenario directories: {', '.join(case_dir.name for case_dir in case_dirs)}",
        f"Role slots: {', '.join(slot_order)}",
        "",
        "Averaging protocol:",
        "- Each source scenario contributes its seed-level histories after remapping actual domains into shared slots: source, target1, target2, ...",
        "- The dataset-level mean/std files average jointly over all seeds and all transfer scenarios in this dataset.",
        "- target_indicator_ranking.csv scores each property by pooled absolute correlation with target performance over all epochs, post-peak epochs, and 10/20-epoch future drops.",
        "",
        "Top target-performance indicators:",
    ]

    valid_predictors = predictor_df[predictor_df["indicator_score"].notna()]
    if valid_predictors.empty:
        lines.append("- Not enough epoch variation was available to compute stable target-indicator rankings.")
    else:
        for _, row in valid_predictors.head(5).iterrows():
            lines.append(
                f"- rank {int(row['rank'])}: {row['label']} "
                f"(score={float(row['indicator_score']):.4f}, "
                f"corr_all={float(row['corr_all']):+.4f}, "
                f"corr_post_peak={float(row['corr_post_peak']):+.4f}, "
                f"corr_future_drop_10={float(row['corr_future_drop_10']):+.4f})"
            )

    lines.extend(
        [
            "",
            "Final epoch snapshot:",
            f"- epoch: {int(final_row['epoch'])}",
            f"- loss: {_fmt('loss')}",
        ]
    )
    for slot_name in slot_order:
        lines.append(f"- micro_f1_{slot_name}: {_fmt(f'micro_f1_{slot_name}')}")
    for slot_name in slot_order:
        if f"dirichlet_energy_{slot_name}" in mean_df.columns:
            lines.append(f"- dirichlet_energy_{slot_name}: {_fmt(f'dirichlet_energy_{slot_name}')}")
    for slot_name in slot_order:
        if f"dirichlet_energy_delta_{slot_name}" in mean_df.columns:
            lines.append(f"- dirichlet_energy_delta_{slot_name}: {_fmt(f'dirichlet_energy_delta_{slot_name}')}")
    for slot_name in slot_order:
        if f"consistency_norm_{slot_name}" in mean_df.columns:
            lines.append(f"- consistency_norm_{slot_name}: {_fmt(f'consistency_norm_{slot_name}')}")
    for slot_name in slot_order:
        if f"graph_signal_denoising_{slot_name}" in mean_df.columns:
            lines.append(f"- graph_signal_denoising_{slot_name}: {_fmt(f'graph_signal_denoising_{slot_name}')}")
    for slot_name in slot_order:
        if f"mmd_source_to_{slot_name}" in mean_df.columns:
            lines.append(f"- mmd_source_to_{slot_name}: {_fmt(f'mmd_source_to_{slot_name}')}")

    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def build_dataset_rollup(
    *,
    dataset: str,
    dataset_out_dir: str | Path,
    case_dirs: list[str | Path],
):
    dataset_out_dir = Path(dataset_out_dir)
    case_dirs = [Path(case_dir) for case_dir in case_dirs]
    if not case_dirs:
        raise ValueError("case_dirs must contain at least one scenario directory.")

    slot_frames = []
    slot_orders = []
    metadata_list = []
    for case_dir in case_dirs:
        metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
        history_df = pd.read_csv(case_dir / "training_diagnostics_all_seeds.csv")
        if "seed" not in history_df.columns:
            history_df["seed"] = 0
        slot_df, slot_order = _map_case_history_to_slots(
            history_df,
            case_name=case_dir.name,
            source_domain=metadata["source_domain"],
            eval_domains=list(metadata["eval_domains"]),
        )
        slot_frames.append(slot_df)
        slot_orders.append(slot_order)
        metadata_list.append(metadata)

    all_df, mean_df, std_df = _aggregate_histories(slot_frames)
    target_slots = sorted(
        {slot_name for slot_order in slot_orders for slot_name in slot_order if slot_name != "source"},
        key=lambda slot_name: int(slot_name.replace("target", "")),
    )
    slot_order = ["source"] + target_slots
    predictor_df = _build_target_indicator_table(all_df, target_slots=target_slots)

    all_path = dataset_out_dir / "training_diagnostics_all_scenarios_seeds.csv"
    mean_path = dataset_out_dir / "training_diagnostics_mean.csv"
    std_path = dataset_out_dir / "training_diagnostics_std.csv"
    predictor_path = dataset_out_dir / "target_indicator_ranking.csv"
    all_df.to_csv(all_path, index=False)
    mean_df.to_csv(mean_path, index=False)
    std_df.to_csv(std_path, index=False)
    predictor_df.to_csv(predictor_path, index=False)

    overview_plot = plot_training_overview(
        mean_df,
        out_dir=dataset_out_dir,
        source_domain="source",
        eval_domains=slot_order,
        std_df=std_df,
    )
    target_quality_plot = plot_target_quality_vs_performance(
        mean_df,
        out_dir=dataset_out_dir,
        source_domain="source",
        target_domains=target_slots,
        std_df=std_df,
    )

    metadata_json = {
        "dataset": str(dataset).lower(),
        "num_scenarios": len(case_dirs),
        "scenario_dirs": [case_dir.name for case_dir in case_dirs],
        "slot_order": slot_order,
        "source_domains": [metadata["source_domain"] for metadata in metadata_list],
        "artifacts": {
            "all_history": all_path.name,
            "mean_history": mean_path.name,
            "std_history": std_path.name,
            "target_indicator_ranking": predictor_path.name,
            "summary": "summary.txt",
            "training_overview_plot": Path(overview_plot).name,
            "target_quality_plot": None if target_quality_plot is None else Path(target_quality_plot).name,
        },
    }
    (dataset_out_dir / "metadata.json").write_text(json.dumps(metadata_json, indent=2), encoding="utf-8")
    _write_dataset_rollup_summary(
        dataset_out_dir,
        dataset_name=str(metadata_list[0]["config"]["data"]["name"]),
        case_dirs=case_dirs,
        mean_df=mean_df,
        std_df=std_df,
        slot_order=slot_order,
        predictor_df=predictor_df,
    )

    print(f"[saved] {all_path}")
    print(f"[saved] {mean_path}")
    print(f"[saved] {std_path}")
    print(f"[saved] {predictor_path}")
    print(f"[saved] {overview_plot}")
    if target_quality_plot is not None:
        print(f"[saved] {target_quality_plot}")
    print(f"[saved] {dataset_out_dir / 'summary.txt'}")
    return dataset_out_dir


def _write_summary(
    out_dir: Path,
    *,
    config: dict,
    history_df: pd.DataFrame,
    std_df: pd.DataFrame | None,
    source_domain: str,
    target_domains: list[str],
    eval_domains: list[str],
    seed_list: list[int] | None = None,
):
    final_row = history_df.iloc[-1]
    final_std_row = None if std_df is None else std_df.iloc[-1]

    def _fmt(metric_name: str):
        mean_value = float(final_row[metric_name])
        if final_std_row is None or metric_name not in final_std_row:
            return f"{mean_value:.4f}"
        return f"{mean_value:.4f} +/- {float(final_std_row[metric_name]):.4f}"

    lines = [
        "Domain generalization tracking",
        "",
        f"Dataset: {config['data']['name']}",
        f"Source domain: {source_domain}",
        f"Target domains: {', '.join(target_domains) if target_domains else 'none'}",
        f"Evaluated domains: {', '.join(eval_domains)}",
        f"Device: {config['expt']['device']}",
        f"Seeds: {seed_list if seed_list is not None else [config['expt']['seed']]}",
        f"Epochs: {config['model']['epochs']}",
        f"GNN layers: {config['model']['num_layers']}",
        f"Hidden dim: {config['model']['hid_dim']}",
        "",
        "Training protocol:",
        f"- The model is a source-only {config['model']['num_layers']}-layer GNN trained on one source domain at a time.",
        "- Domain performance columns are full-graph micro-F1 on each domain, not val-mask metrics.",
        "- MMD is computed between the source-domain bottleneck features and each domain's bottleneck features.",
        "- The source-to-source MMD column is estimated by randomly splitting the source-domain features into two halves each epoch.",
        "- Dirichlet energy is computed on the bottleneck features with an edge-difference formulation.",
        "- Consistency is the Frobenius norm of (A H - H) divided by the number of nodes, using the domain graph adjacency A.",
        "- graph_signal_denoising is the sum of consistency_norm and dirichlet_energy on the final embedding.",
        "- dirichlet_energy_delta is the epoch-to-epoch change in Dirichlet energy.",
        "- training_overview.png compares mean performance, MMD, energy, energy delta, consistency, and graph signal denoising over epochs, with shaded one-standard-deviation bands.",
        "- target_quality_vs_performance.png overlays each target-domain mean micro-F1 with mean MMD, energy, energy delta, consistency, and graph signal denoising on twin y-axes, with shaded one-standard-deviation bands.",
        "",
        "Final epoch snapshot:",
        f"- epoch: {int(final_row['epoch'])}",
        f"- loss: {_fmt('loss')}",
    ]

    for domain in eval_domains:
        lines.append(f"- micro_f1_{domain}: {_fmt(f'micro_f1_{domain}')}")
    for domain in eval_domains:
        lines.append(f"- dirichlet_energy_{domain}: {_fmt(f'dirichlet_energy_{domain}')}")
    for domain in eval_domains:
        lines.append(f"- dirichlet_energy_delta_{domain}: {_fmt(f'dirichlet_energy_delta_{domain}')}")
    for domain in eval_domains:
        lines.append(f"- consistency_norm_{domain}: {_fmt(f'consistency_norm_{domain}')}")
    for domain in eval_domains:
        lines.append(f"- graph_signal_denoising_{domain}: {_fmt(f'graph_signal_denoising_{domain}')}")
    for domain in eval_domains:
        lines.append(f"- mmd_{source_domain}_to_{domain}: {_fmt(f'mmd_{source_domain}_to_{domain}')}")

    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    *,
    dataset: str = "airport",
    source: str = "BRAZIL",
    eval_domains: list[str] | None = None,
    seed: int = 0,
    device: str = "cpu",
    epochs: int = 200,
    verbose: int = 0,
    num_layers: int = DEFAULT_NUM_LAYERS,
    hid_dim: int = DEFAULT_HID_DIM,
    dropout_ratio: float = DEFAULT_DROPOUT,
    out_root: str = "./__saved__/analysis/airport_domain_generalization_tracking",
    out_dir: str | None = None,
):
    start_time = time.time()
    set_seed(seed)

    config = _build_experiment_config(
        dataset=dataset,
        source=source,
        seed=seed,
        device=device,
        epochs=epochs,
        verbose=verbose,
        num_layers=num_layers,
        hid_dim=hid_dim,
        dropout_ratio=dropout_ratio,
    )

    graphs, metadata = _load_dataset_graphs(dataset, config, device=device)
    if eval_domains is None:
        eval_domains = list(metadata["domains"])
    eval_domains = list(eval_domains)
    target_domains = [domain for domain in eval_domains if domain != source]

    config["model"]["in_dim"] = metadata["num_features"]
    config["model"]["num_classes"] = metadata["num_classes"]

    if out_dir is None:
        dataset_out_root = Path(out_root) / str(dataset).lower()
        out_dir = make_output_dir(
            out_root=str(dataset_out_root),
            run_name=(
                f"domain_generalization_tracking_{str(dataset).lower()}_{source}"
                f"_to_{'-'.join(target_domains) if target_domains else 'none'}"
                f"_layers{num_layers}_seed{seed}_{time.strftime('%m%d_%H%M%S')}"
            ),
        )
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=False)

    model = None
    try:
        model = build_model(config, from_pygda=False)
        history_df = _train_source_and_track(
            model,
            graphs[source],
            graphs,
            source_domain=source,
            eval_domains=eval_domains,
            use_mask=True,
            seed=seed,
        )
        history_df = _add_derived_metrics(history_df, eval_domains=eval_domains)

        history_df.to_csv(out_dir / "training_diagnostics.csv", index=False)
        overview_plot = plot_training_overview(
            history_df,
            out_dir=out_dir,
            source_domain=source,
            eval_domains=eval_domains,
        )
        target_quality_plot = plot_target_quality_vs_performance(
            history_df,
            out_dir=out_dir,
            source_domain=source,
            target_domains=target_domains,
        )
        metadata_json = {
            "config": config,
            "dataset_metadata": metadata,
            "eval_domains": eval_domains,
            "target_domains": target_domains,
            "elapsed_seconds": float(time.time() - start_time),
            "artifacts": {
                "history": "training_diagnostics.csv",
                "summary": "summary.txt",
                "training_overview_plot": Path(overview_plot).name,
                "target_quality_plot": None if target_quality_plot is None else Path(target_quality_plot).name,
            },
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata_json, indent=2), encoding="utf-8")
        _write_summary(
            out_dir,
            config=config,
            history_df=history_df,
            std_df=None,
            source_domain=source,
            target_domains=target_domains,
            eval_domains=eval_domains,
        )

        print(f"[saved] {out_dir / 'training_diagnostics.csv'}")
        print(f"[saved] {out_dir / 'summary.txt'}")
        print(f"[saved] {overview_plot}")
        if target_quality_plot is not None:
            print(f"[saved] {target_quality_plot}")
        print(f"results_dir={out_dir}")
        return out_dir
    finally:
        del model
        _cleanup_cuda()


def run_multi_seed_experiment(
    *,
    dataset: str = "airport",
    source: str = "BRAZIL",
    eval_domains: list[str] | None = None,
    seeds: list[int] | None = None,
    device: str = "cpu",
    epochs: int = 200,
    verbose: int = 0,
    num_layers: int = DEFAULT_NUM_LAYERS,
    hid_dim: int = DEFAULT_HID_DIM,
    dropout_ratio: float = DEFAULT_DROPOUT,
    out_root: str = "./__saved__/analysis/airport_domain_generalization_tracking",
    out_dir: str | None = None,
):
    seeds = [0, 1, 2, 3, 4] if seeds is None else [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("seeds must contain at least one seed.")

    if eval_domains is None:
        config_probe = _build_experiment_config(
            dataset=dataset,
            source=source,
            seed=int(seeds[0]),
            device=device,
            epochs=epochs,
            verbose=verbose,
            num_layers=num_layers,
            hid_dim=hid_dim,
            dropout_ratio=dropout_ratio,
        )
        _, metadata = _load_dataset_graphs(dataset, config_probe, device=device)
        eval_domains = list(metadata["domains"])
    eval_domains = list(eval_domains)
    target_domains = [domain for domain in eval_domains if domain != source]

    if out_dir is None:
        dataset_out_root = Path(out_root) / str(dataset).lower()
        out_dir = make_output_dir(
            out_root=str(dataset_out_root),
            run_name=(
                f"domain_generalization_tracking_{str(dataset).lower()}_{source}"
                f"_to_{'-'.join(target_domains) if target_domains else 'none'}"
                f"_layers{num_layers}_multiseed_{time.strftime('%m%d_%H%M%S')}"
            ),
        )
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=False)

    seed_frames = []
    seed_result_dirs = []
    config_for_summary = None

    try:
        for seed in seeds:
            seed_dir = out_dir / f"seed_{seed}"
            seed_result_dir = run_experiment(
                dataset=dataset,
                source=source,
                eval_domains=eval_domains,
                seed=int(seed),
                device=device,
                epochs=epochs,
                verbose=verbose,
                num_layers=num_layers,
                hid_dim=hid_dim,
                dropout_ratio=dropout_ratio,
                out_root=out_root,
                out_dir=str(seed_dir),
            )
            seed_history = pd.read_csv(Path(seed_result_dir) / "training_diagnostics.csv")
            seed_history["seed"] = int(seed)
            seed_frames.append(seed_history)
            seed_result_dirs.append(str(seed_result_dir))
            if config_for_summary is None:
                config_for_summary = json.loads((Path(seed_result_dir) / "metadata.json").read_text(encoding="utf-8"))["config"]

        all_df, mean_df, std_df = _aggregate_histories(seed_frames)
        all_df.to_csv(out_dir / "training_diagnostics_all_seeds.csv", index=False)
        mean_df.to_csv(out_dir / "training_diagnostics_mean.csv", index=False)
        std_df.to_csv(out_dir / "training_diagnostics_std.csv", index=False)

        overview_plot = plot_training_overview(
            mean_df,
            out_dir=out_dir,
            source_domain=source,
            eval_domains=eval_domains,
            std_df=std_df,
        )
        target_quality_plot = plot_target_quality_vs_performance(
            mean_df,
            out_dir=out_dir,
            source_domain=source,
            target_domains=target_domains,
            std_df=std_df,
        )

        metadata_json = {
            "config": config_for_summary,
            "dataset": str(dataset).lower(),
            "source_domain": source,
            "eval_domains": eval_domains,
            "target_domains": target_domains,
            "seeds": seeds,
            "seed_result_dirs": seed_result_dirs,
            "artifacts": {
                "all_seed_history": "training_diagnostics_all_seeds.csv",
                "mean_history": "training_diagnostics_mean.csv",
                "std_history": "training_diagnostics_std.csv",
                "summary": "summary.txt",
                "training_overview_plot": Path(overview_plot).name,
                "target_quality_plot": None if target_quality_plot is None else Path(target_quality_plot).name,
            },
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata_json, indent=2), encoding="utf-8")
        _write_summary(
            out_dir,
            config=config_for_summary,
            history_df=mean_df,
            std_df=std_df,
            source_domain=source,
            target_domains=target_domains,
            eval_domains=eval_domains,
            seed_list=seeds,
        )

        print(f"[saved] {out_dir / 'training_diagnostics_all_seeds.csv'}")
        print(f"[saved] {out_dir / 'training_diagnostics_mean.csv'}")
        print(f"[saved] {out_dir / 'training_diagnostics_std.csv'}")
        print(f"[saved] {overview_plot}")
        if target_quality_plot is not None:
            print(f"[saved] {target_quality_plot}")
        print(f"[saved] {out_dir / 'summary.txt'}")
        print(f"results_dir={out_dir}")
        return out_dir
    finally:
        _cleanup_cuda()
