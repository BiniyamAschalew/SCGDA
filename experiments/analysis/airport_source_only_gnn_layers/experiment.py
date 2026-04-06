from __future__ import annotations

import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score

from  data.build_dataset import build_dataset
from  models.build_model import build_model
from  utils.config_utils import build_config
from  utils.expt_utils import set_seed
from  utils.train_utils.metrics import BaseMetric
from  utils.train_utils.mmd import MMD

from  experiments.analysis.airport_source_only_gnn_layers.plotting import (
    plot_snapshot_metric_overview,
    plot_snapshot_progression,
)
from  experiments.analysis.plain_gnn_cross_domain.log_plot import make_output_dir


DEFAULT_NUM_LAYERS = 5
MAX_TSNE_POINTS_PER_STAGE = 1500


def _snapshot_order(num_layers: int):
    order = ["raw_input"]
    for layer_idx in range(1, int(num_layers) + 1):
        order.append(f"after_linear{layer_idx}")
        order.append(f"after_mp{layer_idx}")
    return order


def _snapshot_titles(num_layers: int):
    titles = {"raw_input": "raw input"}
    for layer_idx in range(1, int(num_layers) + 1):
        titles[f"after_linear{layer_idx}"] = f"after linear {layer_idx}"
        titles[f"after_mp{layer_idx}"] = f"after MP{layer_idx}"
    return titles


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _build_experiment_config(
    *,
    dataset: str,
    source: str,
    target: str,
    seed: int,
    device: str,
    epochs: int,
    verbose: int,
    num_layers: int = DEFAULT_NUM_LAYERS,
):
    config_setup = {
        "model": "source_only_gnn",
        "data": dataset,
        "expt": "default",
    }
    update_config = {
        "expt": {
            "source": source,
            "target": target,
            "device": device,
            "seed": seed,
            "wandb_enabled": False,
            "project": "airport_source_only_gnn_layers",
            "verbose": verbose,
            "metrics": ["micro_f1", "macro_f1"],
        },
        "model": {
            "epochs": epochs,
            "num_layers": num_layers,
            "use_mask": True,
        },
    }
    config = build_config(config_setup, update_config=update_config, use_tuned=0)
    config["model"]["epochs"] = epochs
    config["model"]["num_layers"] = num_layers
    config["model"]["use_mask"] = True
    config["expt"]["metrics"] = ["micro_f1", "macro_f1"]
    return config


def _fit_input_feature_pca(source_data, target_data, *, output_dim: int, seed: int):
    source_x = source_data.x.detach().cpu().numpy()
    target_x = target_data.x.detach().cpu().numpy()
    fit_data = np.concatenate([source_x, target_x], axis=0)
    n_components = min(int(output_dim), int(fit_data.shape[0]), int(fit_data.shape[1]))
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(fit_data)
    return pca


def _apply_input_feature_pca(data, pca: PCA, *, device: str):
    x = data.x.detach().cpu().numpy()
    x_reduced = pca.transform(x)
    data.x = torch.from_numpy(x_reduced).to(device=device, dtype=torch.float)
    return data


def _load_pair(config: dict):
    source_dataset, target_dataset = build_dataset(config)
    source_data = source_dataset[0].to(config["expt"]["device"])
    target_data = target_dataset[0].to(config["expt"]["device"])

    if source_data.edge_index is not None:
        source_data.edge_index = source_data.edge_index.contiguous()
    if target_data.edge_index is not None:
        target_data.edge_index = target_data.edge_index.contiguous()

    input_pca = _fit_input_feature_pca(
        source_data,
        target_data,
        output_dim=int(config["model"]["hid_dim"]),
        seed=int(config["expt"]["seed"]),
    )
    _apply_input_feature_pca(source_data, input_pca, device=config["expt"]["device"])
    _apply_input_feature_pca(target_data, input_pca, device=config["expt"]["device"])

    num_features = int(source_data.x.size(1))
    num_classes = int(torch.unique(torch.cat([source_data.y, target_data.y])).numel())
    config["model"]["in_dim"] = num_features
    config["model"]["num_classes"] = num_classes
    return source_data, target_data, input_pca


def _collect_snapshots(model, data):
    return _collect_snapshots_from_tensors(model, data.x, data.edge_index)


def _collect_snapshots_from_tensors(model, x, edge_index):
    model.gnn.eval()
    with torch.no_grad():
        _, snapshots = model.gnn.feat_bottleneck(
            x,
            edge_index,
            return_snapshots=True,
            include_raw_input=True,
        )
    return {name: tensor.detach().cpu() for name, tensor in snapshots.items()}


def _sample_gaussian_input_like(reference_x, *, mean: float, std: float, seed: int):
    rng = np.random.default_rng(seed)
    gaussian_x = rng.normal(loc=mean, scale=std, size=tuple(reference_x.shape)).astype(np.float32)
    return torch.from_numpy(gaussian_x).to(device=reference_x.device, dtype=reference_x.dtype)


def _collect_gaussian_reference_snapshots(
    model,
    source_data,
    target_data,
    *,
    mean: float,
    std: float,
    seed: int,
):
    source_x = _sample_gaussian_input_like(source_data.x, mean=mean, std=std, seed=seed + 1001)
    target_x = _sample_gaussian_input_like(target_data.x, mean=mean, std=std, seed=seed + 2001)
    source_snapshots = _collect_snapshots_from_tensors(model, source_x, source_data.edge_index)
    target_snapshots = _collect_snapshots_from_tensors(model, target_x, target_data.edge_index)
    return source_snapshots, target_snapshots


def _stage_normalize_snapshots(source_snapshots, target_snapshots, *, snapshot_order: list[str]):
    normalized_source = {}
    normalized_target = {}
    stats_by_snapshot = {}

    for snapshot_name in snapshot_order:
        source_feat = source_snapshots[snapshot_name].numpy()
        target_feat = target_snapshots[snapshot_name].numpy()
        merged_feat = np.concatenate([source_feat, target_feat], axis=0)
        mean = merged_feat.mean(axis=0, keepdims=True)
        std = merged_feat.std(axis=0, keepdims=True)
        std = np.where(std <= 1e-12, 1.0, std)

        normalized_source[snapshot_name] = (source_feat - mean) / std
        normalized_target[snapshot_name] = (target_feat - mean) / std
        stats_by_snapshot[snapshot_name] = {"mean": mean, "std": std}

    return normalized_source, normalized_target, stats_by_snapshot


def _tsne_perplexity(num_points: int):
    if int(num_points) <= 3:
        return 2
    return max(2, min(20, int(num_points) // 8, int(num_points) - 1))


def _balanced_domain_subsample_indices(source_size: int, target_size: int, *, max_points: int, seed: int):
    total_size = int(source_size) + int(target_size)
    if total_size <= int(max_points):
        return np.arange(int(source_size)), np.arange(int(target_size))

    rng = np.random.default_rng(seed)
    source_take = min(int(source_size), max(1, int(max_points) // 2))
    target_take = min(int(target_size), max(1, int(max_points) // 2))
    remaining = int(max_points) - source_take - target_take

    if remaining > 0:
        extra_source = min(int(source_size) - source_take, remaining)
        source_take += extra_source
        remaining -= extra_source
    if remaining > 0:
        extra_target = min(int(target_size) - target_take, remaining)
        target_take += extra_target

    source_idx = np.sort(rng.choice(int(source_size), size=source_take, replace=False))
    target_idx = np.sort(rng.choice(int(target_size), size=target_take, replace=False))
    return source_idx, target_idx


def _split_column(data):
    out = np.full(int(data.num_nodes), "other", dtype=object)
    if getattr(data, "train_mask", None) is not None:
        out[data.train_mask.detach().cpu().numpy().astype(bool)] = "train"
    if getattr(data, "val_mask", None) is not None:
        out[data.val_mask.detach().cpu().numpy().astype(bool)] = "val"
    if getattr(data, "test_mask", None) is not None:
        out[data.test_mask.detach().cpu().numpy().astype(bool)] = "test"
    return out


def _build_tsne_projection_frames(
    source_snapshots: dict[str, np.ndarray],
    target_snapshots: dict[str, np.ndarray],
    *,
    snapshot_order: list[str],
    snapshot_titles: dict[str, str],
    source_domain: str,
    target_domain: str,
    source_data,
    target_data,
    seed: int,
):
    rows = []
    source_labels = source_data.y.detach().cpu().numpy()
    target_labels = target_data.y.detach().cpu().numpy()
    source_split = _split_column(source_data)
    target_split = _split_column(target_data)

    for offset, snapshot_name in enumerate(snapshot_order):
        source_feat = source_snapshots[snapshot_name]
        target_feat = target_snapshots[snapshot_name]
        source_idx, target_idx = _balanced_domain_subsample_indices(
            int(source_feat.shape[0]),
            int(target_feat.shape[0]),
            max_points=MAX_TSNE_POINTS_PER_STAGE,
            seed=int(seed + offset),
        )
        source_feat = source_feat[source_idx]
        target_feat = target_feat[target_idx]
        fit_data = np.concatenate([source_feat, target_feat], axis=0)
        source_size = int(source_feat.shape[0])
        reduce_dim = min(20, int(fit_data.shape[1]), int(fit_data.shape[0]))
        if int(fit_data.shape[1]) > reduce_dim:
            fit_data = PCA(
                n_components=reduce_dim,
                svd_solver="randomized",
                random_state=int(seed + offset),
            ).fit_transform(fit_data)

        tsne = TSNE(
            n_components=2,
            perplexity=_tsne_perplexity(int(fit_data.shape[0])),
            init="pca",
            learning_rate="auto",
            max_iter=350,
            method="barnes_hut",
            angle=0.5,
            random_state=int(seed + offset),
        )
        embedding = tsne.fit_transform(fit_data)
        source_proj = embedding[:source_size]
        target_proj = embedding[source_size:]

        for node_idx in range(source_proj.shape[0]):
            rows.append(
                {
                    "domain": source_domain,
                    "snapshot": snapshot_name,
                    "snapshot_title": snapshot_titles[snapshot_name],
                    "node_idx": int(source_idx[node_idx]),
                    "label": int(source_labels[source_idx[node_idx]]),
                    "split": str(source_split[source_idx[node_idx]]),
                    "pc1": float(source_proj[node_idx, 0]),
                    "pc2": float(source_proj[node_idx, 1]),
                }
            )

        for node_idx in range(target_proj.shape[0]):
            rows.append(
                {
                    "domain": target_domain,
                    "snapshot": snapshot_name,
                    "snapshot_title": snapshot_titles[snapshot_name],
                    "node_idx": int(target_idx[node_idx]),
                    "label": int(target_labels[target_idx[node_idx]]),
                    "split": str(target_split[target_idx[node_idx]]),
                    "pc1": float(target_proj[node_idx, 0]),
                    "pc2": float(target_proj[node_idx, 1]),
                }
            )

    return pd.DataFrame(rows)


def _snapshot_shift_table(source_snapshots, target_snapshots, *, snapshot_order: list[str], snapshot_titles: dict[str, str]):
    rows = []
    for snapshot_name in snapshot_order:
        source_feat = source_snapshots[snapshot_name]
        target_feat = target_snapshots[snapshot_name]
        rows.append(
            {
                "snapshot": snapshot_name,
                "snapshot_title": snapshot_titles[snapshot_name],
                "source_nodes": int(source_feat.size(0)),
                "target_nodes": int(target_feat.size(0)),
                "source_dim": int(source_feat.size(1)),
                "target_dim": int(target_feat.size(1)),
                "mmd": float(MMD(source_feat, target_feat).detach().cpu().item()),
                "centroid_gap": float(torch.linalg.norm(source_feat.mean(0) - target_feat.mean(0)).item()),
            }
        )
    return pd.DataFrame(rows)


def _mask_numpy(data, mask_name: str):
    mask = getattr(data, mask_name, None)
    if mask is None:
        return np.ones(int(data.num_nodes), dtype=bool)
    return mask.detach().cpu().numpy().astype(bool)


def _f1_scores(y_true, y_pred):
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _snapshot_transferability_table(
    source_snapshots,
    target_snapshots,
    source_data,
    target_data,
    *,
    snapshot_order: list[str],
    snapshot_titles: dict[str, str],
    seed: int,
):
    source_y = source_data.y.detach().cpu().numpy()
    target_y = target_data.y.detach().cpu().numpy()
    source_train_mask = _mask_numpy(source_data, "train_mask")
    source_val_mask = _mask_numpy(source_data, "val_mask")
    target_train_mask = _mask_numpy(target_data, "train_mask")
    target_val_mask = _mask_numpy(target_data, "val_mask")

    if int(source_train_mask.sum()) == 0 or int(source_val_mask.sum()) == 0:
        raise RuntimeError("Source train/val masks must be non-empty for transferability evaluation.")
    if int(target_train_mask.sum()) == 0 or int(target_val_mask.sum()) == 0:
        raise RuntimeError("Target train/val masks must be non-empty for transferability evaluation.")

    all_labels = np.unique(np.concatenate([source_y, target_y], axis=0))
    rng = np.random.default_rng(seed)
    rows = []

    for snapshot_name in snapshot_order:
        source_feat = source_snapshots[snapshot_name].numpy()
        target_feat = target_snapshots[snapshot_name].numpy()

        source_train_x = source_feat[source_train_mask]
        source_train_y = source_y[source_train_mask]
        source_val_x = source_feat[source_val_mask]
        source_val_y = source_y[source_val_mask]
        target_train_x = target_feat[target_train_mask]
        target_train_y = target_y[target_train_mask]
        target_val_x = target_feat[target_val_mask]
        target_val_y = target_y[target_val_mask]

        transfer_clf = RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
        )
        transfer_clf.fit(source_train_x, source_train_y)
        transfer_pred = transfer_clf.predict(target_val_x)
        source_self_pred = transfer_clf.predict(source_val_x)

        oracle_clf = RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
        )
        oracle_clf.fit(target_train_x, target_train_y)
        oracle_pred = oracle_clf.predict(target_val_x)

        random_pred = rng.choice(all_labels, size=target_val_y.shape[0], replace=True)

        transfer_scores = _f1_scores(target_val_y, transfer_pred)
        random_scores = _f1_scores(target_val_y, random_pred)
        oracle_scores = _f1_scores(target_val_y, oracle_pred)
        source_self_scores = _f1_scores(source_val_y, source_self_pred)

        rows.append(
            {
                "snapshot": snapshot_name,
                "snapshot_title": snapshot_titles[snapshot_name],
                "source_self_micro_f1": source_self_scores["micro_f1"],
                # "source_self_macro_f1": source_self_scores["macro_f1"],
                "transfer_micro_f1": transfer_scores["micro_f1"],
                # "transfer_macro_f1": transfer_scores["macro_f1"],
                "random_micro_f1": random_scores["micro_f1"],
                # "random_macro_f1": random_scores["macro_f1"],
                "oracle_micro_f1": oracle_scores["micro_f1"],
                # "oracle_macro_f1": oracle_scores["macro_f1"],
                "transfer_vs_random_micro_gain": transfer_scores["micro_f1"] - random_scores["micro_f1"],
                "oracle_vs_transfer_micro_gap": oracle_scores["micro_f1"] - transfer_scores["micro_f1"],
            }
        )

    return pd.DataFrame(rows)


def _describe_delta(delta: float, *, atol: float = 1e-5) -> str:
    if abs(delta) <= atol:
        return "was nearly unchanged"
    if delta > 0:
        return f"increased by {delta:.4f}"
    return f"decreased by {abs(delta):.4f}"


def _interpret_shift_table(shift_df: pd.DataFrame, *, num_layers: int):
    mmd_map = dict(zip(shift_df["snapshot"], shift_df["mmd"]))
    steps = [("raw_input", "after_linear1", "Linear/nonlinearity block 1")]
    for layer_idx in range(1, int(num_layers) + 1):
        steps.append((f"after_linear{layer_idx}", f"after_mp{layer_idx}", f"Message passing {layer_idx}"))
        if layer_idx < int(num_layers):
            next_idx = layer_idx + 1
            steps.append((f"after_mp{layer_idx}", f"after_linear{next_idx}", f"Linear/nonlinearity block {next_idx}"))

    lines = []
    for left, right, label in steps:
        delta = float(mmd_map[right] - mmd_map[left])
        lines.append(f"- {label}: MMD {_describe_delta(delta)} ({mmd_map[left]:.4f} -> {mmd_map[right]:.4f})")

    min_row = shift_df.loc[shift_df["mmd"].idxmin()]
    max_row = shift_df.loc[shift_df["mmd"].idxmax()]
    lines.append(
        f"- Smallest overall source-target shift: {min_row['snapshot_title']} (MMD={float(min_row['mmd']):.4f})"
    )
    lines.append(
        f"- Largest overall source-target shift: {max_row['snapshot_title']} (MMD={float(max_row['mmd']):.4f})"
    )
    return lines


def _write_summary(
    out_dir: Path,
    *,
    config: dict,
    source_metrics: dict,
    target_metrics: dict,
    shift_df: pd.DataFrame,
    gaussian_shift_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
    input_pca: PCA,
    snapshot_order: list[str],
    gaussian_mean: float,
    gaussian_std: float,
):
    dataset_name = str(config["data"]["name"])
    input_pca_total = float(np.sum(input_pca.explained_variance_ratio_))

    lines = [
        f"{dataset_name} source-only GNN layerwise snapshot analysis",
        "",
        f"Dataset: {dataset_name}",
        f"Source -> Target: {config['expt']['source']} -> {config['expt']['target']}",
        f"Device: {config['expt']['device']}",
        f"Seed: {config['expt']['seed']}",
        f"Epochs: {config['model']['epochs']}",
        "",
        "Training protocol:",
        "- Neural training used only the source-domain train mask.",
        "- Before training, source and target raw features were jointly reduced with an unlabeled input PCA to the model input dimension.",
        "- Target labels were not used during neural training or input PCA fitting.",
        "- Target train labels were used only for the oracle Random-Forest transferability baseline.",
        "",
        "Validation metrics:",
        f"- Source micro-F1: {float(source_metrics['micro_f1']):.4f}",
        # f"- Source macro-F1: {float(source_metrics['macro_f1']):.4f}",
        f"- Target micro-F1: {float(target_metrics['micro_f1']):.4f}",
        # f"- Target macro-F1: {float(target_metrics['macro_f1']):.4f}",
        "",
        "Input feature PCA:",
        f"- Raw input features were jointly reduced to {input_pca.n_components_} dimensions before training.",
        f"- Total explained variance kept by the input PCA: {input_pca_total:.4f}",
        "",
        "Projection basis:",
        f"- For each snapshot, the {config['expt']['source']} and {config['expt']['target']} features were merged and standardized feature-wise before t-SNE fitting.",
        "- A separate 2D t-SNE embedding was fit for each snapshot.",
        "- The plotted coordinates are useful within each panel, but not directly comparable across stages.",
        "- MMD and centroid-gap values were computed from the original hidden features, not from the t-SNE coordinates.",
        f"- For speed, each stage used a balanced plotting subsample of at most {MAX_TSNE_POINTS_PER_STAGE} total nodes before t-SNE.",
        "- t-SNE then used a fast PCA pre-reduction to at most 20 dimensions, followed by `init='pca'`, `learning_rate='auto'`, `max_iter=350`, and Barnes-Hut updates.",
        (
            f"- The Gaussian-reference figure uses separate synthetic source/target inputs sampled i.i.d. from "
            f"N({gaussian_mean:.1f}, {gaussian_std:.1f}^2) in the model-input space, followed by the same per-stage normalization and per-stage t-SNE procedure."
        ),
        "",
        "Layerwise shift (MMD):",
        shift_df[["snapshot_title", "mmd", "centroid_gap"]].to_string(index=False),
        "",
        "Gaussian-reference shift (MMD):",
        gaussian_shift_df[["snapshot_title", "mmd", "centroid_gap"]].to_string(index=False),
        "",
        "Transferability (Random Forest):",
        transfer_df[
            [
                "snapshot_title",
                "transfer_micro_f1",
                "random_micro_f1",
                "oracle_micro_f1",
                "source_self_micro_f1",
            ]
        ].to_string(index=False),
        "",
        "Interpretation:",
        *_interpret_shift_table(shift_df, num_layers=int(config["model"]["num_layers"])),
        "",
        "Snapshot definitions:",
        "- raw input: PCA-reduced input features used by the model.",
        "- after linear k: after the k-th linear + non-linearity block.",
        "- after MPk: immediately after the k-th graph propagation step.",
    ]

    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    *,
    dataset: str = "airport",
    source: str = "USA",
    target: str = "BRAZIL",
    seed: int = 0,
    device: str = "cpu",
    epochs: int = 200,
    verbose: int = 0,
    out_root: str = "./__saved__/analysis/airport_source_only_gnn_layers",
):
    start_time = time.time()
    set_seed(seed)
    gaussian_mean = 1.0
    gaussian_std = 0.5

    config = _build_experiment_config(
        dataset=dataset,
        source=source,
        target=target,
        seed=seed,
        device=device,
        epochs=epochs,
        verbose=verbose,
        num_layers=DEFAULT_NUM_LAYERS,
    )
    snapshot_order = _snapshot_order(int(config["model"]["num_layers"]))
    snapshot_titles = _snapshot_titles(int(config["model"]["num_layers"]))

    dataset_out_root = Path(out_root) / str(dataset).lower()
    out_dir = make_output_dir(
        out_root=str(dataset_out_root),
        run_name=f"source_only_gnn_layers_{str(dataset).lower()}_{source}_{target}_seed{seed}_{time.strftime('%m%d_%H%M%S')}",
    )

    model = None
    try:
        source_data, target_data, input_pca = _load_pair(config)
        model = build_model(config, from_pygda=False)
        model.fit(source_data, target_data, use_mask=True)

        metrics = BaseMetric(config)
        source_logits, source_labels = model.predict(source_data, use_mask=True)
        target_logits, target_labels = model.predict(target_data, use_mask=True)
        source_metrics = metrics(source_logits, source_labels)
        target_metrics = metrics(target_logits, target_labels)

        source_snapshots = _collect_snapshots(model, source_data)
        target_snapshots = _collect_snapshots(model, target_data)
        normalized_source_snapshots, normalized_target_snapshots, _ = _stage_normalize_snapshots(
            source_snapshots,
            target_snapshots,
            snapshot_order=snapshot_order,
        )
        gaussian_source_snapshots, gaussian_target_snapshots = _collect_gaussian_reference_snapshots(
            model,
            source_data,
            target_data,
            mean=gaussian_mean,
            std=gaussian_std,
            seed=seed,
        )
        gaussian_normalized_source_snapshots, gaussian_normalized_target_snapshots, _ = _stage_normalize_snapshots(
            gaussian_source_snapshots,
            gaussian_target_snapshots,
            snapshot_order=snapshot_order,
        )

        shift_df = _snapshot_shift_table(
            source_snapshots,
            target_snapshots,
            snapshot_order=snapshot_order,
            snapshot_titles=snapshot_titles,
        )
        gaussian_shift_df = _snapshot_shift_table(
            gaussian_source_snapshots,
            gaussian_target_snapshots,
            snapshot_order=snapshot_order,
            snapshot_titles=snapshot_titles,
        )
        transfer_df = _snapshot_transferability_table(
            source_snapshots,
            target_snapshots,
            source_data,
            target_data,
            snapshot_order=snapshot_order,
            snapshot_titles=snapshot_titles,
            seed=seed,
        )
        projected_df = _build_tsne_projection_frames(
            normalized_source_snapshots,
            normalized_target_snapshots,
            snapshot_order=snapshot_order,
            snapshot_titles=snapshot_titles,
            source_domain=source,
            target_domain=target,
            source_data=source_data,
            target_data=target_data,
            seed=seed,
        )
        gaussian_projected_df = _build_tsne_projection_frames(
            gaussian_normalized_source_snapshots,
            gaussian_normalized_target_snapshots,
            snapshot_order=snapshot_order,
            snapshot_titles=snapshot_titles,
            source_domain=source,
            target_domain=target,
            source_data=source_data,
            target_data=target_data,
            seed=seed + 100,
        )

        plot_snapshot_progression(
            projected_df,
            out_dir / "per_snapshot_pca_snapshots_observed_only.png",
            snapshot_order=snapshot_order,
            snapshot_titles=snapshot_titles,
            dataset_name=str(config["data"]["name"]),
            source_domain=source,
            target_domain=target,
        )
        plot_snapshot_progression(
            gaussian_projected_df,
            out_dir / "per_snapshot_pca_snapshots.png",
            snapshot_order=snapshot_order,
            snapshot_titles=snapshot_titles,
            dataset_name=str(config["data"]["name"]),
            source_domain=source,
            target_domain=target,
        )
        plot_snapshot_metric_overview(
            shift_df,
            transfer_df,
            out_dir / "snapshot_metric_overview.png",
            snapshot_order=snapshot_order,
            dataset_name=str(config["data"]["name"]),
            source_domain=source,
            target_domain=target,
        )

        shift_df.to_csv(out_dir / "snapshot_shift_mmd.csv", index=False)
        gaussian_shift_df.to_csv(out_dir / "gaussian_reference_snapshot_shift_mmd.csv", index=False)
        transfer_df.to_csv(out_dir / "snapshot_transferability.csv", index=False)
        projected_df.to_csv(out_dir / "snapshot_projections.csv", index=False)
        gaussian_projected_df.to_csv(out_dir / "gaussian_reference_snapshot_projections.csv", index=False)
        torch.save(
            {
                "dataset": dataset,
                "input_pca_dim": int(input_pca.n_components_),
                "source_domain": source,
                "target_domain": target,
                "source_snapshots": source_snapshots,
                "target_snapshots": target_snapshots,
                "gaussian_reference_source_snapshots": gaussian_source_snapshots,
                "gaussian_reference_target_snapshots": gaussian_target_snapshots,
                "snapshot_order": snapshot_order,
            },
            out_dir / "snapshot_features.pt",
        )
        torch.save(
            {
                "components": torch.from_numpy(input_pca.components_),
                "mean": torch.from_numpy(input_pca.mean_),
                "explained_variance_ratio": torch.from_numpy(input_pca.explained_variance_ratio_),
                "n_components": int(input_pca.n_components_),
            },
            out_dir / "input_feature_pca.pt",
        )
        metadata = {
            "config": config,
            "source_metrics": {k: float(v) for k, v in source_metrics.items()},
            "target_metrics": {k: float(v) for k, v in target_metrics.items()},
            "elapsed_seconds": float(time.time() - start_time),
            "input_feature_pca_dim": int(input_pca.n_components_),
            "projection_normalization": "feature-wise stage normalization on merged source+target features before per-stage t-SNE fitting",
            "projection_subsample_limit": int(MAX_TSNE_POINTS_PER_STAGE),
            "gaussian_reference": {
                "mean": gaussian_mean,
                "std": gaussian_std,
                "description": "Synthetic i.i.d. Gaussian inputs in the trained model-input space, propagated on source and target graphs.",
            },
            "artifacts": {
                "figure": "per_snapshot_pca_snapshots.png",
                "observed_only_figure": "per_snapshot_pca_snapshots_observed_only.png",
                "snapshot_metric_overview": "snapshot_metric_overview.png",
                "shift_table": "snapshot_shift_mmd.csv",
                "gaussian_reference_shift_table": "gaussian_reference_snapshot_shift_mmd.csv",
                "transferability_table": "snapshot_transferability.csv",
                "projections": "snapshot_projections.csv",
                "gaussian_reference_projections": "gaussian_reference_snapshot_projections.csv",
                "snapshots": "snapshot_features.pt",
                "input_pca": "input_feature_pca.pt",
                "summary": "summary.txt",
            },
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _write_summary(
            out_dir,
            config=config,
            source_metrics=source_metrics,
            target_metrics=target_metrics,
            shift_df=shift_df,
            gaussian_shift_df=gaussian_shift_df,
            transfer_df=transfer_df,
            input_pca=input_pca,
            snapshot_order=snapshot_order,
            gaussian_mean=gaussian_mean,
            gaussian_std=gaussian_std,
        )

        print(f"[saved] {out_dir / 'per_snapshot_pca_snapshots.png'}")
        print(f"[saved] {out_dir / 'per_snapshot_pca_snapshots_observed_only.png'}")
        print(f"[saved] {out_dir / 'snapshot_metric_overview.png'}")
        print(f"[saved] {out_dir / 'snapshot_shift_mmd.csv'}")
        print(f"[saved] {out_dir / 'gaussian_reference_snapshot_shift_mmd.csv'}")
        print(f"[saved] {out_dir / 'snapshot_transferability.csv'}")
        print(f"[saved] {out_dir / 'summary.txt'}")
        print(f"results_dir={out_dir}")
        return out_dir
    finally:
        del model
        _cleanup_cuda()
