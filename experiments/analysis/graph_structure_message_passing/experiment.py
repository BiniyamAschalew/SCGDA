from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch_geometric.transforms import OneHotDegree

from Learn.Clean_SCGDA.data.build_dataset import get_dataset, get_max_degree
from Learn.Clean_SCGDA.experiments.analysis.graph_structure_message_passing.plotting import (
    CASE_LABELS,
    CASE_ORDER,
    plot_case_summary,
    plot_domain_metric_grid,
)
from Learn.Clean_SCGDA.experiments.analysis.plain_gnn_cross_domain.log_plot import make_output_dir
from Learn.Clean_SCGDA.utils.ablation_utils.propagation import Propagation
from Learn.Clean_SCGDA.utils.config_utils import load_config


DEFAULT_DATASETS = ("airport", "blog", "citation")
DEFAULT_MAX_LAYERS = 10
DEFAULT_NUM_SEEDS = 3


def _resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _load_dataset_graphs(dataset_name: str, device: str):
    data_cfg = load_config(f"./configs/data_configs/{dataset_name}.yaml")
    config = {
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
        "model": {"name": "test"},
        "expt": {"verbose": 0},
    }

    graphs = {}
    max_degree = 0
    if config["data"]["name"].lower() == "airport":
        max_degree = int(get_max_degree(config))

    for domain in config["data"]["domains"]:
        dataset = get_dataset(domain, config)
        if config["data"]["name"].lower() == "airport":
            dataset.transform = OneHotDegree(max_degree)
        data = dataset[0].to(device)
        if data.edge_index is not None:
            data.edge_index = data.edge_index.contiguous()
        graphs[domain] = data

    return graphs, {
        "dataset_name": str(config["data"]["name"]),
        "domains": list(config["data"]["domains"]),
        "max_degree": max_degree,
    }


def _project_features_to_k(x: torch.Tensor, out_dim: int, seed: int) -> torch.Tensor:
    x_np = x.detach().cpu().numpy()
    n_components = min(int(out_dim), int(x_np.shape[0]), int(x_np.shape[1]))
    pca = PCA(n_components=n_components, random_state=seed)
    projected = pca.fit_transform(x_np).astype(np.float32)
    if projected.shape[1] < int(out_dim):
        padded = np.zeros((projected.shape[0], int(out_dim)), dtype=np.float32)
        padded[:, : projected.shape[1]] = projected
        projected = padded
    return torch.from_numpy(projected).to(device=x.device, dtype=torch.float32)


def _build_case_features(case_name: str, data, num_classes: int, seed: int) -> torch.Tensor:
    if case_name == "label_one_hot":
        return F.one_hot(data.y, num_classes=num_classes).to(dtype=torch.float32)

    if case_name == "random_noise":
        noise = torch.randn((int(data.num_nodes), int(num_classes)), dtype=torch.float32)
        return noise.to(data.y.device)

    if case_name == "pca_projection":
        return _project_features_to_k(data.x.to(torch.float32), out_dim=num_classes, seed=seed)

    raise ValueError(f"Unknown case '{case_name}'")


def _class_frequency_rank(labels: np.ndarray, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(labels, minlength=int(num_classes))
    ranked = np.array(
        sorted(range(int(num_classes)), key=lambda cls: (-int(counts[cls]), int(cls))),
        dtype=np.int64,
    )
    return ranked, counts.astype(np.int64)


def _aligned_prediction_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
) -> dict[str, float]:
    preds = logits.argmax(dim=1)
    pred_np = preds.detach().cpu().numpy()
    label_np = labels.detach().cpu().numpy()

    pred_rank, pred_counts = _class_frequency_rank(pred_np, num_classes)
    true_rank, true_counts = _class_frequency_rank(label_np, num_classes)

    true_to_pred = np.empty(int(num_classes), dtype=np.int64)
    for pred_cls, true_cls in zip(pred_rank.tolist(), true_rank.tolist()):
        true_to_pred[int(true_cls)] = int(pred_cls)

    aligned_index = torch.from_numpy(true_to_pred).to(device=logits.device, dtype=torch.long)
    aligned_logits = logits[:, aligned_index]
    aligned_preds = aligned_logits.argmax(dim=1)
    aligned_accuracy = float((aligned_preds == labels).to(torch.float32).mean().item())
    classification_loss = float(F.cross_entropy(aligned_logits, labels).item())

    return {
        "aligned_accuracy": aligned_accuracy,
        "classification_loss": classification_loss,
        "pred_num_clusters": int(np.unique(pred_np).size),
        "pred_top_class_count": int(pred_counts[pred_rank[0]]) if len(pred_rank) else 0,
        "true_top_class_count": int(true_counts[true_rank[0]]) if len(true_rank) else 0,
    }


def _propagation_rows(
    data,
    x0: torch.Tensor,
    *,
    max_layers: int,
    prop: Propagation,
):
    rows = []
    labels = data.y.detach().cpu().numpy()
    cur = x0

    with torch.no_grad():
        for step in range(int(max_layers) + 1):
            preds = cur.argmax(dim=1).detach().cpu().numpy()
            aligned_metrics = _aligned_prediction_metrics(
                cur,
                data.y,
                num_classes=int(cur.size(1)),
            )
            rows.append(
                {
                    "step": int(step),
                    "ari": float(adjusted_rand_score(labels, preds)),
                    "nmi": float(normalized_mutual_info_score(labels, preds)),
                    **aligned_metrics,
                }
            )
            if step < int(max_layers):
                cur = prop(cur, data.edge_index)

    return rows


def _summarize_by_domain(raw_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        raw_df.groupby(["dataset", "dataset_name", "domain", "case", "step"], as_index=False)
        .agg(
            num_seeds=("seed", "nunique"),
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            nmi_mean=("nmi", "mean"),
            nmi_std=("nmi", "std"),
            aligned_accuracy_mean=("aligned_accuracy", "mean"),
            aligned_accuracy_std=("aligned_accuracy", "std"),
            classification_loss_mean=("classification_loss", "mean"),
            classification_loss_std=("classification_loss", "std"),
        )
        .sort_values(["dataset", "domain", "case", "step"])
    )
    for col in (
        "ari_std",
        "nmi_std",
        "aligned_accuracy_std",
        "classification_loss_std",
    ):
        summary_df[col] = summary_df[col].fillna(0.0)
    return summary_df


def _summarize_by_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        raw_df.groupby(["dataset", "dataset_name", "case", "step"], as_index=False)
        .agg(
            num_domains=("domain", "nunique"),
            num_seeds=("seed", "nunique"),
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            nmi_mean=("nmi", "mean"),
            nmi_std=("nmi", "std"),
            aligned_accuracy_mean=("aligned_accuracy", "mean"),
            aligned_accuracy_std=("aligned_accuracy", "std"),
            classification_loss_mean=("classification_loss", "mean"),
            classification_loss_std=("classification_loss", "std"),
        )
        .sort_values(["dataset", "case", "step"])
    )
    for col in (
        "ari_std",
        "nmi_std",
        "aligned_accuracy_std",
        "classification_loss_std",
    ):
        summary_df[col] = summary_df[col].fillna(0.0)
    return summary_df


def run_experiment(
    *,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    max_layers: int = DEFAULT_MAX_LAYERS,
    seed: int = 0,
    num_seeds: int = DEFAULT_NUM_SEEDS,
    device: str = "cpu",
    out_root: str = "./__saved__/analysis/graph_structure_message_passing",
    domain: str | None = None,
):
    device = _resolve_device(device)
    datasets = tuple(str(dataset).lower() for dataset in datasets)
    if domain is not None and len(datasets) != 1:
        raise ValueError("A single domain can only be requested with a single dataset.")

    _set_seed(seed)
    seeds = tuple(int(seed) + offset for offset in range(int(num_seeds)))
    out_dir = make_output_dir(
        out_root=str(Path(out_root)),
        run_name=(
            f"graph_structure_message_passing_{time.strftime('%m%d_%H%M%S')}"
            f"_seeds{seeds[0]}to{seeds[-1]}_L{int(max_layers)}"
        ),
    )
    prop = Propagation().to(device)

    rows = []
    metadata_rows = []
    started_at = time.time()

    for dataset in datasets:
        graphs, dataset_meta = _load_dataset_graphs(dataset, device=device)
        domains = list(dataset_meta["domains"])
        if domain is not None:
            if domain not in domains:
                raise ValueError(f"Domain '{domain}' not found for dataset '{dataset}'. Available: {domains}")
            domains = [domain]

        print(f"\nDataset: {dataset_meta['dataset_name']} | domains={domains} | seeds={list(seeds)}")
        dataset_rows = []
        for domain_idx, domain_name in enumerate(domains):
            data = graphs[domain_name]
            num_classes = int(torch.unique(data.y).numel())
            num_features = int(data.x.size(1)) if data.x is not None else 0

            metadata_rows.append(
                {
                    "dataset": dataset,
                    "domain": domain_name,
                    "num_nodes": int(data.num_nodes),
                    "num_edges": int(data.edge_index.size(1)),
                    "num_input_features": num_features,
                    "num_classes": num_classes,
                }
            )

            for seed_idx, run_seed in enumerate(seeds):
                for case_idx, case_name in enumerate(CASE_ORDER):
                    case_seed = int(run_seed) + 1000 * domain_idx + 10000 * case_idx
                    _set_seed(case_seed)
                    x0 = _build_case_features(case_name, data, num_classes=num_classes, seed=case_seed)
                    case_rows = _propagation_rows(
                        data,
                        x0,
                        max_layers=max_layers,
                        prop=prop,
                    )

                    for row in case_rows:
                        row.update(
                            {
                                "dataset": dataset,
                                "dataset_name": dataset_meta["dataset_name"],
                                "domain": domain_name,
                                "seed": int(run_seed),
                                "seed_index": int(seed_idx),
                                "case": case_name,
                                "case_label": CASE_LABELS[case_name],
                                "num_nodes": int(data.num_nodes),
                                "num_edges": int(data.edge_index.size(1)),
                                "num_input_features": num_features,
                                "num_classes": num_classes,
                            }
                        )
                    rows.extend(case_rows)
                    dataset_rows.extend(case_rows)

            domain_case_df = pd.DataFrame(
                [
                    row
                    for row in dataset_rows
                    if row["domain"] == domain_name
                ]
            )
            domain_case_summary = _summarize_by_domain(domain_case_df)
            for case_name in CASE_ORDER:
                case_summary = domain_case_summary[
                    (domain_case_summary["domain"] == domain_name)
                    & (domain_case_summary["case"] == case_name)
                ].sort_values("step")
                if case_summary.empty:
                    continue
                print(
                    f"  {domain_name:>10s} | {CASE_LABELS[case_name]:<14s} "
                    f"| ARI(t=0)->ARI(t={int(max_layers)}): "
                    f"{case_summary.iloc[0]['ari_mean']:.4f}->{case_summary.iloc[-1]['ari_mean']:.4f} "
                    f"| ACC(t=0)->ACC(t={int(max_layers)}): "
                    f"{case_summary.iloc[0]['aligned_accuracy_mean']:.4f}->{case_summary.iloc[-1]['aligned_accuracy_mean']:.4f}"
                )

        dataset_df = pd.DataFrame(dataset_rows)
        dataset_domain_summary_df = _summarize_by_domain(dataset_df)
        dataset_summary_df = _summarize_by_dataset(dataset_df)
        plot_domain_metric_grid(
            dataset_domain_summary_df,
            out_dir=out_dir,
            dataset_name=dataset_meta["dataset_name"],
            domains=domains,
        )
        plot_case_summary(
            dataset_summary_df,
            out_dir=out_dir,
            dataset_name=dataset_meta["dataset_name"],
        )

    raw_df = pd.DataFrame(rows).sort_values(["dataset", "domain", "seed", "case", "step"])
    metadata_df = pd.DataFrame(metadata_rows).sort_values(["dataset", "domain"])
    summary_domain_df = _summarize_by_domain(raw_df)
    summary_df = _summarize_by_dataset(raw_df)

    raw_df.to_csv(Path(out_dir) / "raw_results.csv", index=False)
    summary_domain_df.to_csv(Path(out_dir) / "summary_by_dataset_domain_case_step.csv", index=False)
    summary_df.to_csv(Path(out_dir) / "summary_by_dataset_case_step.csv", index=False)
    metadata_df.to_csv(Path(out_dir) / "domain_metadata.csv", index=False)
    (Path(out_dir) / "config.json").write_text(
        json.dumps(
            {
                "datasets": list(datasets),
                "domain": domain,
                "max_layers": int(max_layers),
                "seed_start": int(seed),
                "num_seeds": int(num_seeds),
                "seeds": list(seeds),
                "device": device,
                "metrics": [
                    "ari",
                    "nmi",
                    "aligned_accuracy",
                    "classification_loss",
                ],
                "cases": list(CASE_ORDER),
                "elapsed_seconds": float(time.time() - started_at),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nresults_dir={out_dir}")
    return out_dir
