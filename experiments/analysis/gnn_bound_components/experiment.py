from __future__ import annotations

"""
Plain-GNN bound-components experiment.

For each dataset and seed:
1. Train one source-only GNN per domain on `train_mask`.
2. Evaluate each trained source model on its own domain and every other domain on `val_mask`.
3. Train one target-self GNN per domain so we can measure target difficulty when the target is treated as a source.
4. Train one oracle GNN per ordered pair on both source and target labels.
5. Save one raw row per ordered pair containing the main quantities needed to study:

   target_micro_f1 ~ f(
       source_micro_f1,
       source_random_micro_f1,
       target_random_micro_f1,
       mmd_shift,
       smoothness_grad_max,
       oracle_target_micro_f1,
   )

`mmd_shift` and `w1_shift` are defined here as the source-trained model's own
source-vs-target validation embedding shift, measured on the bottleneck
representations right before the final classifier.

We also keep `source_model_domain_mmd` and `source_model_domain_w1` as legacy
aliases of the same quantity in the raw CSV.
"""

import gc
import json
import math
import os
from pathlib import Path
import time
import warnings

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import pandas as pd
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.transforms import OneHotDegree

torch.set_num_threads(4)

from Learn.Clean_SCGDA.data.build_dataset import get_dataset, get_max_degree
from Learn.Clean_SCGDA.models.build_model import build_model
from Learn.Clean_SCGDA.utils.config_utils import build_config, load_config
from Learn.Clean_SCGDA.utils.expt_utils import set_seed
from Learn.Clean_SCGDA.utils.train_utils.metrics import BaseMetric
from Learn.Clean_SCGDA.utils.train_utils.mmd import MMD
from Learn.Clean_SCGDA.utils.train_utils.sinkhorn import get_Sinkhorn

from Learn.Clean_SCGDA.experiments.analysis.gnn_bound_components.log_plot import (
    finalize_outputs,
    make_output_dir,
)


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message="The `pre_transform` argument differs from the one used in the pre-processed version of this dataset.*",
    category=UserWarning,
)


DEFAULT_DATASETS = ("citation", "airport", "blog")
DEFAULT_SEEDS = (2025, 2026, 2027, 2028, 2029)
DEFAULT_NUM_LAYERS = (2, 3, 4)
DEFAULT_STRUCTURAL_LAYERS = 3
DEFAULT_W1_MAX_SAMPLES = 512
DEFAULT_MODELS = ("gnn", "simgda")


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _parse_csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _parse_int_csv_env(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _build_train_config(
    *,
    model_name: str,
    dataset: str,
    source: str,
    target: str,
    num_layers: int,
    seed: int,
    device: str,
    epochs: int,
    verbose: int,
):
    config_setup = {
        "model": model_name,
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
            "project": "gnn_bound_components",
            "verbose": verbose,
            "metrics": ["micro_f1"],
        },
        "model": {
            "epochs": epochs,
            "use_mask": True,
            "num_layers": num_layers,
        },
    }
    config = build_config(config_setup, update_config=update_config, use_tuned=0)
    config["model"]["epochs"] = epochs
    config["model"]["use_mask"] = True
    config["model"]["num_layers"] = num_layers
    config["expt"]["metrics"] = ["micro_f1"]
    return config


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
        "model": {"name": "gnn"},
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

    num_features = int(next(iter(graphs.values())).x.size(1))
    num_classes = int(torch.unique(torch.cat([graph.y for graph in graphs.values()])).numel())
    return graphs, {
        "domains": list(config["data"]["domains"]),
        "num_features": num_features,
        "num_classes": num_classes,
        "max_degree": max_degree,
    }


def _get_backbone(model):
    if hasattr(model, "gnn") and model.gnn is not None:
        return model.gnn
    if hasattr(model, "simgda") and model.simgda is not None:
        return model.simgda
    raise ValueError(f"Unsupported model backbone for {type(model).__name__}")


def _set_backbone(model, backbone):
    if hasattr(model, "gnn"):
        model.gnn = backbone
        return
    if hasattr(model, "simgda"):
        model.simgda = backbone
        return
    raise ValueError(f"Unsupported model backbone for {type(model).__name__}")


def _feature_mask(data):
    mask = getattr(data, "val_mask", None)
    if mask is None:
        return torch.ones_like(data.y, dtype=torch.bool, device=data.y.device)
    return mask.bool()


def _predict_micro_f1(model, data, metrics: BaseMetric) -> float:
    logits, labels = model.predict(data, use_mask=True)
    return float(metrics(logits, labels).get("micro_f1", 0.0))


def _extract_val_features(model, data):
    backbone = _get_backbone(model)
    backbone.eval()
    batch = getattr(data, "batch", None)
    with torch.no_grad():
        features = backbone.feat_bottleneck(data.x, data.edge_index, batch=batch)
    return features[_feature_mask(data)]


def _mmd_value(features_a, features_b) -> float:
    return float(MMD(features_a, features_b).detach().cpu().item())


def _subsample_rows(features, max_rows: int):
    if max_rows <= 0 or features.size(0) <= max_rows:
        return features
    idx = torch.linspace(
        0,
        features.size(0) - 1,
        steps=max_rows,
        device=features.device,
    ).round().long()
    return features[idx]


def _w1_value(features_a, features_b, *, max_samples: int) -> float:
    features_a = _subsample_rows(features_a, max_samples)
    features_b = _subsample_rows(features_b, max_samples)
    return float(
        get_Sinkhorn(
            features_a,
            features_b,
            p=1,
            blur=0.05,
            scaling=0.9,
            debias=True,
            backend="tensorized",
        ).detach().cpu().item()
    )


def _propagate_with_norm(edge_index, edge_weight, x, num_nodes):
    if edge_index.numel() == 0:
        return x
    adj = torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        size=(num_nodes, num_nodes),
        device=x.device,
        dtype=x.dtype,
    ).coalesce()
    return torch.sparse.mm(adj, x)


def _structural_trajectory(data, *, num_steps: int):
    x = torch.ones((data.x.size(0), 1), device=data.x.device, dtype=data.x.dtype)
    edge_index, edge_weight = gcn_norm(
        data.edge_index,
        None,
        num_nodes=data.x.size(0),
        add_self_loops=True,
        dtype=data.x.dtype,
    )

    states = []
    cur = x
    for _ in range(num_steps):
        cur = _propagate_with_norm(edge_index, edge_weight, cur, data.x.size(0))
        states.append(cur)

    if not states:
        return x
    return torch.cat(states, dim=1)


def _spectral_lipschitz_proxy(model) -> dict[str, float]:
    backbone = _get_backbone(model)
    log10_sum = 0.0
    matrix_count = 0

    for name, param in backbone.named_parameters():
        if param.ndim != 2:
            continue
        sigma = torch.linalg.matrix_norm(param.detach(), ord=2)
        sigma_val = max(float(sigma.detach().cpu().item()), 1e-12)
        log10_sum += math.log10(sigma_val)
        matrix_count += 1

    if matrix_count == 0:
        return {
            "spectral_lipschitz_upper": 0.0,
            "spectral_lipschitz_log10": float("-inf"),
            "spectral_matrix_count": 0,
        }

    if log10_sum > 30:
        upper = float("inf")
    else:
        upper = float(10.0 ** log10_sum)

    return {
        "spectral_lipschitz_upper": upper,
        "spectral_lipschitz_log10": float(log10_sum),
        "spectral_matrix_count": int(matrix_count),
    }


def _smoothness_proxy(model, data, *, max_nodes: int):
    mask = _feature_mask(data)
    node_idx = mask.nonzero(as_tuple=False).view(-1)
    if max_nodes > 0 and node_idx.numel() > max_nodes:
        node_idx = node_idx[:max_nodes]

    if node_idx.numel() == 0:
        return {
            "smoothness_grad_max": 0.0,
            "smoothness_grad_mean": 0.0,
            "smoothness_nodes_evaluated": 0,
        }

    x = data.x.detach().clone().requires_grad_(True)
    batch = getattr(data, "batch", None)
    backbone = _get_backbone(model)
    logits = backbone(x, data.edge_index, batch=batch)
    pred = logits.detach().argmax(dim=1)

    grad_max_values = []
    for step, idx in enumerate(node_idx.tolist()):
        score = logits[idx, pred[idx]]
        grad = torch.autograd.grad(
            score,
            x,
            retain_graph=step < (len(node_idx) - 1),
            create_graph=False,
        )[0]
        grad_norms = grad.norm(dim=1)
        grad_max_values.append(float(grad_norms.max().detach().cpu().item()))

    return {
        "smoothness_grad_max": max(grad_max_values),
        "smoothness_grad_mean": float(sum(grad_max_values) / len(grad_max_values)),
        "smoothness_nodes_evaluated": int(len(grad_max_values)),
    }


def _prepare_model_config(config: dict, num_features: int, num_classes: int):
    config["model"]["in_dim"] = num_features
    config["model"]["num_classes"] = num_classes
    config["expt"]["metrics"] = ["micro_f1"]
    return config


def _build_random_model(config: dict):
    model = build_model(config, from_pygda=False)
    _set_backbone(model, model.init_model(**model.kwargs))
    return model


def run_experiment():
    models = _parse_csv_env("GNN_BOUND_MODELS", DEFAULT_MODELS)
    datasets = _parse_csv_env("GNN_BOUND_DATASETS", DEFAULT_DATASETS)
    seeds = _parse_int_csv_env("GNN_BOUND_SEEDS", DEFAULT_SEEDS)
    num_layers_values = _parse_int_csv_env("GNN_BOUND_NUM_LAYERS", DEFAULT_NUM_LAYERS)
    device = os.getenv("GNN_BOUND_DEVICE", "cuda:7" if torch.cuda.is_available() else "cpu")
    epochs = int(os.getenv("GNN_BOUND_EPOCHS", "200"))
    verbose = int(os.getenv("GNN_BOUND_VERBOSE", "0"))
    smoothness_nodes = int(os.getenv("GNN_BOUND_SMOOTHNESS_NODES", "64"))
    structural_layers = int(os.getenv("GNN_BOUND_STRUCTURAL_LAYERS", str(DEFAULT_STRUCTURAL_LAYERS)))
    w1_max_samples = int(os.getenv("GNN_BOUND_W1_MAX_SAMPLES", str(DEFAULT_W1_MAX_SAMPLES)))
    out_root = os.getenv("GNN_BOUND_OUT_ROOT", "./__saved__/analysis/gnn_bound_components")
    run_name = os.getenv("GNN_BOUND_RUN_NAME", f"gnn_bound_components_{time.strftime('%m%d_%H%M%S')}")

    out_dir = make_output_dir(out_root=out_root, run_name=run_name)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "datasets": list(datasets),
                "models": list(models),
                "seeds": list(seeds),
                "num_layers_values": list(num_layers_values),
                "device": device,
                "epochs": epochs,
                "verbose": verbose,
                "smoothness_nodes": smoothness_nodes,
                "structural_layers": structural_layers,
                "w1_max_samples": w1_max_samples,
                "mmd_shift_definition": "MMD(source-trained model bottleneck features on source-val vs target-val)",
                "source_model_domain_mmd_definition": "Legacy alias of mmd_shift",
                "w1_shift_definition": "Sinkhorn W1 proxy(source-trained model bottleneck features on source-val vs target-val)",
                "source_model_domain_w1_definition": "Legacy alias of w1_shift",
                "structural_mmd_shift_definition": "MMD between propagated-ones structural trajectories for source and target graphs",
                "structural_w1_shift_definition": "Sinkhorn W1 proxy between propagated-ones structural trajectories for source and target graphs",
                "spectral_lipschitz_definition": "Product of spectral norms of all 2D parameter matrices in the GNN",
            },
            f,
            indent=2,
        )

    total_pairs = 0
    for dataset in datasets:
        domains = list(load_config(f"./configs/data_configs/{dataset}.yaml")["domains"])
        total_pairs += len(models) * len(num_layers_values) * len(seeds) * len(domains) * max(len(domains) - 1, 0)

    rows = []
    pair_idx = 0

    for dataset in datasets:
        graphs, metadata = _load_dataset_graphs(dataset, device=device)
        domains = metadata["domains"]
        structural_cache = {
            domain: _structural_trajectory(graphs[domain], num_steps=structural_layers).detach().cpu()
            for domain in domains
        }

        for model_name in models:
            for num_layers in num_layers_values:
                for seed in seeds:
                    base_config = _prepare_model_config(
                        _build_train_config(
                            model_name=model_name,
                            dataset=dataset,
                            source=domains[0],
                            target=domains[0],
                            num_layers=num_layers,
                            seed=seed,
                            device=device,
                            epochs=epochs,
                            verbose=verbose,
                        ),
                        metadata["num_features"],
                        metadata["num_classes"],
                    )
                    metrics = BaseMetric(base_config)

                    set_seed(seed)
                    random_model = _build_random_model(base_config)
                    random_scores = {
                        domain: _predict_micro_f1(random_model, graphs[domain], metrics)
                        for domain in domains
                    }
                    del random_model
                    _cleanup_cuda()

                    self_cache = {}
                    pair_cache = {}

                    for source in domains:
                        set_seed(seed)
                        config = _prepare_model_config(
                            _build_train_config(
                                model_name=model_name,
                                dataset=dataset,
                                source=source,
                                target=source,
                                num_layers=num_layers,
                                seed=seed,
                                device=device,
                                epochs=epochs,
                                verbose=verbose,
                            ),
                            metadata["num_features"],
                            metadata["num_classes"],
                        )

                        model = build_model(config, from_pygda=False)
                        source_data = graphs[source]

                        train_start = time.time()
                        model.fit(source_data, source_data, use_mask=True, oracle=False)
                        train_time_self = time.time() - train_start

                        source_self_micro = _predict_micro_f1(model, source_data, metrics)
                        source_home_features = _extract_val_features(model, source_data).detach().cpu()
                        source_smoothness = _smoothness_proxy(
                            model,
                            source_data,
                            max_nodes=smoothness_nodes,
                        )
                        spectral_proxy = _spectral_lipschitz_proxy(model)

                        self_cache[source] = {
                            "self_micro_f1": source_self_micro,
                            "random_micro_f1": random_scores[source],
                            "home_features": source_home_features,
                            "smoothness_source_grad_max": source_smoothness["smoothness_grad_max"],
                            "smoothness_source_grad_mean": source_smoothness["smoothness_grad_mean"],
                            "smoothness_source_nodes_evaluated": source_smoothness["smoothness_nodes_evaluated"],
                            "spectral_lipschitz_upper": spectral_proxy["spectral_lipschitz_upper"],
                            "spectral_lipschitz_log10": spectral_proxy["spectral_lipschitz_log10"],
                            "spectral_matrix_count": spectral_proxy["spectral_matrix_count"],
                            "train_time_self_model": float(train_time_self),
                        }

                        for target in domains:
                            if target == source:
                                continue

                            target_data = graphs[target]
                            target_micro = _predict_micro_f1(model, target_data, metrics)
                            target_features = _extract_val_features(model, target_data).detach().cpu()
                            target_smoothness = _smoothness_proxy(
                                model,
                                target_data,
                                max_nodes=smoothness_nodes,
                            )

                            pair_cache[(source, target)] = {
                                "target_micro_f1": target_micro,
                                "source_model_domain_mmd": _mmd_value(source_home_features, target_features),
                                "source_model_domain_w1": _w1_value(
                                    source_home_features,
                                    target_features,
                                    max_samples=w1_max_samples,
                                ),
                                "smoothness_source_grad_max": source_smoothness["smoothness_grad_max"],
                                "smoothness_source_grad_mean": source_smoothness["smoothness_grad_mean"],
                                "smoothness_source_nodes_evaluated": source_smoothness["smoothness_nodes_evaluated"],
                                "smoothness_target_grad_max": target_smoothness["smoothness_grad_max"],
                                "smoothness_target_grad_mean": target_smoothness["smoothness_grad_mean"],
                                "smoothness_target_nodes_evaluated": target_smoothness["smoothness_nodes_evaluated"],
                                "smoothness_grad_max": max(
                                    source_smoothness["smoothness_grad_max"],
                                    target_smoothness["smoothness_grad_max"],
                                ),
                                "smoothness_grad_mean_avg": float(
                                    0.5
                                    * (
                                        source_smoothness["smoothness_grad_mean"]
                                        + target_smoothness["smoothness_grad_mean"]
                                    )
                                ),
                                "spectral_lipschitz_upper": spectral_proxy["spectral_lipschitz_upper"],
                                "spectral_lipschitz_log10": spectral_proxy["spectral_lipschitz_log10"],
                                "spectral_matrix_count": spectral_proxy["spectral_matrix_count"],
                                "train_time_self_model": float(train_time_self),
                            }

                        del model
                        _cleanup_cuda()

                    oracle_cache = {}
                    for source in domains:
                        for target in domains:
                            if target == source:
                                continue

                            set_seed(seed)
                            config = _prepare_model_config(
                                _build_train_config(
                                    model_name=model_name,
                                    dataset=dataset,
                                    source=source,
                                    target=target,
                                    num_layers=num_layers,
                                    seed=seed,
                                    device=device,
                                    epochs=epochs,
                                    verbose=verbose,
                                ),
                                metadata["num_features"],
                                metadata["num_classes"],
                            )

                            model = build_model(config, from_pygda=False)
                            train_start = time.time()
                            model.fit(graphs[source], graphs[target], use_mask=True, oracle=True)
                            train_time_oracle = time.time() - train_start

                            oracle_cache[(source, target)] = {
                                "oracle_source_micro_f1": _predict_micro_f1(model, graphs[source], metrics),
                                "oracle_target_micro_f1": _predict_micro_f1(model, graphs[target], metrics),
                                "train_time_oracle_model": float(train_time_oracle),
                            }

                            del model
                            _cleanup_cuda()

                    for source in domains:
                        for target in domains:
                            if target == source:
                                continue

                            source_stats = self_cache[source]
                            target_stats = self_cache[target]
                            pair_stats = pair_cache[(source, target)]
                            oracle_stats = oracle_cache[(source, target)]

                            mmd_shift = pair_stats["source_model_domain_mmd"]
                            w1_shift = pair_stats["source_model_domain_w1"]
                            structural_mmd_shift = _mmd_value(
                                structural_cache[source],
                                structural_cache[target],
                            )
                            structural_w1_shift = _w1_value(
                                structural_cache[source],
                                structural_cache[target],
                                max_samples=w1_max_samples,
                            )

                            row = {
                                "dataset": dataset,
                                "model": model_name,
                                "num_layers": num_layers,
                                "seed": seed,
                                "source": source,
                                "target": target,
                                "source_micro_f1": source_stats["self_micro_f1"],
                                "target_micro_f1": pair_stats["target_micro_f1"],
                                "target_self_micro_f1": target_stats["self_micro_f1"],
                                "source_random_micro_f1": source_stats["random_micro_f1"],
                                "target_random_micro_f1": target_stats["random_micro_f1"],
                                "mmd_shift": mmd_shift,
                                "w1_shift": w1_shift,
                                "structural_mmd_shift": structural_mmd_shift,
                                "structural_w1_shift": structural_w1_shift,
                                "source_model_domain_mmd": pair_stats["source_model_domain_mmd"],
                                "source_model_domain_w1": pair_stats["source_model_domain_w1"],
                                "smoothness_source_grad_max": pair_stats["smoothness_source_grad_max"],
                                "smoothness_source_grad_mean": pair_stats["smoothness_source_grad_mean"],
                                "smoothness_source_nodes_evaluated": pair_stats["smoothness_source_nodes_evaluated"],
                                "smoothness_target_grad_max": pair_stats["smoothness_target_grad_max"],
                                "smoothness_target_grad_mean": pair_stats["smoothness_target_grad_mean"],
                                "smoothness_target_nodes_evaluated": pair_stats["smoothness_target_nodes_evaluated"],
                                "smoothness_grad_max": pair_stats["smoothness_grad_max"],
                                "smoothness_grad_mean_avg": pair_stats["smoothness_grad_mean_avg"],
                                "spectral_lipschitz_upper": pair_stats["spectral_lipschitz_upper"],
                                "spectral_lipschitz_log10": pair_stats["spectral_lipschitz_log10"],
                                "spectral_matrix_count": pair_stats["spectral_matrix_count"],
                                "oracle_source_micro_f1": oracle_stats["oracle_source_micro_f1"],
                                "oracle_target_micro_f1": oracle_stats["oracle_target_micro_f1"],
                                "train_time_self_model": pair_stats["train_time_self_model"],
                                "train_time_oracle_model": oracle_stats["train_time_oracle_model"],
                            }
                            row["source_gain_above_random"] = (
                                row["source_micro_f1"] - row["source_random_micro_f1"]
                            )
                            row["target_gain_above_random"] = (
                                row["target_self_micro_f1"] - row["target_random_micro_f1"]
                            )
                            row["transfer_gap"] = row["source_micro_f1"] - row["target_micro_f1"]
                            row["oracle_headroom"] = (
                                row["oracle_target_micro_f1"] - row["target_micro_f1"]
                            )

                            rows.append(row)
                            pair_idx += 1
                            print(
                                f"[{pair_idx}/{total_pairs}] {dataset} {model_name} L={num_layers} seed={seed} "
                                f"{source}->{target} src={row['source_micro_f1']:.4f} "
                                f"tgt={row['target_micro_f1']:.4f} "
                                f"w1={row['w1_shift']:.4f} "
                                f"smooth={row['smoothness_grad_max']:.4f} "
                                f"oracle={row['oracle_target_micro_f1']:.4f}"
                            )

                    _cleanup_cuda()

    raw_df = pd.DataFrame(rows)
    finalize_outputs(raw_df, out_dir)
    print(f"results_dir=\n\n{out_dir}")
    return out_dir


if __name__ == "__main__":
    run_experiment()
