from __future__ import annotations

import gc
import json
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
from torch_geometric.transforms import OneHotDegree

torch.set_num_threads(4)

from  data.build_dataset import get_dataset, get_max_degree
from  models.build_model import build_model
from  utils.config_utils import build_config, load_config
from  utils.expt_utils import set_seed
from  utils.train_utils.metrics import BaseMetric
from  utils.train_utils.mmd import MMD

from  experiments.analysis.plain_gnn_cross_domain.log_plot import (
    finalize_outputs,
    make_output_dir,
)


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message="The `pre_transform` argument differs from the one used in the pre-processed version of this dataset.*",
    category=UserWarning,
)


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _metric_error(metrics: dict) -> float:
    return 1.0 - float(metrics.get("micro_f1", 0.0))


def _build_train_config(
    *,
    dataset: str,
    source: str,
    seed: int,
    device: str,
    epochs: int,
    verbose: int,
):
    config_setup = {
        "model": "gnn",
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
            "project": "plain_gnn_cross_domain",
            "verbose": verbose,
        },
        "model": {
            "epochs": epochs,
            "use_mask": True,
        },
    }
    config = build_config(config_setup, update_config=update_config, use_tuned=0)
    config["model"]["epochs"] = epochs
    config["model"]["use_mask"] = True
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


def _extract_val_features(model, data):
    model.gnn.eval()
    with torch.no_grad():
        features = model.gnn.feat_bottleneck(data.x, data.edge_index)
    mask = data.val_mask.bool()
    return features[mask]


def run_experiment():
    # datasets = ("citation", "airport", "blog", "mag", "twitch")
    datasets = ("citation", "airport", "blog")

    seeds = (2025, 2026, 2027, 2028, 2029)
    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    epochs = 200
    verbose = 0

    out_dir = make_output_dir(
        out_root="./__saved__/analysis/plain_gnn_cross_domain",
        run_name=f"plain_gnn_cross_domain_{time.strftime('%m%d_%H%M%S')}",
    )
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "datasets": datasets,
                "seeds": list(seeds),
                "device": device,
                "epochs": epochs,
                "verbose": verbose,
            },
            f,
            indent=2,
        )

    rows = []
    total_runs = 0
    for dataset in datasets:
        domains = list(load_config(f"./configs/data_configs/{dataset}.yaml")["domains"])
        total_runs += len(seeds) * len(domains) * max(len(domains) - 1, 0)

    run_idx = 0
    for dataset in datasets:
        graphs, metadata = _load_dataset_graphs(dataset, device=device)
        domains = metadata["domains"]
        for seed in seeds:
            for source in domains:
                set_seed(seed)
                config = _build_train_config(
                    dataset=dataset,
                    source=source,
                    seed=seed,
                    device=device,
                    epochs=epochs,
                    verbose=verbose,
                )
                config["model"]["in_dim"] = metadata["num_features"]
                config["model"]["num_classes"] = metadata["num_classes"]

                model = build_model(config, from_pygda=False)
                source_data = graphs[source]

                start = time.time()
                model.fit(source_data, source_data, use_mask=True, oracle=False)
                train_time = time.time() - start

                metrics = BaseMetric(config)
                source_logits, source_labels = model.predict(source_data, use_mask=True)
                source_metrics = metrics(source_logits, source_labels)
                source_metrics["error"] = _metric_error(source_metrics)
                source_features = _extract_val_features(model, source_data)

                for target in domains:
                    if target == source:
                        continue

                    target_data = graphs[target]
                    target_logits, target_labels = model.predict(target_data, use_mask=True)
                    target_metrics = metrics(target_logits, target_labels)
                    target_metrics["error"] = _metric_error(target_metrics)
                    target_features = _extract_val_features(model, target_data)

                    row = {
                        "dataset": dataset,
                        "seed": seed,
                        "source": source,
                        "target": target,
                        "source_micro_f1": source_metrics["micro_f1"],
                        "source_macro_f1": source_metrics["macro_f1"],
                        "source_error": source_metrics["error"],
                        "target_micro_f1": target_metrics["micro_f1"],
                        "target_macro_f1": target_metrics["macro_f1"],
                        "target_error": target_metrics["error"],
                        "feature_mmd": float(MMD(source_features, target_features).detach().cpu().item()),
                        "train_time": float(train_time),
                    }
                    row["generalization_gap"] = row["source_micro_f1"] - row["target_micro_f1"]
                    rows.append(row)
                    run_idx += 1
                    print(
                        f"[{run_idx}/{total_runs}] {dataset} "
                        f"{source}->{target} src={row['source_micro_f1']:.4f} "
                        f"tgt={row['target_micro_f1']:.4f} mmd={row['feature_mmd']:.4f}"
                    )

                del model
                _cleanup_cuda()

    raw_df = pd.DataFrame(rows)
    finalize_outputs(raw_df, out_dir)
    print(f"results_dir={out_dir}")
    return out_dir


if __name__ == "__main__":
    run_experiment()
