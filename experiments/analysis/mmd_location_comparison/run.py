from __future__ import annotations

import gc
import json
from pathlib import Path
import sys
import time
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch_geometric.transforms import OneHotDegree

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Learn.Clean_SCGDA.data.build_dataset import get_dataset, get_max_degree
from Learn.Clean_SCGDA.models.baselines.gnn.gnn_base import GNNBase
from Learn.Clean_SCGDA.utils.config_utils import load_config
from Learn.Clean_SCGDA.utils.expt_utils import set_seed
from Learn.Clean_SCGDA.utils.train_utils.metrics import BaseMetric
from Learn.Clean_SCGDA.utils.train_utils.mmd import MMD
from Learn.Clean_SCGDA.utils.train_utils.sinkhorn import get_Sinkhorn

from .log_plot import finalize_outputs, make_output_dir


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message="The `pre_transform` argument differs from the one used in the pre-processed version of this dataset.*",
    category=UserWarning,
)


VARIANT_SPECS = (
    {"key": "encoder_mmd", "label": "Encoder MMD", "space": "encoder"},
    {"key": "logit_mmd", "label": "Logit MMD", "space": "logit"},
)


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


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
        "model": {"name": "simgda"},
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


def _mask(data, mask_name: str):
    mask = getattr(data, mask_name, None)
    if mask is None:
        return torch.ones_like(data.y, dtype=torch.bool, device=data.y.device)
    return mask.bool()


def _build_model_config(
    *,
    num_features: int,
    num_classes: int,
    epochs: int,
    device: str,
    verbose: int,
):
    model_cfg = OmegaConf.to_container(
        load_config("./configs/model_configs/baselines/simgda.yaml"),
        resolve=True,
    )
    model_cfg["name"] = "SimGDA"
    model_cfg["in_dim"] = int(num_features)
    model_cfg["num_classes"] = int(num_classes)
    model_cfg["epochs"] = int(epochs)
    model_cfg["use_mask"] = True
    return {
        "model": model_cfg,
        "expt": {
            "metrics": ["micro_f1"],
            "verbose": int(verbose),
            "device": device,
        },
    }


def _extract_eval_spaces(model: GNNBase, data):
    mask = _mask(data, "val_mask")
    model.eval()
    with torch.no_grad():
        encoder = model.feat_bottleneck(data.x, data.edge_index)
        logits = model.feat_classifier(encoder, data.edge_index)
    return encoder[mask], logits[mask]


def _predict_micro_f1(model: GNNBase, data, metrics: BaseMetric) -> float:
    mask = _mask(data, "val_mask")
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    return float(metrics(logits[mask], data.y[mask]).get("micro_f1", 0.0))


def _safe_mmd(source_repr, target_repr):
    if int(source_repr.size(0)) == 0 or int(target_repr.size(0)) == 0:
        return source_repr.new_tensor(0.0)
    return MMD(source_repr, target_repr)


def _safe_w1(source_repr, target_repr):
    if int(source_repr.size(0)) == 0 or int(target_repr.size(0)) == 0:
        return 0.0
    return float(
        get_Sinkhorn(
            source_repr,
            target_repr,
            p=1,
            blur=0.05,
            scaling=0.9,
            debias=True,
            backend="tensorized",
        ).detach().cpu().item()
    )


def _train_variant(
    *,
    source_data,
    target_data,
    dataset: str,
    seed: int,
    source: str,
    target: str,
    variant: dict[str, str],
    model_cfg: dict,
):
    set_seed(seed)
    metrics = BaseMetric(model_cfg)
    model = GNNBase(model_cfg).to(model_cfg["expt"]["device"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_cfg["model"]["lr"]),
        weight_decay=float(model_cfg["model"]["weight_decay"]),
    )

    source_train_mask = _mask(source_data, "train_mask")
    target_train_mask = _mask(target_data, "train_mask")
    history_rows = []
    final_ce_loss = 0.0
    final_train_mmd = 0.0

    start = time.time()
    for epoch in range(int(model_cfg["model"]["epochs"])):
        model.train()
        source_encoder = model.feat_bottleneck(source_data.x, source_data.edge_index)
        target_encoder = model.feat_bottleneck(target_data.x, target_data.edge_index)
        source_logits_raw = model.feat_classifier(source_encoder, source_data.edge_index)
        target_logits_raw = model.feat_classifier(target_encoder, target_data.edge_index)
        source_log_probs = F.log_softmax(source_logits_raw, dim=1)

        ce_loss = F.nll_loss(source_log_probs[source_train_mask], source_data.y[source_train_mask])
        if variant["space"] == "encoder":
            mmd_source = source_encoder[source_train_mask]
            mmd_target = target_encoder[target_train_mask]
        else:
            mmd_source = source_logits_raw[source_train_mask]
            mmd_target = target_logits_raw[target_train_mask]
        train_mmd = _safe_mmd(mmd_source, mmd_target)
        total_loss = ce_loss + float(model_cfg["model"]["mmd_weight"]) * train_mmd

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        final_ce_loss = float(ce_loss.detach().cpu().item())
        final_train_mmd = float(train_mmd.detach().cpu().item())
        history_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "source": source,
                "target": target,
                "variant_key": variant["key"],
                "variant_label": variant["label"],
                "epoch": epoch + 1,
                "total_loss": float(total_loss.detach().cpu().item()),
                "ce_loss": final_ce_loss,
                "train_mmd": final_train_mmd,
                "source_val_micro_f1": _predict_micro_f1(model, source_data, metrics),
                "target_val_micro_f1": _predict_micro_f1(model, target_data, metrics),
            }
        )

    train_time = time.time() - start

    source_micro_f1 = _predict_micro_f1(model, source_data, metrics)
    target_micro_f1 = _predict_micro_f1(model, target_data, metrics)
    source_encoder_val, source_logits_val = _extract_eval_spaces(model, source_data)
    target_encoder_val, target_logits_val = _extract_eval_spaces(model, target_data)

    row = {
        "dataset": dataset,
        "seed": seed,
        "source": source,
        "target": target,
        "variant_key": variant["key"],
        "variant_label": variant["label"],
        "source_micro_f1": source_micro_f1,
        "target_micro_f1": target_micro_f1,
        "source_target_gap": source_micro_f1 - target_micro_f1,
        "encoder_val_mmd": float(_safe_mmd(source_encoder_val, target_encoder_val).detach().cpu().item()),
        "logit_val_mmd": float(_safe_mmd(source_logits_val, target_logits_val).detach().cpu().item()),
        "encoder_val_w1": _safe_w1(source_encoder_val, target_encoder_val),
        "logit_val_w1": _safe_w1(source_logits_val, target_logits_val),
        "final_ce_loss": final_ce_loss,
        "final_train_mmd": final_train_mmd,
        "train_time": float(train_time),
    }

    del model
    _cleanup_cuda()
    return row, history_rows


def run_experiment(
    *,
    datasets: tuple[str, ...],
    seeds: tuple[int, ...],
    device: str,
    epochs: int,
    out_root: str,
    run_name: str,
    verbose: int,
):
    out_dir = make_output_dir(out_root=out_root, run_name=run_name)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "datasets": list(datasets),
                "seeds": list(seeds),
                "device": device,
                "epochs": epochs,
                "variants": list(VARIANT_SPECS),
            },
            f,
            indent=2,
        )

    rows = []
    history_rows = []
    total_runs = 0
    for dataset in datasets:
        domains = list(load_config(f"./configs/data_configs/{dataset}.yaml")["domains"])
        total_runs += len(seeds) * len(VARIANT_SPECS) * len(
            [(src, tgt) for src in domains for tgt in domains if src != tgt]
        )

    run_idx = 0
    for dataset in datasets:
        graphs, metadata = _load_dataset_graphs(dataset, device=device)
        domains = metadata["domains"]
        ordered_pairs = [(src, tgt) for src in domains for tgt in domains if src != tgt]
        model_cfg = _build_model_config(
            num_features=metadata["num_features"],
            num_classes=metadata["num_classes"],
            epochs=epochs,
            device=device,
            verbose=0,
        )

        for seed in seeds:
            for source, target in ordered_pairs:
                for variant in VARIANT_SPECS:
                    row, cur_history = _train_variant(
                        source_data=graphs[source],
                        target_data=graphs[target],
                        dataset=dataset,
                        seed=seed,
                        source=source,
                        target=target,
                        variant=variant,
                        model_cfg=model_cfg,
                    )
                    rows.append(row)
                    history_rows.extend(cur_history)
                    run_idx += 1
                    if verbose:
                        print(
                            f"[{run_idx}/{total_runs}] dataset={dataset} seed={seed} "
                            f"{source}->{target} variant={variant['label']} "
                            f"src_micro_f1={row['source_micro_f1']:.4f} "
                            f"target_micro_f1={row['target_micro_f1']:.4f}"
                        )

    raw_df = pd.DataFrame(rows)
    history_df = pd.DataFrame(history_rows)
    finalize_outputs(raw_df, history_df, out_dir)
    print(f"results_dir=\n\n{out_dir}")
    return out_dir

