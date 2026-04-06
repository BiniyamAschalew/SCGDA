"""
We compare different models in terms of the main transfer-bound components:

E_tgt <= E_src + d_HΔH(src, tgt) + λ

This runner evaluates, for each model and transfer pair:

1. Base case: train on (Xs, Ys, Xs) so the model never sees the true target domain.
2. Normal GDA: train on (Xs, Ys, Xt).
3. Oracle: train on (Xs, Ys, Xt, Yt).

All reported model performance metrics are computed on `val_mask`.
Domain divergence is measured with MMD over bottleneck features.
Smoothness is measured with an input-gradient proxy on validation nodes.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import time

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import pandas as pd
import torch

torch.set_num_threads(4)

from Learn.Clean_SCGDA.data.build_dataset import build_dataset
from Learn.Clean_SCGDA.models.build_model import build_model
from Learn.Clean_SCGDA.utils.config_utils import build_config
from Learn.Clean_SCGDA.utils.expt_utils import set_seed
from Learn.Clean_SCGDA.utils.train_utils.metrics import BaseMetric
from Learn.Clean_SCGDA.utils.train_utils.mmd import MMD

from Learn.Clean_SCGDA.experiments.analysis.bound_importance.log_plot import (
    finalize_outputs,
    make_output_dir,
)


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _predict_model(model, model_name: str, data, *, source_domain: bool, use_mask: bool):
    model_name = model_name.lower()
    if model_name == "a2gnn":
        return model.predict(data, source=source_domain, use_mask=use_mask)
    return model.predict(data, use_mask=use_mask)


def _feature_mask(data, use_mask: bool, mask_name: str = "val_mask"):
    if not use_mask:
        return torch.ones_like(data.y, dtype=torch.bool, device=data.y.device)
    mask = getattr(data, mask_name, None)
    if mask is None:
        return torch.ones_like(data.y, dtype=torch.bool, device=data.y.device)
    return mask.bool()


def _extract_bottleneck_features(model, model_name: str, data, *, source_domain: bool, use_mask: bool):
    model_name = model_name.lower()
    batch = getattr(data, "batch", None)

    with torch.no_grad():
        if model_name == "gnn":
            features = model.gnn.feat_bottleneck(data.x, data.edge_index, batch=batch)
        elif model_name == "simgda":
            features = model.simgda.feat_bottleneck(data.x, data.edge_index, batch=batch)
        elif model_name == "a2gnn":
            prop_nums = model.s_pnums if source_domain else model.t_pnums
            features = model.a2gnn.feat_bottleneck(data.x, data.edge_index, batch, prop_nums=prop_nums)
        elif model_name == "adagcn":
            features = model.adagcn(data)
        else:
            raise ValueError(f"Unsupported model for feature extraction: {model_name}")

    mask = _feature_mask(data, use_mask=use_mask, mask_name="val_mask")
    return features[mask]


def _clone_with_x(data, x):
    cloned = data.clone().to(x.device)
    cloned.x = x
    if getattr(cloned, "edge_index", None) is not None:
        cloned.edge_index = cloned.edge_index.contiguous()
    return cloned


def _forward_logits_for_grad(model, model_name: str, data, *, source_domain: bool, x):
    model_name = model_name.lower()
    if model_name == "gnn":
        return model.gnn(x, data.edge_index)
    if model_name == "simgda":
        return model.simgda(x, data.edge_index)
    if model_name == "a2gnn":
        prop_nums = model.s_pnums if source_domain else model.t_pnums
        data_with_x = _clone_with_x(data, x)
        return model.a2gnn(data_with_x, prop_nums)
    if model_name == "adagcn":
        data_with_x = _clone_with_x(data, x)
        encoded = model.adagcn(data_with_x)
        return model.adagcn.cls_model(encoded)
    raise ValueError(f"Unsupported model for gradient smoothness: {model_name}")


def _smoothness_proxy(
    model,
    model_name: str,
    data,
    *,
    source_domain: bool,
    use_mask: bool,
    max_nodes: int,
):
    mask = _feature_mask(data, use_mask=use_mask, mask_name="val_mask")
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
    logits = _forward_logits_for_grad(
        model,
        model_name,
        data,
        source_domain=source_domain,
        x=x,
    )
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


def _micro_error(metrics: dict) -> float:
    return 1.0 - float(metrics.get("micro_f1", 0.0))


def _normalized_gain(base_value: float, normal_value: float, oracle_value: float) -> float:
    denom = float(oracle_value) - float(base_value)
    if abs(denom) < 1e-12:
        return float("nan")
    return (float(normal_value) - float(base_value)) / denom


def _evaluate_variant(
    model,
    model_name: str,
    train_source_data,
    train_target_data,
    eval_source_data,
    eval_target_data,
    metrics: BaseMetric,
    *,
    use_mask: bool,
    oracle: bool,
    smoothness_max_nodes: int,
):
    start = time.time()
    model.fit(train_source_data, train_target_data, use_mask=use_mask, oracle=oracle)
    train_time = time.time() - start

    source_logits, source_labels = _predict_model(
        model,
        model_name,
        eval_source_data,
        source_domain=True,
        use_mask=use_mask,
    )
    target_logits, target_labels = _predict_model(
        model,
        model_name,
        eval_target_data,
        source_domain=False,
        use_mask=use_mask,
    )

    source_metrics = metrics(source_logits, source_labels)
    target_metrics = metrics(target_logits, target_labels)
    source_metrics["error"] = _micro_error(source_metrics)
    target_metrics["error"] = _micro_error(target_metrics)

    source_features = _extract_bottleneck_features(
        model,
        model_name,
        eval_source_data,
        source_domain=True,
        use_mask=use_mask,
    )
    target_features = _extract_bottleneck_features(
        model,
        model_name,
        eval_target_data,
        source_domain=False,
        use_mask=use_mask,
    )

    divergence = float(MMD(source_features, target_features).detach().cpu().item())
    smoothness = _smoothness_proxy(
        model,
        model_name,
        eval_target_data,
        source_domain=False,
        use_mask=use_mask,
        max_nodes=smoothness_max_nodes,
    )

    return {
        "source_metrics": source_metrics,
        "target_metrics": target_metrics,
        "domain_mmd": divergence,
        "smoothness": smoothness,
        "train_time": float(train_time),
    }


def _build_pair_config(
    *,
    model_name: str,
    dataset: str,
    source: str,
    target: str,
    seed: int,
    device: str,
    use_tuned: int,
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
            "project": "bound_importance",
            "verbose": verbose,
        },
        "model": {
            "epochs": epochs,
            "use_mask": True,
        },
    }

    try:
        config = build_config(config_setup, update_config, use_tuned=use_tuned)
    except FileNotFoundError:
        config = build_config(config_setup, update_config, use_tuned=0)

    config["model"]["epochs"] = epochs
    config["model"]["use_mask"] = True
    return config


def run_expt(
    config: dict,
    *,
    model_name: str,
    dataset: str,
    source: str,
    target: str,
    seed: int,
    from_pygda: bool,
    smoothness_max_nodes: int,
):
    set_seed(config["expt"]["seed"])
    device = config["expt"]["device"]

    source_dataset, target_dataset = build_dataset(config)
    source_data = source_dataset[0].to(device)
    target_data = target_dataset[0].to(device)

    if source_data.edge_index is not None:
        source_data.edge_index = source_data.edge_index.contiguous()
    if target_data.edge_index is not None:
        target_data.edge_index = target_data.edge_index.contiguous()

    num_features = int(source_data.x.shape[1])
    num_classes = int(torch.unique(torch.cat([source_data.y, target_data.y])).numel())
    config["model"]["in_dim"] = num_features
    config["model"]["num_classes"] = num_classes

    metrics = BaseMetric(config)
    base_model = build_model(config, from_pygda=from_pygda)
    normal_model = build_model(config, from_pygda=from_pygda)
    oracle_model = build_model(config, from_pygda=from_pygda)

    base = _evaluate_variant(
        base_model,
        model_name,
        source_data,
        source_data,
        source_data,
        target_data,
        metrics,
        use_mask=True,
        oracle=False,
        smoothness_max_nodes=smoothness_max_nodes,
    )
    normal = _evaluate_variant(
        normal_model,
        model_name,
        source_data,
        target_data,
        source_data,
        target_data,
        metrics,
        use_mask=True,
        oracle=False,
        smoothness_max_nodes=smoothness_max_nodes,
    )
    oracle = _evaluate_variant(
        oracle_model,
        model_name,
        source_data,
        target_data,
        source_data,
        target_data,
        metrics,
        use_mask=True,
        oracle=True,
        smoothness_max_nodes=smoothness_max_nodes,
    )

    row = {
        "dataset": dataset,
        "source": source,
        "target": target,
        "pair": f"{source}->{target}",
        "model": model_name,
        "seed": seed,
        "base_source_micro_f1": base["source_metrics"]["micro_f1"],
        "base_source_macro_f1": base["source_metrics"]["macro_f1"],
        "base_source_error": base["source_metrics"]["error"],
        "base_target_micro_f1": base["target_metrics"]["micro_f1"],
        "base_target_macro_f1": base["target_metrics"]["macro_f1"],
        "base_target_error": base["target_metrics"]["error"],
        "base_domain_mmd": base["domain_mmd"],
        "base_smoothness_grad_max": base["smoothness"]["smoothness_grad_max"],
        "base_smoothness_grad_mean": base["smoothness"]["smoothness_grad_mean"],
        "base_smoothness_nodes_evaluated": base["smoothness"]["smoothness_nodes_evaluated"],
        "base_train_time": base["train_time"],
        "normal_source_micro_f1": normal["source_metrics"]["micro_f1"],
        "normal_source_macro_f1": normal["source_metrics"]["macro_f1"],
        "normal_source_error": normal["source_metrics"]["error"],
        "normal_target_micro_f1": normal["target_metrics"]["micro_f1"],
        "normal_target_macro_f1": normal["target_metrics"]["macro_f1"],
        "normal_target_error": normal["target_metrics"]["error"],
        "normal_domain_mmd": normal["domain_mmd"],
        "normal_smoothness_grad_max": normal["smoothness"]["smoothness_grad_max"],
        "normal_smoothness_grad_mean": normal["smoothness"]["smoothness_grad_mean"],
        "normal_smoothness_nodes_evaluated": normal["smoothness"]["smoothness_nodes_evaluated"],
        "normal_train_time": normal["train_time"],
        "oracle_source_micro_f1": oracle["source_metrics"]["micro_f1"],
        "oracle_source_macro_f1": oracle["source_metrics"]["macro_f1"],
        "oracle_source_error": oracle["source_metrics"]["error"],
        "oracle_target_micro_f1": oracle["target_metrics"]["micro_f1"],
        "oracle_target_macro_f1": oracle["target_metrics"]["macro_f1"],
        "oracle_target_error": oracle["target_metrics"]["error"],
        "oracle_domain_mmd": oracle["domain_mmd"],
        "oracle_smoothness_grad_max": oracle["smoothness"]["smoothness_grad_max"],
        "oracle_smoothness_grad_mean": oracle["smoothness"]["smoothness_grad_mean"],
        "oracle_smoothness_nodes_evaluated": oracle["smoothness"]["smoothness_nodes_evaluated"],
        "oracle_train_time": oracle["train_time"],
    }
    row["oracle_joint_error"] = 0.5 * (
        row["oracle_source_error"] + row["oracle_target_error"]
    )
    row["normalized_da_gain"] = _normalized_gain(
        row["base_target_micro_f1"],
        row["normal_target_micro_f1"],
        row["oracle_target_micro_f1"],
    )
    row["absolute_da_gain"] = row["normal_target_micro_f1"] - row["base_target_micro_f1"]
    row["oracle_headroom"] = row["oracle_target_micro_f1"] - row["base_target_micro_f1"]
    row["bound_rhs_proxy"] = (
        row["normal_source_error"] + row["normal_domain_mmd"] + row["oracle_joint_error"]
    )
    row["bound_gap_proxy"] = row["bound_rhs_proxy"] - row["normal_target_error"]
    row["oracle_target_gap"] = (
        row["oracle_target_micro_f1"] - row["normal_target_micro_f1"]
    )
    return row


def main():
    from_pygda = False
    use_tuned = 2
    epochs = 200
    device = "cuda:0"
    verbose = 0
    smoothness_max_nodes = 128
    seeds = (2025,)
    models = ["gnn", "simgda", "a2gnn", "adagcn"]
    transfer_settings = {
        "citation": [
            ("ACMv9", "Citationv1"),
            ("ACMv9", "DBLPv7"),
        ],
        "airport": [
            ("BRAZIL", "EUROPE"),
            ("BRAZIL", "USA"),
        ],
        "blog": [
            ("Blog1", "Blog2"),
            ("Blog2", "Blog1"),
        ],
    }

    out_dir = make_output_dir(
        out_root="./__saved__/analysis/bound_importance",
        run_name=f"bound_importance_{time.strftime('%m%d_%H%M%S')}",
    )
    config_payload = {
        "from_pygda": from_pygda,
        "use_tuned": use_tuned,
        "epochs": epochs,
        "device": device,
        "verbose": verbose,
        "smoothness_max_nodes": smoothness_max_nodes,
        "seeds": list(seeds),
        "models": models,
        "transfer_settings": transfer_settings,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    rows = []
    total_runs = sum(len(pairs) for pairs in transfer_settings.values()) * len(models) * len(seeds)
    run_idx = 0

    for seed in seeds:
        for dataset, pairs in transfer_settings.items():
            for source, target in pairs:
                for model_name in models:
                    config = _build_pair_config(
                        model_name=model_name,
                        dataset=dataset,
                        source=source,
                        target=target,
                        seed=seed,
                        device=device,
                        use_tuned=use_tuned,
                        epochs=epochs,
                        verbose=verbose,
                    )
                    row = run_expt(
                        config,
                        model_name=model_name,
                        dataset=dataset,
                        source=source,
                        target=target,
                        seed=seed,
                        from_pygda=from_pygda,
                        smoothness_max_nodes=smoothness_max_nodes,
                    )
                    rows.append(row)
                    run_idx += 1
                    print(
                        f"[{run_idx}/{total_runs}] "
                        f"{dataset} {source}->{target} model={model_name} "
                        f"base={row['base_target_micro_f1']:.4f} "
                        f"target={row['normal_target_micro_f1']:.4f} "
                        f"oracle={row['oracle_target_micro_f1']:.4f}"
                    )
                    _cleanup_cuda()

    raw_df = pd.DataFrame(rows)
    finalize_outputs(raw_df, out_dir)
    print(f"results_dir={out_dir}")


if __name__ == "__main__":
    main()
