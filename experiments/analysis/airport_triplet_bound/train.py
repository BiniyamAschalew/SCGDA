from __future__ import annotations

from dataclasses import dataclass
import random
import time

import numpy as np
import torch

from .config import CaseSpec, ExperimentConfig, TripletSpec, model_config
from .metrics import (
    gradient_lipschitz_proxy,
    mask_tensor,
    masked_classification_metrics,
    masked_cross_entropy,
    masked_mmd,
    pairwise_mmds,
)
from .model import TripletSimGDAModel


@dataclass
class CaseRunOutput:
    row: dict
    history: list[dict]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def _weighted_supervised_loss(
    case: CaseSpec,
    source_ce: torch.Tensor,
    target_ce: torch.Tensor,
    source_count: int,
    target_count: int,
) -> torch.Tensor:
    total_count = (
        case.source_ce_weight * max(int(source_count), 0)
        + case.target_ce_weight * max(int(target_count), 0)
    )
    if total_count <= 0:
        return torch.zeros_like(source_ce)
    weighted_sum = (
        case.source_ce_weight * source_ce * max(int(source_count), 0)
        + case.target_ce_weight * target_ce * max(int(target_count), 0)
    )
    return weighted_sum / total_count


def train_case(
    *,
    exp_cfg: ExperimentConfig,
    case: CaseSpec,
    triplet: TripletSpec,
    graphs: dict[str, object],
    metadata: dict[str, int],
    case_hparams: dict,
    seed: int,
) -> CaseRunOutput:
    seed_everything(seed)

    config = model_config(
        in_dim=metadata["num_features"],
        num_classes=metadata["num_classes"],
        hid_dim=case_hparams["hid_dim"],
        num_layers=case_hparams["num_layers"],
        dropout_ratio=case_hparams["dropout_ratio"],
        gnn=case_hparams["gnn"],
        activation=case_hparams["activation"],
    )
    device = exp_cfg.resolved_device()
    model = TripletSimGDAModel(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=case_hparams["lr"],
        weight_decay=case_hparams["weight_decay"],
    )
    scheduler = None
    if str(case_hparams.get("scheduler", "")).lower() == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(case_hparams.get("step_size", 10)),
            gamma=float(case_hparams.get("gamma", 0.1)),
        )

    source_data = graphs["source"]
    target_data = graphs["target"]
    reference_data = graphs["reference"]
    source_train_mask = mask_tensor(source_data, "train_mask")
    target_train_mask = mask_tensor(target_data, "train_mask")
    source_val_mask = mask_tensor(source_data, "val_mask")
    target_val_mask = mask_tensor(target_data, "val_mask")
    source_train_count = int(source_train_mask.sum().item())
    target_train_count = int(target_train_mask.sum().item())

    history = []
    start = time.time()
    for epoch in range(exp_cfg.epochs):
        model.train()
        optimizer.zero_grad()

        source_logits, source_features = model(source_data)
        target_logits, target_features = model(target_data)

        source_ce = masked_cross_entropy(source_logits, source_data.y, source_train_mask)
        target_ce = masked_cross_entropy(target_logits, target_data.y, target_train_mask)
        train_mmd_tensor = masked_mmd(
            source_features,
            target_features,
            source_train_mask,
            target_train_mask,
            max_samples=exp_cfg.metric_sample_size,
        )
        st_train_mmd = float(train_mmd_tensor.detach().cpu().item())

        total_loss = _weighted_supervised_loss(
            case,
            source_ce,
            target_ce,
            source_train_count,
            target_train_count,
        ) + case.mmd_weight * train_mmd_tensor
        total_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        history.append(
            {
                "seed": seed,
                "triplet_key": triplet.key,
                "triplet_label": triplet.label,
                "source": triplet.source,
                "target": triplet.target,
                "reference": triplet.reference,
                "case_key": case.key,
                "case_label": case.label,
                "epoch": epoch + 1,
                "total_loss": float(total_loss.detach().cpu().item()),
                "source_ce": float(source_ce.detach().cpu().item()),
                "target_ce": float(target_ce.detach().cpu().item()),
                "train_st_mmd": float(st_train_mmd),
            }
        )

    train_time = time.time() - start

    model.eval()
    with torch.no_grad():
        source_logits, source_features = model(source_data)
        target_logits, target_features = model(target_data)
        reference_logits, reference_features = model(reference_data)

    domain_payload = {}
    val_features_by_domain = {}
    val_metrics = {
        "source": masked_classification_metrics(source_logits, source_data, "val_mask"),
        "target": masked_classification_metrics(target_logits, target_data, "val_mask"),
        "reference": masked_classification_metrics(reference_logits, reference_data, "val_mask"),
    }
    val_masks = {
        "source": mask_tensor(source_data, "val_mask"),
        "target": mask_tensor(target_data, "val_mask"),
        "reference": mask_tensor(reference_data, "val_mask"),
    }
    feature_map = {
        "source": source_features,
        "target": target_features,
        "reference": reference_features,
    }
    for domain_name, metrics in val_metrics.items():
        for metric_name, value in metrics.items():
            domain_payload[f"{domain_name}_{metric_name}"] = value
        val_features_by_domain[domain_name] = feature_map[domain_name][val_masks[domain_name]]

    pairwise_payload = pairwise_mmds(
        val_features_by_domain,
        max_samples=exp_cfg.metric_sample_size,
    )
    shift_source_train_source_val = float(
        masked_mmd(
            source_features,
            source_features,
            source_train_mask,
            source_val_mask,
            max_samples=exp_cfg.metric_sample_size,
        ).detach().cpu().item()
    )
    shift_source_train_target_val = float(
        masked_mmd(
            source_features,
            target_features,
            source_train_mask,
            target_val_mask,
            max_samples=exp_cfg.metric_sample_size,
        ).detach().cpu().item()
    )
    final_train_st_mmd = float(
        masked_mmd(
            source_features,
            target_features,
            source_train_mask,
            target_train_mask,
            max_samples=exp_cfg.metric_sample_size,
        ).detach().cpu().item()
    )

    gradient_payload = {}
    gradient_domains = {
        "source": source_data,
        "target": target_data,
        "reference": reference_data,
    }
    global_grad_input_max = 0.0
    for domain_name, domain_data in gradient_domains.items():
        grad_stats = gradient_lipschitz_proxy(
            model,
            domain_data,
            mask_name="val_mask",
            max_nodes=exp_cfg.grad_nodes_cap,
        )
        for key, value in grad_stats.items():
            gradient_payload[f"{domain_name}_{key}"] = value
        global_grad_input_max = max(global_grad_input_max, grad_stats["grad_input_max"])

    row = {
        "seed": seed,
        "triplet_key": triplet.key,
        "triplet_label": triplet.label,
        "dataset": exp_cfg.dataset.lower(),
        "source": triplet.source,
        "target": triplet.target,
        "reference": triplet.reference,
        "case_key": case.key,
        "case_label": case.label,
        "epochs": exp_cfg.epochs,
        "lr": case_hparams["lr"],
        "weight_decay": case_hparams["weight_decay"],
        "hid_dim": case_hparams["hid_dim"],
        "num_layers": case_hparams["num_layers"],
        "dropout_ratio": case_hparams["dropout_ratio"],
        "gnn": case_hparams["gnn"],
        "tuned_model_family": case_hparams.get("tuned_model_family", "manual"),
        "mmd_weight": case.mmd_weight,
        "train_st_mmd": float(final_train_st_mmd),
        "shift_mmd_source_train_source_val": float(shift_source_train_source_val),
        "shift_mmd_source_train_target_val": float(shift_source_train_target_val),
        "shift_gap_target_minus_source": float(
            shift_source_train_target_val - shift_source_train_source_val
        ),
        "train_time": float(train_time),
        "global_grad_input_max": float(global_grad_input_max),
        "max_degree": metadata["max_degree"],
        "num_features": metadata["num_features"],
        "num_classes": metadata["num_classes"],
    }
    row.update(domain_payload)
    row.update(pairwise_payload)
    row.update(gradient_payload)
    return CaseRunOutput(row=row, history=history)
