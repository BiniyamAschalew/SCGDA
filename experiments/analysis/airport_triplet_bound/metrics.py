from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F

from Learn.Clean_SCGDA.utils.filter_utils import mmd_rbf


def mask_tensor(data, name: str) -> torch.Tensor:
    mask = getattr(data, name, None)
    if mask is None:
        return torch.ones(data.y.size(0), dtype=torch.bool, device=data.y.device)
    return mask.bool()


def _maybe_subsample(x: torch.Tensor, max_samples: int) -> torch.Tensor:
    if max_samples <= 0 or x.size(0) <= max_samples:
        return x
    idx = torch.randperm(x.size(0), device=x.device)[:max_samples]
    return x[idx]


def masked_cross_entropy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits[mask], y[mask])


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    if logits.numel() == 0 or labels.numel() == 0:
        return {
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "error": 1.0,
        }
    pred = logits.argmax(dim=1).detach().cpu().numpy()
    true = labels.detach().cpu().numpy()
    micro = float(f1_score(true, pred, average="micro", zero_division=0))
    macro = float(f1_score(true, pred, average="macro", zero_division=0))
    return {
        "micro_f1": micro,
        "macro_f1": macro,
        "error": 1.0 - micro,
    }


def masked_classification_metrics(logits: torch.Tensor, data, mask_name: str = "val_mask") -> dict[str, float]:
    mask = mask_tensor(data, mask_name)
    masked_logits = logits[mask]
    masked_labels = data.y[mask]
    metrics = classification_metrics(masked_logits, masked_labels)
    metrics["count"] = int(mask.sum().item())
    return metrics


def pairwise_mmds(
    features_by_domain: dict[str, torch.Tensor],
    *,
    max_samples: int = 0,
) -> dict[str, float]:
    pairs = {
        "source_target": ("source", "target"),
        "source_reference": ("source", "reference"),
        "target_reference": ("target", "reference"),
    }
    results = {}
    for key, (left_name, right_name) in pairs.items():
        left = _maybe_subsample(features_by_domain[left_name], max_samples)
        right = _maybe_subsample(features_by_domain[right_name], max_samples)
        if left.size(0) == 0 or right.size(0) == 0:
            results[f"val_mmd_{key}"] = 0.0
            continue
        results[f"val_mmd_{key}"] = float(mmd_rbf(left, right).detach().cpu().item())
    return results


def masked_mmd(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    max_samples: int = 0,
) -> torch.Tensor:
    left = _maybe_subsample(source_features[source_mask], max_samples)
    right = _maybe_subsample(target_features[target_mask], max_samples)
    if left.size(0) == 0 or right.size(0) == 0:
        return torch.zeros((), device=source_features.device, dtype=source_features.dtype)
    return mmd_rbf(left, right)


def gradient_lipschitz_proxy(
    model,
    data,
    *,
    mask_name: str = "val_mask",
    max_nodes: int = 0,
) -> dict[str, float]:
    model.eval()
    mask = mask_tensor(data, mask_name)
    node_idx = mask.nonzero(as_tuple=False).view(-1)
    if max_nodes > 0 and node_idx.numel() > max_nodes:
        node_idx = node_idx[:max_nodes]

    x = data.x.detach().clone().requires_grad_(True)
    logits, _ = model(data, x=x)
    pred = logits.detach().argmax(dim=1)

    grad_max_values = []
    grad_self_values = []
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
        grad_self_values.append(float(grad_norms[idx].detach().cpu().item()))

    if not grad_max_values:
        return {
            "grad_input_max": 0.0,
            "grad_input_mean": 0.0,
            "grad_self_max": 0.0,
            "grad_self_mean": 0.0,
            "grad_nodes_evaluated": 0,
        }

    return {
        "grad_input_max": float(np.max(grad_max_values)),
        "grad_input_mean": float(np.mean(grad_max_values)),
        "grad_self_max": float(np.max(grad_self_values)),
        "grad_self_mean": float(np.mean(grad_self_values)),
        "grad_nodes_evaluated": int(len(grad_max_values)),
    }
