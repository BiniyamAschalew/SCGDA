from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from utils.expt_utils import set_seed
from utils.ablation_utils.propagation import (
    WeightedPropagation,
    propagate_fixed_cheb_layers,
    propagate_weighted_layers,
)


class TransferMLP(nn.Module):
    """Small MLP for source-train / target-test transfer evaluation."""

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers not in (2, 3):
            raise ValueError(f"num_layers must be 2 or 3, got {num_layers}")

        layers = [nn.Linear(in_dim, hid_dim), nn.ReLU()]
        if num_layers == 3:
            layers += [nn.Dropout(float(dropout)), nn.Linear(hid_dim, hid_dim), nn.ReLU()]
        layers += [nn.Dropout(float(dropout)), nn.Linear(hid_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_source_train_mask(data) -> torch.Tensor:
    if hasattr(data, "train_mask") and data.train_mask is not None:
        mask = data.train_mask.view(-1).bool()
        if int(mask.sum().item()) > 0:
            return mask
    return torch.ones_like(data.y, dtype=torch.bool)


def macro_f1_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    max_cls = int(torch.max(torch.cat([pred, labels], dim=0)).item())
    f1_vals = []
    for cls in range(max_cls + 1):
        c = torch.tensor(cls, device=labels.device)
        tp = ((pred == c) & (labels == c)).sum().float()
        fp = ((pred == c) & (labels != c)).sum().float()
        fn = ((pred != c) & (labels == c)).sum().float()
        denom = 2.0 * tp + fp + fn
        f1 = torch.where(denom > 0, (2.0 * tp) / denom, torch.tensor(0.0, device=labels.device))
        f1_vals.append(f1)
    return float(torch.stack(f1_vals).mean().item())


def train_eval_transfer_once(
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    source_y: torch.Tensor,
    target_y: torch.Tensor,
    source_train_mask: torch.Tensor,
    cfg: dict,
    seed: int,
) -> dict[str, float]:
    set_seed(int(seed))
    in_dim = int(source_feat.size(1))
    num_classes = int(torch.max(torch.cat([source_y, target_y], dim=0)).item()) + 1
    model = TransferMLP(
        in_dim=in_dim,
        hid_dim=int(cfg["mlp_hid_dim"]),
        out_dim=num_classes,
        num_layers=int(cfg["mlp_layers"]),
        dropout=float(cfg["mlp_dropout"]),
    ).to(source_feat.device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["mlp_lr"]),
        weight_decay=float(cfg["mlp_weight_decay"]),
    )

    for _ in range(int(cfg["mlp_epochs"])):
        model.train()
        opt.zero_grad()
        logits = model(source_feat)
        loss = F.cross_entropy(logits[source_train_mask], source_y[source_train_mask])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        src_logits = model(source_feat)
        tgt_logits = model(target_feat)
        src_acc = float((src_logits.argmax(dim=1) == source_y).float().mean().item())
        tgt_acc = float((tgt_logits.argmax(dim=1) == target_y).float().mean().item())
        src_macro = macro_f1_from_logits(src_logits, source_y)
        tgt_macro = macro_f1_from_logits(tgt_logits, target_y)

    return {
        "source_acc": src_acc,
        "target_acc": tgt_acc,
        "source_macro_f1": src_macro,
        "target_macro_f1": tgt_macro,
    }


def build_transfer_layers(source_data, target_data, align_state: dict, cfg: dict) -> dict[str, list[torch.Tensor]]:
    source_x = source_data.x.detach().float()
    target_x = target_data.x.detach().float()

    aligned_prop = WeightedPropagation().to(source_x.device)
    normal_cheb_k = int(cfg.get("normal_cheb_k", cfg["cheb_k"]))

    return {
        "normal_source": propagate_fixed_cheb_layers(
            source_x,
            source_data,
            cfg["max_layers"],
            cheb_k=normal_cheb_k,
            lambda_max=cfg["cheb_lambda_max"],
        ),
        "normal_target": propagate_fixed_cheb_layers(
            target_x,
            target_data,
            cfg["max_layers"],
            cheb_k=normal_cheb_k,
            lambda_max=cfg["cheb_lambda_max"],
        ),
        "aligned_source": propagate_weighted_layers(
            source_x,
            align_state["source_aligned_edge_index"],
            align_state["source_aligned_edge_weight"],
            cfg["max_layers"],
            aligned_prop,
        ),
        "aligned_target": propagate_weighted_layers(
            target_x,
            align_state["target_aligned_edge_index"],
            align_state["target_aligned_edge_weight"],
            cfg["max_layers"],
            aligned_prop,
        ),
    }


def evaluate_transferability(
    source_data,
    target_data,
    align_state: dict,
    cfg: dict,
) -> tuple[list[dict], list[dict]]:
    layers = build_transfer_layers(source_data, target_data, align_state, cfg)
    train_mask = get_source_train_mask(source_data).to(source_data.x.device)

    run_rows = []
    cases = [
        ("normal", "normal_source", "normal_target"),
        ("aligned", "aligned_source", "aligned_target"),
    ]
    for case, src_key, tgt_key in cases:
        for layer in range(int(cfg["max_layers"]) + 1):
            src_feat = layers[src_key][layer].detach()
            tgt_feat = layers[tgt_key][layer].detach()
            for rep in range(int(cfg["mlp_repeats"])):
                rep_seed = int(cfg["seed"]) + 1000 * layer + 100 * rep + (0 if case == "normal" else 50000)
                metrics = train_eval_transfer_once(
                    src_feat,
                    tgt_feat,
                    source_data.y,
                    target_data.y,
                    train_mask,
                    cfg,
                    seed=rep_seed,
                )
                run_rows.append({"case": case, "layer": layer, "repeat": rep, **metrics})

    summary_rows = []
    metrics_keys = ["source_acc", "target_acc", "source_macro_f1", "target_macro_f1"]
    for case in ("normal", "aligned"):
        for layer in range(int(cfg["max_layers"]) + 1):
            subset = [r for r in run_rows if r["case"] == case and int(r["layer"]) == layer]
            if not subset:
                continue
            row = {"case": case, "layer": layer, "n_runs": len(subset)}
            for metric_key in metrics_keys:
                vals = torch.tensor([float(r[metric_key]) for r in subset], dtype=torch.float32)
                row[f"{metric_key}_mean"] = float(vals.mean().item())
                row[f"{metric_key}_std"] = float(vals.std(unbiased=False).item())
            summary_rows.append(row)

    return run_rows, summary_rows


def write_transferability_protocol(path: Path) -> None:
    text = (
        "Transferability evaluation protocol:\n"
        "1) For normal/original graphs, compute features using fixed Chebyshev propagation.\n"
        "2) For aligned graphs, compute features using aligned weighted propagation.\n"
        "3) Build features for k=0..K (A^k X-style progression).\n"
        "4) For each k and case, train an MLP on source features/labels only.\n"
        "5) Evaluate the trained MLP on target features/labels with no target supervision.\n"
        "6) Repeat training multiple times (mlp_repeats) with different seeds.\n"
        "7) Report mean/std across repeats (target accuracy and target macro-F1).\n"
    )
    path.write_text(text)


def build_average_target_transfer_rows(
    all_scenario_transfer_summaries: list[list[dict]],
    max_layers: int,
) -> list[dict]:
    rows = []
    for layer in range(int(max_layers) + 1):
        paired_vals = []
        for scenario_rows in all_scenario_transfer_summaries:
            normal_row = next(
                (r for r in scenario_rows if r["case"] == "normal" and int(r["layer"]) == layer),
                None,
            )
            aligned_row = next(
                (r for r in scenario_rows if r["case"] == "aligned" and int(r["layer"]) == layer),
                None,
            )
            if normal_row is None or aligned_row is None:
                continue
            paired_vals.append(
                (
                    float(normal_row["target_acc_mean"]),
                    float(aligned_row["target_acc_mean"]),
                )
            )

        if not paired_vals:
            continue

        normal_vals = torch.tensor([v[0] for v in paired_vals], dtype=torch.float32)
        aligned_vals = torch.tensor([v[1] for v in paired_vals], dtype=torch.float32)
        rows.append(
            {
                "layer": layer,
                "n_scenarios": int(normal_vals.numel()),
                "normal_target_acc_mean": float(normal_vals.mean().item()),
                "normal_target_acc_std": float(normal_vals.std(unbiased=False).item()),
                "aligned_target_acc_mean": float(aligned_vals.mean().item()),
                "aligned_target_acc_std": float(aligned_vals.std(unbiased=False).item()),
                "delta_target_acc_mean": float((aligned_vals - normal_vals).mean().item()),
            }
        )
    return rows
