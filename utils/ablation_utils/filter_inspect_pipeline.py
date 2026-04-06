"""Pipeline for filter-learning inspection experiment."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from Learn.Clean_SCGDA.models.__components.chebprop import ChebProp
from Learn.Clean_SCGDA.utils.ablation_utils.alignment import build_struct_positive_edges, edge_bpr_structure_loss
from Learn.Clean_SCGDA.utils.ablation_utils.common import as_float, load_pair, sample_idx
from Learn.Clean_SCGDA.utils.ablation_utils.propagation import (
    Propagation,
    apply_cheb_once,
    propagate_layers,
    uniform_cheb_temp,
)
from Learn.Clean_SCGDA.utils.filter_utils import cheb_to_monomial, mmd_rbf


def load_scenario_graphs(dataset: str, source: str, target: str, cfg: dict):
    """Load source/target graph objects for one scenario."""
    return load_pair(dataset, source, target, cfg["device"], cfg["seed"])


def build_target_polynomial_reference(
    target_data,
    cfg: dict,
):
    """Construct target-side reference features from a fixed polynomial graph filter."""
    target_x = target_data.x.detach().float()
    device = target_x.device
    dtype = target_x.dtype

    coeff_cfg = cfg.get("target_poly_coeffs", [0.0, 0.2, 0.5, 0.2])
    coeff = torch.tensor(coeff_cfg, device=device, dtype=dtype)
    if coeff.dim() != 1:
        raise ValueError(f"target_poly_coeffs must be 1D, got shape={tuple(coeff.shape)}")

    max_power = max(int(cfg.get("target_max_power", coeff.numel() - 1)), int(coeff.numel() - 1))
    prop = Propagation().to(device)
    target_layers = propagate_layers(target_x, target_data.edge_index, max_power, prop)

    coeff_padded = torch.zeros(max_power + 1, device=device, dtype=dtype)
    coeff_padded[: coeff.numel()] = coeff

    target_reference = torch.zeros_like(target_x)
    for power in range(max_power + 1):
        target_reference = target_reference + coeff_padded[power] * target_layers[power]

    return {
        "target_reference": target_reference.detach(),
        "target_poly_coeffs": coeff_padded.detach(),
        "target_max_power": int(max_power),
    }


def initialize_source_learnable_filter(
    source_data,
    cfg: dict,
):
    """Initialize learnable source-side Chebyshev filter parameters."""
    device = source_data.x.device
    k = int(cfg["cheb_k"])
    dtype = source_data.x.dtype if source_data.x.dtype.is_floating_point else torch.float32

    source_temp = nn.Parameter(uniform_cheb_temp(k, device=device, dtype=dtype).clone())
    source_edge_logits = nn.Parameter(torch.zeros(source_data.edge_index.size(1), device=device, dtype=dtype))

    optimizer = torch.optim.Adam(
        [source_temp, source_edge_logits],
        lr=float(cfg["train_lr"]),
        weight_decay=float(cfg["train_weight_decay"]),
    )
    cheb_prop = ChebProp(k, is_source_domain=False).to(device)

    struct_weight = float(cfg.get("struct_bpr_weight", 0.0))
    struct_pos_edges = None
    if struct_weight > 0:
        struct_pos_edges = build_struct_positive_edges(source_data, cfg)

    return {
        "source_temp": source_temp,
        "source_edge_logits": source_edge_logits,
        "optimizer": optimizer,
        "cheb_prop": cheb_prop,
        "struct_pos_edges": struct_pos_edges,
    }


def train_source_filter_to_target_reference(
    source_data,
    target_reference: dict,
    train_state: dict,
    cfg: dict,
):
    """Train source Chebyshev filter so its propagated response matches target reference."""
    source_x = source_data.x.detach().float()
    target_ref = target_reference["target_reference"].detach().to(device=source_x.device, dtype=source_x.dtype)

    source_temp = train_state["source_temp"]
    source_edge_logits = train_state["source_edge_logits"]
    optimizer = train_state["optimizer"]
    cheb_prop = train_state["cheb_prop"]
    struct_pos_edges = train_state["struct_pos_edges"]

    edge_eps = float(cfg.get("edge_eps", 1e-6))
    struct_weight = float(cfg.get("struct_bpr_weight", 0.0))
    struct_samples = int(cfg.get("struct_bpr_samples", 2048))
    struct_margin = float(cfg.get("struct_bpr_margin", 0.0))

    history_rows = []
    cheb_rows = []

    epochs = int(cfg["train_epochs"])
    log_interval = int(cfg.get("train_log_interval", 10))
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        source_w = F.softplus(source_edge_logits) + edge_eps
        source_out = apply_cheb_once(
            source_x,
            source_data,
            cheb_prop,
            source_temp,
            source_w,
            float(cfg["cheb_lambda_max"]),
        )

        idx_s = sample_idx(source_out.size(0), int(cfg["metric_sample_size"]), source_out.device)
        idx_t = sample_idx(target_ref.size(0), int(cfg["metric_sample_size"]), target_ref.device)
        mmd_loss = mmd_rbf(
            source_out[idx_s],
            target_ref[idx_t],
            kernel_mul=float(cfg["kernel_mul"]),
            kernel_num=int(cfg["kernel_num"]),
            fix_sigma=cfg.get("fix_sigma"),
        )

        source_temp_pos = F.relu(source_temp)
        edge_reg = float(cfg.get("edge_reg", 0.0)) * source_w.pow(2).mean()
        temp_reg = float(cfg.get("temp_reg", 0.0)) * source_temp_pos.pow(2).mean()

        if struct_weight > 0 and struct_pos_edges is not None:
            bpr_loss = edge_bpr_structure_loss(
                source_out,
                struct_pos_edges,
                num_samples=struct_samples,
                margin=struct_margin,
            )
        else:
            bpr_loss = source_out.new_tensor(0.0)

        struct_loss = struct_weight * bpr_loss
        loss = mmd_loss + edge_reg + temp_reg + struct_loss
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            with torch.no_grad():
                source_temp_now = F.relu(source_temp).detach()
                cheb_row = {"epoch": epoch}
                for i, val in enumerate(source_temp_now):
                    cheb_row[f"cheb_{i}"] = float(val.item())
                cheb_rows.append(cheb_row)

                history_rows.append(
                    {
                        "epoch": epoch,
                        "loss": as_float(loss),
                        "mmd_loss": as_float(mmd_loss),
                        "edge_reg": as_float(edge_reg),
                        "temp_reg": as_float(temp_reg),
                        "bpr_loss": as_float(bpr_loss),
                        "struct_loss": as_float(struct_loss),
                    }
                )

    with torch.no_grad():
        final_temp = F.relu(source_temp).detach()
        final_edge_weight = F.softplus(source_edge_logits).detach() + edge_eps
        final_source_out = apply_cheb_once(
            source_x,
            source_data,
            cheb_prop,
            final_temp,
            final_edge_weight,
            float(cfg["cheb_lambda_max"]),
        ).detach()

    return {
        "history_rows": history_rows,
        "cheb_rows": cheb_rows,
        "final_temp": final_temp,
        "final_edge_weight": final_edge_weight,
        "final_source_out": final_source_out,
    }


def convert_cheb_history_to_polynomial_history(
    cheb_history,
    cfg: dict,
):
    """Convert tracked Chebyshev coefficients to polynomial (power-basis) coefficients."""
    if not cheb_history:
        return {"poly_rows": []}

    cheb_cols = sorted(
        [k for k in cheb_history[0].keys() if k.startswith("cheb_")],
        key=lambda x: int(x.split("_")[1]),
    )

    max_order_cfg = int(cfg.get("coeff_plot_max_order", -1))
    poly_rows = []
    for row in cheb_history:
        cheb_vec = torch.tensor([float(row[c]) for c in cheb_cols], dtype=torch.float32)
        poly_vec = cheb_to_monomial(cheb_vec)
        if max_order_cfg >= 0:
            poly_vec = poly_vec[: max_order_cfg + 1]

        poly_row = {"epoch": int(row["epoch"])}
        for i, val in enumerate(poly_vec):
            poly_row[f"poly_{i}"] = float(val.item())
        poly_rows.append(poly_row)

    return {"poly_rows": poly_rows}


def build_scenario_payload(dataset: str, source: str, target: str, cfg: dict) -> dict:
    """Run one scenario pipeline and return all in-memory outputs."""
    source_data, target_data = load_scenario_graphs(dataset, source, target, cfg)
    target_reference = build_target_polynomial_reference(target_data, cfg)
    train_state = initialize_source_learnable_filter(source_data, cfg)
    train_result = train_source_filter_to_target_reference(source_data, target_reference, train_state, cfg)
    poly_history = convert_cheb_history_to_polynomial_history(train_result["cheb_rows"], cfg)

    final_source_out = train_result["final_source_out"]
    target_ref = target_reference["target_reference"]
    idx_s = sample_idx(final_source_out.size(0), int(cfg["metric_sample_size"]), final_source_out.device)
    idx_t = sample_idx(target_ref.size(0), int(cfg["metric_sample_size"]), target_ref.device)
    final_mmd = mmd_rbf(
        final_source_out[idx_s],
        target_ref[idx_t],
        kernel_mul=float(cfg["kernel_mul"]),
        kernel_num=int(cfg["kernel_num"]),
        fix_sigma=cfg.get("fix_sigma"),
    )

    final_cheb = train_result["final_temp"].detach()
    final_poly = cheb_to_monomial(final_cheb)
    target_poly = target_reference["target_poly_coeffs"].to(device=final_poly.device, dtype=final_poly.dtype)

    target_poly_aligned = torch.zeros_like(final_poly)
    n = min(target_poly.numel(), final_poly.numel())
    target_poly_aligned[:n] = target_poly[:n]

    poly_l2 = torch.norm(final_poly - target_poly_aligned, p=2)
    if torch.norm(final_poly, p=2) > 0 and torch.norm(target_poly_aligned, p=2) > 0:
        poly_cos = F.cosine_similarity(final_poly, target_poly_aligned, dim=0)
    else:
        poly_cos = final_poly.new_tensor(0.0)

    summary = {
        "dataset": dataset,
        "source": source,
        "target": target,
        "final_mmd": as_float(final_mmd),
        "poly_l2_to_target": as_float(poly_l2),
        "poly_cos_to_target": as_float(poly_cos),
        "final_epoch": int(cfg["train_epochs"]),
    }

    return {
        "dataset": dataset,
        "source": source,
        "target": target,
        "summary": summary,
        "training_rows": train_result["history_rows"],
        "cheb_coeff_rows": train_result["cheb_rows"],
        "poly_coeff_rows": poly_history["poly_rows"],
        "target_poly_coeffs": target_poly_aligned.detach().cpu(),
        "final_cheb_coeffs": final_cheb.detach().cpu(),
        "final_poly_coeffs": final_poly.detach().cpu(),
        "final_edge_weight": train_result["final_edge_weight"].detach().cpu(),
    }
