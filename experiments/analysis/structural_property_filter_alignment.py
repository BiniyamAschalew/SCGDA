import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from scipy.stats import ks_2samp, wasserstein_distance
from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Learn.Clean_SCGDA.data.build_dataset import build_dataset
from Learn.Clean_SCGDA.models.__components.chebprop import ChebProp
from Learn.Clean_SCGDA.utils.config_utils import build_config
from Learn.Clean_SCGDA.utils.expt_utils import set_seed


def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True).transpose(0, 1)
    dist = x_norm + y_norm - 2.0 * (x @ y.transpose(0, 1))
    return torch.clamp(dist, min=0.0)


def _median_bandwidth(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    all_feat = torch.cat([x, y], dim=0)
    dists = _pairwise_sq_dist(all_feat, all_feat).detach()
    n = dists.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=dists.device)
    vals = dists[mask]
    if vals.numel() == 0:
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)
    return vals.median().clamp_min(eps)


def mmd_rbf(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    bandwidth = (
        torch.as_tensor(fix_sigma, device=source.device, dtype=source.dtype)
        if fix_sigma is not None
        else _median_bandwidth(source, target, eps=eps)
    )
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))

    xx = _pairwise_sq_dist(source, source)
    yy = _pairwise_sq_dist(target, target)
    xy = _pairwise_sq_dist(source, target)

    k_xx = 0.0
    k_yy = 0.0
    k_xy = 0.0
    for i in range(kernel_num):
        bw = (bandwidth * (kernel_mul**i)).clamp_min(eps)
        k_xx = k_xx + torch.exp(-xx / bw)
        k_yy = k_yy + torch.exp(-yy / bw)
        k_xy = k_xy + torch.exp(-xy / bw)

    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


def conditional_mmd(
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    source_y: torch.Tensor,
    target_y: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | None = None,
) -> torch.Tensor:
    classes = torch.unique(torch.cat([source_y, target_y], dim=0))
    per_class = []
    for cls in classes:
        src_mask = source_y == cls
        tgt_mask = target_y == cls
        if int(src_mask.sum().item()) == 0 or int(tgt_mask.sum().item()) == 0:
            continue
        per_class.append(
            mmd_rbf(
                source_feat[src_mask],
                target_feat[tgt_mask],
                kernel_mul=kernel_mul,
                kernel_num=kernel_num,
                fix_sigma=fix_sigma,
            )
        )
    if not per_class:
        return torch.tensor(0.0, device=source_feat.device, dtype=source_feat.dtype)
    return torch.stack(per_class).mean()


def _resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _load_pair(dataset: str, source: str, target: str, device: str, seed: int):
    config_setup = {"data": dataset, "expt": "default", "model": "test"}
    update_config = {
        "expt": {
            "source": source,
            "target": target,
            "device": device,
            "seed": seed,
            "verbose": 0,
            "wandb_enabled": False,
        }
    }
    config = build_config(config_setup, update_config)
    source_dataset, target_dataset = build_dataset(config)
    source_data = source_dataset[0].to(device)
    target_data = target_dataset[0].to(device)

    if source_data.x is None or target_data.x is None:
        raise ValueError("Source/target features are required.")
    if int(source_data.x.size(1)) != int(target_data.x.size(1)):
        raise ValueError(
            f"Feature dim mismatch: source={source_data.x.size(1)}, target={target_data.x.size(1)}"
        )
    return source_data, target_data


def _edge_index_to_adj(edge_index: torch.Tensor, num_nodes: int) -> sp.csr_matrix:
    edge_np = edge_index.detach().cpu().numpy()
    rows = edge_np[0]
    cols = edge_np[1]
    data = np.ones(rows.shape[0], dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32).tocsr()
    adj.setdiag(0.0)
    adj.eliminate_zeros()
    adj = adj.maximum(adj.T)
    adj.data[:] = 1.0
    return adj


def _extract_properties(adj: sp.csr_matrix) -> dict[str, np.ndarray]:
    degree = np.asarray(adj.sum(axis=1)).ravel().astype(np.float32)
    a = adj.astype(np.float32)
    a2 = a @ a
    two_step_walk = np.asarray(a2.sum(axis=1)).ravel().astype(np.float32)

    a3_diag = (a2 @ a).diagonal().astype(np.float32)
    triangles = np.clip(a3_diag / 2.0, 0.0, None)
    denom = degree * np.clip(degree - 1.0, 0.0, None)
    clustering = np.zeros_like(degree)
    mask = denom > 0
    clustering[mask] = np.clip(2.0 * triangles[mask] / denom[mask], 0.0, 1.0)

    open_wedges = np.clip(degree * np.clip(degree - 1.0, 0.0, None) / 2.0 - triangles, 0.0, None)

    return {
        "degree": degree,
        "log_degree": np.log1p(degree),
        "clustering": clustering,
        "triangles_log": np.log1p(triangles),
        "open_wedges_log": np.log1p(open_wedges),
        "two_step_walk_log": np.log1p(two_step_walk),
    }


def _to_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def _apply_operator(
    x: torch.Tensor,
    data,
    temp: torch.Tensor,
    prop: ChebProp,
    lambda_max: float,
) -> torch.Tensor:
    edge_weight = getattr(data, "edge_weight", None)
    batch = getattr(data, "batch", None)
    return prop(
        x,
        data.edge_index,
        edge_weight=edge_weight,
        batch=batch,
        lambda_max=lambda_max,
        temp=temp,
    )


def _subsample_for_mmd(
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    source_y: torch.Tensor,
    target_y: torch.Tensor,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if max_samples <= 0:
        return source_feat, target_feat, source_y, target_y

    ns = source_feat.size(0)
    nt = target_feat.size(0)
    ks = min(max_samples, ns)
    kt = min(max_samples, nt)
    idx_s = torch.randperm(ns, device=source_feat.device)[:ks]
    idx_t = torch.randperm(nt, device=target_feat.device)[:kt]
    return source_feat[idx_s], target_feat[idx_t], source_y[idx_s], target_y[idx_t]


def _js_divergence(x: np.ndarray, y: np.ndarray, bins: int, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lo = min(float(x.min()), float(y.min()))
    hi = max(float(x.max()), float(y.max()))
    if hi <= lo:
        return 0.0
    px, _ = np.histogram(x, bins=bins, range=(lo, hi))
    py, _ = np.histogram(y, bins=bins, range=(lo, hi))
    p = px.astype(np.float64) + eps
    q = py.astype(np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    js = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
    return float(js)


def _compute_distances(
    source_vec: np.ndarray,
    target_vec: np.ndarray,
    source_y: torch.Tensor,
    target_y: torch.Tensor,
    *,
    kernel_mul: float,
    kernel_num: int,
    fix_sigma: float | None,
    metric_sample_size: int,
    hist_bins: int,
    device: str,
) -> dict[str, float]:
    xs = torch.as_tensor(source_vec[:, None], dtype=torch.float32, device=device)
    xt = torch.as_tensor(target_vec[:, None], dtype=torch.float32, device=device)

    xs_m, xt_m, ys_m, yt_m = _subsample_for_mmd(
        xs, xt, source_y, target_y, metric_sample_size
    )
    feat_mmd = mmd_rbf(
        xs_m,
        xt_m,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )
    cond_mmd = conditional_mmd(
        xs_m,
        xt_m,
        ys_m,
        yt_m,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )

    wasser = float(wasserstein_distance(source_vec, target_vec))
    ks_stat = float(ks_2samp(source_vec, target_vec).statistic)
    js_div = _js_divergence(source_vec, target_vec, bins=hist_bins)

    return {
        "feature_mmd": _to_float(feat_mmd),
        "conditional_mmd": _to_float(cond_mmd),
        "wasserstein": wasser,
        "ks_stat": ks_stat,
        "js_divergence": js_div,
        "source_mean": float(np.mean(source_vec)),
        "target_mean": float(np.mean(target_vec)),
        "source_std": float(np.std(source_vec)),
        "target_std": float(np.std(target_vec)),
    }


def _warmup_r_filters(
    *,
    source_x: torch.Tensor,
    target_x: torch.Tensor,
    source_data,
    target_data,
    prop: ChebProp,
    poly_order: int,
    warmup_epochs: int,
    align_lr: float,
    align_weight_decay: float,
    cheb_lambda_max: float,
    kernel_mul: float,
    kernel_num: int,
    fix_sigma: float | None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict], list[dict]]:
    num_terms = poly_order + 1
    source_temp = nn.Parameter(torch.full((num_terms,), 1.0 / float(num_terms), device=source_x.device))
    target_temp = nn.Parameter(torch.full((num_terms,), 1.0 / float(num_terms), device=source_x.device))
    optimizer = torch.optim.Adam(
        [source_temp, target_temp], lr=align_lr, weight_decay=align_weight_decay
    )

    warmup_rows = []
    coeff_rows = []

    all_features = torch.cat([source_x, target_x], dim=0).detach()
    probe_mean = all_features.mean(dim=0, keepdim=True)
    probe_std = all_features.std(dim=0, keepdim=True) + 1e-6

    for epoch in range(1, warmup_epochs + 1):
        optimizer.zero_grad()

        probe_features = torch.randn_like(all_features) * probe_std + probe_mean
        source_probe = probe_features[: source_x.size(0)]
        target_probe = probe_features[source_x.size(0) :]
        source_probe_out = _apply_operator(
            source_probe, source_data, source_temp, prop, cheb_lambda_max
        )
        target_probe_out = _apply_operator(
            target_probe, target_data, target_temp, prop, cheb_lambda_max
        )

        probe_mmd = mmd_rbf(
            source_probe_out,
            target_probe_out,
            kernel_mul=kernel_mul,
            kernel_num=kernel_num,
            fix_sigma=fix_sigma,
        )
        probe_mmd.backward()
        optimizer.step()

        with torch.no_grad():
            src_coeff = torch.relu(source_temp).detach()
            tgt_coeff = torch.relu(target_temp).detach()
            coeff_gap = torch.norm(src_coeff - tgt_coeff, p=2)
            warmup_rows.append(
                {
                    "epoch": epoch,
                    "probe_mmd": _to_float(probe_mmd),
                    "coeff_gap_l2": _to_float(coeff_gap),
                    "source_coeff_l1": float(src_coeff.abs().sum().item()),
                    "target_coeff_l1": float(tgt_coeff.abs().sum().item()),
                }
            )
            for k in range(num_terms):
                coeff_rows.append(
                    {
                        "epoch": epoch,
                        "domain": "source",
                        "basis": k,
                        "coeff": float(src_coeff[k].item()),
                    }
                )
                coeff_rows.append(
                    {
                        "epoch": epoch,
                        "domain": "target",
                        "basis": k,
                        "coeff": float(tgt_coeff[k].item()),
                    }
                )

    return torch.relu(source_temp).detach(), torch.relu(target_temp).detach(), warmup_rows, coeff_rows


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_warmup(rows: list[dict], out_png: str) -> None:
    if not rows:
        return
    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(epochs, [r["probe_mmd"] for r in rows], linewidth=1.8, label="Probe MMD")
    axes[0].set_ylabel("MMD")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [r["coeff_gap_l2"] for r in rows], linewidth=1.8, label="Coeff gap L2")
    axes[1].set_xlabel("Warm-up Epoch")
    axes[1].set_ylabel("Gap")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_coefficients(rows: list[dict], out_png: str) -> None:
    if not rows:
        return
    domains = ["source", "target"]
    bases = sorted({int(r["basis"]) for r in rows})
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    domain_to_ax = {"source": axes[0], "target": axes[1]}

    for domain in domains:
        ax = domain_to_ax[domain]
        for b in bases:
            vals = [r for r in rows if r["domain"] == domain and int(r["basis"]) == b]
            vals.sort(key=lambda x: x["epoch"])
            ax.plot(
                [r["epoch"] for r in vals],
                [r["coeff"] for r in vals],
                linewidth=1.7,
                color=colors[b % len(colors)],
                label=f"T{b}",
            )
        ax.set_title(f"{domain.capitalize()} R Coefficients")
        ax.set_ylabel("Coeff")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Warm-up Epoch")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=max(1, min(6, len(labels))), loc="upper center")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_metric_grouped(
    rows: list[dict],
    metric: str,
    title: str,
    out_png: str,
) -> None:
    operators = ["raw", "P_fixed", "R_aligned"]
    properties = sorted({r["property"] for r in rows})
    x = np.arange(len(properties))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 4.8))
    for i, op in enumerate(operators):
        vals = []
        for prop in properties:
            row = next(r for r in rows if r["property"] == prop and r["operator"] == op)
            vals.append(row[metric])
        ax.bar(x + (i - 1) * width, vals, width=width, label=op)

    ax.set_xticks(x, properties, rotation=20)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_degree_histograms(
    degree_distributions: dict[str, dict[str, np.ndarray]],
    out_png: str,
    bins: int,
) -> None:
    operators = ["raw", "P_fixed", "R_aligned"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for i, op in enumerate(operators):
        src = degree_distributions[op]["source"]
        tgt = degree_distributions[op]["target"]
        lo = min(float(src.min()), float(tgt.min()))
        hi = max(float(src.max()), float(tgt.max()))
        if hi <= lo:
            hi = lo + 1.0
        axes[i].hist(src, bins=bins, range=(lo, hi), density=True, alpha=0.5, label="source")
        axes[i].hist(tgt, bins=bins, range=(lo, hi), density=True, alpha=0.5, label="target")
        axes[i].set_title(op)
        axes[i].set_xlabel("Signal Value")
        axes[i].grid(alpha=0.3)
    axes[0].set_ylabel("Density")
    axes[0].legend()

    fig.suptitle("Degree-Signal Distribution Shift: Raw vs P vs R")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def run(args):
    device = _resolve_device(args.device)
    set_seed(args.seed)

    source_data, target_data = _load_pair(
        dataset=args.dataset,
        source=args.source,
        target=args.target,
        device=device,
        seed=args.seed,
    )
    source_adj = _edge_index_to_adj(source_data.edge_index, int(source_data.num_nodes))
    target_adj = _edge_index_to_adj(target_data.edge_index, int(target_data.num_nodes))
    source_props = _extract_properties(source_adj)
    target_props = _extract_properties(target_adj)

    available_properties = list(source_props.keys())
    requested_properties = [p.strip() for p in args.properties.split(",") if p.strip()]
    for prop in requested_properties:
        if prop not in available_properties:
            raise ValueError(f"Unknown property '{prop}'. Available: {available_properties}")

    num_terms = args.poly_order + 1
    cheb_prop = ChebProp(num_terms, is_source_domain=False).to(device)

    r_source_temp, r_target_temp, warmup_rows, coeff_rows = _warmup_r_filters(
        source_x=source_data.x.detach(),
        target_x=target_data.x.detach(),
        source_data=source_data,
        target_data=target_data,
        prop=cheb_prop,
        poly_order=args.poly_order,
        warmup_epochs=args.warmup_epochs,
        align_lr=args.align_lr,
        align_weight_decay=args.align_weight_decay,
        cheb_lambda_max=args.cheb_lambda_max,
        kernel_mul=args.kernel_mul,
        kernel_num=args.kernel_num,
        fix_sigma=args.fix_sigma,
    )

    p_temp = torch.zeros((num_terms,), device=device)
    p_temp[-1] = 1.0

    metric_rows = []
    degree_distributions = {}

    for prop_name in requested_properties:
        src_raw = source_props[prop_name]
        tgt_raw = target_props[prop_name]
        src_signal = torch.as_tensor(src_raw[:, None], dtype=torch.float32, device=device)
        tgt_signal = torch.as_tensor(tgt_raw[:, None], dtype=torch.float32, device=device)

        with torch.no_grad():
            src_p = _apply_operator(src_signal, source_data, p_temp, cheb_prop, args.cheb_lambda_max)
            tgt_p = _apply_operator(tgt_signal, target_data, p_temp, cheb_prop, args.cheb_lambda_max)
            src_r = _apply_operator(src_signal, source_data, r_source_temp, cheb_prop, args.cheb_lambda_max)
            tgt_r = _apply_operator(tgt_signal, target_data, r_target_temp, cheb_prop, args.cheb_lambda_max)

        src_p_np = src_p.squeeze(1).detach().cpu().numpy()
        tgt_p_np = tgt_p.squeeze(1).detach().cpu().numpy()
        src_r_np = src_r.squeeze(1).detach().cpu().numpy()
        tgt_r_np = tgt_r.squeeze(1).detach().cpu().numpy()

        raw_metrics = _compute_distances(
            src_raw,
            tgt_raw,
            source_data.y,
            target_data.y,
            kernel_mul=args.kernel_mul,
            kernel_num=args.kernel_num,
            fix_sigma=args.fix_sigma,
            metric_sample_size=args.metric_sample_size,
            hist_bins=args.hist_bins,
            device=device,
        )
        p_metrics = _compute_distances(
            src_p_np,
            tgt_p_np,
            source_data.y,
            target_data.y,
            kernel_mul=args.kernel_mul,
            kernel_num=args.kernel_num,
            fix_sigma=args.fix_sigma,
            metric_sample_size=args.metric_sample_size,
            hist_bins=args.hist_bins,
            device=device,
        )
        r_metrics = _compute_distances(
            src_r_np,
            tgt_r_np,
            source_data.y,
            target_data.y,
            kernel_mul=args.kernel_mul,
            kernel_num=args.kernel_num,
            fix_sigma=args.fix_sigma,
            metric_sample_size=args.metric_sample_size,
            hist_bins=args.hist_bins,
            device=device,
        )

        metric_rows.append({"property": prop_name, "operator": "raw", **raw_metrics})
        metric_rows.append({"property": prop_name, "operator": "P_fixed", **p_metrics})
        metric_rows.append({"property": prop_name, "operator": "R_aligned", **r_metrics})

        if prop_name == "degree":
            degree_distributions = {
                "raw": {"source": src_raw, "target": tgt_raw},
                "P_fixed": {"source": src_p_np, "target": tgt_p_np},
                "R_aligned": {"source": src_r_np, "target": tgt_r_np},
            }

    reduction_rows = []
    metrics_of_interest = ["feature_mmd", "conditional_mmd", "wasserstein", "ks_stat", "js_divergence"]
    for prop_name in requested_properties:
        by_op = {
            row["operator"]: row
            for row in metric_rows
            if row["property"] == prop_name
        }
        for metric_name in metrics_of_interest:
            raw_val = float(by_op["raw"][metric_name])
            p_val = float(by_op["P_fixed"][metric_name])
            r_val = float(by_op["R_aligned"][metric_name])
            reduction_rows.append(
                {
                    "property": prop_name,
                    "metric": metric_name,
                    "raw": raw_val,
                    "P_fixed": p_val,
                    "R_aligned": r_val,
                    "R_minus_P": r_val - p_val,
                    "R_reduction_vs_P_percent": ((p_val - r_val) / (abs(p_val) + 1e-12)) * 100.0,
                    "R_minus_raw": r_val - raw_val,
                }
            )

    os.makedirs(args.out_dir, exist_ok=True)
    warmup_csv = os.path.join(args.out_dir, "probe_warmup_metrics.csv")
    coeff_csv = os.path.join(args.out_dir, "aligned_coefficients.csv")
    metric_csv = os.path.join(args.out_dir, "structural_shift_metrics.csv")
    reduction_csv = os.path.join(args.out_dir, "structural_shift_reduction.csv")

    _write_csv(
        warmup_csv,
        warmup_rows,
        fieldnames=["epoch", "probe_mmd", "coeff_gap_l2", "source_coeff_l1", "target_coeff_l1"],
    )
    _write_csv(coeff_csv, coeff_rows, fieldnames=["epoch", "domain", "basis", "coeff"])
    _write_csv(
        metric_csv,
        metric_rows,
        fieldnames=[
            "property",
            "operator",
            "feature_mmd",
            "conditional_mmd",
            "wasserstein",
            "ks_stat",
            "js_divergence",
            "source_mean",
            "target_mean",
            "source_std",
            "target_std",
        ],
    )
    _write_csv(
        reduction_csv,
        reduction_rows,
        fieldnames=[
            "property",
            "metric",
            "raw",
            "P_fixed",
            "R_aligned",
            "R_minus_P",
            "R_reduction_vs_P_percent",
            "R_minus_raw",
        ],
    )

    warmup_png = os.path.join(args.out_dir, "probe_warmup_curves.png")
    coeff_png = os.path.join(args.out_dir, "aligned_coefficients.png")
    feat_mmd_png = os.path.join(args.out_dir, "feature_mmd_by_property.png")
    cond_mmd_png = os.path.join(args.out_dir, "conditional_mmd_by_property.png")
    wasser_png = os.path.join(args.out_dir, "wasserstein_by_property.png")
    degree_hist_png = os.path.join(args.out_dir, "degree_shift_histograms.png")

    _plot_warmup(warmup_rows, warmup_png)
    _plot_coefficients(coeff_rows, coeff_png)
    _plot_metric_grouped(
        metric_rows,
        metric="feature_mmd",
        title="Feature MMD Shift by Structural Property",
        out_png=feat_mmd_png,
    )
    _plot_metric_grouped(
        metric_rows,
        metric="conditional_mmd",
        title="Conditional MMD Shift by Structural Property",
        out_png=cond_mmd_png,
    )
    _plot_metric_grouped(
        metric_rows,
        metric="wasserstein",
        title="Wasserstein Shift by Structural Property",
        out_png=wasser_png,
    )
    if degree_distributions:
        _plot_degree_histograms(degree_distributions, degree_hist_png, bins=args.hist_bins)

    print(f"Saved warm-up CSV: {warmup_csv}")
    print(f"Saved coefficient CSV: {coeff_csv}")
    print(f"Saved structural metrics CSV: {metric_csv}")
    print(f"Saved reduction CSV: {reduction_csv}")
    print(f"Saved warm-up plot: {warmup_png}")
    print(f"Saved coefficient plot: {coeff_png}")
    print(f"Saved feature-MMD plot: {feat_mmd_png}")
    print(f"Saved conditional-MMD plot: {cond_mmd_png}")
    print(f"Saved wasserstein plot: {wasser_png}")
    if degree_distributions:
        print(f"Saved degree histogram plot: {degree_hist_png}")

    degree_feat = [
        row for row in reduction_rows if row["property"] == "degree" and row["metric"] == "feature_mmd"
    ]
    degree_cond = [
        row
        for row in reduction_rows
        if row["property"] == "degree" and row["metric"] == "conditional_mmd"
    ]
    if degree_feat:
        print("Degree feature-MMD reduction summary:", degree_feat[0])
    if degree_cond:
        print("Degree conditional-MMD reduction summary:", degree_cond[0])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test whether probe-aligned filters (R) reduce structural-property shifts "
            "vs fixed highest-degree filters (P)."
        )
    )
    parser.add_argument("--dataset", type=str, default="airport")
    parser.add_argument("--source", type=str, default="BRAZIL")
    parser.add_argument("--target", type=str, default="USA")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--poly_order", type=int, default=3)
    parser.add_argument("--cheb_lambda_max", type=float, default=2.0)

    parser.add_argument("--warmup_epochs", type=int, default=100)
    parser.add_argument("--align_lr", type=float, default=0.01)
    parser.add_argument("--align_weight_decay", type=float, default=0.0)

    parser.add_argument("--kernel_mul", type=float, default=2.0)
    parser.add_argument("--kernel_num", type=int, default=5)
    parser.add_argument("--fix_sigma", type=float, default=None)
    parser.add_argument("--metric_sample_size", type=int, default=2000)
    parser.add_argument("--hist_bins", type=int, default=40)

    parser.add_argument(
        "--properties",
        type=str,
        default="degree,log_degree,clustering,triangles_log,open_wedges_log,two_step_walk_log",
        help="Comma-separated property names.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./__saved__/analysis/structural_property_filter_alignment",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
