import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Learn.Clean_SCGDA.data.build_dataset import build_dataset
from Learn.Clean_SCGDA.models.__components.chebprop import ChebProp
from Learn.Clean_SCGDA.models.__layers.build_layer import build_activation
from Learn.Clean_SCGDA.utils.config_utils import build_config
from Learn.Clean_SCGDA.utils.expt_utils import set_seed


class FeatureMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        num_classes: int,
        num_layers: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.dropout = float(dropout)
        self.act = build_activation(activation)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hid_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hid_dim, hid_dim))
        self.cls = nn.Linear(hid_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.cls(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        logits = self.classify(z)
        return logits, z


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
        raise ValueError("Source/target features are required. For airport this should be one-hot degree.")
    if int(source_data.x.size(1)) != int(target_data.x.size(1)):
        raise ValueError(
            f"Feature dim mismatch: source={source_data.x.size(1)}, target={target_data.x.size(1)}"
        )
    return source_data, target_data


def _source_train_mask(data, use_train_mask: bool) -> torch.Tensor:
    if use_train_mask and hasattr(data, "train_mask") and data.train_mask is not None:
        if int(data.train_mask.sum().item()) > 0:
            return data.train_mask
    return torch.ones_like(data.y, dtype=torch.bool)


def _to_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


def _subsample_for_metrics(
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


def _compute_mmd_cmmd(
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    source_y: torch.Tensor,
    target_y: torch.Tensor,
    *,
    kernel_mul: float,
    kernel_num: int,
    fix_sigma: float | None,
    metric_sample_size: int,
) -> tuple[float, float]:
    xs, xt, ys, yt = _subsample_for_metrics(
        source_feat, target_feat, source_y, target_y, metric_sample_size
    )
    mmd_val = mmd_rbf(xs, xt, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    cmmd_val = conditional_mmd(
        xs,
        xt,
        ys,
        yt,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )
    return _to_float(mmd_val), _to_float(cmmd_val)


def _cmmd_target_labels(mode: str, target_logits: torch.Tensor, target_true: torch.Tensor) -> torch.Tensor:
    if mode == "true":
        return target_true
    if mode == "pseudo":
        return target_logits.argmax(dim=1)
    raise ValueError(f"Unknown cmmd_target_labels mode: {mode}")


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def _apply_monomial_average_operator(
    x: torch.Tensor,
    data,
    prop: ChebProp,
    lambda_max: float,
    max_degree: int = 3,
) -> torch.Tensor:
    """Apply fixed monomial-average operator (A + A^2 + ... + A^max_degree)/max_degree."""
    if max_degree < 1:
        raise ValueError(f"max_degree must be >= 1, got {max_degree}")

    edge_weight = getattr(data, "edge_weight", None)
    batch = getattr(data, "batch", None)
    edge_index1, norm1 = prop.__norm__(
        data.edge_index,
        x.size(0),
        edge_weight,
        lambda_max=lambda_max,
        dtype=x.dtype,
        batch=batch,
    )

    terms = []
    h = x
    for _ in range(max_degree):
        h = prop.propagate(edge_index1, x=h, norm=norm1, size=None)
        terms.append(h)

    return torch.stack(terms, dim=0).mean(dim=0)


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
    metric_sample_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict], list[dict]]:
    num_terms = poly_order + 1  # degree 0..poly_order (poly_order=3 includes degree 3 / A^3-like term)

    source_temp = nn.Parameter(torch.full((num_terms,), 1.0 / float(num_terms), device=source_x.device))
    target_temp = nn.Parameter(torch.full((num_terms,), 1.0 / float(num_terms), device=source_x.device))
    optimizer = torch.optim.Adam(
        [source_temp, target_temp],
        lr=align_lr,
        weight_decay=align_weight_decay,
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
            rs = _apply_operator(source_x, source_data, source_temp, prop, cheb_lambda_max)
            rt = _apply_operator(target_x, target_data, target_temp, prop, cheb_lambda_max)
            real_mmd, real_cmmd = _compute_mmd_cmmd(
                rs,
                rt,
                source_data.y,
                target_data.y,
                kernel_mul=kernel_mul,
                kernel_num=kernel_num,
                fix_sigma=fix_sigma,
                metric_sample_size=metric_sample_size,
            )

            src_coeff = torch.relu(source_temp).detach()
            tgt_coeff = torch.relu(target_temp).detach()
            warmup_rows.append(
                {
                    "epoch": epoch,
                    "probe_mmd": _to_float(probe_mmd),
                    "real_mmd": real_mmd,
                    "real_cmmd": real_cmmd,
                    "source_coeff_l1": float(src_coeff.abs().sum().item()),
                    "target_coeff_l1": float(tgt_coeff.abs().sum().item()),
                }
            )
            for k in range(num_terms):
                coeff_rows.append(
                    {"epoch": epoch, "domain": "source", "basis": k, "coeff": float(src_coeff[k].item())}
                )
                coeff_rows.append(
                    {"epoch": epoch, "domain": "target", "basis": k, "coeff": float(tgt_coeff[k].item())}
                )

    return torch.relu(source_temp).detach(), torch.relu(target_temp).detach(), warmup_rows, coeff_rows


def _train_classifier_on_operator(
    *,
    operator_name: str,
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    source_data,
    target_data,
    source_train_mask: torch.Tensor,
    hid_dim: int,
    mlp_layers: int,
    dropout: float,
    activation: str,
    mlp_epochs: int,
    lr: float,
    weight_decay: float,
    kernel_mul: float,
    kernel_num: int,
    fix_sigma: float | None,
    metric_sample_size: int,
    cmmd_target_labels_mode: str,
) -> tuple[list[dict], dict]:
    model = FeatureMLP(
        in_dim=int(source_feat.size(1)),
        hid_dim=hid_dim,
        num_classes=int(torch.cat([source_data.y, target_data.y], dim=0).max().item()) + 1,
        num_layers=mlp_layers,
        dropout=dropout,
        activation=activation,
    ).to(source_feat.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    input_mmd, input_cmmd = _compute_mmd_cmmd(
        source_feat,
        target_feat,
        source_data.y,
        target_data.y,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
        metric_sample_size=metric_sample_size,
    )

    epoch_rows = []
    final_source_acc = 0.0
    final_target_acc = 0.0
    final_latent_mmd = 0.0
    final_latent_cmmd = 0.0

    for epoch in range(1, mlp_epochs + 1):
        model.train()
        optimizer.zero_grad()

        source_logits, _ = model(source_feat)
        cls_loss = F.cross_entropy(source_logits[source_train_mask], source_data.y[source_train_mask])
        cls_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            source_logits, source_z = model(source_feat)
            target_logits, target_z = model(target_feat)
            target_y_for_cmmd = _cmmd_target_labels(
                cmmd_target_labels_mode, target_logits, target_data.y
            )
            latent_mmd, latent_cmmd = _compute_mmd_cmmd(
                source_z,
                target_z,
                source_data.y,
                target_y_for_cmmd,
                kernel_mul=kernel_mul,
                kernel_num=kernel_num,
                fix_sigma=fix_sigma,
                metric_sample_size=metric_sample_size,
            )

            source_acc = _accuracy(source_logits, source_data.y)
            target_acc = _accuracy(target_logits, target_data.y)

            epoch_rows.append(
                {
                    "operator": operator_name,
                    "epoch": epoch,
                    "cls_loss": _to_float(cls_loss),
                    "source_acc": source_acc,
                    "target_acc": target_acc,
                    "latent_mmd": latent_mmd,
                    "latent_cmmd": latent_cmmd,
                }
            )

            final_source_acc = source_acc
            final_target_acc = target_acc
            final_latent_mmd = latent_mmd
            final_latent_cmmd = latent_cmmd

    summary = {
        "operator": operator_name,
        "input_feature_mmd": input_mmd,
        "input_conditional_mmd": input_cmmd,
        "final_source_acc": final_source_acc,
        "final_target_acc": final_target_acc,
        "final_acc_degradation_source_minus_target": final_source_acc - final_target_acc,
        "final_latent_feature_mmd": final_latent_mmd,
        "final_latent_conditional_mmd": final_latent_cmmd,
    }
    return epoch_rows, summary


def _plot_warmup(rows: list[dict], out_png: str) -> None:
    if not rows:
        return
    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(epochs, [r["probe_mmd"] for r in rows], linewidth=1.8, label="Probe MMD")
    axes[0].plot(epochs, [r["real_mmd"] for r in rows], linewidth=1.8, label="Real Feature MMD")
    axes[0].plot(epochs, [r["real_cmmd"] for r in rows], linewidth=1.8, label="Real Conditional MMD")
    axes[0].set_ylabel("Distance")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [r["source_coeff_l1"] for r in rows], linewidth=1.8, label="Source coeff L1")
    axes[1].plot(epochs, [r["target_coeff_l1"] for r in rows], linewidth=1.8, label="Target coeff L1")
    axes[1].set_ylabel("L1 Norm")
    axes[1].set_xlabel("Warm-up Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_coeffs(rows: list[dict], out_png: str) -> None:
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
        ax.set_title(f"{domain.capitalize()} R-operator Coefficients")
        ax.set_ylabel("Coeff")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Warm-up Epoch")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=max(1, min(6, len(labels))), loc="upper center")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_classifier_curves(rows: list[dict], out_png: str) -> None:
    operators = sorted({r["operator"] for r in rows})
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    for op in operators:
        series = [r for r in rows if r["operator"] == op]
        series.sort(key=lambda x: x["epoch"])
        x = [r["epoch"] for r in series]
        axes[0].plot(x, [r["target_acc"] for r in series], linewidth=1.8, label=op)
        axes[1].plot(x, [r["latent_mmd"] for r in series], linewidth=1.8, label=op)
        axes[2].plot(x, [r["latent_cmmd"] for r in series], linewidth=1.8, label=op)

    axes[0].set_ylabel("Target Acc")
    axes[1].set_ylabel("Latent Feature MMD")
    axes[2].set_ylabel("Latent Conditional MMD")
    axes[2].set_xlabel("Epoch")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_summary(rows: list[dict], out_png: str) -> None:
    names = [r["operator"] for r in rows]
    x = torch.arange(len(names)).numpy()
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(x, [r["final_target_acc"] for r in rows], color="#1f77b4")
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Final Target Accuracy")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(
        x - width / 2,
        [r["final_latent_feature_mmd"] for r in rows],
        width=width,
        label="Feature MMD",
    )
    axes[1].bar(
        x + width / 2,
        [r["final_latent_conditional_mmd"] for r in rows],
        width=width,
        label="Conditional MMD",
    )
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("Distance")
    axes[1].set_title("Final Latent Distances")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
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
    source_train_mask = _source_train_mask(source_data, args.use_source_train_mask)

    source_x = source_data.x.detach()
    target_x = target_data.x.detach()

    cheb_basis_size = args.poly_order + 1
    cheb_prop = ChebProp(cheb_basis_size, is_source_domain=False).to(device)

    r_source_temp, r_target_temp, warmup_rows, coeff_rows = _warmup_r_filters(
        source_x=source_x,
        target_x=target_x,
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
        metric_sample_size=args.metric_sample_size,
    )

    p_temp = torch.zeros((cheb_basis_size,), device=device)
    p_temp[-1] = 1.0  # fixed highest-degree term only (degree = poly_order; A^3-like when poly_order=3)

    with torch.no_grad():
        rs_x = _apply_operator(source_x, source_data, r_source_temp, cheb_prop, args.cheb_lambda_max).detach()
        rt_x = _apply_operator(target_x, target_data, r_target_temp, cheb_prop, args.cheb_lambda_max).detach()
        ps_x = _apply_operator(source_x, source_data, p_temp, cheb_prop, args.cheb_lambda_max).detach()
        pt_x = _apply_operator(target_x, target_data, p_temp, cheb_prop, args.cheb_lambda_max).detach()
        ps_avg_x = _apply_monomial_average_operator(
            source_x, source_data, cheb_prop, args.cheb_lambda_max, max_degree=args.p_avg_degree
        ).detach()
        pt_avg_x = _apply_monomial_average_operator(
            target_x, target_data, cheb_prop, args.cheb_lambda_max, max_degree=args.p_avg_degree
        ).detach()

    operator_rows = []
    r_input_mmd, r_input_cmmd = _compute_mmd_cmmd(
        rs_x,
        rt_x,
        source_data.y,
        target_data.y,
        kernel_mul=args.kernel_mul,
        kernel_num=args.kernel_num,
        fix_sigma=args.fix_sigma,
        metric_sample_size=args.metric_sample_size,
    )
    operator_rows.append(
        {"operator": "R_aligned", "feature_mmd": r_input_mmd, "conditional_mmd": r_input_cmmd}
    )
    p_input_mmd, p_input_cmmd = _compute_mmd_cmmd(
        ps_x,
        pt_x,
        source_data.y,
        target_data.y,
        kernel_mul=args.kernel_mul,
        kernel_num=args.kernel_num,
        fix_sigma=args.fix_sigma,
        metric_sample_size=args.metric_sample_size,
    )
    operator_rows.append(
        {"operator": "P_fixed_highest_degree", "feature_mmd": p_input_mmd, "conditional_mmd": p_input_cmmd}
    )
    p_avg_input_mmd, p_avg_input_cmmd = _compute_mmd_cmmd(
        ps_avg_x,
        pt_avg_x,
        source_data.y,
        target_data.y,
        kernel_mul=args.kernel_mul,
        kernel_num=args.kernel_num,
        fix_sigma=args.fix_sigma,
        metric_sample_size=args.metric_sample_size,
    )
    operator_rows.append(
        {
            "operator": "P_fixed_avg_monomial",
            "feature_mmd": p_avg_input_mmd,
            "conditional_mmd": p_avg_input_cmmd,
        }
    )

    r_train_rows, r_summary = _train_classifier_on_operator(
        operator_name="R_aligned",
        source_feat=rs_x,
        target_feat=rt_x,
        source_data=source_data,
        target_data=target_data,
        source_train_mask=source_train_mask,
        hid_dim=args.hid_dim,
        mlp_layers=args.mlp_layers,
        dropout=args.dropout,
        activation=args.activation,
        mlp_epochs=args.mlp_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        kernel_mul=args.kernel_mul,
        kernel_num=args.kernel_num,
        fix_sigma=args.fix_sigma,
        metric_sample_size=args.metric_sample_size,
        cmmd_target_labels_mode=args.cmmd_target_labels,
    )
    p_train_rows, p_summary = _train_classifier_on_operator(
        operator_name="P_fixed_highest_degree",
        source_feat=ps_x,
        target_feat=pt_x,
        source_data=source_data,
        target_data=target_data,
        source_train_mask=source_train_mask,
        hid_dim=args.hid_dim,
        mlp_layers=args.mlp_layers,
        dropout=args.dropout,
        activation=args.activation,
        mlp_epochs=args.mlp_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        kernel_mul=args.kernel_mul,
        kernel_num=args.kernel_num,
        fix_sigma=args.fix_sigma,
        metric_sample_size=args.metric_sample_size,
        cmmd_target_labels_mode=args.cmmd_target_labels,
    )
    p_avg_train_rows, p_avg_summary = _train_classifier_on_operator(
        operator_name="P_fixed_avg_monomial",
        source_feat=ps_avg_x,
        target_feat=pt_avg_x,
        source_data=source_data,
        target_data=target_data,
        source_train_mask=source_train_mask,
        hid_dim=args.hid_dim,
        mlp_layers=args.mlp_layers,
        dropout=args.dropout,
        activation=args.activation,
        mlp_epochs=args.mlp_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        kernel_mul=args.kernel_mul,
        kernel_num=args.kernel_num,
        fix_sigma=args.fix_sigma,
        metric_sample_size=args.metric_sample_size,
        cmmd_target_labels_mode=args.cmmd_target_labels,
    )

    summary_rows = [r_summary, p_summary, p_avg_summary]
    best_r_target = r_summary["final_target_acc"]
    for row in summary_rows:
        row["target_acc_drop_vs_R"] = best_r_target - row["final_target_acc"]
        row["feature_mmd_delta_vs_R"] = row["final_latent_feature_mmd"] - r_summary["final_latent_feature_mmd"]
        row["conditional_mmd_delta_vs_R"] = (
            row["final_latent_conditional_mmd"] - r_summary["final_latent_conditional_mmd"]
        )

    os.makedirs(args.out_dir, exist_ok=True)
    warmup_csv = os.path.join(args.out_dir, "r_warmup_metrics.csv")
    coeff_csv = os.path.join(args.out_dir, "r_coefficients.csv")
    operator_csv = os.path.join(args.out_dir, "operator_space_mmd_cmmd.csv")
    train_csv = os.path.join(args.out_dir, "classifier_training_metrics.csv")
    summary_csv = os.path.join(args.out_dir, "classifier_summary.csv")
    summary_main_csv = os.path.join(args.out_dir, "classifier_summary_main.csv")

    _write_csv(
        warmup_csv,
        warmup_rows,
        fieldnames=[
            "epoch",
            "probe_mmd",
            "real_mmd",
            "real_cmmd",
            "source_coeff_l1",
            "target_coeff_l1",
        ],
    )
    _write_csv(coeff_csv, coeff_rows, fieldnames=["epoch", "domain", "basis", "coeff"])
    _write_csv(
        operator_csv,
        operator_rows,
        fieldnames=["operator", "feature_mmd", "conditional_mmd"],
    )
    _write_csv(
        train_csv,
        r_train_rows + p_train_rows + p_avg_train_rows,
        fieldnames=[
            "operator",
            "epoch",
            "cls_loss",
            "source_acc",
            "target_acc",
            "latent_mmd",
            "latent_cmmd",
        ],
    )
    summary_main_rows = [
        {
            "op": row["operator"],
            "tacc": row["final_target_acc"],
            "acc_drop": row["target_acc_drop_vs_R"],
            "fmmd": row["final_latent_feature_mmd"],
            "cmmd": row["final_latent_conditional_mmd"],
            "dfmmd": row["feature_mmd_delta_vs_R"],
            "dcmmd": row["conditional_mmd_delta_vs_R"],
        }
        for row in summary_rows
    ]
    _write_csv(
        summary_main_csv,
        summary_main_rows,
        fieldnames=["op", "tacc", "acc_drop", "fmmd", "cmmd", "dfmmd", "dcmmd"],
    )
    _write_csv(
        summary_csv,
        summary_rows,
        fieldnames=[
            "operator",
            "input_feature_mmd",
            "input_conditional_mmd",
            "final_source_acc",
            "final_target_acc",
            "final_acc_degradation_source_minus_target",
            "final_latent_feature_mmd",
            "final_latent_conditional_mmd",
            "target_acc_drop_vs_R",
            "feature_mmd_delta_vs_R",
            "conditional_mmd_delta_vs_R",
        ],
    )

    warmup_png = os.path.join(args.out_dir, "r_warmup_curves.png")
    coeff_png = os.path.join(args.out_dir, "r_coefficients.png")
    train_png = os.path.join(args.out_dir, "classifier_curves.png")
    summary_png = os.path.join(args.out_dir, "classifier_summary.png")
    _plot_warmup(warmup_rows, warmup_png)
    _plot_coeffs(coeff_rows, coeff_png)
    _plot_classifier_curves(r_train_rows + p_train_rows + p_avg_train_rows, train_png)
    _plot_summary(summary_rows, summary_png)

    print(f"Saved warm-up CSV: {warmup_csv}")
    print(f"Saved coefficient CSV: {coeff_csv}")
    print(f"Saved operator-space MMD/cMMD CSV: {operator_csv}")
    print(f"Saved classifier training CSV: {train_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved concise summary CSV: {summary_main_csv}")
    print(f"Saved warm-up plot: {warmup_png}")
    print(f"Saved coefficient plot: {coeff_png}")
    print(f"Saved classifier curves plot: {train_png}")
    print(f"Saved summary plot: {summary_png}")
    print("Final summary:")
    for row in summary_rows:
        print(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Airport-default ablation: compare R (probe-aligned Chebyshev operator) vs "
            "P (fixed highest-degree operator), then freeze operators and train source-only MLP."
        )
    )
    parser.add_argument("--dataset", type=str, default="airport")
    parser.add_argument("--source", type=str, default="BRAZIL")
    parser.add_argument("--target", type=str, default="USA")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--poly_order", type=int, default=3)
    parser.add_argument(
        "--p_avg_degree",
        type=int,
        default=3,
        help="Degree used in fixed monomial-average operator: (A + ... + A^d)/d.",
    )
    parser.add_argument("--cheb_lambda_max", type=float, default=2.0)

    parser.add_argument("--warmup_epochs", type=int, default=100)
    parser.add_argument("--align_lr", type=float, default=0.01)
    parser.add_argument("--align_weight_decay", type=float, default=0.0)

    parser.add_argument("--mlp_layers", type=int, default=2, choices=[2, 3])
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--mlp_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--use_source_train_mask", action="store_true")

    parser.add_argument("--kernel_mul", type=float, default=2.0)
    parser.add_argument("--kernel_num", type=int, default=5)
    parser.add_argument("--fix_sigma", type=float, default=None)
    parser.add_argument("--metric_sample_size", type=int, default=2000)
    parser.add_argument("--cmmd_target_labels", type=str, default="true", choices=["true", "pseudo"])

    parser.add_argument(
        "--out_dir",
        type=str,
        default="./__saved__/analysis/airport_cheb_rp_ablation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
