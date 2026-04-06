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
        raise ValueError("This ablation requires explicit node features in both source and target.")
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


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


def _to_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def _build_props(num_layers: int, cheb_k: int, device: str):
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    return nn.ModuleList(
        [ChebProp(cheb_k, is_source_domain=False).to(device) for _ in range(num_layers)]
    )


def _apply_prop(
    x: torch.Tensor,
    data,
    temp: torch.Tensor,
    props: nn.ModuleList,
    lambda_max: float,
) -> torch.Tensor:
    out = x
    edge_weight = getattr(data, "edge_weight", None)
    batch = getattr(data, "batch", None)
    for prop in props:
        out = prop(
            out,
            data.edge_index,
            edge_weight=edge_weight,
            batch=batch,
            lambda_max=lambda_max,
            temp=temp,
        )
    return out


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


def _cmmd_target_labels(
    mode: str,
    target_logits: torch.Tensor,
    target_true: torch.Tensor,
) -> torch.Tensor:
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


def _plot_case_training(rows: list[dict], out_png: str) -> None:
    cases = sorted({r["case"] for r in rows})
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    for case in cases:
        series = [r for r in rows if r["case"] == case]
        series.sort(key=lambda x: x["epoch"])
        x = [r["epoch"] for r in series]
        axes[0].plot(x, [r["target_acc"] for r in series], linewidth=1.8, label=case)
        axes[1].plot(x, [r["latent_mmd"] for r in series], linewidth=1.8, label=case)
        axes[2].plot(x, [r["latent_cmmd"] for r in series], linewidth=1.8, label=case)

    axes[0].set_ylabel("Target Acc")
    axes[1].set_ylabel("Latent MMD")
    axes[2].set_ylabel("Latent cMMD")
    axes[2].set_xlabel("Epoch")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_case_summary(rows: list[dict], out_png: str) -> None:
    names = [r["case"] for r in rows]
    x = torch.arange(len(names)).numpy()
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(x, [r["final_target_acc"] for r in rows], color="#1f77b4")
    axes[0].set_xticks(x, names, rotation=15)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Final Target Accuracy")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x - width / 2, [r["final_latent_mmd"] for r in rows], width=width, label="MMD")
    axes[1].bar(
        x + width / 2, [r["final_latent_cmmd"] for r in rows], width=width, label="Conditional MMD"
    )
    axes[1].set_xticks(x, names, rotation=15)
    axes[1].set_ylabel("Distance")
    axes[1].set_title("Final Latent Distances")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_warmup(rows: list[dict], out_png: str) -> None:
    if not rows:
        return
    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(epochs, [r["probe_mmd"] for r in rows], label="Probe MMD", linewidth=1.8)
    axes[0].plot(epochs, [r["real_mmd"] for r in rows], label="Real MMD", linewidth=1.8)
    axes[0].plot(epochs, [r["real_cmmd"] for r in rows], label="Real cMMD", linewidth=1.8)
    axes[0].set_ylabel("Distance")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [r["source_coeff_l1"] for r in rows], label="Source coeff L1", linewidth=1.8)
    axes[1].plot(epochs, [r["target_coeff_l1"] for r in rows], label="Target coeff L1", linewidth=1.8)
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
            series = [r for r in rows if r["domain"] == domain and int(r["basis"]) == b]
            series.sort(key=lambda x: x["epoch"])
            ax.plot(
                [r["epoch"] for r in series],
                [r["coeff"] for r in series],
                linewidth=1.7,
                color=colors[b % len(colors)],
                label=f"T{b}",
            )
        ax.set_title(f"{domain.capitalize()} Aligned Filter Coefficients")
        ax.set_ylabel("Coeff")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Warm-up Epoch")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=max(1, min(6, len(labels))), loc="upper center")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _train_case(
    *,
    case_name: str,
    source_input: torch.Tensor,
    target_input: torch.Tensor,
    source_data,
    target_data,
    source_train_mask: torch.Tensor,
    args,
) -> tuple[list[dict], dict]:
    model = FeatureMLP(
        in_dim=int(source_input.size(1)),
        hid_dim=args.hid_dim,
        num_classes=int(torch.cat([source_data.y, target_data.y], dim=0).max().item()) + 1,
        num_layers=args.mlp_layers,
        dropout=args.dropout,
        activation=args.activation,
    ).to(source_input.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    training_rows = []
    final_latent_mmd = 0.0
    final_latent_cmmd = 0.0
    final_source_acc = 0.0
    final_target_acc = 0.0

    input_mmd, input_cmmd = _compute_mmd_cmmd(
        source_input,
        target_input,
        source_data.y,
        target_data.y,
        kernel_mul=args.kernel_mul,
        kernel_num=args.kernel_num,
        fix_sigma=args.fix_sigma,
        metric_sample_size=args.metric_sample_size,
    )

    for epoch in range(1, args.mlp_epochs + 1):
        model.train()
        optimizer.zero_grad()
        source_logits, _ = model(source_input)
        cls_loss = F.cross_entropy(source_logits[source_train_mask], source_data.y[source_train_mask])
        cls_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            source_logits, source_z = model(source_input)
            target_logits, target_z = model(target_input)

            target_y_for_cmmd = _cmmd_target_labels(
                args.cmmd_target_labels, target_logits, target_data.y
            )
            latent_mmd, latent_cmmd = _compute_mmd_cmmd(
                source_z,
                target_z,
                source_data.y,
                target_y_for_cmmd,
                kernel_mul=args.kernel_mul,
                kernel_num=args.kernel_num,
                fix_sigma=args.fix_sigma,
                metric_sample_size=args.metric_sample_size,
            )
            source_acc = _accuracy(source_logits, source_data.y)
            target_acc = _accuracy(target_logits, target_data.y)

            training_rows.append(
                {
                    "case": case_name,
                    "epoch": epoch,
                    "cls_loss": _to_float(cls_loss),
                    "source_acc": source_acc,
                    "target_acc": target_acc,
                    "latent_mmd": latent_mmd,
                    "latent_cmmd": latent_cmmd,
                }
            )

            final_latent_mmd = latent_mmd
            final_latent_cmmd = latent_cmmd
            final_source_acc = source_acc
            final_target_acc = target_acc

    summary = {
        "case": case_name,
        "input_mmd": input_mmd,
        "input_cmmd": input_cmmd,
        "final_source_acc": final_source_acc,
        "final_target_acc": final_target_acc,
        "final_latent_mmd": final_latent_mmd,
        "final_latent_cmmd": final_latent_cmmd,
    }
    return training_rows, summary


def _warmup_align_filters(
    *,
    source_input: torch.Tensor,
    target_input: torch.Tensor,
    source_data,
    target_data,
    props: nn.ModuleList,
    args,
) -> tuple[torch.Tensor, torch.Tensor, list[dict], list[dict]]:
    source_temp = nn.Parameter(
        torch.full((args.cheb_k,), 1.0 / float(args.cheb_k), device=source_input.device)
    )
    target_temp = nn.Parameter(
        torch.full((args.cheb_k,), 1.0 / float(args.cheb_k), device=source_input.device)
    )
    optimizer = torch.optim.Adam(
        [source_temp, target_temp], lr=args.align_lr, weight_decay=args.align_weight_decay
    )

    warmup_rows = []
    coeff_rows = []

    all_features = torch.cat([source_input, target_input], dim=0).detach()
    probe_mean = all_features.mean(dim=0, keepdim=True)
    probe_std = all_features.std(dim=0, keepdim=True) + 1e-6

    for epoch in range(1, args.warmup_epochs + 1):
        optimizer.zero_grad()

        probe_features = torch.randn_like(all_features) * probe_std + probe_mean
        probe_source = probe_features[: source_input.size(0)]
        probe_target = probe_features[source_input.size(0) :]

        probe_source_out = _apply_prop(
            probe_source,
            source_data,
            temp=source_temp,
            props=props,
            lambda_max=args.cheb_lambda_max,
        )
        probe_target_out = _apply_prop(
            probe_target,
            target_data,
            temp=target_temp,
            props=props,
            lambda_max=args.cheb_lambda_max,
        )
        probe_mmd = mmd_rbf(
            probe_source_out,
            probe_target_out,
            kernel_mul=args.kernel_mul,
            kernel_num=args.kernel_num,
            fix_sigma=args.fix_sigma,
        )
        probe_mmd.backward()
        optimizer.step()

        with torch.no_grad():
            source_real = _apply_prop(
                source_input,
                source_data,
                temp=source_temp,
                props=props,
                lambda_max=args.cheb_lambda_max,
            )
            target_real = _apply_prop(
                target_input,
                target_data,
                temp=target_temp,
                props=props,
                lambda_max=args.cheb_lambda_max,
            )
            real_mmd, real_cmmd = _compute_mmd_cmmd(
                source_real,
                target_real,
                source_data.y,
                target_data.y,
                kernel_mul=args.kernel_mul,
                kernel_num=args.kernel_num,
                fix_sigma=args.fix_sigma,
                metric_sample_size=args.metric_sample_size,
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
            for k in range(args.cheb_k):
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

    return source_temp.detach(), target_temp.detach(), warmup_rows, coeff_rows


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
    props = _build_props(args.prop_layers, args.cheb_k, device)

    with torch.no_grad():
        source_raw = source_data.x.detach()
        target_raw = target_data.x.detach()

    all_training_rows = []
    summary_rows = []

    case1_rows, case1_summary = _train_case(
        case_name="case1_features_raw",
        source_input=source_raw,
        target_input=target_raw,
        source_data=source_data,
        target_data=target_data,
        source_train_mask=source_train_mask,
        args=args,
    )
    all_training_rows.extend(case1_rows)
    summary_rows.append(case1_summary)

    unaligned_temp = torch.full((args.cheb_k,), 1.0 / float(args.cheb_k), device=device)
    with torch.no_grad():
        source_unaligned = _apply_prop(
            source_raw,
            source_data,
            temp=unaligned_temp,
            props=props,
            lambda_max=args.cheb_lambda_max,
        ).detach()
        target_unaligned = _apply_prop(
            target_raw,
            target_data,
            temp=unaligned_temp,
            props=props,
            lambda_max=args.cheb_lambda_max,
        ).detach()

    case2_rows, case2_summary = _train_case(
        case_name="case2_unaligned_prop_frozen",
        source_input=source_unaligned,
        target_input=target_unaligned,
        source_data=source_data,
        target_data=target_data,
        source_train_mask=source_train_mask,
        args=args,
    )
    all_training_rows.extend(case2_rows)
    summary_rows.append(case2_summary)

    source_aligned_temp, target_aligned_temp, warmup_rows, coeff_rows = _warmup_align_filters(
        source_input=source_raw,
        target_input=target_raw,
        source_data=source_data,
        target_data=target_data,
        props=props,
        args=args,
    )
    with torch.no_grad():
        source_aligned = _apply_prop(
            source_raw,
            source_data,
            temp=source_aligned_temp,
            props=props,
            lambda_max=args.cheb_lambda_max,
        ).detach()
        target_aligned = _apply_prop(
            target_raw,
            target_data,
            temp=target_aligned_temp,
            props=props,
            lambda_max=args.cheb_lambda_max,
        ).detach()

    case3_rows, case3_summary = _train_case(
        case_name="case3_aligned_prop_warmup_frozen",
        source_input=source_aligned,
        target_input=target_aligned,
        source_data=source_data,
        target_data=target_data,
        source_train_mask=source_train_mask,
        args=args,
    )
    all_training_rows.extend(case3_rows)
    summary_rows.append(case3_summary)

    base_target_acc = case1_summary["final_target_acc"]
    for row in summary_rows:
        row["target_acc_drop_vs_case1"] = base_target_acc - row["final_target_acc"]

    os.makedirs(args.out_dir, exist_ok=True)
    training_csv = os.path.join(args.out_dir, "case_training_metrics.csv")
    summary_csv = os.path.join(args.out_dir, "case_summary.csv")
    warmup_csv = os.path.join(args.out_dir, "filter_warmup_metrics.csv")
    coeff_csv = os.path.join(args.out_dir, "filter_warmup_coefficients.csv")

    _write_csv(
        training_csv,
        all_training_rows,
        fieldnames=["case", "epoch", "cls_loss", "source_acc", "target_acc", "latent_mmd", "latent_cmmd"],
    )
    _write_csv(
        summary_csv,
        summary_rows,
        fieldnames=[
            "case",
            "input_mmd",
            "input_cmmd",
            "final_source_acc",
            "final_target_acc",
            "final_latent_mmd",
            "final_latent_cmmd",
            "target_acc_drop_vs_case1",
        ],
    )
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
    _write_csv(
        coeff_csv,
        coeff_rows,
        fieldnames=["epoch", "domain", "basis", "coeff"],
    )

    case_training_png = os.path.join(args.out_dir, "case_training_curves.png")
    case_summary_png = os.path.join(args.out_dir, "case_summary.png")
    warmup_png = os.path.join(args.out_dir, "filter_warmup_curves.png")
    coeff_png = os.path.join(args.out_dir, "filter_warmup_coefficients.png")
    _plot_case_training(all_training_rows, case_training_png)
    _plot_case_summary(summary_rows, case_summary_png)
    _plot_warmup(warmup_rows, warmup_png)
    _plot_coeffs(coeff_rows, coeff_png)

    print(f"Saved case training CSV: {training_csv}")
    print(f"Saved case summary CSV: {summary_csv}")
    print(f"Saved warm-up CSV: {warmup_csv}")
    print(f"Saved warm-up coeff CSV: {coeff_csv}")
    print(f"Saved case training plot: {case_training_png}")
    print(f"Saved case summary plot: {case_summary_png}")
    print(f"Saved warm-up plot: {warmup_png}")
    print(f"Saved coeff plot: {coeff_png}")
    print("Final case summary:")
    for row in summary_rows:
        print(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Source-only supervised ablation with three cases: raw features, "
            "unaligned propagated features (frozen), aligned propagated features "
            "(probe warm-up then frozen)."
        )
    )
    parser.add_argument("--dataset", type=str, default="citation")
    parser.add_argument("--source", type=str, default="ACMv9")
    parser.add_argument("--target", type=str, default="Citationv1")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--mlp_layers", type=int, default=2, choices=[2, 3])
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--mlp_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--use_source_train_mask", action="store_true")

    parser.add_argument("--prop_layers", type=int, default=2)
    parser.add_argument("--cheb_k", type=int, default=5)
    parser.add_argument("--cheb_lambda_max", type=float, default=2.0)

    parser.add_argument("--warmup_epochs", type=int, default=100)
    parser.add_argument("--align_lr", type=float, default=0.01)
    parser.add_argument("--align_weight_decay", type=float, default=0.0)

    parser.add_argument("--kernel_mul", type=float, default=2.0)
    parser.add_argument("--kernel_num", type=int, default=5)
    parser.add_argument("--fix_sigma", type=float, default=None)
    parser.add_argument("--metric_sample_size", type=int, default=2000)
    parser.add_argument("--cmmd_target_labels", type=str, default="true", choices=["true", "pseudo"])

    parser.add_argument(
        "--out_dir",
        type=str,
        default="./__saved__/analysis/mlp_structure_mmd_cmmd_ablation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
