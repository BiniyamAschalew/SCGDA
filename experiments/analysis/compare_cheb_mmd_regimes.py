import argparse
import csv
import os
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from  data.build_dataset import build_dataset
from  models.ours.test.test_base import TestBase
from  utils.config_utils import build_config
from  utils.expt_utils import set_seed
from  utils.filter_utils import cheb_to_monomial, tensor_to_float_list
from  utils.train_utils.mmd import MMD


def _edge_discriminability_loss(z, edge_index, num_samples=2048, margin=1.0):
    """
    Encourage edge pairs to be closer than random non-edge pairs.
    Loss: mean(relu(margin + d_pos - d_neg)).
    """
    if edge_index.numel() == 0 or z.size(0) <= 1:
        return z.new_tensor(0.0)

    e = edge_index.size(1)
    k = min(num_samples, e)
    edge_ids = torch.randint(e, (k,), device=edge_index.device)
    pos_u = edge_index[0, edge_ids]
    pos_v = edge_index[1, edge_ids]

    num_nodes = z.size(0)
    neg_u = torch.randint(num_nodes, (k,), device=z.device)
    neg_v = torch.randint(num_nodes, (k,), device=z.device)
    same = neg_u == neg_v
    neg_v = torch.where(same, (neg_v + 1) % num_nodes, neg_v)

    pos_dist = (z[pos_u] - z[pos_v]).pow(2).sum(dim=1)
    neg_dist = (z[neg_u] - z[neg_v]).pow(2).sum(dim=1)
    return F.relu(margin + pos_dist - neg_dist).mean()


def _load_blog_pair(source, target, device, seed):
    config_setup = {
        "data": "blog",
        "expt": "default",
        "model": "test",
    }
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

    config["model"]["in_dim"] = source_data.x.shape[1]
    config["model"]["num_classes"] = int(source_data.y.unique().numel())

    return config, source_data, target_data


def _forward_and_losses(
    model,
    source_data,
    target_data,
    regime,
    mmd_weight,
    cls_weight,
    edge_disc_weight,
    edge_disc_margin,
    edge_disc_samples,
):
    source_batch = getattr(source_data, "batch", None)
    target_batch = getattr(target_data, "batch", None)
    source_edge_weight = getattr(source_data, "edge_weight", None)
    target_edge_weight = getattr(target_data, "edge_weight", None)

    source_features = model.feat_bottleneck(
        source_data.x,
        source_data.edge_index,
        edge_weight=source_edge_weight,
        batch=source_batch,
        domain="source",
    )
    target_features = model.feat_bottleneck(
        target_data.x,
        target_data.edge_index,
        edge_weight=target_edge_weight,
        batch=target_batch,
        domain="target",
    )

    source_logits = model.feat_classifier(
        source_features,
        source_data.edge_index,
        edge_weight=source_edge_weight,
        domain="source",
    )
    source_logits = F.log_softmax(source_logits, dim=1)
    cls_loss = F.nll_loss(source_logits, source_data.y)

    if regime == "feature_mmd":
        mmd_loss = MMD(source_features, target_features)
        edge_disc_loss = source_features.new_tensor(0.0)
    elif regime == "filter_mmd":
        # Same probe construction style as current test.py implementation.
        all_features = torch.cat((source_features, target_features), dim=0).detach()
        probe_mean = all_features.mean(dim=0, keepdim=True)
        probe_std = all_features.std(dim=0, keepdim=True) + 1e-6
        probe_features = torch.randn_like(all_features) * probe_std + probe_mean
        source_probe = probe_features[: source_features.size(0)]
        target_probe = probe_features[source_features.size(0) :]

        source_probe_out = model.filter_bottleneck(
            source_probe,
            source_data.edge_index,
            edge_weight=source_edge_weight,
            batch=source_batch,
            domain="source",
        )
        target_probe_out = model.filter_bottleneck(
            target_probe,
            target_data.edge_index,
            edge_weight=target_edge_weight,
            batch=target_batch,
            domain="target",
        )
        mmd_loss = MMD(source_probe_out, target_probe_out)
        edge_disc_src = _edge_discriminability_loss(
            source_probe_out,
            source_data.edge_index,
            num_samples=edge_disc_samples,
            margin=edge_disc_margin,
        )
        edge_disc_tgt = _edge_discriminability_loss(
            target_probe_out,
            target_data.edge_index,
            num_samples=edge_disc_samples,
            margin=edge_disc_margin,
        )
        edge_disc_loss = 0.5 * (edge_disc_src + edge_disc_tgt)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    total_loss = cls_weight * cls_loss + mmd_weight * mmd_loss + edge_disc_weight * edge_disc_loss

    return total_loss, cls_loss.detach(), mmd_loss.detach(), edge_disc_loss.detach()


def _train_regime(
    base_config,
    source_data,
    target_data,
    regime,
    epochs,
    lr,
    weight_decay,
    mmd_weight,
    cls_weight,
    edge_disc_weight,
    edge_disc_margin,
    edge_disc_samples,
    seed,
):
    set_seed(seed)
    config = deepcopy(base_config)
    model = TestBase(config).to(config["expt"]["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    weight_rows = []
    metric_rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, cls_loss, mmd_loss, edge_disc_loss = _forward_and_losses(
            model,
            source_data,
            target_data,
            regime=regime,
            mmd_weight=mmd_weight,
            cls_weight=cls_weight,
            edge_disc_weight=edge_disc_weight if regime == "filter_mmd" else 0.0,
            edge_disc_margin=edge_disc_margin,
            edge_disc_samples=edge_disc_samples,
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            source_coeff = torch.relu(model.source_temp).detach().cpu()
            target_coeff = torch.relu(model.target_temp).detach().cpu()

            source_mono = cheb_to_monomial(source_coeff)
            target_mono = cheb_to_monomial(target_coeff)

            metric_rows.append(
                {
                    "regime": regime,
                    "epoch": epoch,
                    "total_loss": float(total_loss.detach().cpu().item()),
                    "cls_loss": float(cls_loss.cpu().item()),
                    "mmd_loss": float(mmd_loss.cpu().item()),
                    "edge_disc_loss": float(edge_disc_loss.cpu().item()),
                    "source_coeff_l1": float(source_coeff.abs().sum().item()),
                    "target_coeff_l1": float(target_coeff.abs().sum().item()),
                    "source_mono_l1": float(source_mono.abs().sum().item()),
                    "target_mono_l1": float(target_mono.abs().sum().item()),
                }
            )

            for k in range(source_coeff.numel()):
                weight_rows.append(
                    {
                        "regime": regime,
                        "epoch": epoch,
                        "domain": "source",
                        "basis": k,
                        "cheb_coeff": float(source_coeff[k].item()),
                        "mono_coeff": float(source_mono[k].item()),
                    }
                )
                weight_rows.append(
                    {
                        "regime": regime,
                        "epoch": epoch,
                        "domain": "target",
                        "basis": k,
                        "cheb_coeff": float(target_coeff[k].item()),
                        "mono_coeff": float(target_mono[k].item()),
                    }
                )

    final_source = torch.relu(model.source_temp.detach()).cpu()
    final_target = torch.relu(model.target_temp.detach()).cpu()
    print(
        f"[{regime}] final source cheb: {tensor_to_float_list(final_source)} | "
        f"target cheb: {tensor_to_float_list(final_target)}"
    )

    return weight_rows, metric_rows


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_weight_trajectories(weight_rows, out_png):
    regimes = sorted({r["regime"] for r in weight_rows})
    bases = sorted({r["basis"] for r in weight_rows})
    domains = ["source", "target"]
    colors = plt.cm.tab10.colors
    regime_style = {"feature_mmd": "-", "filter_mmd": "--"}

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    domain_to_ax = {"source": axes[0], "target": axes[1]}

    for domain in domains:
        ax = domain_to_ax[domain]
        for regime in regimes:
            for b in bases:
                rows = [
                    r
                    for r in weight_rows
                    if r["domain"] == domain and r["regime"] == regime and r["basis"] == b
                ]
                rows.sort(key=lambda x: x["epoch"])
                x = [r["epoch"] for r in rows]
                y = [r["cheb_coeff"] for r in rows]
                label = f"{regime}-T{b}"
                ax.plot(
                    x,
                    y,
                    linestyle=regime_style.get(regime, "-"),
                    color=colors[b % len(colors)],
                    linewidth=1.7,
                    label=label,
                )
        ax.set_title(f"{domain.capitalize()} Cheb Coefficients")
        ax.set_ylabel("Coefficient")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Epoch")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_loss_curves(metric_rows, out_png):
    regimes = sorted({r["regime"] for r in metric_rows})
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    keys = ["total_loss", "cls_loss", "mmd_loss", "edge_disc_loss"]

    for i, key in enumerate(keys):
        ax = axes[i]
        for regime in regimes:
            rows = [r for r in metric_rows if r["regime"] == regime]
            rows.sort(key=lambda x: x["epoch"])
            ax.plot(
                [r["epoch"] for r in rows],
                [r[key] for r in rows],
                label=regime,
                linewidth=1.8,
            )
        ax.set_ylabel(key)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Epoch")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(args):
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    config, source_data, target_data = _load_blog_pair(
        source=args.source,
        target=args.target,
        device=device,
        seed=args.seed,
    )

    # Force settings from CLI for fair comparison.
    config["model"]["num_layers"] = args.num_layers
    config["model"]["hid_dim"] = args.hid_dim
    config["model"]["dropout_ratio"] = args.dropout
    config["model"]["cheb_K"] = args.cheb_k
    config["model"]["cheb_lambda_max"] = args.cheb_lambda_max
    config["model"]["activation"] = args.activation
    config["model"]["mode"] = "node"

    os.makedirs(args.out_dir, exist_ok=True)

    all_weight_rows = []
    all_metric_rows = []

        for regime in ("feature_mmd", "filter_mmd"):
        w_rows, m_rows = _train_regime(
            base_config=config,
            source_data=source_data,
            target_data=target_data,
            regime=regime,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            mmd_weight=args.mmd_weight,
            cls_weight=args.cls_weight,
            edge_disc_weight=args.edge_disc_weight,
            edge_disc_margin=args.edge_disc_margin,
            edge_disc_samples=args.edge_disc_samples,
            seed=args.seed,
        )
        all_weight_rows.extend(w_rows)
        all_metric_rows.extend(m_rows)

    weights_csv = os.path.join(args.out_dir, "basis_weight_trajectories.csv")
    metrics_csv = os.path.join(args.out_dir, "training_metrics.csv")
    weights_png = os.path.join(args.out_dir, "basis_weight_trajectories.png")
    metrics_png = os.path.join(args.out_dir, "training_losses.png")

    _write_csv(
        weights_csv,
        all_weight_rows,
        fieldnames=["regime", "epoch", "domain", "basis", "cheb_coeff", "mono_coeff"],
    )
    _write_csv(
        metrics_csv,
        all_metric_rows,
        fieldnames=[
            "regime",
            "epoch",
            "total_loss",
            "cls_loss",
            "mmd_loss",
            "edge_disc_loss",
            "source_coeff_l1",
            "target_coeff_l1",
            "source_mono_l1",
            "target_mono_l1",
        ],
    )
    _plot_weight_trajectories(all_weight_rows, weights_png)
    _plot_loss_curves(all_metric_rows, metrics_png)

    print(f"Saved weights CSV: {weights_csv}")
    print(f"Saved metrics CSV: {metrics_csv}")
    print(f"Saved weights plot: {weights_png}")
    print(f"Saved loss plot: {metrics_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare Cheb filter collapse behavior under two DA objectives: "
            "feature-MMD vs filter-output-MMD."
        )
    )
    parser.add_argument("--source", type=str, default="Blog1")
    parser.add_argument("--target", type=str, default="Blog2")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--mmd_weight", type=float, default=0.1)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--edge_disc_weight", type=float, default=0.1)
    parser.add_argument("--edge_disc_margin", type=float, default=1.0)
    parser.add_argument("--edge_disc_samples", type=int, default=2048)

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--cheb_k", type=int, default=5)
    parser.add_argument("--cheb_lambda_max", type=float, default=2.0)

    parser.add_argument(
        "--out_dir",
        type=str,
        default="./__saved__/analysis/cheb_filter_mmd_comparison",
    )

    main(parser.parse_args())
