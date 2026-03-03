"""Fit mono/cheb/bern FDA filter heads to a black-box oracle in FDA embedding space."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.__filters.bern import BernProp
from models.__filters.cheb import ChebProp
from models.__filters.mono import MonoProp
from models.ours.fda.fda_base import FDABase
from utils.ablation_utils.common import load_pair
from utils.filter_utils import monomial_to_cheb

CFG = {
    "dataset": "blog",
    "domain": "Blog1",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "seed": 7,
    "k": 4,
    "epochs": 100,
    "lr": 5e-2,
    "batch_size": 16,
    "sample_mean_scale": 1.0,
    "sample_std_scale": 1.0,
    "out_root": "__saved__/analysis/filter_basis_fit_sanity",
    "tag": "",
    # FDA backbone settings (frozen during fitting)
    "backbone_gnn": "gcn",
    "backbone_hid_dim": 64,
    "backbone_layers": 2,
    "backbone_dropout": 0.0,
    "backbone_activation": "relu",
    # Synthetic fallback (used when dataset/domain is empty)
    "synthetic_num_nodes": 128,
    "synthetic_feat_dim": 32,
    "synthetic_num_classes": 3,
    "synthetic_feature_mean": 1.0,
    "synthetic_feature_std": 1.0,
    "synthetic_extra_edges": 512,
}


def bern_from_mono(mono_coeffs: torch.Tensor) -> torch.Tensor:
    k = int(mono_coeffs.numel()) - 1
    cols = []
    for i in range(k + 1):
        unit = torch.zeros(k + 1, dtype=mono_coeffs.dtype, device=mono_coeffs.device)
        unit[i] = 1.0
        cols.append(BernProp.to_polynomial(unit, enforce_nonneg=False))
    basis = torch.stack(cols, dim=1)
    return torch.linalg.solve(basis, mono_coeffs)


def build_synthetic_edge_index(
    num_nodes: int,
    extra_edges: int,
    generator: torch.Generator,
) -> torch.Tensor:
    ring_src = torch.arange(num_nodes, dtype=torch.long)
    ring_dst = (ring_src + 1) % num_nodes
    ring_edges = torch.stack(
        [
            torch.cat([ring_src, ring_dst], dim=0),
            torch.cat([ring_dst, ring_src], dim=0),
        ],
        dim=0,
    )

    rand_src = torch.randint(num_nodes, (extra_edges,), generator=generator, dtype=torch.long)
    rand_dst = torch.randint(num_nodes, (extra_edges,), generator=generator, dtype=torch.long)
    mask = rand_src != rand_dst
    rand_src = rand_src[mask]
    rand_dst = rand_dst[mask]
    rand_edges = torch.stack(
        [
            torch.cat([rand_src, rand_dst], dim=0),
            torch.cat([rand_dst, rand_src], dim=0),
        ],
        dim=0,
    )

    edge_index = torch.cat([ring_edges, rand_edges], dim=1)
    lin = edge_index[0] * num_nodes + edge_index[1]
    uniq = torch.unique(lin)
    return torch.stack([uniq // num_nodes, uniq % num_nodes], dim=0).contiguous()


def build_synthetic_source_data(cfg: dict):
    seed = int(cfg["seed"])
    num_nodes = int(cfg["synthetic_num_nodes"])
    feat_dim = int(cfg["synthetic_feat_dim"])
    num_classes = int(cfg["synthetic_num_classes"])
    mean = float(cfg["synthetic_feature_mean"])
    std = float(cfg["synthetic_feature_std"])
    extra_edges = int(cfg["synthetic_extra_edges"])

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    x = torch.randn((num_nodes, feat_dim), generator=g, dtype=torch.float32) * std + mean
    y = torch.randint(num_classes, (num_nodes,), generator=g, dtype=torch.long)
    edge_index = build_synthetic_edge_index(num_nodes, extra_edges, g)

    device = str(cfg["device"])
    return SimpleNamespace(
        x=x.to(device),
        y=y.to(device),
        edge_index=edge_index.to(device),
        num_nodes=num_nodes,
    )


def build_fda_backbone(source_data, cfg: dict) -> FDABase:
    model_cfg = {
        "model": {
            "in_dim": int(source_data.x.size(1)),
            "hid_dim": int(cfg["backbone_hid_dim"]),
            "num_classes": int(source_data.y.max().item()) + 1,
            "num_layers": int(cfg["backbone_layers"]),
            "dropout_ratio": float(cfg["backbone_dropout"]),
            "gnn": str(cfg["backbone_gnn"]),
            "activation": str(cfg["backbone_activation"]),
            "mode": "node",
        }
    }
    backbone = FDABase(model_cfg).to(source_data.x.device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


def encode_batch(backbone: FDABase, x_batch: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    batch_emb = []
    with torch.no_grad():
        for x in x_batch:
            batch_emb.append(backbone.feat_bottleneck(x, edge_index))
    return torch.stack(batch_emb, dim=0)


def filter_batch_output(
    filter_op,
    x_batch: torch.Tensor,
    edge_index: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    ys = [filter_op(x, edge_index, params) for x in x_batch]
    return torch.stack(ys, dim=0)


def sample_batch_from_stats(
    batch_size: int,
    num_nodes: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    x = torch.randn(
        (batch_size, num_nodes, mean.numel()),
        dtype=mean.dtype,
        generator=generator,
    )
    x = x * std.view(1, 1, -1) + mean.view(1, 1, -1)
    return x


def train_one(
    name: str,
    filter_op,
    to_mono_fn,
    edge_index: torch.Tensor,
    backbone: FDABase,
    k: int,
    epochs: int,
    lr: float,
    oracle_filter_op,
    oracle_params: torch.Tensor,
    target_mono: torch.Tensor,
    batch_size: int,
    num_nodes: int,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    sample_seed: int,
):
    params = torch.nn.Parameter(torch.empty(k, device=target_mono.device, dtype=target_mono.dtype))
    with torch.no_grad():
        params.normal_(mean=0.0, std=0.2)
    opt = torch.optim.Adam([params], lr=lr)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(sample_seed))

    history = []
    epoch_iter = tqdm(
        range(1, epochs + 1),
        desc=f"{name} fit",
        leave=False,
    )
    for epoch in epoch_iter:
        x_batch = sample_batch_from_stats(
            batch_size=batch_size,
            num_nodes=num_nodes,
            mean=feature_mean.detach().cpu(),
            std=feature_std.detach().cpu(),
            generator=gen,
        ).to(target_mono.device)

        z_batch = encode_batch(backbone, x_batch, edge_index)

        with torch.no_grad():
            y_target = filter_batch_output(oracle_filter_op, z_batch, edge_index, oracle_params)

        opt.zero_grad()
        y_pred = filter_batch_output(filter_op, z_batch, edge_index, params)
        loss = F.mse_loss(y_pred, y_target)
        loss.backward()
        opt.step()

        mono_now = to_mono_fn(params.detach())
        l2_now = float(torch.norm(mono_now - target_mono, p=2).item())
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.item()),
                "l2": l2_now,
                "mono_est": mono_now.detach().cpu().clone(),
            }
        )
        epoch_iter.set_postfix(loss=f"{loss.item():.3e}", l2=f"{l2_now:.3e}")

    return {
        "name": name,
        "params": params.detach().cpu(),
        "history": history,
    }


def _history_coeff_tensor(history: list[dict]) -> torch.Tensor:
    return torch.stack([row["mono_est"] for row in history], dim=0)


def save_history_csv(fits: list[dict], target_mono: torch.Tensor, out_csv: Path) -> None:
    k = int(target_mono.numel())
    epochs = len(fits[0]["history"])

    fieldnames = ["epoch"]
    fieldnames += [f"oracle_c{i}" for i in range(k)]
    for fit in fits:
        fieldnames += [f"{fit['name']}_c{i}" for i in range(k)]
        fieldnames += [f"{fit['name']}_loss", f"{fit['name']}_l2"]

    target_cpu = target_mono.detach().cpu()
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ei in range(epochs):
            row = {"epoch": ei + 1}
            for i in range(k):
                row[f"oracle_c{i}"] = float(target_cpu[i].item())

            for fit in fits:
                h = fit["history"][ei]
                coeff = h["mono_est"]
                for i in range(k):
                    row[f"{fit['name']}_c{i}"] = float(coeff[i].item())
                row[f"{fit['name']}_loss"] = float(h["loss"])
                row[f"{fit['name']}_l2"] = float(h["l2"])
            writer.writerow(row)


def plot_coeff_trajectories(fits: list[dict], target_mono: torch.Tensor, out_path: Path) -> None:
    k = int(target_mono.numel())
    epochs = torch.arange(1, len(fits[0]["history"]) + 1)

    ncols = min(3, k)
    nrows = (k + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)

    for i in range(k):
        ax = axes[i // ncols][i % ncols]
        for fit in fits:
            coeff = _history_coeff_tensor(fit["history"])[:, i]
            ax.plot(epochs.numpy(), coeff.numpy(), label=fit["name"], linewidth=1.8)
        ax.axhline(float(target_mono[i].item()), color="black", linestyle="--", linewidth=1.3, label="oracle" if i == 0 else None)
        ax.set_title(f"Coeff c{i}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25)

    for j in range(k, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_l2_vs_oracle(fits: list[dict], out_path: Path) -> None:
    epochs = torch.arange(1, len(fits[0]["history"]) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for fit in fits:
        l2 = [row["l2"] for row in fit["history"]]
        ax.plot(epochs.numpy(), l2, linewidth=2.0, label=fit["name"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2(mapped coeffs, oracle)")
    ax.set_title("Monomial-Coefficient Distance to Oracle")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_final_coefficients(fits: list[dict], target_mono: torch.Tensor, out_path: Path) -> None:
    k = int(target_mono.numel())
    x = torch.arange(k).numpy()
    width = 0.2

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - 1.5 * width, target_mono.detach().cpu().numpy(), width=width, label="oracle")

    shifts = [-0.5, 0.5, 1.5]
    for shift, fit in zip(shifts, fits):
        est = fit["history"][-1]["mono_est"].numpy()
        ax.bar(x + shift * width, est, width=width, label=fit["name"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"c{i}" for i in range(k)])
    ax.set_ylabel("Value")
    ax.set_title("Final Monomial Coefficients")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    torch.manual_seed(int(CFG["seed"]))

    dataset_cfg = CFG.get("dataset")
    domain_cfg = CFG.get("domain")
    use_synthetic = (dataset_cfg is None or str(dataset_cfg).strip() == "") or (
        domain_cfg is None or str(domain_cfg).strip() == ""
    )

    if use_synthetic:
        source_data = build_synthetic_source_data(CFG)
        dataset_name = "synthetic"
        domain_name = "synthetic"
        print("Using synthetic fallback data (CFG['dataset'] or CFG['domain'] missing).")
    else:
        source_data, _ = load_pair(
            dataset=str(dataset_cfg),
            source=str(domain_cfg),
            target=str(domain_cfg),
            device=str(CFG["device"]),
            seed=int(CFG["seed"]),
        )
        dataset_name = str(dataset_cfg)
        domain_name = str(domain_cfg)

    edge_index = source_data.edge_index.contiguous()

    feature_mean = source_data.x.mean(dim=0) * float(CFG["sample_mean_scale"])
    feature_std = source_data.x.std(dim=0).clamp_min(1e-6) * float(CFG["sample_std_scale"])

    backbone = build_fda_backbone(source_data, CFG)
    with torch.no_grad():
        embed_dim = int(backbone.feat_bottleneck(source_data.x, edge_index).size(1))

    k = int(CFG["k"])
    target_mono = torch.rand(k, dtype=source_data.x.dtype, device=source_data.x.device)

    oracle_filter_op = MonoProp()
    oracle_params = target_mono.detach().clone()

    print(f"Dataset/domain: {dataset_name}/{domain_name}")
    print(f"Graph size: num_nodes={source_data.num_nodes}, num_edges={edge_index.size(1)}")
    print(f"FDA embedding dim: {embed_dim}")
    print("Oracle monomial coeffs:", target_mono.detach().cpu().tolist())
    print("Equivalent Cheb coeffs:", monomial_to_cheb(target_mono).detach().cpu().tolist())
    print("Equivalent Bern coeffs:", bern_from_mono(target_mono).detach().cpu().tolist())

    common_args = {
        "edge_index": edge_index,
        "backbone": backbone,
        "k": k,
        "epochs": int(CFG["epochs"]),
        "lr": float(CFG["lr"]),
        "oracle_filter_op": oracle_filter_op,
        "oracle_params": oracle_params,
        "target_mono": target_mono,
        "batch_size": int(CFG["batch_size"]),
        "num_nodes": int(source_data.num_nodes),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }

    mono_fit = train_one(
        name="mono",
        filter_op=MonoProp(),
        to_mono_fn=lambda p: MonoProp.to_polynomial(p),
        sample_seed=int(CFG["seed"]) + 123,
        **common_args,
    )
    cheb_fit = train_one(
        name="cheb",
        filter_op=ChebProp(),
        to_mono_fn=lambda p: ChebProp.to_polynomial(p),
        sample_seed=int(CFG["seed"]) + 123,
        **common_args,
    )
    bern_fit = train_one(
        name="bern",
        filter_op=BernProp(),
        to_mono_fn=lambda p: BernProp.to_polynomial(p, enforce_nonneg=False),
        sample_seed=int(CFG["seed"]) + 123,
        **common_args,
    )

    fits = [mono_fit, cheb_fit, bern_fit]
    for fit in fits:
        final_loss = fit["history"][-1]["loss"]
        final_l2 = fit["history"][-1]["l2"]
        mapped = fit["history"][-1]["mono_est"].tolist()
        print(f"[{fit['name']}] final_loss={final_loss:.6e}, coeff_L2={final_l2:.6e}")
        print(f"[{fit['name']}] mapped_mono={mapped}")

    tag = str(CFG["tag"]).strip() or datetime.now().strftime("%m%d_%H%M%S")
    out_dir = Path(str(CFG["out_root"])) / dataset_name.lower() / domain_name.upper() / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "fit_trace.csv"
    coeff_plot = out_dir / "coeff_trajectories.png"
    l2_plot = out_dir / "coeff_l2_to_oracle.png"
    final_plot = out_dir / "final_coefficients.png"

    save_history_csv(fits, target_mono, csv_path)
    plot_coeff_trajectories(fits, target_mono, coeff_plot)
    plot_l2_vs_oracle(fits, l2_plot)
    plot_final_coefficients(fits, target_mono, final_plot)

    print(f"Saved traces and plots under: {out_dir}")


if __name__ == "__main__":
    main()
