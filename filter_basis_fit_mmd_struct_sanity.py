"""Fit a source filter to a frozen target mono oracle with feature-MMD + probe-MMD + structural BPR."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from filter_basis_fit_sanity import (
    bern_from_mono,
    build_fda_backbone,
    build_synthetic_source_data,
    encode_batch,
    plot_coeff_trajectories,
    plot_final_coefficients,
    plot_l2_vs_oracle,
    sample_batch_from_stats,
)
from models.__filters.bern import BernProp
from models.__filters.cheb import ChebProp
from models.__filters.mono import MonoProp
from utils.ablation_utils.alignment import build_struct_positive_edges, edge_bpr_structure_loss
from utils.ablation_utils.common import load_pair
from utils.ablation_utils.transfer import get_source_train_mask, train_eval_transfer_once
from utils.filter_utils import make_gaussian_probe, mmd_rbf, monomial_to_cheb

CFG = {
    # Real-data mode (if dataset/source/target are set):
    "dataset": "airport",
    "source": "USA",
    "target": "EUROPE",
    # Shorthand (same source/target):
    # "domain": "Blog1",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "seed": 7,
    "k": 4,
    "epochs": 200,
    "lr": 5e-2,
    "batch_size": 16,  # only used in synthetic fallback
    "feature_node_batch_size": 4096,
    "probe_node_batch_size": 4096,
    "probe_refresh_every": 0,  # 0 => keep one fixed probe pool
    "transfer_eval_every": 20,
    "transfer_repeats": 5,
    "out_root": "__saved__/analysis/filter_basis_fit_mmd_struct_sanity",
    "tag": "",
    # Frozen FDA backbone config
    "backbone_gnn": "gcn",
    "backbone_hid_dim": 64,
    "backbone_layers": 2,
    "backbone_dropout": 0.0,
    "backbone_activation": "relu",
    # Loss weights
    "feature_mmd_weight": 1.0,
    "probe_mmd_weight": 1.0,
    "struct_weight": 0.0,
    "param_reg_weight": 1e-4,
    # MMD settings
    "mmd_sample_size": 2048,
    "kernel_mul": 2.0,
    "kernel_num": 5,
    "fix_sigma": None,
    # Structural loss settings
    "struct_loss_mode": "adjacency",
    "struct_bpr_samples": 2048,
    "struct_bpr_margin": 0.0,
    "ppmi_path_len": 5,
    "ppmi_pos_threshold": 0.0,
    # MLP transferability eval settings
    "mlp_hid_dim": 64,
    "mlp_layers": 2,
    "mlp_dropout": 0.0,
    "mlp_lr": 1e-2,
    "mlp_weight_decay": 5e-4,
    "mlp_epochs": 100,
    # Synthetic fallback settings
    "synthetic_num_nodes": 128,
    "synthetic_feat_dim": 32,
    "synthetic_num_classes": 3,
    "synthetic_feature_mean": 1.0,
    "synthetic_feature_std": 1.0,
    "synthetic_target_mean_shift": 0.5,
    "synthetic_extra_edges": 512,
}


def _clean(x) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def load_pair_or_synth(cfg: dict):
    dataset = _clean(cfg.get("dataset"))
    source = _clean(cfg.get("source")) or _clean(cfg.get("domain"))
    target = _clean(cfg.get("target")) or _clean(cfg.get("domain"))

    if dataset and source and target:
        src, tgt = load_pair(
            dataset=dataset,
            source=source,
            target=target,
            device=str(cfg["device"]),
            seed=int(cfg["seed"]),
        )
        return src, tgt, dataset, source, target

    base_mean = float(cfg["synthetic_feature_mean"])
    src_cfg = dict(cfg)
    src_cfg["seed"] = int(cfg["seed"]) + 17
    src_cfg["synthetic_feature_mean"] = base_mean

    tgt_cfg = dict(cfg)
    tgt_cfg["seed"] = int(cfg["seed"]) + 91
    tgt_cfg["synthetic_feature_mean"] = base_mean + float(cfg["synthetic_target_mean_shift"])

    src = build_synthetic_source_data(src_cfg)
    tgt = build_synthetic_source_data(tgt_cfg)
    return src, tgt, "synthetic", "synthetic_src", "synthetic_tgt"


def _sample_indices(n: int, m: int, device: torch.device, gen: torch.Generator) -> torch.Tensor:
    if m <= 0 or n <= m:
        return torch.arange(n, device=device)
    idx = torch.randperm(n, generator=gen)[:m]
    return idx.to(device)


def _subsample(x: torch.Tensor, n: int) -> torch.Tensor:
    if n <= 0 or x.size(0) <= n:
        return x
    idx = torch.randperm(x.size(0), device=x.device)[:n]
    return x[idx]


def _mmd(a: torch.Tensor, b: torch.Tensor, cfg: dict) -> torch.Tensor:
    a = _subsample(a, int(cfg["mmd_sample_size"]))
    b = _subsample(b, int(cfg["mmd_sample_size"]))
    n = min(a.size(0), b.size(0))
    return mmd_rbf(
        a[:n],
        b[:n],
        kernel_mul=float(cfg["kernel_mul"]),
        kernel_num=int(cfg["kernel_num"]),
        fix_sigma=cfg["fix_sigma"],
    )


def make_train_param(k: int, ref: torch.Tensor, seed: int) -> torch.nn.Parameter:
    g = torch.Generator(device=ref.device)
    g.manual_seed(int(seed))
    p = torch.empty(k, device=ref.device, dtype=ref.dtype)
    p.normal_(mean=0.0, std=0.2, generator=g)
    return torch.nn.Parameter(p)


def _transfer_seed(base_seed: int, epoch: int, rep: int, filter_name: str) -> int:
    offset = {"mono": 0, "cheb": 200_000, "bern": 400_000}.get(filter_name.lower(), 600_000)
    return int(base_seed) + offset + 10_000 * int(epoch) + int(rep)


def train_filter(
    name: str,
    filter_op,
    to_mono_fn,
    train_params: torch.nn.Parameter,
    oracle_mono: torch.Tensor,
    backbone,
    source_data,
    target_data,
    source_pos_edges: torch.Tensor,
    cfg: dict,
    sample_seed: int,
) -> dict:
    oracle_filter = MonoProp()
    opt = torch.optim.Adam([train_params], lr=float(cfg["lr"]))

    is_synth = str(_clean(cfg.get("dataset")) or "").lower() in {"", "synthetic"}
    feature_batch = int(cfg.get("feature_node_batch_size", 0))
    probe_batch = int(cfg.get("probe_node_batch_size", 0))
    probe_refresh_every = int(cfg.get("probe_refresh_every", 0))

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(sample_seed))
    source_train_mask = get_source_train_mask(source_data).to(source_data.x.device)

    src_mean = source_data.x.mean(dim=0)
    src_std = source_data.x.std(dim=0).clamp_min(1e-6)
    tgt_mean = target_data.x.mean(dim=0)
    tgt_std = target_data.x.std(dim=0).clamp_min(1e-6)

    with torch.no_grad():
        src_embed_full = backbone.feat_bottleneck(source_data.x, source_data.edge_index)
        tgt_embed_full = backbone.feat_bottleneck(target_data.x, target_data.edge_index)
        src_oracle_feat_full = oracle_filter(src_embed_full, source_data.edge_index, oracle_mono)
        tgt_feat_push_full = oracle_filter(tgt_embed_full, target_data.edge_index, oracle_mono)
        src_probe_pool, tgt_probe_pool = make_gaussian_probe(src_embed_full.detach(), tgt_embed_full.detach())
        tgt_probe_push_full = oracle_filter(tgt_probe_pool, target_data.edge_index, oracle_mono)

    history = []
    transfer_run_rows = []
    transfer_summary_rows = []
    eval_every = int(cfg.get("transfer_eval_every", 0))
    repeats = int(cfg.get("transfer_repeats", 0))

    def _run_transfer_eval(epoch_idx: int) -> None:
        if repeats <= 0:
            return
        with torch.no_grad():
            tgt_learned_feat = filter_op(tgt_embed_full, target_data.edge_index, train_params)
        run_metrics = []
        for rep in range(repeats):
            metrics = train_eval_transfer_once(
                source_feat=src_oracle_feat_full.detach(),
                target_feat=tgt_learned_feat.detach(),
                source_y=source_data.y,
                target_y=target_data.y,
                source_train_mask=source_train_mask,
                cfg=cfg,
                seed=_transfer_seed(int(cfg["seed"]), epoch_idx, rep, name),
            )
            run_row = {"filter": name, "epoch": epoch_idx, "repeat": rep, **metrics}
            transfer_run_rows.append(run_row)
            run_metrics.append(metrics)

        target_acc_vals = torch.tensor([float(m["target_acc"]) for m in run_metrics], dtype=torch.float32)
        target_f1_vals = torch.tensor([float(m["target_macro_f1"]) for m in run_metrics], dtype=torch.float32)
        transfer_summary_rows.append(
            {
                "filter": name,
                "epoch": epoch_idx,
                "n_runs": int(repeats),
                "target_acc_mean": float(target_acc_vals.mean().item()),
                "target_acc_std": float(target_acc_vals.std(unbiased=False).item()),
                "target_macro_f1_mean": float(target_f1_vals.mean().item()),
                "target_macro_f1_std": float(target_f1_vals.std(unbiased=False).item()),
            }
        )

    # Evaluate transferability at epoch 0 before any training updates.
    if eval_every > 0:
        _run_transfer_eval(epoch_idx=0)

    for epoch in tqdm(range(1, int(cfg["epochs"]) + 1), desc=f"{name} fit", leave=False):
        if is_synth:
            src_x = sample_batch_from_stats(
                batch_size=int(cfg["batch_size"]),
                num_nodes=int(source_data.num_nodes),
                mean=src_mean.detach().cpu(),
                std=src_std.detach().cpu(),
                generator=gen,
            ).to(oracle_mono.device)
            tgt_x = sample_batch_from_stats(
                batch_size=int(cfg["batch_size"]),
                num_nodes=int(target_data.num_nodes),
                mean=tgt_mean.detach().cpu(),
                std=tgt_std.detach().cpu(),
                generator=gen,
            ).to(oracle_mono.device)
            src_z = encode_batch(backbone, src_x, source_data.edge_index).mean(dim=0)
            tgt_z = encode_batch(backbone, tgt_x, target_data.edge_index).mean(dim=0)
            with torch.no_grad():
                tgt_feat_push_full = oracle_filter(tgt_z, target_data.edge_index, oracle_mono)
            src_feat_push_full = filter_op(src_z, source_data.edge_index, train_params)
        else:
            src_feat_push_full = filter_op(src_embed_full, source_data.edge_index, train_params)

        src_idx = _sample_indices(src_feat_push_full.size(0), feature_batch, src_feat_push_full.device, gen)
        tgt_idx = _sample_indices(tgt_feat_push_full.size(0), feature_batch, tgt_feat_push_full.device, gen)
        feature_mmd_loss = _mmd(src_feat_push_full[src_idx], tgt_feat_push_full[tgt_idx], cfg)

        if is_synth or (probe_refresh_every > 0 and epoch % probe_refresh_every == 0):
            if is_synth:
                src_probe_pool, tgt_probe_pool = make_gaussian_probe(src_z.detach(), tgt_z.detach())
            else:
                src_probe_pool, tgt_probe_pool = make_gaussian_probe(src_embed_full.detach(), tgt_embed_full.detach())
            with torch.no_grad():
                tgt_probe_push_full = oracle_filter(tgt_probe_pool, target_data.edge_index, oracle_mono)

        src_probe_push_full = filter_op(src_probe_pool, source_data.edge_index, train_params)
        src_probe_idx = _sample_indices(src_probe_push_full.size(0), probe_batch, src_probe_push_full.device, gen)
        tgt_probe_idx = _sample_indices(tgt_probe_push_full.size(0), probe_batch, tgt_probe_push_full.device, gen)
        probe_mmd_loss = _mmd(
            src_probe_push_full[src_probe_idx],
            tgt_probe_push_full[tgt_probe_idx],
            cfg,
        )

        struct_loss = edge_bpr_structure_loss(
            src_probe_push_full,
            source_pos_edges,
            num_samples=int(cfg["struct_bpr_samples"]),
            margin=float(cfg["struct_bpr_margin"]),
        )

        reg = train_params.pow(2).mean()
        loss = (
            float(cfg["feature_mmd_weight"]) * feature_mmd_loss
            + float(cfg["probe_mmd_weight"]) * probe_mmd_loss
            + float(cfg["struct_weight"]) * struct_loss
            + float(cfg["param_reg_weight"]) * reg
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        mono_est = to_mono_fn(train_params.detach())
        l2 = float(torch.norm(mono_est - oracle_mono, p=2).item())
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.item()),
                "feature_mmd_loss": float(feature_mmd_loss.item()),
                "probe_mmd_loss": float(probe_mmd_loss.item()),
                "struct_loss": float(struct_loss.item()),
                "l2": l2,
                "mono_est": mono_est.detach().cpu().clone(),
            }
        )

        if eval_every > 0 and epoch % eval_every == 0:
            _run_transfer_eval(epoch_idx=epoch)

    with torch.no_grad():
        src_feat_eval = filter_op(src_embed_full, source_data.edge_index, train_params)
        src_probe_eval = filter_op(src_probe_pool, source_data.edge_index, train_params)
        eval_feature_mmd = float(_mmd(src_feat_eval, tgt_feat_push_full, cfg).item())
        eval_probe_mmd = float(_mmd(src_probe_eval, tgt_probe_push_full, cfg).item())
        eval_struct = float(
            edge_bpr_structure_loss(
                src_probe_eval,
                source_pos_edges,
                num_samples=int(cfg["struct_bpr_samples"]),
                margin=float(cfg["struct_bpr_margin"]),
            ).item()
        )
        eval_l2 = float(torch.norm(to_mono_fn(train_params.detach()) - oracle_mono, p=2).item())

    return {
        "name": name,
        "params": train_params.detach().cpu(),
        "history": history,
        "transfer_runs": transfer_run_rows,
        "transfer_summary": transfer_summary_rows,
        "eval": {
            "feature_mmd": eval_feature_mmd,
            "probe_mmd": eval_probe_mmd,
            "struct_loss": eval_struct,
            "l2": eval_l2,
        },
    }


def save_trace_csv(path: Path, fits: list[dict], oracle_mono: torch.Tensor) -> None:
    k = int(oracle_mono.numel())
    epochs = len(fits[0]["history"])

    fieldnames = ["epoch"] + [f"oracle_c{i}" for i in range(k)]
    for fit in fits:
        fieldnames += [f"{fit['name']}_c{i}" for i in range(k)]
        fieldnames += [
            f"{fit['name']}_loss",
            f"{fit['name']}_feature_mmd_loss",
            f"{fit['name']}_probe_mmd_loss",
            f"{fit['name']}_struct_loss",
            f"{fit['name']}_l2",
        ]

    oracle_cpu = oracle_mono.detach().cpu()
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ei in range(epochs):
            row = {"epoch": ei + 1}
            for i in range(k):
                row[f"oracle_c{i}"] = float(oracle_cpu[i].item())

            for fit in fits:
                h = fit["history"][ei]
                for i in range(k):
                    row[f"{fit['name']}_c{i}"] = float(h["mono_est"][i].item())
                row[f"{fit['name']}_loss"] = float(h["loss"])
                row[f"{fit['name']}_feature_mmd_loss"] = float(h["feature_mmd_loss"])
                row[f"{fit['name']}_probe_mmd_loss"] = float(h["probe_mmd_loss"])
                row[f"{fit['name']}_struct_loss"] = float(h["struct_loss"])
                row[f"{fit['name']}_l2"] = float(h["l2"])
            writer.writerow(row)


def save_eval_csv(path: Path, fits: list[dict]) -> None:
    rows = []
    for fit in fits:
        row = {"filter": fit["name"]}
        row.update(fit["eval"])
        rows.append(row)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filter", "feature_mmd", "probe_mmd", "struct_loss", "l2"],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_transfer_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_transferability_summary(path: Path, summary_rows: list[dict], title: str) -> None:
    if not summary_rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for filter_name in sorted({str(r["filter"]) for r in summary_rows}):
        rows = sorted(
            [r for r in summary_rows if str(r["filter"]) == filter_name],
            key=lambda r: int(r["epoch"]),
        )
        x = [int(r["epoch"]) for r in rows]
        acc_mean = [float(r["target_acc_mean"]) for r in rows]
        acc_std = [float(r["target_acc_std"]) for r in rows]
        f1_mean = [float(r["target_macro_f1_mean"]) for r in rows]
        f1_std = [float(r["target_macro_f1_std"]) for r in rows]

        axes[0].plot(x, acc_mean, marker="o", label=filter_name)
        axes[0].fill_between(
            x,
            [m - s for m, s in zip(acc_mean, acc_std)],
            [m + s for m, s in zip(acc_mean, acc_std)],
            alpha=0.2,
        )
        axes[1].plot(x, f1_mean, marker="o", label=filter_name)
        axes[1].fill_between(
            x,
            [m - s for m, s in zip(f1_mean, f1_std)],
            [m + s for m, s in zip(f1_mean, f1_std)],
            alpha=0.2,
        )

    axes[0].set_title("Target Transfer Accuracy")
    axes[0].set_xlabel("Training epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Target Transfer Macro-F1")
    axes[1].set_xlabel("Training epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    torch.manual_seed(int(CFG["seed"]))

    source_data, target_data, dataset, source, target = load_pair_or_synth(CFG)
    backbone = build_fda_backbone(source_data, CFG)
    source_pos_edges = build_struct_positive_edges(source_data, CFG)

    k = int(CFG["k"])
    oracle_mono = torch.rand(k, device=source_data.x.device, dtype=source_data.x.dtype)

    # Explicit parameters in main file.
    train_mono = make_train_param(k, oracle_mono, seed=int(CFG["seed"]) + 11)
    train_cheb = make_train_param(k, oracle_mono, seed=int(CFG["seed"]) + 13)
    train_bern = make_train_param(k, oracle_mono, seed=int(CFG["seed"]) + 17)

    print(f"Scenario: {dataset} {source}->{target}")
    print("oracle_mono:", oracle_mono.detach().cpu().tolist())
    print("oracle_cheb:", monomial_to_cheb(oracle_mono).detach().cpu().tolist())
    print("oracle_bern:", bern_from_mono(oracle_mono).detach().cpu().tolist())

    fits = [
        train_filter(
            name="mono",
            filter_op=MonoProp(),
            to_mono_fn=lambda p: MonoProp.to_polynomial(p),
            train_params=train_mono,
            oracle_mono=oracle_mono,
            backbone=backbone,
            source_data=source_data,
            target_data=target_data,
            source_pos_edges=source_pos_edges,
            cfg=CFG,
            sample_seed=int(CFG["seed"]) + 123,
        ),
        train_filter(
            name="cheb",
            filter_op=ChebProp(),
            to_mono_fn=lambda p: ChebProp.to_polynomial(p),
            train_params=train_cheb,
            oracle_mono=oracle_mono,
            backbone=backbone,
            source_data=source_data,
            target_data=target_data,
            source_pos_edges=source_pos_edges,
            cfg=CFG,
            sample_seed=int(CFG["seed"]) + 123,
        ),
        train_filter(
            name="bern",
            filter_op=BernProp(),
            to_mono_fn=lambda p: BernProp.to_polynomial(p, enforce_nonneg=False),
            train_params=train_bern,
            oracle_mono=oracle_mono,
            backbone=backbone,
            source_data=source_data,
            target_data=target_data,
            source_pos_edges=source_pos_edges,
            cfg=CFG,
            sample_seed=int(CFG["seed"]) + 123,
        ),
    ]

    for fit in fits:
        h = fit["history"][-1]
        e = fit["eval"]
        ts = fit["transfer_summary"][-1] if fit["transfer_summary"] else None
        print(
            f"[{fit['name']}] loss={h['loss']:.3e}, feat_mmd={h['feature_mmd_loss']:.3e}, "
            f"probe_mmd={h['probe_mmd_loss']:.3e}, struct={h['struct_loss']:.3e}, l2={h['l2']:.3e}"
        )
        print(
            f"[{fit['name']}] eval(feature_mmd={e['feature_mmd']:.3e}, "
            f"probe_mmd={e['probe_mmd']:.3e}, struct={e['struct_loss']:.3e}, l2={e['l2']:.3e})"
        )
        print(f"[{fit['name']}] params={fit['params'].tolist()}")
        print(f"[{fit['name']}] mono={h['mono_est'].tolist()}")
        if ts is not None:
            print(
                f"[{fit['name']}] transfer@{ts['epoch']}: "
                f"target_acc={ts['target_acc_mean']:.4f}±{ts['target_acc_std']:.4f}, "
                f"target_f1={ts['target_macro_f1_mean']:.4f}±{ts['target_macro_f1_std']:.4f}"
            )

    tag = _clean(CFG.get("tag")) or datetime.now().strftime("%m%d_%H%M%S")
    out_dir = Path(str(CFG["out_root"])) / dataset.lower() / f"{source}_{target}" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    save_trace_csv(out_dir / "fit_trace.csv", fits, oracle_mono)
    save_eval_csv(out_dir / "final_eval.csv", fits)
    transfer_run_rows = []
    transfer_summary_rows = []
    for fit in fits:
        transfer_run_rows.extend(fit["transfer_runs"])
        transfer_summary_rows.extend(fit["transfer_summary"])
    if transfer_run_rows:
        save_transfer_csv(
            out_dir / "transferability_runs.csv",
            transfer_run_rows,
            fieldnames=[
                "filter",
                "epoch",
                "repeat",
                "source_acc",
                "target_acc",
                "source_macro_f1",
                "target_macro_f1",
            ],
        )
    if transfer_summary_rows:
        save_transfer_csv(
            out_dir / "transferability_summary.csv",
            transfer_summary_rows,
            fieldnames=[
                "filter",
                "epoch",
                "n_runs",
                "target_acc_mean",
                "target_acc_std",
                "target_macro_f1_mean",
                "target_macro_f1_std",
            ],
        )
        plot_transferability_summary(
            out_dir / "transferability_summary.png",
            transfer_summary_rows,
            title=f"Transferability over training ({dataset}: {source}->{target})",
        )
    plot_coeff_trajectories(fits, oracle_mono, out_dir / "coeff_trajectories.png")
    plot_l2_vs_oracle(fits, out_dir / "coeff_l2_to_oracle.png")
    plot_final_coefficients(fits, oracle_mono, out_dir / "final_coefficients.png")

    print(f"Saved traces and plots under: {out_dir}")


if __name__ == "__main__":
    main()
