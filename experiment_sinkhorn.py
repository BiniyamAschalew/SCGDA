import argparse
import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from  data.build_dataset import build_dataset
from  models.baselines.gnn.gnn_base import GNNBase
from  utils.config_utils import build_config
from  utils.expt_utils import set_seed
from  utils.train_utils.metrics import BaseMetric
from  utils.train_utils.mmd import MMD, Sinkhorn as SinkhornImported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simple comparison: GNN+MMD, GNN+Sinkhorn(imported geomloss), "
            "GNN+Sinkhorn(implementation)"
        )
    )
    parser.add_argument("--dataset", type=str, default="airport")
    parser.add_argument("--source", type=str, default="BRAZIL")
    parser.add_argument("--target", type=str, default="USA")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--aligners", type=str, default="mmd,sinkhorn_imported,sinkhorn_impl")
    parser.add_argument("--align-weights", type=str, default="0.1,0.2,0.5")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-3)

    parser.add_argument("--sampling-num", type=int, default=512)
    parser.add_argument("--times", type=int, default=3)

    parser.add_argument("--sinkhorn-eps", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--sinkhorn-tol", type=float, default=1e-8)
    parser.add_argument("--sinkhorn-debias", action="store_true", default=True)

    parser.add_argument("--use-mask", action="store_true", default=True)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./__saved__/results/ablation/sinkhorn_vs_mmd",
    )
    return parser.parse_args()


def parse_seeds(raw: str) -> list[int]:
    seeds = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("No valid seeds found. Use --seeds like '0,1,2'.")
    return seeds


def parse_float_list(raw: str) -> list[float]:
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("No valid float values found.")
    return values


def parse_aligners(raw: str) -> list[str]:
    valid = {"mmd", "sinkhorn_imported", "sinkhorn_impl"}
    values = []
    for token in str(raw).split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in valid:
            raise ValueError(f"Invalid aligner '{token}'. Valid: {sorted(valid)}")
        values.append(token)
    if not values:
        raise ValueError("No valid aligners provided.")
    return values


def resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(device)


def get_train_mask(data, use_mask: bool) -> torch.Tensor:
    if use_mask and hasattr(data, "train_mask") and data.train_mask is not None:
        return data.train_mask.bool()
    return torch.ones_like(data.y, dtype=torch.bool, device=data.y.device)


def sampled_mmd(source_feat: torch.Tensor, target_feat: torch.Tensor, sampling_num: int, times: int) -> torch.Tensor:
    return MMD(
        source_feat,
        target_feat,
        sampling_num=int(sampling_num),
        times=int(times),
    )


def _sinkhorn_uniform_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    iters: int,
    tol: float,
) -> torch.Tensor:
    n = int(x.size(0))
    m = int(y.size(0))
    if n == 0 or m == 0:
        return x.new_tensor(0.0)

    cost = torch.cdist(x, y, p=2).pow(2)
    reg = max(float(eps), 1e-6)
    K = torch.exp(-cost / reg).clamp_min(float(tol))

    a = torch.full((n,), 1.0 / float(n), device=x.device, dtype=x.dtype)
    b = torch.full((m,), 1.0 / float(m), device=x.device, dtype=x.dtype)
    u = torch.full_like(a, 1.0 / float(n))
    v = torch.full_like(b, 1.0 / float(m))

    for _ in range(max(1, int(iters))):
        Kv = torch.matmul(K, v).clamp_min(float(tol))
        u = a / Kv
        KTu = torch.matmul(K.t(), u).clamp_min(float(tol))
        v = b / KTu

    transport = u.unsqueeze(1) * K * v.unsqueeze(0)
    return torch.sum(transport * cost)


def sampled_sinkhorn(
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    sampling_num: int,
    times: int,
    eps: float,
    iters: int,
    tol: float,
    debias: bool,
) -> torch.Tensor:
    if times <= 0:
        raise ValueError(f"times must be > 0, got {times}")
    if sampling_num <= 0:
        raise ValueError(f"sampling_num must be > 0, got {sampling_num}")

    s_num = int(source_feat.size(0))
    t_num = int(target_feat.size(0))
    device = source_feat.device

    out = source_feat.new_tensor(0.0)
    for _ in range(times):
        s_idx = torch.randint(s_num, (sampling_num,), device=device)
        t_idx = torch.randint(t_num, (sampling_num,), device=device)
        xs = source_feat[s_idx]
        xt = target_feat[t_idx]

        st = _sinkhorn_uniform_cost(xs, xt, eps=eps, iters=iters, tol=tol)
        if not debias:
            out = out + st
            continue

        ss = _sinkhorn_uniform_cost(xs, xs, eps=eps, iters=iters, tol=tol)
        tt = _sinkhorn_uniform_cost(xt, xt, eps=eps, iters=iters, tol=tol)
        out = out + (st - 0.5 * (ss + tt))

    return out / float(times)


def run_one_seed(
    base_config: dict,
    args: argparse.Namespace,
    aligner: str,
    align_weight: float,
    seed: int,
) -> dict:
    config = deepcopy(base_config)
    config["expt"]["seed"] = int(seed)
    set_seed(int(seed))

    device = resolve_device(config["expt"]["device"])
    config["expt"]["device"] = device

    source_dataset, target_dataset = build_dataset(config)
    source_data = source_dataset[0].to(device)
    target_data = target_dataset[0].to(device)

    if source_data.edge_index is not None:
        source_data.edge_index = source_data.edge_index.contiguous()
    if target_data.edge_index is not None:
        target_data.edge_index = target_data.edge_index.contiguous()

    config["model"]["in_dim"] = int(source_data.x.size(1))
    config["model"]["num_classes"] = int(source_data.y.unique().numel())

    model = GNNBase(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    source_mask = get_train_mask(source_data, bool(args.use_mask))

    start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        optimizer.zero_grad()

        source_feat = model.feat_bottleneck(source_data.x, source_data.edge_index)
        target_feat = model.feat_bottleneck(target_data.x, target_data.edge_index)

        source_logits = model.feat_classifier(source_feat, source_data.edge_index)
        source_log_probs = F.log_softmax(source_logits, dim=1)

        cls_loss = F.nll_loss(source_log_probs[source_mask], source_data.y[source_mask])

        if aligner == "mmd":
            align_loss = sampled_mmd(
                source_feat,
                target_feat,
                sampling_num=int(args.sampling_num),
                times=int(args.times),
            )
        elif aligner == "sinkhorn_imported":
            align_loss = SinkhornImported(
                source_feat,
                target_feat,
                sampling_num=int(args.sampling_num),
                times=int(args.times),
                blur=float(args.sinkhorn_eps),
                p=2,
                scaling=0.9,
                debias=bool(args.sinkhorn_debias),
                backend="auto",
            )
        elif aligner == "sinkhorn_impl":
            align_loss = sampled_sinkhorn(
                source_feat,
                target_feat,
                sampling_num=int(args.sampling_num),
                times=int(args.times),
                eps=float(args.sinkhorn_eps),
                iters=int(args.sinkhorn_iters),
                tol=float(args.sinkhorn_tol),
                debias=bool(args.sinkhorn_debias),
            )
        else:
            raise ValueError(f"Unsupported aligner: {aligner}")

        loss = cls_loss + float(align_weight) * align_loss
        loss.backward()
        optimizer.step()

        if args.log_interval > 0 and (epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs):
            print(
                f"[{aligner}][seed={seed}] epoch {epoch:03d} "
                f"cls={cls_loss.item():.4f} align={align_loss.item():.4f} total={loss.item():.4f}"
            )

    train_time = time.time() - start

    model.eval()
    with torch.no_grad():
        target_logits = model(target_data.x, target_data.edge_index)

    metrics = BaseMetric(config)(target_logits, target_data.y)
    return {
        "aligner": aligner,
        "seed": int(seed),
        "source": config["expt"]["source"],
        "target": config["expt"]["target"],
        "dataset": config["data"]["name"],
        "epochs": int(args.epochs),
        "align_weight": float(align_weight),
        "sampling_num": int(args.sampling_num),
        "times": int(args.times),
        "sinkhorn_eps": float(args.sinkhorn_eps),
        "sinkhorn_iters": int(args.sinkhorn_iters),
        "train_time": float(train_time),
        **{k: float(v) for k, v in metrics.items()},
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in ["micro_f1", "macro_f1", "accuracy", "train_time"] if c in df.columns]
    if not metric_cols:
        return pd.DataFrame()

    grouped = df.groupby(["aligner", "align_weight"], as_index=False)[metric_cols].agg(["mean", "std"])
    grouped.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
        for col in grouped.columns.to_flat_index()
    ]
    return grouped


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    aligners = parse_aligners(args.aligners)
    align_weights = parse_float_list(args.align_weights)
    args.device = resolve_device(args.device)

    config_setup = {
        "data": args.dataset,
        "expt": "default",
        "model": "gnn",
    }
    update_config = {
        "expt": {
            "source": args.source,
            "target": args.target,
            "device": args.device,
            "seed": int(seeds[0]),
            "verbose": 0,
            "wandb_enabled": False,
        },
        "model": {
            "epochs": int(args.epochs),
            "use_mask": bool(args.use_mask),
            "batch_size": 0,
        },
    }
    base_config = build_config(config_setup, update_config=update_config)

    rows = []
    for align_weight in align_weights:
        for aligner in aligners:
            for seed in seeds:
                row = run_one_seed(
                    base_config,
                    args,
                    aligner=aligner,
                    align_weight=float(align_weight),
                    seed=seed,
                )
                rows.append(row)

    results_df = pd.DataFrame(rows)
    summary_df = summarize(results_df)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir) / f"{args.dataset}_{args.source}_{args.target}_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    results_csv = out_root / "results_by_seed.csv"
    summary_csv = out_root / "summary.csv"
    args_json = out_root / "config.json"

    results_df.to_csv(results_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    args_payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    args_payload["seeds"] = seeds
    args_payload["aligners"] = aligners
    args_payload["align_weights"] = align_weights
    args_json.write_text(json.dumps(args_payload, indent=2), encoding="utf-8")

    print("\n=== Per-seed results ===")
    print(results_df.to_string(index=False))
    print("\n=== Summary (mean/std) ===")
    if summary_df.empty:
        print("No metrics to summarize.")
    else:
        print(summary_df.to_string(index=False))

    print(f"\nSaved: {results_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {args_json}")


if __name__ == "__main__":
    main()
