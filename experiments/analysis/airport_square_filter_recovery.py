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

from data.build_dataset import build_dataset
from models.__components.chebprop import ChebProp
from utils.config_utils import build_config
from utils.expt_utils import set_seed
from utils.filter_utils import cheb_to_monomial, monomial_to_cheb, tensor_to_float_list


def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True).transpose(0, 1)
    return torch.clamp(x_norm + y_norm - 2.0 * (x @ y.transpose(0, 1)), min=0.0)


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


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(x))


def _resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _load_domain_data(domain: str, device: str, seed: int):
    config_setup = {"data": "airport", "expt": "default", "model": "test"}
    update_config = {
        "expt": {
            "source": domain,
            "target": domain,
            "device": device,
            "seed": seed,
            "verbose": 0,
            "wandb_enabled": False,
        }
    }
    config = build_config(config_setup, update_config)
    source_dataset, _ = build_dataset(config)
    data = source_dataset[0].to(device)
    if data.x is None:
        raise ValueError("Airport dataset must provide transformed node features.")
    return data


def _apply_operator(
    prop: ChebProp,
    x: torch.Tensor,
    data,
    temp: torch.Tensor,
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


def _edge_bpr_loss(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    num_samples: int = 2048,
    margin: float = 1.0,
) -> torch.Tensor:
    """BPR-like edge discriminability loss: relu(margin + d_pos - d_neg)."""
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


def _build_target_monomial(num_terms: int, target_fn: str, custom: str | None, device: str) -> torch.Tensor:
    target_mono = torch.zeros(num_terms, device=device, dtype=torch.float32)
    fn = target_fn.lower().strip()
    if fn == "square":
        if num_terms <= 2:
            raise ValueError("square target requires K>=3 (terms 0..2).")
        target_mono[2] = 1.0
        return target_mono

    if fn in {"a3_plus_half_a", "a3_plus_0.5a"}:
        if num_terms <= 3:
            raise ValueError("A^3 + 0.5A target requires K>=4 (terms 0..3).")
        target_mono[1] = 0.5
        target_mono[3] = 1.0
        return target_mono

    if fn == "custom":
        if custom is None:
            raise ValueError("custom target function requires --target_monomial_custom.")
        vals = [float(v.strip()) for v in custom.split(",") if v.strip()]
        if len(vals) > num_terms:
            raise ValueError(f"Custom monomial length {len(vals)} exceeds K={num_terms}.")
        target_mono[: len(vals)] = torch.tensor(vals, dtype=torch.float32, device=device)
        return target_mono

    raise ValueError(f"Unknown target_fn: {target_fn}")


def _build_source_init_monomial(num_terms: int, source_init: str, device: str) -> torch.Tensor:
    source_mono = torch.zeros(num_terms, device=device, dtype=torch.float32)
    key = source_init.lower().strip()
    if key == "a":
        if num_terms <= 1:
            raise ValueError("source init A requires K>=2.")
        source_mono[1] = 1.0
        return source_mono
    if key == "identity":
        source_mono[0] = 1.0
        return source_mono
    raise ValueError(f"Unknown source_init: {source_init}")


def _build_reference_coeffs(
    num_terms: int,
    source_init: str,
    target_fn: str,
    target_custom: str | None,
    device: str,
):
    source_mono = _build_source_init_monomial(num_terms, source_init, device)
    target_mono = _build_target_monomial(num_terms, target_fn, target_custom, device)
    source_cheb = monomial_to_cheb(source_mono).to(device=device, dtype=torch.float32)
    target_cheb = monomial_to_cheb(target_mono).to(device=device, dtype=torch.float32)
    return source_cheb, source_mono, target_cheb, target_mono


def _to_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_domain(
    domain: str,
    data,
    args,
    seed: int,
) -> tuple[dict, list[dict]]:
    set_seed(seed)
    num_terms = args.cheb_k
    source_init_cheb, source_init_mono, target_cheb, target_mono = _build_reference_coeffs(
        num_terms=num_terms,
        source_init=args.source_init,
        target_fn=args.target_fn,
        target_custom=args.target_monomial_custom,
        device=data.x.device,
    )

    prop = ChebProp(num_terms, is_source_domain=False).to(data.x.device)
    source_init_pos = source_init_cheb.clamp_min(args.init_eps)
    source_raw = nn.Parameter(_inverse_softplus(source_init_pos))
    optimizer = torch.optim.Adam([source_raw], lr=args.lr, weight_decay=args.weight_decay)

    all_features = data.x.detach()
    probe_mean = all_features.mean(dim=0, keepdim=True)
    probe_std = all_features.std(dim=0, keepdim=True) + 1e-6

    trace_rows = []
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        probe = torch.randn_like(all_features) * probe_std + probe_mean
        source_temp = F.softplus(source_raw)

        source_probe_out = _apply_operator(
            prop=prop,
            x=probe,
            data=data,
            temp=source_temp,
            lambda_max=args.cheb_lambda_max,
        )
        target_probe_out = _apply_operator(
            prop=prop,
            x=probe,
            data=data,
            temp=target_cheb,
            lambda_max=args.cheb_lambda_max,
        )
        mmd_loss = mmd_rbf(
            source_probe_out,
            target_probe_out,
            kernel_mul=args.kernel_mul,
            kernel_num=args.kernel_num,
            fix_sigma=args.fix_sigma,
        )

        # Optional structural regularizer on the learned source filter output.
        source_real_out = _apply_operator(
            prop=prop,
            x=all_features,
            data=data,
            temp=source_temp,
            lambda_max=args.cheb_lambda_max,
        )
        bpr_loss = _edge_bpr_loss(
            source_real_out,
            data.edge_index,
            num_samples=args.bpr_samples,
            margin=args.bpr_margin,
        )
        total_loss = mmd_loss + args.bpr_weight * bpr_loss
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            src_cheb = F.softplus(source_raw).detach()
            src_mono = cheb_to_monomial(src_cheb)
            cheb_l2 = torch.norm(src_cheb - target_cheb, p=2)
            mono_l2 = torch.norm(src_mono - target_mono, p=2)
            trace_rows.append(
                {
                    "domain": domain,
                    "epoch": epoch,
                    "total_loss": _to_float(total_loss),
                    "probe_mmd": _to_float(mmd_loss),
                    "bpr_loss": _to_float(bpr_loss),
                    "cheb_l2_to_target": _to_float(cheb_l2),
                    "mono_l2_to_target": _to_float(mono_l2),
                }
            )

    with torch.no_grad():
        src_cheb = F.softplus(source_raw).detach()
        src_mono = cheb_to_monomial(src_cheb)
        cheb_l2 = torch.norm(src_cheb - target_cheb, p=2)
        mono_l2 = torch.norm(src_mono - target_mono, p=2)
        mono_cos = torch.nn.functional.cosine_similarity(src_mono, target_mono, dim=0)
        recovered_degree = int(torch.argmax(src_mono.abs()).item())
        target_degree = int(torch.argmax(target_mono.abs()).item())

        support = target_mono.abs() > 1e-8
        support_l1_err = float((src_mono[support] - target_mono[support]).abs().sum().item())
        nonsupport_l1 = float(src_mono[~support].abs().sum().item())

    result_row = {
        "domain": domain,
        "target_fn": args.target_fn,
        "probe_mmd_final": trace_rows[-1]["probe_mmd"] if trace_rows else float("nan"),
        "bpr_loss_final": trace_rows[-1]["bpr_loss"] if trace_rows else float("nan"),
        "cheb_l2_to_target": _to_float(cheb_l2),
        "mono_l2_to_target": _to_float(mono_l2),
        "mono_cos_to_target": _to_float(mono_cos),
        "recovered_degree": recovered_degree,
        "target_degree": target_degree,
        "support_l1_error": support_l1_err,
        "nonsupport_l1": nonsupport_l1,
        "source_cheb": tensor_to_float_list(src_cheb),
        "source_mono": tensor_to_float_list(src_mono),
        "source_init_cheb": tensor_to_float_list(source_init_pos),
        "source_init_mono": tensor_to_float_list(source_init_mono),
        "target_cheb": tensor_to_float_list(target_cheb),
        "target_mono": tensor_to_float_list(target_mono),
    }
    return result_row, trace_rows


def _plot_domain_errors(result_rows: list[dict], out_png: str) -> None:
    domains = [r["domain"] for r in result_rows]
    x = torch.arange(len(domains)).numpy()
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, [r["cheb_l2_to_target"] for r in result_rows], width=width, label="Cheb L2")
    ax.bar(x + width / 2, [r["mono_l2_to_target"] for r in result_rows], width=width, label="Mono L2")
    ax.set_xticks(x, domains)
    ax.set_ylabel("Distance to target mapping")
    ax.set_title("Filter-Recovery Error by Domain")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_support_errors(result_rows: list[dict], out_png: str) -> None:
    domains = [r["domain"] for r in result_rows]
    x = torch.arange(len(domains)).numpy()
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, [r["support_l1_error"] for r in result_rows], width=width, label="support L1 err")
    ax.bar(x + width / 2, [r["nonsupport_l1"] for r in result_rows], width=width, label="off-support L1")
    ax.set_xticks(x, domains)
    ax.set_ylabel("L1")
    ax.set_title("Target-Support vs Off-Support Error")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def run(args):
    device = _resolve_device(args.device)
    domain_list = [d.strip() for d in args.domains.split(",") if d.strip()]
    if not domain_list:
        raise ValueError("No domains provided.")
    if args.cheb_k < 4 and args.target_fn in {"a3_plus_half_a", "a3_plus_0.5a"}:
        raise ValueError("A^3 + 0.5A requires --cheb_k >= 4.")

    result_rows = []
    trace_rows = []
    for i, domain in enumerate(domain_list):
        data = _load_domain_data(domain, device=device, seed=args.seed + i)
        result_row, domain_trace = _run_domain(
            domain=domain,
            data=data,
            args=args,
            seed=args.seed + i,
        )
        result_rows.append(result_row)
        trace_rows.extend(domain_trace)

    os.makedirs(args.out_dir, exist_ok=True)
    results_csv = os.path.join(args.out_dir, "filter_recovery_results.csv")
    trace_csv = os.path.join(args.out_dir, "filter_recovery_training_trace.csv")
    err_png = os.path.join(args.out_dir, "filter_recovery_error_by_domain.png")
    support_png = os.path.join(args.out_dir, "filter_recovery_support_error.png")

    _write_csv(
        results_csv,
        result_rows,
        fieldnames=[
            "domain",
            "target_fn",
            "probe_mmd_final",
            "bpr_loss_final",
            "cheb_l2_to_target",
            "mono_l2_to_target",
            "mono_cos_to_target",
            "recovered_degree",
            "target_degree",
            "support_l1_error",
            "nonsupport_l1",
            "source_cheb",
            "source_mono",
            "source_init_cheb",
            "source_init_mono",
            "target_cheb",
            "target_mono",
        ],
    )
    _write_csv(
        trace_csv,
        trace_rows,
        fieldnames=[
            "domain",
            "epoch",
            "total_loss",
            "probe_mmd",
            "bpr_loss",
            "cheb_l2_to_target",
            "mono_l2_to_target",
        ],
    )
    _plot_domain_errors(result_rows, err_png)
    _plot_support_errors(result_rows, support_png)

    print(f"Saved domain results CSV: {results_csv}")
    print(f"Saved training trace CSV: {trace_csv}")
    print(f"Saved recovery error plot: {err_png}")
    print(f"Saved support/off-support error plot: {support_png}")
    print("Per-domain recovery summary:")
    for row in result_rows:
        print(
            {
                "domain": row["domain"],
                "target_fn": row["target_fn"],
                "mono_l2_to_target": row["mono_l2_to_target"],
                "mono_cos_to_target": row["mono_cos_to_target"],
                "recovered_degree": row["recovered_degree"],
                "target_degree": row["target_degree"],
                "bpr_loss_final": row["bpr_loss_final"],
            }
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Airport ablation: source starts as A, target mapping is fixed (e.g., A^3 + 0.5A), "
            "learn only source Chebyshev coefficients via probe MMD (+ optional BPR structure loss)."
        )
    )
    parser.add_argument("--domains", type=str, default="BRAZIL,EUROPE,USA")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)

    # K Chebyshev basis terms (0..K-1). K=4 includes up to degree 3.
    parser.add_argument("--cheb_k", type=int, default=4)
    parser.add_argument("--cheb_lambda_max", type=float, default=2.0)
    parser.add_argument("--source_init", type=str, default="A", choices=["A", "identity"])
    parser.add_argument(
        "--target_fn",
        type=str,
        default="a3_plus_half_a",
        choices=["square", "a3_plus_half_a", "a3_plus_0.5a", "custom"],
    )
    parser.add_argument(
        "--target_monomial_custom",
        type=str,
        default=None,
        help="Used when target_fn=custom. Example for A^3+0.5A: '0,0.5,0,1'.",
    )

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--init_eps",
        type=float,
        default=1e-3,
        help="Floor for source init coefficients to avoid dead ReLU coordinates inside ChebProp.",
    )

    parser.add_argument("--kernel_mul", type=float, default=2.0)
    parser.add_argument("--kernel_num", type=int, default=5)
    parser.add_argument("--fix_sigma", type=float, default=None)

    # Optional structural regularization on learned source filter output.
    parser.add_argument("--bpr_weight", type=float, default=0.1)
    parser.add_argument("--bpr_samples", type=int, default=2048)
    parser.add_argument("--bpr_margin", type=float, default=1.0)

    parser.add_argument(
        "--out_dir",
        type=str,
        default="../__saved__/analysis/airport_square_filter_recovery",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
