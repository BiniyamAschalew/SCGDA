"""Sanity test: SimGDA-Filter base vs A2GNN base.

Goal:
For each (s_pnums, t_pnums), verify SimGDAFilterBase outputs match A2GNNBase
when both networks share identical affine weights.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data

from models.baselines.a2gnn.a2gnn_base import A2GNNBase
from models.ours.simgda_filter.simgda_filter_base import SimGDAFilterBase


def random_graph(num_nodes: int, num_edges: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    src = torch.randint(num_nodes, (num_edges,), generator=g, dtype=torch.long)
    dst = torch.randint(num_nodes, (num_edges,), generator=g, dtype=torch.long)
    mask = src != dst
    src, dst = src[mask], dst[mask]

    edge_index = torch.stack(
        [torch.cat([src, dst], dim=0), torch.cat([dst, src], dim=0)],
        dim=0,
    )

    lin = edge_index[0] * num_nodes + edge_index[1]
    uniq = torch.unique(lin)
    edge_index = torch.stack([uniq // num_nodes, uniq % num_nodes], dim=0).contiguous()
    return edge_index


def make_cfg(gnn_type: str, *, in_dim: int, hid_dim: int, num_classes: int, num_layers: int, k: int) -> dict:
    return {
        "model": {
            "in_dim": in_dim,
            "hid_dim": hid_dim,
            "num_classes": num_classes,
            "num_layers": num_layers,
            "dropout_ratio": 0.0,
            "mode": "node",
            "activation": "relu",
            "adv": False,
            "gnn": gnn_type,
            "K": k,
            "cls_pnums": 1,
        }
    }


def copy_a2_to_filter(a2: A2GNNBase, filt: SimGDAFilterBase) -> None:
    with torch.no_grad():
        for a_conv, f_conv in zip(a2.convs, filt.convs):
            f_conv.lin.weight.copy_(a_conv.lin.weight)
            if a_conv.bias is not None and f_conv.bias is not None:
                f_conv.bias.copy_(a_conv.bias)

        if filt.mode == "node":
            f_cls = filt.cls
            a_cls = a2.cls
            f_cls.lin.weight.copy_(a_cls.lin.weight)
            if a_cls.bias is not None and f_cls.bias is not None:
                f_cls.bias.copy_(a_cls.bias)
        else:
            filt.cls.weight.copy_(a2.cls.weight)
            filt.cls.bias.copy_(a2.cls.bias)


def abs_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = (a - b).abs()
    return float(d.max().item()), float(d.mean().item())


def compare_pair(
    a2: A2GNNBase,
    filt: SimGDAFilterBase,
    a2_rand: A2GNNBase,
    source_data: Data,
    target_data: Data,
    s_pnums: int,
    t_pnums: int,
) -> dict[str, float]:
    src_param = torch.zeros(filt.k, device=source_data.x.device, dtype=source_data.x.dtype)
    src_param[int(s_pnums)] = 1.0
    tgt_param = torch.zeros(filt.k, device=target_data.x.device, dtype=target_data.x.dtype)
    tgt_param[int(t_pnums)] = 1.0

    with torch.no_grad():
        src_a2 = a2(source_data, s_pnums)
        src_f = filt(source_data, src_param)
        src_rand = a2_rand(source_data, s_pnums)

        tgt_a2 = a2(target_data, t_pnums)
        tgt_f = filt(target_data, tgt_param)
        tgt_rand = a2_rand(target_data, t_pnums)

    src_max, src_mean = abs_stats(src_a2, src_f)
    tgt_max, tgt_mean = abs_stats(tgt_a2, tgt_f)
    src_rand_mean = float((src_a2 - src_rand).abs().mean().item())
    tgt_rand_mean = float((tgt_a2 - tgt_rand).abs().mean().item())

    return {
        "s_pnums": int(s_pnums),
        "t_pnums": int(t_pnums),
        "src_max_abs_diff_match": src_max,
        "src_mean_abs_diff_match": src_mean,
        "tgt_max_abs_diff_match": tgt_max,
        "tgt_mean_abs_diff_match": tgt_mean,
        "src_mean_abs_diff_random": src_rand_mean,
        "tgt_mean_abs_diff_random": tgt_rand_mean,
        "src_ratio_match_to_random": float(src_mean / (src_rand_mean + 1e-12)),
        "tgt_ratio_match_to_random": float(tgt_mean / (tgt_rand_mean + 1e-12)),
        "src_allclose_1e6": float(torch.allclose(src_a2, src_f, atol=1e-6, rtol=1e-6)),
        "src_rand_allclose_1e6": float(torch.allclose(src_a2, src_rand, atol=1e-6, rtol=1e-6)),
        "tgt_allclose_1e6": float(torch.allclose(tgt_a2, tgt_f, atol=1e-6, rtol=1e-6)),
        "tgt_rand_allclose_1e6": float(torch.allclose(tgt_a2, tgt_rand, atol=1e-6, rtol=1e-6)),
        "src_allclose_1e5": float(torch.allclose(src_a2, src_f, atol=1e-5, rtol=1e-5)),
        "src_rand_allclose_1e5": float(torch.allclose(src_a2, src_rand, atol=1e-5, rtol=1e-5)),
        "tgt_allclose_1e5": float(torch.allclose(tgt_a2, tgt_f, atol=1e-5, rtol=1e-5)),
        "tgt_rand_allclose_1e5": float(torch.allclose(tgt_a2, tgt_rand, atol=1e-5, rtol=1e-5)),
    }


def save_rows_csv(rows: list[dict[str, float]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_heatmap(rows: list[dict[str, float]], k: int, metric: str, out_path: Path) -> None:
    grid = torch.full((k, k), float("nan"))
    for row in rows:
        grid[int(row["s_pnums"]), int(row["t_pnums"])] = float(row[metric])

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(grid.numpy(), cmap="viridis")
    ax.set_xlabel("t_pnums")
    ax.set_ylabel("s_pnums")
    ax.set_title(metric)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_layer_sweep(device: torch.device, *, k: int, num_layers: int, seed: int) -> list[dict[str, float]]:
    in_dim = 64
    hid_dim = 48
    num_classes = 7

    cfg_a2 = make_cfg("prop", in_dim=in_dim, hid_dim=hid_dim, num_classes=num_classes, num_layers=num_layers, k=k)
    cfg_f = make_cfg("filter_mono", in_dim=in_dim, hid_dim=hid_dim, num_classes=num_classes, num_layers=num_layers, k=k)

    n_src, n_tgt = 320, 280
    e_src, e_tgt = 2200, 1800

    source_data = Data(
        x=torch.randn(n_src, in_dim, device=device),
        edge_index=random_graph(n_src, e_src, seed=seed + 11).to(device),
        y=torch.randint(0, num_classes, (n_src,), device=device),
    ).to(device)
    target_data = Data(
        x=torch.randn(n_tgt, in_dim, device=device),
        edge_index=random_graph(n_tgt, e_tgt, seed=seed + 23).to(device),
        y=torch.randint(0, num_classes, (n_tgt,), device=device),
    ).to(device)

    torch.manual_seed(seed + 101)
    a2 = A2GNNBase(cfg_a2).to(device).eval()
    filt = SimGDAFilterBase(cfg_f).to(device).eval()
    copy_a2_to_filter(a2, filt)

    torch.manual_seed(seed + 707)
    a2_rand = A2GNNBase(cfg_a2).to(device).eval()

    rows = []
    for s_pnums in range(k):
        for t_pnums in range(k):
            r = compare_pair(a2, filt, a2_rand, source_data, target_data, s_pnums, t_pnums)
            r["layers"] = int(num_layers)
            rows.append(r)
    return rows


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 37
    k = 6
    layers_to_test = [1, 2, 3]

    all_rows: list[dict[str, float]] = []
    for layers in layers_to_test:
        all_rows.extend(run_layer_sweep(device, k=k, num_layers=layers, seed=seed + layers * 97))

    out_dir = Path("__saved__/analysis/simgda_filter_vs_a2gnn_pair_sanity")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pair_results.csv"
    save_rows_csv(all_rows, csv_path)

    for layers in layers_to_test:
        rows = [r for r in all_rows if int(r["layers"]) == layers]
        save_heatmap(rows, k, "src_mean_abs_diff_match", out_dir / f"src_mean_abs_diff_match_L{layers}.png")
        save_heatmap(rows, k, "tgt_mean_abs_diff_match", out_dir / f"tgt_mean_abs_diff_match_L{layers}.png")
        save_heatmap(rows, k, "src_ratio_match_to_random", out_dir / f"src_ratio_match_to_random_L{layers}.png")
        save_heatmap(rows, k, "tgt_ratio_match_to_random", out_dir / f"tgt_ratio_match_to_random_L{layers}.png")

    total = len(all_rows)
    src_ok = sum(int(r["src_allclose_1e5"]) for r in all_rows)
    tgt_ok = sum(int(r["tgt_allclose_1e5"]) for r in all_rows)
    worst_src = max(float(r["src_max_abs_diff_match"]) for r in all_rows)
    worst_tgt = max(float(r["tgt_max_abs_diff_match"]) for r in all_rows)
    mean_src_ratio = sum(float(r["src_ratio_match_to_random"]) for r in all_rows) / total
    mean_tgt_ratio = sum(float(r["tgt_ratio_match_to_random"]) for r in all_rows) / total

    print("=== SimGDA-Filter vs A2GNN Pair Sanity ===")
    print(f"device: {device}")
    print(f"layers tested: {layers_to_test}")
    print(f"K: {k}, pairs tested: {total}")
    print(f"allclose@1e-5 src: {src_ok}/{total}")
    print(f"allclose@1e-5 tgt: {tgt_ok}/{total}")
    print(f"worst src max abs diff: {worst_src:.10f}")
    print(f"worst tgt max abs diff: {worst_tgt:.10f}")
    print(f"mean src ratio(match/random): {mean_src_ratio:.10f}")
    print(f"mean tgt ratio(match/random): {mean_tgt_ratio:.10f}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
