"""Sanity test for SimGDA-Filter basis-index mapping.

Checks that for many (s_pnums, t_pnums) pairs:
1) SimGDAFilterBase output matches a PropGCNConv reference network
   with identical weights.
2) The matched error is much smaller than a random-reference baseline.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data

from models.__layers.prop_gcn_conv import PropGCNConv
from models.ours.simgda_filter.simgda_filter_base import SimGDAFilterBase
from models.__layers.build_layer import build_activation


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


class PropA2Reference(nn.Module):
    """A2GNNBase-style network using PropGCNConv with prop_nums."""

    def __init__(self, config: dict):
        super().__init__()

        self.in_dim = config["model"]["in_dim"]
        self.hid_dim = config["model"]["hid_dim"]
        self.num_classes = config["model"]["num_classes"]
        self.num_layers = config["model"]["num_layers"]
        self.dropout = config["model"]["dropout_ratio"]
        self.mode = config["model"]["mode"]
        self.cls_pnums = int(config["model"].get("cls_pnums", 1))

        self.act = build_activation(config["model"]["activation"])

        self.convs = nn.ModuleList()
        self.convs.append(PropGCNConv(self.in_dim, self.hid_dim))
        for _ in range(self.num_layers - 1):
            self.convs.append(PropGCNConv(self.hid_dim, self.hid_dim))

        if self.mode == "node":
            self.cls = PropGCNConv(self.hid_dim, self.num_classes)
        else:
            self.cls = nn.Linear(self.hid_dim, self.num_classes)

    def forward(self, data: Data, prop_nums: int) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index, prop_nums=prop_nums)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.mode == "node":
            x = self.cls(x, edge_index, prop_nums=self.cls_pnums)
        else:
            x = self.cls(x)
        return x


def copy_weights(filter_model: SimGDAFilterBase, ref_model: PropA2Reference) -> None:
    with torch.no_grad():
        for f_conv, r_conv in zip(filter_model.convs, ref_model.convs):
            r_conv.lin.weight.copy_(f_conv.lin.weight)
            if f_conv.bias is not None and r_conv.bias is not None:
                r_conv.bias.copy_(f_conv.bias)

        if filter_model.mode == "node":
            r_cls = ref_model.cls
            f_cls = filter_model.cls
            r_cls.lin.weight.copy_(f_cls.lin.weight)
            if f_cls.bias is not None and r_cls.bias is not None:
                r_cls.bias.copy_(f_cls.bias)
        else:
            ref_model.cls.weight.copy_(filter_model.cls.weight)
            ref_model.cls.bias.copy_(filter_model.cls.bias)


def _abs_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = (a - b).abs()
    return float(d.max().item()), float(d.mean().item())


def compare_pair(
    config: dict,
    data: Data,
    s_pnums: int,
    t_pnums: int,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    filter_net = SimGDAFilterBase(config).to(data.x.device).eval()
    ref_net = PropA2Reference(config).to(data.x.device).eval()
    copy_weights(filter_net, ref_net)

    torch.manual_seed(seed + 999)
    rand_ref = PropA2Reference(config).to(data.x.device).eval()

    src_param = torch.zeros(int(config["model"]["K"]), device=data.x.device, dtype=data.x.dtype)
    src_param[int(s_pnums)] = 1.0
    tgt_param = torch.zeros(int(config["model"]["K"]), device=data.x.device, dtype=data.x.dtype)
    tgt_param[int(t_pnums)] = 1.0

    with torch.no_grad():
        src_filter = filter_net(data, src_param)
        src_ref = ref_net(data, s_pnums)
        src_rand = rand_ref(data, s_pnums)

        tgt_filter = filter_net(data, tgt_param)
        tgt_ref = ref_net(data, t_pnums)
        tgt_rand = rand_ref(data, t_pnums)

    src_max, src_mean = _abs_stats(src_filter, src_ref)
    tgt_max, tgt_mean = _abs_stats(tgt_filter, tgt_ref)
    src_rand_mean = float((src_filter - src_rand).abs().mean().item())
    tgt_rand_mean = float((tgt_filter - tgt_rand).abs().mean().item())

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
        "src_allclose_1e6": float(torch.allclose(src_filter, src_ref, atol=1e-6, rtol=1e-6)),
        "tgt_allclose_1e6": float(torch.allclose(tgt_filter, tgt_ref, atol=1e-6, rtol=1e-6)),
        "src_allclose_1e5": float(torch.allclose(src_filter, src_ref, atol=1e-5, rtol=1e-5)),
        "tgt_allclose_1e5": float(torch.allclose(tgt_filter, tgt_ref, atol=1e-5, rtol=1e-5)),
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
    if not rows:
        return
    grid = torch.full((k, k), float("nan"))
    for row in rows:
        grid[int(row["s_pnums"]), int(row["t_pnums"])] = float(row[metric])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(grid.numpy(), cmap="viridis")
    ax.set_title(metric)
    ax.set_xlabel("t_pnums")
    ax.set_ylabel("s_pnums")
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 23
    torch.manual_seed(seed)

    k = 6
    cfg = {
        "model": {
            "in_dim": 64,
            "hid_dim": 32,
            "num_classes": 7,
            "num_layers": 2,
            "dropout_ratio": 0.0,
            "activation": "relu",
            "mode": "node",
            "gnn": "filter_mono",
            "K": k,
            "cls_pnums": 1,
            "adv": False,
        }
    }

    n = 384
    e = 2400
    x = torch.randn(n, cfg["model"]["in_dim"], device=device)
    edge_index = random_graph(n, e, seed=seed).to(device)
    y = torch.randint(0, cfg["model"]["num_classes"], (n,), device=device)
    data = Data(x=x, edge_index=edge_index, y=y).to(device)

    rows = []
    for s_pnums in range(k):
        for t_pnums in range(k):
            stats = compare_pair(cfg, data, s_pnums, t_pnums, seed + s_pnums * 31 + t_pnums * 17)
            rows.append(stats)

    out_dir = Path("__saved__/analysis/simgda_filter_pair_sanity")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "pair_results.csv"
    save_rows_csv(rows, csv_path)

    save_heatmap(rows, k, "src_mean_abs_diff_match", out_dir / "src_mean_abs_diff_match.png")
    save_heatmap(rows, k, "tgt_mean_abs_diff_match", out_dir / "tgt_mean_abs_diff_match.png")
    save_heatmap(rows, k, "src_ratio_match_to_random", out_dir / "src_ratio_match_to_random.png")
    save_heatmap(rows, k, "tgt_ratio_match_to_random", out_dir / "tgt_ratio_match_to_random.png")

    src_ok = sum(int(r["src_allclose_1e5"]) for r in rows)
    tgt_ok = sum(int(r["tgt_allclose_1e5"]) for r in rows)
    total = len(rows)
    worst_src = max(float(r["src_max_abs_diff_match"]) for r in rows)
    worst_tgt = max(float(r["tgt_max_abs_diff_match"]) for r in rows)
    mean_src_ratio = sum(float(r["src_ratio_match_to_random"]) for r in rows) / total
    mean_tgt_ratio = sum(float(r["tgt_ratio_match_to_random"]) for r in rows) / total

    print("=== SimGDA-Filter sp/tp Pair Sanity ===")
    print(f"device: {device}")
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
