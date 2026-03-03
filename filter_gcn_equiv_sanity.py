"""Sanity check: multi-layer GCN vs FilterGCNConv(mono).

Expectation:
1) With identical affine weights and filter_param=[1, 0, 0, ...],
   FilterGCNConv(mono) should match one-hop GCNConv exactly.
2) This should hold for 1/2/3 stacked layers.
3) Matched-network distance should be much smaller than random-network distance.
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
from torch_geometric.nn import GCNConv

from models.__layers.filter_gcn_conv import FilterGCNConv


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

    # deduplicate
    lin = edge_index[0] * num_nodes + edge_index[1]
    uniq = torch.unique(lin)
    edge_index = torch.stack([uniq // num_nodes, uniq % num_nodes], dim=0).contiguous()
    return edge_index


class GCNStack(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GCNConv(dims[i], dims[i + 1], add_self_loops=True, normalize=True, bias=True)
                for i in range(len(dims) - 1)
            ]
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class FilterStack(nn.Module):
    def __init__(self, dims: list[int], mono_param: torch.Tensor):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FilterGCNConv(
                    dims[i],
                    dims[i + 1],
                    filter_type="mono",
                    add_self_loops=True,
                    prepend_zero_to_filter=True,
                    bias=True,
                )
                for i in range(len(dims) - 1)
            ]
        )
        self.register_buffer("mono_param", mono_param)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, filter_param=self.mono_param)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


def copy_gcn_to_filter(gcn_stack: GCNStack, filter_stack: FilterStack) -> None:
    with torch.no_grad():
        for g_layer, f_layer in zip(gcn_stack.layers, filter_stack.layers):
            f_layer.lin.weight.copy_(g_layer.lin.weight)
            if g_layer.bias is not None and f_layer.bias is not None:
                f_layer.bias.copy_(g_layer.bias)


def compare_stacks(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    dims: list[int],
    mono_param: torch.Tensor,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    gcn = GCNStack(dims).to(x.device)
    filt = FilterStack(dims, mono_param).to(x.device)
    copy_gcn_to_filter(gcn, filt)

    # Baseline network (same architecture, different random init).
    torch.manual_seed(seed + 999)
    gcn_rand = GCNStack(dims).to(x.device)

    gcn.eval()
    filt.eval()
    gcn_rand.eval()
    with torch.no_grad():
        out_gcn = gcn(x, edge_index)
        out_filt = filt(x, edge_index)
        out_rand = gcn_rand(x, edge_index)

    diff_match = (out_gcn - out_filt).abs()
    diff_rand = (out_gcn - out_rand).abs()

    return {
        "max_abs_diff_match": float(diff_match.max().item()),
        "mean_abs_diff_match": float(diff_match.mean().item()),
        "mean_abs_diff_random": float(diff_rand.mean().item()),
        "ratio_match_to_random": float((diff_match.mean() / (diff_rand.mean() + 1e-12)).item()),
        "gcn_norm": float(out_gcn.norm().item()),
        "filter_norm": float(out_filt.norm().item()),
        "rand_norm": float(out_rand.norm().item()),
        "gcn_std": float(out_gcn.std().item()),
        "filter_std": float(out_filt.std().item()),
        "rand_std": float(out_rand.std().item()),
        "allclose_1e6": float(torch.allclose(out_gcn, out_filt, atol=1e-6, rtol=1e-6)),
        "allclose_1e5": float(torch.allclose(out_gcn, out_filt, atol=1e-5, rtol=1e-5)),
    }


def save_rows_csv(rows: list[dict[str, float]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict[str, float]], out_path: Path) -> None:
    if not rows:
        return

    x = [int(r["layers"]) for r in rows]
    mean_match = [float(r["mean_abs_diff_match"]) for r in rows]
    mean_rand = [float(r["mean_abs_diff_random"]) for r in rows]

    gcn_norm = [float(r["gcn_norm"]) for r in rows]
    filter_norm = [float(r["filter_norm"]) for r in rows]
    rand_norm = [float(r["rand_norm"]) for r in rows]

    gcn_std = [float(r["gcn_std"]) for r in rows]
    filter_std = [float(r["filter_std"]) for r in rows]
    rand_std = [float(r["rand_std"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].plot(x, mean_match, marker="o", linewidth=2.0, label="GCN vs Filter (matched)")
    axes[0].plot(x, mean_rand, marker="o", linewidth=2.0, label="GCN vs Random")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Number of layers")
    axes[0].set_ylabel("Mean abs diff (log scale)")
    axes[0].set_title("Distance Across Layers")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, gcn_norm, marker="o", linewidth=2.0, label="GCN norm")
    axes[1].plot(x, filter_norm, marker="o", linewidth=2.0, label="Filter norm")
    axes[1].plot(x, rand_norm, marker="o", linewidth=2.0, label="Random norm")
    axes[1].set_xlabel("Number of layers")
    axes[1].set_ylabel("Output norm")
    axes[1].set_title("Output Norms")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(x, gcn_std, marker="o", linewidth=2.0, label="GCN std")
    axes[2].plot(x, filter_std, marker="o", linewidth=2.0, label="Filter std")
    axes[2].plot(x, rand_std, marker="o", linewidth=2.0, label="Random std")
    axes[2].set_xlabel("Number of layers")
    axes[2].set_ylabel("Output std")
    axes[2].set_title("Output STDs")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.suptitle("GCN vs FilterGCNConv(mono=[1,0,0,...]) Sanity")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 17
    torch.manual_seed(seed)

    n = 512
    in_dim = 128
    hid_dim = 64
    e = 4000

    x = torch.randn(n, in_dim, device=device)
    edge_index = random_graph(n, e, seed=seed).to(device)

    mono_param = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    rows = []
    for layers in (1, 2, 3):
        dims = [in_dim] + [hid_dim] * layers
        stats = compare_stacks(
            x=x,
            edge_index=edge_index,
            dims=dims,
            mono_param=mono_param,
            seed=seed + 13 * layers,
        )
        rows.append({"layers": layers, **stats})
        print(f"\n=== {layers}-layer sanity ===")
        print("max_abs_diff_match:", stats["max_abs_diff_match"])
        print("mean_abs_diff_match:", stats["mean_abs_diff_match"])
        print("mean_abs_diff_random:", stats["mean_abs_diff_random"])
        print("ratio_match_to_random:", stats["ratio_match_to_random"])
        print(
            "norms(gcn/filter/rand):",
            stats["gcn_norm"],
            stats["filter_norm"],
            stats["rand_norm"],
        )
        print(
            "stds(gcn/filter/rand):",
            stats["gcn_std"],
            stats["filter_std"],
            stats["rand_std"],
        )
        print("allclose(1e-6):", bool(stats["allclose_1e6"]))
        print("allclose(1e-5):", bool(stats["allclose_1e5"]))

    out_dir = Path("__saved__/analysis/filter_gcn_equiv_sanity")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "layer_comparison.csv"
    png_path = out_dir / "layer_comparison.png"
    save_rows_csv(rows, csv_path)
    save_plot(rows, png_path)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
