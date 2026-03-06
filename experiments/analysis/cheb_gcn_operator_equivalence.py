"""
Operator-level sanity checks for Chebyshev-vs-GCN equivalence.

Key claim under test:
- With symmetric normalization, lambda_max=2, and Cheb coefficients [0, -1, 0, 0],
  Chebyshev T1 branch equals normalized adjacency propagation (no added self-loops).
"""

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import coalesce, get_laplacian, remove_self_loops, to_undirected


def get_cheb_norm_laplacian_like_chebprop(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
    lambda_max: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float64,
):
    """Exact ChebProp normalization (matches models/__components/chebprop.py::__norm__)."""
    edge_index, edge_weight = get_laplacian(
        edge_index,
        edge_weight,
        normalization="sym",
        dtype=dtype,
        num_nodes=num_nodes,
    )
    assert edge_weight is not None

    if lambda_max is None:
        lambda_max = 2.0 * edge_weight.max()
    elif not isinstance(lambda_max, torch.Tensor):
        lambda_max = torch.tensor(lambda_max, dtype=dtype, device=edge_index.device)

    edge_weight = (2.0 * edge_weight) / lambda_max
    edge_weight.masked_fill_(edge_weight == float("inf"), 0)

    loop_mask = edge_index[0] == edge_index[1]
    edge_weight[loop_mask] -= 1.0

    # ChebProp uses normalized adjacency A_norm as the base operator.
    edge_weight = -edge_weight

    return edge_index, edge_weight


def get_gcn_norm_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
    add_self_loops: bool = False,
    dtype: torch.dtype = torch.float64,
):
    """GCN normalization coefficients (same as torch_geometric.nn.GCNConv)."""
    return gcn_norm(
        edge_index,
        edge_weight=edge_weight,
        num_nodes=num_nodes,
        improved=False,
        add_self_loops=add_self_loops,
        flow="source_to_target",
        dtype=dtype,
    )


def edge_to_dense_operator(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    dtype: torch.dtype = torch.float64,
):
    """
    Convert (edge_index, edge_weight) into dense operator matrix M such that:
    out = M @ x, following MessagePassing(source_to_target):
    out[target] += weight * x[source].
    """
    mat = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=edge_index.device)
    src = edge_index[0]
    dst = edge_index[1]
    mat[dst, src] += edge_weight
    return mat


def cheb_filter_operator_from_lhat(lhat: torch.Tensor, coeffs: torch.Tensor):
    """
    Build filter operator H = sum_k coeffs[k] * T_k(lhat) with Chebyshev recurrence.
    """
    k_terms = coeffs.numel()
    n = lhat.size(0)
    eye = torch.eye(n, dtype=lhat.dtype, device=lhat.device)

    t0 = eye
    out = coeffs[0] * t0
    if k_terms == 1:
        return out

    t1 = lhat
    out = out + coeffs[1] * t1

    for k in range(2, k_terms):
        t2 = 2.0 * (lhat @ t1) - t0
        out = out + coeffs[k] * t2
        t0, t1 = t1, t2

    return out


def make_graph(
    graph_type: str,
    num_nodes: int,
    num_edges: int,
    seed: int,
    dtype: torch.dtype = torch.float64,
):
    gen = torch.Generator()
    gen.manual_seed(seed)

    row = torch.randint(0, num_nodes, (num_edges,), generator=gen)
    col = torch.randint(0, num_nodes, (num_edges,), generator=gen)
    edge_index = torch.stack([row, col], dim=0)

    edge_weight = None
    if "weighted" in graph_type:
        edge_weight = torch.rand(num_edges, generator=gen, dtype=dtype) + 0.1

    if "undirected" in graph_type:
        edge_index, edge_weight = to_undirected(
            edge_index,
            edge_attr=edge_weight,
            num_nodes=num_nodes,
            reduce="add",
        )

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, edge_weight = coalesce(
        edge_index,
        edge_weight,
        num_nodes=num_nodes,
        reduce="sum",
    )

    return edge_index, edge_weight


@dataclass
class Case:
    name: str
    graph_type: str
    coeffs: torch.Tensor
    lambda_max: Optional[float]
    gcn_add_self_loops: bool
    expected_equiv: bool


def evaluate_case(case: Case, num_trials: int, num_nodes: int, num_edges: int):
    matrix_rel_errors = []
    feature_rel_errors = []

    for seed in range(num_trials):
        edge_index, edge_weight = make_graph(case.graph_type, num_nodes, num_edges, seed)

        cheb_edge_index, cheb_w = get_cheb_norm_laplacian_like_chebprop(
            edge_index=edge_index,
            num_nodes=num_nodes,
            edge_weight=edge_weight,
            lambda_max=case.lambda_max,
            dtype=torch.float64,
        )
        gcn_edge_index, gcn_w = get_gcn_norm_adjacency(
            edge_index=edge_index,
            num_nodes=num_nodes,
            edge_weight=edge_weight,
            add_self_loops=case.gcn_add_self_loops,
            dtype=torch.float64,
        )

        lhat = edge_to_dense_operator(cheb_edge_index, cheb_w, num_nodes, dtype=torch.float64)
        gcn_adj = edge_to_dense_operator(gcn_edge_index, gcn_w, num_nodes, dtype=torch.float64)
        cheb_op = cheb_filter_operator_from_lhat(lhat, case.coeffs)

        matrix_rel = (cheb_op - gcn_adj).norm() / (gcn_adj.norm() + 1e-15)
        matrix_rel_errors.append(float(matrix_rel.item()))

        x = torch.randn((num_nodes, 32), dtype=torch.float64)
        out_cheb = cheb_op @ x
        out_gcn = gcn_adj @ x
        feature_rel = (out_cheb - out_gcn).norm() / (out_gcn.norm() + 1e-15)
        feature_rel_errors.append(float(feature_rel.item()))

    return matrix_rel_errors, feature_rel_errors


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)

    coeff_pos = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    coeff_neg = torch.tensor([0.0, -1.0, 0.0, 0.0], dtype=torch.float64)

    cases = [
        Case(
            name="Undirected + [0, 1,0,0] + no loops + lambda=2",
            graph_type="undirected_unweighted",
            coeffs=coeff_pos,
            lambda_max=2.0,
            gcn_add_self_loops=False,
            expected_equiv=True,
        ),
        Case(
            name="Undirected + [0,-1,0,0] + no loops + lambda=2",
            graph_type="undirected_unweighted",
            coeffs=coeff_neg,
            lambda_max=2.0,
            gcn_add_self_loops=False,
            expected_equiv=False,
        ),
        Case(
            name="Undirected + [0,-1,0,0] + gcn loops + lambda=2",
            graph_type="undirected_unweighted",
            coeffs=coeff_neg,
            lambda_max=2.0,
            gcn_add_self_loops=True,
            expected_equiv=False,
        ),
        Case(
            name="Directed + [0,-1,0,0] + no loops + lambda=2",
            graph_type="directed_unweighted",
            coeffs=coeff_neg,
            lambda_max=2.0,
            gcn_add_self_loops=False,
            expected_equiv=False,
        ),
        Case(
            name="Undirected weighted + [0, 1,0,0] + no loops + lambda=2",
            graph_type="undirected_weighted",
            coeffs=coeff_pos,
            lambda_max=2.0,
            gcn_add_self_loops=False,
            expected_equiv=True,
        ),
    ]

    rows = []
    summary = []
    for case in cases:
        matrix_rel, feature_rel = evaluate_case(
            case=case,
            num_trials=args.num_trials,
            num_nodes=args.num_nodes,
            num_edges=args.num_edges,
        )

        for i, (m_err, f_err) in enumerate(zip(matrix_rel, feature_rel)):
            rows.append({
                "case": case.name,
                "trial": i,
                "expected_equiv": int(case.expected_equiv),
                "matrix_rel_error": m_err,
                "feature_rel_error": f_err,
            })

        summary.append({
            "case": case.name,
            "expected_equiv": case.expected_equiv,
            "matrix_rel_mean": float(torch.tensor(matrix_rel).mean().item()),
            "matrix_rel_std": float(torch.tensor(matrix_rel).std().item()),
            "feature_rel_mean": float(torch.tensor(feature_rel).mean().item()),
            "feature_rel_std": float(torch.tensor(feature_rel).std().item()),
        })

    csv_path = os.path.join(args.out_dir, "cheb_gcn_operator_equivalence.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "trial",
                "expected_equiv",
                "matrix_rel_error",
                "feature_rel_error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    labels = [s["case"] for s in summary]
    mat_means = [s["matrix_rel_mean"] for s in summary]
    feat_means = [s["feature_rel_mean"] for s in summary]
    eps = 1e-16
    mat_means_plot = [max(v, eps) for v in mat_means]
    feat_means_plot = [max(v, eps) for v in feat_means]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    x = torch.arange(len(labels)).numpy()

    axes[0].bar(x, mat_means_plot)
    axes[0].set_title("Matrix-Level Relative Error: ||H_cheb - A_gcn|| / ||A_gcn||")
    axes[0].set_ylabel("Relative error")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_yscale("log")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)

    axes[1].bar(x, feat_means_plot)
    axes[1].set_title("Feature-Level Relative Error: ||H_cheb X - A_gcn X|| / ||A_gcn X||")
    axes[1].set_ylabel("Relative error")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_yscale("log")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)

    fig_path = os.path.join(args.out_dir, "cheb_gcn_operator_equivalence.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print("\nSummary (mean +- std):")
    for s in summary:
        print(
            f"- {s['case']}\n"
            f"  expected_equiv={s['expected_equiv']} | "
            f"matrix_rel={s['matrix_rel_mean']:.6e} +- {s['matrix_rel_std']:.3e} | "
            f"feature_rel={s['feature_rel_mean']:.6e} +- {s['feature_rel_std']:.3e}"
        )

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Plot: {fig_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Chebyshev and GCN normalized propagation operators."
    )
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--num_nodes", type=int, default=80)
    parser.add_argument("--num_edges", type=int, default=320)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="../__saved__/analysis/cheb_gcn_operator_equivalence",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
