from pathlib import Path

import matplotlib.pyplot as plt
import torch

from utils.ablation_utils.common import save_table


def edge_distance_metrics(
    normal_edge_index: torch.Tensor,
    aligned_edge_index: torch.Tensor,
    aligned_edge_weight: torch.Tensor,
    num_nodes: int,
) -> dict[str, float]:
    """Distance/overlap stats between original binary adjacency and aligned weighted adjacency."""
    normal_idx = normal_edge_index.detach().cpu().long()
    aligned_idx = aligned_edge_index.detach().cpu().long()
    aligned_w = aligned_edge_weight.detach().cpu().float()
    normal_w = torch.ones(normal_idx.size(1), dtype=torch.float32)

    idx_cat = torch.cat([normal_idx, aligned_idx], dim=1)
    val_cat = torch.cat([normal_w, -aligned_w], dim=0)
    diff = torch.sparse_coo_tensor(
        idx_cat,
        val_cat,
        size=(num_nodes, num_nodes),
        dtype=torch.float32,
    ).coalesce()
    l1 = float(diff.values().abs().sum().item())
    l2 = float(torch.sqrt((diff.values() ** 2).sum()).item())

    normal_lin = (normal_idx[0] * num_nodes + normal_idx[1]).tolist()
    aligned_lin = (aligned_idx[0] * num_nodes + aligned_idx[1]).tolist()
    set_normal = set(normal_lin)
    set_aligned = set(aligned_lin)
    inter = len(set_normal & set_aligned)
    union = len(set_normal | set_aligned)
    jaccard = float(inter / union) if union > 0 else 1.0

    return {"l1": l1, "l2": l2, "jaccard": jaccard}


def build_structural_measures(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
    pagerank_iters: int = 50,
    eig_iters: int = 50,
    pagerank_alpha: float = 0.85,
) -> dict[str, torch.Tensor]:
    """Compute structural distributions on CPU for one graph."""
    idx = edge_index.detach().cpu().long()
    if edge_weight is None:
        w = torch.ones(idx.size(1), dtype=torch.float32)
    else:
        w = edge_weight.detach().cpu().float()

    src = idx[0]
    dst = idx[1]

    degree_w = torch.zeros(num_nodes, dtype=torch.float32)
    degree_w = degree_w.index_add(0, src, w)
    degree_centrality = degree_w / max(1, num_nodes - 1)

    out_deg = degree_w.clamp_min(1e-12)
    trans = w / out_deg[src]
    pr = torch.full((num_nodes,), 1.0 / float(num_nodes), dtype=torch.float32)
    teleport = (1.0 - pagerank_alpha) / float(num_nodes)
    dangling_mask = degree_w <= 0
    for _ in range(max(1, pagerank_iters)):
        nxt = torch.zeros_like(pr)
        nxt = nxt.index_add(0, dst, pr[src] * trans)
        dangling_mass = pr[dangling_mask].sum() / float(num_nodes)
        pr = pagerank_alpha * (nxt + dangling_mass) + teleport

    eig = torch.full((num_nodes,), 1.0 / float(num_nodes), dtype=torch.float32)
    for _ in range(max(1, eig_iters)):
        nxt = torch.zeros_like(eig)
        nxt = nxt.index_add(0, dst, eig[src] * w)
        norm = torch.norm(nxt, p=2).clamp_min(1e-12)
        eig = nxt / norm

    return {
        "degree": degree_w,
        "degree_centrality": degree_centrality,
        "pagerank": pr,
        "eigenvector": eig.abs(),
    }


def _dist_stats(values: torch.Tensor) -> dict[str, float]:
    vals = values.detach().cpu().float().flatten()
    if vals.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    q = torch.quantile(vals, torch.tensor([0.5, 0.9]))
    return {
        "mean": float(vals.mean().item()),
        "std": float(vals.std(unbiased=False).item()),
        "min": float(vals.min().item()),
        "median": float(q[0].item()),
        "p90": float(q[1].item()),
        "max": float(vals.max().item()),
    }


def save_structural_analysis(
    path_png: Path,
    path_csv: Path,
    source_data,
    target_data,
    align_state: dict,
    cfg: dict,
) -> None:
    """Compare source/target structural distributions for normal vs aligned graphs."""
    source_normal = build_structural_measures(
        source_data.edge_index,
        edge_weight=None,
        num_nodes=int(source_data.x.size(0)),
        pagerank_iters=cfg["struct_pagerank_iters"],
        eig_iters=cfg["struct_eig_iters"],
        pagerank_alpha=cfg["struct_pagerank_alpha"],
    )
    source_aligned = build_structural_measures(
        align_state["source_aligned_edge_index"],
        edge_weight=align_state["source_aligned_edge_weight"],
        num_nodes=int(source_data.x.size(0)),
        pagerank_iters=cfg["struct_pagerank_iters"],
        eig_iters=cfg["struct_eig_iters"],
        pagerank_alpha=cfg["struct_pagerank_alpha"],
    )
    target_normal = build_structural_measures(
        target_data.edge_index,
        edge_weight=None,
        num_nodes=int(target_data.x.size(0)),
        pagerank_iters=cfg["struct_pagerank_iters"],
        eig_iters=cfg["struct_eig_iters"],
        pagerank_alpha=cfg["struct_pagerank_alpha"],
    )
    target_aligned = build_structural_measures(
        align_state["target_aligned_edge_index"],
        edge_weight=align_state["target_aligned_edge_weight"],
        num_nodes=int(target_data.x.size(0)),
        pagerank_iters=cfg["struct_pagerank_iters"],
        eig_iters=cfg["struct_eig_iters"],
        pagerank_alpha=cfg["struct_pagerank_alpha"],
    )

    measures = ["degree", "pagerank", "eigenvector"]
    fig, axes = plt.subplots(2, len(measures), figsize=(5 * len(measures), 8))
    if len(measures) == 1:
        axes = axes.reshape(2, 1)

    for j, measure in enumerate(measures):
        ax_normal = axes[0, j]
        ax_aligned = axes[1, j]

        ax_normal.hist(
            source_normal[measure].numpy(),
            bins=cfg["struct_bins"],
            alpha=0.55,
            density=True,
            label="source normal",
        )
        ax_normal.hist(
            target_normal[measure].numpy(),
            bins=cfg["struct_bins"],
            alpha=0.55,
            density=True,
            label="target normal",
        )
        ax_normal.set_title(f"Normal {measure}")
        ax_normal.grid(alpha=0.25)
        ax_normal.legend()

        ax_aligned.hist(
            source_aligned[measure].numpy(),
            bins=cfg["struct_bins"],
            alpha=0.55,
            density=True,
            label="source aligned",
        )
        ax_aligned.hist(
            target_aligned[measure].numpy(),
            bins=cfg["struct_bins"],
            alpha=0.55,
            density=True,
            label="target aligned",
        )
        ax_aligned.set_title(f"Aligned {measure}")
        ax_aligned.grid(alpha=0.25)
        ax_aligned.legend()

    fig.suptitle("Structural distribution comparison: normal vs aligned")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path_png, dpi=200)
    plt.close(fig)

    stats_rows = []
    payload = {
        ("source", "normal"): source_normal,
        ("source", "aligned"): source_aligned,
        ("target", "normal"): target_normal,
        ("target", "aligned"): target_aligned,
    }
    for (domain, graph_type), data_map in payload.items():
        for measure_name, values in data_map.items():
            stats = _dist_stats(values)
            stats_rows.append(
                {
                    "domain": domain,
                    "graph": graph_type,
                    "measure": measure_name,
                    **stats,
                }
            )

    save_table(
        path_csv,
        stats_rows,
        fieldnames=["domain", "graph", "measure", "mean", "std", "min", "median", "p90", "max"],
    )
