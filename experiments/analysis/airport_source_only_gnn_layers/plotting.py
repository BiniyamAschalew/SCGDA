from __future__ import annotations

import math
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _domain_styles(source_domain: str, target_domain: str):
    return {
        source_domain: {"color": "#2E8B57", "marker": "o"},
        target_domain: {"color": "#C62828", "marker": "^"},
    }


def _draw_snapshot_panel(
    ax,
    view: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    source_domain: str,
    target_domain: str,
    domain_styles: dict[str, dict[str, str]],
    show_labels: bool,
):
    for domain in (source_domain, target_domain):
        style = domain_styles.get(domain, {"color": "#333333", "marker": "o"})
        cur = view[view["domain"] == domain]
        ax.scatter(
            cur[x_col],
            cur[y_col],
            s=16,
            alpha=0.72,
            c=style["color"],
            marker=style["marker"],
            label=domain if show_labels else None,
            linewidths=0.0,
        )

    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")


def plot_snapshot_progression(
    projected_df: pd.DataFrame,
    out_path: Path,
    *,
    snapshot_order: list[str],
    snapshot_titles: dict[str, str],
    dataset_name: str,
    source_domain: str,
    target_domain: str,
) -> None:
    domain_styles = _domain_styles(source_domain, target_domain)
    x_min = float(projected_df["pc1"].min())
    x_max = float(projected_df["pc1"].max())
    y_min = float(projected_df["pc2"].min())
    y_max = float(projected_df["pc2"].max())
    x_pad = max((x_max - x_min) * 0.08, 1e-3)
    y_pad = max((y_max - y_min) * 0.08, 1e-3)
    x_limits = (x_min - x_pad, x_max + x_pad)
    y_limits = (y_min - y_pad, y_max + y_pad)

    n_panels = len(snapshot_order)
    ncols = 3
    nrows = int(math.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 4.2 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for idx, snapshot_key in enumerate(snapshot_order):
        ax = axes[idx]
        view = projected_df[projected_df["snapshot"] == snapshot_key]
        _draw_snapshot_panel(
            ax,
            view,
            x_col="pc1",
            y_col="pc2",
            x_limits=x_limits,
            y_limits=y_limits,
            source_domain=source_domain,
            target_domain=target_domain,
            domain_styles=domain_styles,
            show_labels=(idx == 0),
        )

        ax.set_title(snapshot_titles.get(snapshot_key, snapshot_key))

    for idx in range(n_panels, len(axes)):
        axes[idx].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    fig.suptitle(
        f"{dataset_name}: Per-Snapshot t-SNE of Source-Only GNN Snapshots ({source_domain} -> {target_domain})",
        y=1.02,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _snapshot_metric_frame(
    shift_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
    *,
    snapshot_order: list[str],
) -> pd.DataFrame:
    merged = shift_df[
        [
            "snapshot",
            "snapshot_title",
            "mmd",
        ]
    ].merge(
        transfer_df[
            [
                "snapshot",
                "source_self_micro_f1",
                "transfer_micro_f1",
                "random_micro_f1",
                "oracle_micro_f1",
                "transfer_vs_random_micro_gain",
                "oracle_vs_transfer_micro_gap",
            ]
        ],
        on="snapshot",
        how="inner",
    )
    merged["snapshot"] = pd.Categorical(
        merged["snapshot"],
        categories=list(snapshot_order),
        ordered=True,
    )
    merged = merged.sort_values("snapshot", kind="stable").reset_index(drop=True)
    merged["snapshot_idx"] = range(len(merged))
    return merged


def _plot_snapshot_line(ax, x, y, *, color: str, label: str, marker: str = "o", linestyle: str = "-"):
    ax.plot(
        x,
        y,
        color=color,
        marker=marker,
        markersize=5.5,
        linewidth=2.0,
        linestyle=linestyle,
        label=label,
    )


def plot_snapshot_metric_overview(
    shift_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
    out_path: Path,
    *,
    snapshot_order: list[str],
    dataset_name: str,
    source_domain: str,
    target_domain: str,
) -> None:
    metrics_df = _snapshot_metric_frame(
        shift_df,
        transfer_df,
        snapshot_order=snapshot_order,
    )
    x = metrics_df["snapshot_idx"].to_numpy()
    x_labels = metrics_df["snapshot_title"].astype(str).tolist()

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14.0, 11.2),
        sharex=True,
        constrained_layout=True,
    )
    ax_mmd, ax_transfer, ax_domain = axes

    _plot_snapshot_line(
        ax_mmd,
        x,
        metrics_df["mmd"].to_numpy(),
        color="#C62828",
        label="source-target MMD",
    )
    ax_mmd.set_title("Source-Target Shift")
    ax_mmd.set_ylabel("MMD")
    ax_mmd.grid(alpha=0.25)
    ax_mmd.legend(frameon=False, fontsize=9)

    _plot_snapshot_line(
        ax_transfer,
        x,
        metrics_df["transfer_micro_f1"].to_numpy(),
        color="#1f77b4",
        label=f"{source_domain}->{target_domain} transfer",
    )
    _plot_snapshot_line(
        ax_transfer,
        x,
        metrics_df["random_micro_f1"].to_numpy(),
        color="#7f7f7f",
        label="random baseline",
        marker="d",
        linestyle="--",
    )
    _plot_snapshot_line(
        ax_transfer,
        x,
        metrics_df["transfer_vs_random_micro_gain"].to_numpy(),
        color="#ff7f0e",
        label="transfer - random",
        marker="s",
    )
    _plot_snapshot_line(
        ax_transfer,
        x,
        metrics_df["oracle_vs_transfer_micro_gap"].to_numpy(),
        color="#6a3d9a",
        label="oracle - transfer",
        marker="P",
    )
    ax_transfer.set_title("Transferability Scores")
    ax_transfer.set_ylabel("Micro-F1 / gap")
    ax_transfer.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--", alpha=0.6)
    ax_transfer.grid(alpha=0.25)
    ax_transfer.legend(frameon=False, fontsize=9)

    _plot_snapshot_line(
        ax_domain,
        x,
        metrics_df["source_self_micro_f1"].to_numpy(),
        color="#2E8B57",
        label=f"{source_domain} self micro-F1",
    )
    _plot_snapshot_line(
        ax_domain,
        x,
        metrics_df["transfer_micro_f1"].to_numpy(),
        color="#1f77b4",
        label=f"{target_domain} transfer micro-F1",
        marker="^",
    )
    _plot_snapshot_line(
        ax_domain,
        x,
        metrics_df["oracle_micro_f1"].to_numpy(),
        color="#C62828",
        label=f"{target_domain} oracle/self micro-F1",
        marker="s",
    )
    ax_domain.set_title("Domain Micro-F1")
    ax_domain.set_xlabel("Snapshot")
    ax_domain.set_ylabel("Micro-F1")
    ax_domain.grid(alpha=0.25)
    ax_domain.legend(frameon=False, fontsize=9)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(x_labels, rotation=32, ha="right")

    fig.suptitle(
        f"{dataset_name}: Snapshot Metrics ({source_domain} -> {target_domain})",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
