from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PALETTE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
]


def _domain_colors(domains: list[str]):
    return {domain: PALETTE[idx % len(PALETTE)] for idx, domain in enumerate(domains)}


def _plot_mean_with_band(ax, x, mean_values, std_values, *, color: str, label: str, linewidth: float = 2.0, linestyle: str = "-"):
    ax.plot(
        x,
        mean_values,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )
    if std_values is not None:
        lower = mean_values - std_values
        upper = mean_values + std_values
        ax.fill_between(x, lower, upper, color=color, alpha=0.16, linewidth=0)


def plot_training_overview(
    history_df,
    *,
    out_dir: str | Path,
    source_domain: str,
    eval_domains: list[str],
    std_df=None,
):
    out_dir = Path(out_dir)
    colors = _domain_colors(eval_domains)
    epochs = history_df["epoch"].to_numpy()
    target_domains = [domain for domain in eval_domains if domain != source_domain]

    fig, axes = plt.subplots(3, 2, figsize=(13.8, 11.8), sharex=True)
    ax_perf, ax_mmd, ax_energy, ax_energy_delta, ax_consistency, ax_denoising = axes.reshape(-1)

    for domain in eval_domains:
        _plot_mean_with_band(
            ax_perf,
            epochs,
            history_df[f"micro_f1_{domain}"].to_numpy(),
            None if std_df is None else std_df[f"micro_f1_{domain}"].to_numpy(),
            color=colors[domain],
            label=domain,
        )
    ax_perf.set_title("Performance")
    ax_perf.set_ylabel("Micro-F1")
    ax_perf.grid(alpha=0.25)
    ax_perf.legend(frameon=False, fontsize=9)

    for domain in target_domains:
        _plot_mean_with_band(
            ax_mmd,
            epochs,
            history_df[f"mmd_{source_domain}_to_{domain}"].to_numpy(),
            None if std_df is None else std_df[f"mmd_{source_domain}_to_{domain}"].to_numpy(),
            color=colors[domain],
            label=f"{source_domain}->{domain}",
        )
    _plot_mean_with_band(
        ax_mmd,
        epochs,
        history_df[f"mmd_{source_domain}_to_{source_domain}"].to_numpy(),
        None if std_df is None else std_df[f"mmd_{source_domain}_to_{source_domain}"].to_numpy(),
        color=colors[source_domain],
        label=f"{source_domain}->{source_domain} (half split)",
        linestyle="--",
    )
    ax_mmd.set_title("MMD To Source")
    ax_mmd.set_ylabel("MMD")
    ax_mmd.grid(alpha=0.25)
    if target_domains or f"mmd_{source_domain}_to_{source_domain}" in history_df.columns:
        ax_mmd.legend(frameon=False, fontsize=9)

    for domain in eval_domains:
        _plot_mean_with_band(
            ax_energy,
            epochs,
            history_df[f"dirichlet_energy_{domain}"].to_numpy(),
            None if std_df is None else std_df[f"dirichlet_energy_{domain}"].to_numpy(),
            color=colors[domain],
            label=domain,
        )
    ax_energy.set_title("Dirichlet Energy")
    ax_energy.set_xlabel("Epoch")
    ax_energy.set_ylabel("Energy / node")
    ax_energy.grid(alpha=0.25)

    for domain in eval_domains:
        _plot_mean_with_band(
            ax_energy_delta,
            epochs,
            history_df[f"dirichlet_energy_delta_{domain}"].to_numpy(),
            None if std_df is None else std_df[f"dirichlet_energy_delta_{domain}"].to_numpy(),
            color=colors[domain],
            label=domain,
        )
    ax_energy_delta.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_energy_delta.set_title("Dirichlet Energy Delta")
    ax_energy_delta.set_xlabel("Epoch")
    ax_energy_delta.set_ylabel("Delta From Prev Epoch")
    ax_energy_delta.grid(alpha=0.25)

    for domain in eval_domains:
        _plot_mean_with_band(
            ax_consistency,
            epochs,
            history_df[f"consistency_norm_{domain}"].to_numpy(),
            None if std_df is None else std_df[f"consistency_norm_{domain}"].to_numpy(),
            color=colors[domain],
            label=domain,
        )
    ax_consistency.set_title("Consistency")
    ax_consistency.set_xlabel("Epoch")
    ax_consistency.set_ylabel("||AH - H|| / N")
    ax_consistency.grid(alpha=0.25)

    for domain in eval_domains:
        _plot_mean_with_band(
            ax_denoising,
            epochs,
            history_df[f"graph_signal_denoising_{domain}"].to_numpy(),
            None if std_df is None else std_df[f"graph_signal_denoising_{domain}"].to_numpy(),
            color=colors[domain],
            label=domain,
        )
    ax_denoising.set_title("Graph Signal Denoising")
    ax_denoising.set_xlabel("Epoch")
    ax_denoising.set_ylabel("Energy + Consistency")
    ax_denoising.grid(alpha=0.25)

    fig.suptitle(
        f"Training Dynamics: source={source_domain}, targets={', '.join(target_domains) if target_domains else 'none'}",
        fontsize=13,
    )
    fig.tight_layout()
    out_path = out_dir / "training_overview.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_target_quality_vs_performance(
    history_df,
    *,
    out_dir: str | Path,
    source_domain: str,
    target_domains: list[str],
    std_df=None,
):
    out_dir = Path(out_dir)
    if not target_domains:
        return None

    epochs = history_df["epoch"].to_numpy()
    fig, axes = plt.subplots(
        len(target_domains),
        5,
        figsize=(24.0, 3.9 * len(target_domains)),
        sharex=True,
        squeeze=False,
    )

    metric_specs = [
        ("mmd", "MMD", lambda domain: f"mmd_{source_domain}_to_{domain}", "#d62728"),
        ("energy", "Dirichlet Energy / Node", lambda domain: f"dirichlet_energy_{domain}", "#2ca02c"),
        ("energy_delta", "Dirichlet Energy Delta", lambda domain: f"dirichlet_energy_delta_{domain}", "#9467bd"),
        ("consistency", "Consistency", lambda domain: f"consistency_norm_{domain}", "#ff7f0e"),
        ("denoising", "Graph Signal Denoising", lambda domain: f"graph_signal_denoising_{domain}", "#8c564b"),
    ]

    for row_idx, target_domain in enumerate(target_domains):
        perf_values = history_df[f"micro_f1_{target_domain}"].to_numpy()
        for col_idx, (_, title, column_fn, metric_color) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            ax2 = ax.twinx()

            _plot_mean_with_band(
                ax,
                epochs,
                perf_values,
                None if std_df is None else std_df[f"micro_f1_{target_domain}"].to_numpy(),
                color="#1f77b4",
                label=f"{target_domain} micro-F1",
            )
            _plot_mean_with_band(
                ax2,
                epochs,
                history_df[column_fn(target_domain)].to_numpy(),
                None if std_df is None else std_df[column_fn(target_domain)].to_numpy(),
                color=metric_color,
                label=title,
            )

            ax.set_title(f"{source_domain}->{target_domain}: {title}")
            ax.set_ylabel("Micro-F1", color="#1f77b4")
            ax2.set_ylabel(title, color=metric_color)
            ax.tick_params(axis="y", colors="#1f77b4")
            ax2.tick_params(axis="y", colors=metric_color)
            if "Delta" in title:
                ax2.axhline(0.0, color=metric_color, linewidth=1.0, linestyle="--", alpha=0.7)
            ax.grid(alpha=0.25)
            if row_idx == len(target_domains) - 1:
                ax.set_xlabel("Epoch")

            perf_line = ax.lines[-1]
            metric_line = ax2.lines[-1]
            ax.legend([perf_line, metric_line], [perf_line.get_label(), metric_line.get_label()], frameon=False, fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "target_quality_vs_performance.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path
