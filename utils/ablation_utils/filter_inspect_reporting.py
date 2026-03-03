"""Reporting utilities for filter-learning inspection experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from utils.ablation_utils.common import save_table


def save_coeff_history_csv(out_path: Path, rows: list[dict]) -> None:
    """Save per-epoch coefficient history rows."""
    if not rows:
        return
    save_table(out_path, rows)


def save_training_history_csv(out_path: Path, rows: list[dict]) -> None:
    """Save training losses/metrics tracked over epochs."""
    if not rows:
        return
    save_table(out_path, rows)


def _ordered_cols(rows: list[dict], prefix: str) -> list[str]:
    if not rows:
        return []
    cols = [k for k in rows[0].keys() if k.startswith(prefix)]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def plot_cheb_coeff_trajectories(
    out_path: Path,
    coeff_history,
    title: str,
) -> None:
    """Plot Chebyshev coefficients vs. epoch."""
    rows = coeff_history
    if not rows:
        return

    cols = _ordered_cols(rows, "cheb_")
    x = [int(r["epoch"]) for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for c in cols:
        ax.plot(x, [float(r[c]) for r in rows], marker="o", label=c)
    ax.set_title("Chebyshev coefficient trajectories")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Coefficient value")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_polynomial_coeff_trajectories(
    out_path: Path,
    poly_history,
    title: str,
) -> None:
    """Plot polynomial-basis coefficients vs. epoch."""
    rows = poly_history
    if not rows:
        return

    cols = _ordered_cols(rows, "poly_")
    x = [int(r["epoch"]) for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for c in cols:
        ax.plot(x, [float(r[c]) for r in rows], marker="o", label=c)
    ax.set_title("Polynomial-basis coefficient trajectories")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Coefficient value")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_training_history(out_path: Path, rows: list[dict], title: str) -> None:
    if not rows:
        return

    x = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(x, [float(r["loss"]) for r in rows], marker="o", label="loss")
    axes[0].plot(x, [float(r["mmd_loss"]) for r in rows], marker="o", label="mmd_loss")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, [float(r["edge_reg"]) for r in rows], marker="o", label="edge_reg")
    axes[1].plot(x, [float(r["temp_reg"]) for r in rows], marker="o", label="temp_reg")
    if "struct_loss" in rows[0]:
        axes[1].plot(x, [float(r["struct_loss"]) for r in rows], marker="o", label="struct_loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Regularization")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_scenario_artifacts(
    dataset: str,
    source: str,
    target: str,
    payload: dict,
    cfg: dict,
) -> None:
    """Write all outputs for one scenario (CSV/plots/checkpoints)."""
    out_dir = Path(cfg["out_dir"]) / dataset / f"{source}_{target}"
    out_dir.mkdir(parents=True, exist_ok=True)

    training_csv = out_dir / "training_history.csv"
    cheb_csv = out_dir / "cheb_coeff_history.csv"
    poly_csv = out_dir / "poly_coeff_history.csv"
    cheb_png = out_dir / "cheb_coeff_trajectories.png"
    poly_png = out_dir / "poly_coeff_trajectories.png"
    train_png = out_dir / "training_curves.png"
    final_pt = out_dir / "final_filter_state.pt"

    save_training_history_csv(training_csv, payload["training_rows"])
    save_coeff_history_csv(cheb_csv, payload["cheb_coeff_rows"])
    save_coeff_history_csv(poly_csv, payload["poly_coeff_rows"])

    title = f"{dataset}: {source}->{target}"
    plot_cheb_coeff_trajectories(cheb_png, payload["cheb_coeff_rows"], title)
    plot_polynomial_coeff_trajectories(poly_png, payload["poly_coeff_rows"], title)
    _plot_training_history(train_png, payload["training_rows"], title)

    torch.save(
        {
            "final_cheb_coeffs": payload["final_cheb_coeffs"],
            "final_poly_coeffs": payload["final_poly_coeffs"],
            "target_poly_coeffs": payload["target_poly_coeffs"],
            "final_edge_weight": payload["final_edge_weight"],
        },
        final_pt,
    )

    print(f"[saved] {training_csv}")
    print(f"[saved] {cheb_csv}")
    print(f"[saved] {poly_csv}")
    print(f"[saved] {cheb_png}")
    print(f"[saved] {poly_png}")
    print(f"[saved] {train_png}")
    print(f"[saved] {final_pt}")


def save_aggregate_reports(
    test_config: dict,
    cfg: dict,
    summary_rows: list[dict],
    failed_rows: list[dict],
) -> None:
    """Write cross-scenario aggregate reports."""
    out_root = Path(cfg["out_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    if summary_rows:
        summary_csv = out_root / "summary_final.csv"
        save_table(summary_csv, summary_rows)
        print(f"[saved] {summary_csv}")

        numeric_keys = [
            k for k, v in summary_rows[0].items() if isinstance(v, (float, int)) and k != "final_epoch"
        ]
        dataset_rows = []
        for dataset in test_config.keys():
            subset = [r for r in summary_rows if r["dataset"] == dataset]
            if not subset:
                continue
            row = {"dataset": dataset, "n_scenarios": len(subset)}
            for key in numeric_keys:
                vals = torch.tensor([float(r[key]) for r in subset], dtype=torch.float32)
                row[f"{key}_mean"] = float(vals.mean().item())
                row[f"{key}_std"] = float(vals.std(unbiased=False).item())
            dataset_rows.append(row)

        if dataset_rows:
            dataset_csv = out_root / "summary_by_dataset.csv"
            save_table(dataset_csv, dataset_rows)
            print(f"[saved] {dataset_csv}")

    if failed_rows:
        failed_csv = out_root / "failed_scenarios.csv"
        save_table(failed_csv, failed_rows, fieldnames=["dataset", "source", "target", "error"])
        print(f"[saved] {failed_csv}")
