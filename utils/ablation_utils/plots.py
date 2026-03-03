from pathlib import Path

import matplotlib.pyplot as plt


def save_compare_plot(path: Path, rows: list[dict], title: str) -> None:
    x = [int(r["layer"]) for r in rows]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    axes = axes.reshape(-1)

    axes[0].plot(x, [r["normal_real_mmd"] for r in rows], marker="o", label="normal")
    axes[0].plot(x, [r["aligned_real_mmd"] for r in rows], marker="o", label="aligned")
    axes[0].set_title("Real MMD")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, [r["normal_real_cmmd"] for r in rows], marker="o", label="normal")
    axes[1].plot(x, [r["aligned_real_cmmd"] for r in rows], marker="o", label="aligned")
    axes[1].set_title("Real conditional MMD")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(x, [r["normal_probe_mmd"] for r in rows], marker="o", label="normal")
    axes[2].plot(x, [r["aligned_probe_mmd"] for r in rows], marker="o", label="aligned")
    axes[2].set_title("Probe MMD")
    axes[2].set_xlabel("Layer (0 = initial)")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    axes[3].plot(x, [r["normal_probe_cmmd"] for r in rows], marker="o", label="normal")
    axes[3].plot(x, [r["aligned_probe_cmmd"] for r in rows], marker="o", label="aligned")
    axes[3].set_title("Probe conditional MMD")
    axes[3].set_xlabel("Layer (0 = initial)")
    axes[3].grid(alpha=0.3)
    axes[3].legend()

    axes[4].plot(x, [r["source_edge_l1_distance"] for r in rows], marker="o", label="source L1")
    axes[4].plot(x, [r["target_edge_l1_distance"] for r in rows], marker="o", label="target L1")
    axes[4].plot(x, [r["avg_edge_l1_distance"] for r in rows], marker="o", label="avg L1")
    axes[4].set_title("Edge Distance (L1)")
    axes[4].set_xlabel("Layer (0 = initial)")
    axes[4].grid(alpha=0.3)
    axes[4].legend()

    axes[5].plot(x, [r["source_edge_jaccard"] for r in rows], marker="o", label="source Jaccard")
    axes[5].plot(x, [r["target_edge_jaccard"] for r in rows], marker="o", label="target Jaccard")
    axes[5].plot(x, [r["avg_edge_jaccard"] for r in rows], marker="o", label="avg Jaccard")
    axes[5].set_title("Edge Overlap (Jaccard)")
    axes[5].set_xlabel("Layer (0 = initial)")
    axes[5].grid(alpha=0.3)
    axes[5].legend()

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_align_history_plot(path: Path, rows: list[dict], title: str) -> None:
    if not rows:
        return
    x = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(x, [r["probe_mmd"] for r in rows], marker="o", label="probe_mmd")
    axes[0].plot(x, [r["real_mmd"] for r in rows], marker="o", label="real_mmd")
    axes[0].plot(x, [r["real_cmmd"] for r in rows], marker="o", label="real_cmmd")
    axes[0].set_ylabel("Distance")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, [r["loss"] for r in rows], marker="o", label="align_loss")
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Align epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_transferability_plot(path: Path, summary_rows: list[dict], title: str) -> None:
    if not summary_rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for case in ("normal", "aligned"):
        rows = sorted([r for r in summary_rows if r["case"] == case], key=lambda x: int(x["layer"]))
        x = [int(r["layer"]) for r in rows]
        acc_mean = [float(r["target_acc_mean"]) for r in rows]
        acc_std = [float(r["target_acc_std"]) for r in rows]
        f1_mean = [float(r["target_macro_f1_mean"]) for r in rows]
        f1_std = [float(r["target_macro_f1_std"]) for r in rows]

        axes[0].plot(x, acc_mean, marker="o", label=f"{case} target acc")
        axes[0].fill_between(
            x,
            [m - s for m, s in zip(acc_mean, acc_std)],
            [m + s for m, s in zip(acc_mean, acc_std)],
            alpha=0.2,
        )
        axes[1].plot(x, f1_mean, marker="o", label=f"{case} target macro-f1")
        axes[1].fill_between(
            x,
            [m - s for m, s in zip(f1_mean, f1_std)],
            [m + s for m, s in zip(f1_mean, f1_std)],
            alpha=0.2,
        )

    axes[0].set_title("Transfer Target Accuracy")
    axes[0].set_xlabel("k (A^k X)")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Transfer Target Macro-F1")
    axes[1].set_xlabel("k (A^k X)")
    axes[1].set_ylabel("Macro-F1")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_average_target_transfer_plot(path: Path, rows: list[dict], title: str) -> None:
    if not rows:
        return

    x = [int(r["layer"]) for r in rows]
    normal_mean = [float(r["normal_target_acc_mean"]) for r in rows]
    normal_std = [float(r["normal_target_acc_std"]) for r in rows]
    aligned_mean = [float(r["aligned_target_acc_mean"]) for r in rows]
    aligned_std = [float(r["aligned_target_acc_std"]) for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.8))
    ax.plot(x, normal_mean, marker="o", label="normal avg target acc")
    ax.fill_between(
        x,
        [m - s for m, s in zip(normal_mean, normal_std)],
        [m + s for m, s in zip(normal_mean, normal_std)],
        alpha=0.2,
    )
    ax.plot(x, aligned_mean, marker="o", label="aligned avg target acc")
    ax.fill_between(
        x,
        [m - s for m, s in zip(aligned_mean, aligned_std)],
        [m + s for m, s in zip(aligned_mean, aligned_std)],
        alpha=0.2,
    )
    ax.set_title("Average Target Transfer Accuracy")
    ax.set_xlabel("Layer (k)")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200)
    plt.close(fig)
