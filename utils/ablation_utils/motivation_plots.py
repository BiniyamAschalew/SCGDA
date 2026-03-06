from pathlib import Path

import matplotlib.pyplot as plt


def _extract_with_std(row_map: dict[int, dict], layers: list[int], mean_key: str) -> tuple[list[float], list[float]]:
    vals = [float(row_map[k][mean_key]) for k in layers]
    std_key = mean_key[:-5] + "_std" if mean_key.endswith("_mean") else None
    stds = [float(row_map[k].get(std_key, 0.0)) for k in layers] if std_key else [0.0 for _ in layers]
    return vals, stds


def plot_average_quad(
    path: Path,
    mmd_rows: list[dict],
    transfer_rows: list[dict],
    distortion_rows: list[dict],
    title: str,
) -> bool:
    if not mmd_rows or not transfer_rows or not distortion_rows:
        return False
    mmd_map = {int(r["layer"]): r for r in mmd_rows}
    trf_map = {int(r["layer"]): r for r in transfer_rows}
    dis_map = {int(r["layer"]): r for r in distortion_rows}
    layers = sorted(set(mmd_map.keys()) & set(trf_map.keys()) & set(dis_map.keys()))
    if not layers:
        return False

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    for key, color, label in (
        ("mmd2_total_mean", "tab:blue", "mmd2_total"),
        ("mmd2_norm_mean", "tab:orange", "mmd2_norm (cosine)"),
        ("mmd2_angle_mean", "tab:green", "mmd2_angle"),
    ):
        y, s = _extract_with_std(mmd_map, layers, key)
        ax[0, 0].plot(layers, y, marker="o", color=color, label=label)
        ax[0, 0].fill_between(layers, [a - b for a, b in zip(y, s)], [a + b for a, b in zip(y, s)], alpha=0.2, color=color)
    ax[0, 0].set(title="Average MMD by layer", xlabel="Propagation step", ylabel="MMD")
    ax[0, 0].grid(alpha=0.3)
    ax[0, 0].legend(loc="best")

    for key, color, label in (
        ("source_micro_f1_mean", "tab:blue", "source micro-F1"),
        ("target_micro_f1_mean", "tab:green", "target micro-F1"),
        ("oracle_target_micro_f1_mean", "tab:orange", "oracle target micro-F1"),
    ):
        y, s = _extract_with_std(trf_map, layers, key)
        ax[0, 1].plot(layers, y, marker="o", color=color, label=label)
        ax[0, 1].fill_between(layers, [a - b for a, b in zip(y, s)], [a + b for a, b in zip(y, s)], alpha=0.2, color=color)
    ax[0, 1].set(title="Average transferability by layer", xlabel="Propagation step", ylabel="Micro-F1")
    ax[0, 1].grid(alpha=0.3)
    ax[0, 1].legend(loc="best")

    y_majority, _ = _extract_with_std(trf_map, layers, "majority_source_label_target_f1_mean")
    ax[0, 1].plot(layers, y_majority, marker="d", color="tab:purple", linestyle="--", label="majority source label target baseline")

    y_d, s_d = _extract_with_std(trf_map, layers, "delta_micro_f1_mean")
    y_o, s_o = _extract_with_std(trf_map, layers, "delta_oracle_micro_f1_mean")
    y_a, s_a = _extract_with_std(mmd_map, layers, "mmd2_angle_mean")
    ax[1, 0].plot(layers, y_d, marker="o", color="tab:red", label="delta micro-F1")
    ax[1, 0].fill_between(layers, [a - b for a, b in zip(y_d, s_d)], [a + b for a, b in zip(y_d, s_d)], alpha=0.2, color="tab:red")
    ax[1, 0].plot(layers, y_o, marker="^", color="tab:orange", label="delta oracle")
    ax[1, 0].fill_between(layers, [a - b for a, b in zip(y_o, s_o)], [a + b for a, b in zip(y_o, s_o)], alpha=0.2, color="tab:orange")
    twin = ax[1, 0].twinx()
    twin.plot(layers, y_a, marker="s", color="tab:purple", label="mmd2_angle")
    twin.fill_between(layers, [a - b for a, b in zip(y_a, s_a)], [a + b for a, b in zip(y_a, s_a)], alpha=0.2, color="tab:purple")
    ax[1, 0].set(title="Delta gaps & mmd2_angle", xlabel="Propagation step", ylabel="gap vs target")
    twin.set_ylabel("mmd2_angle")
    ax[1, 0].grid(alpha=0.3)
    lines1, labels1 = ax[1, 0].get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax[1, 0].legend(lines1 + lines2, labels1 + labels2, loc="best")

    y_s, s_s = _extract_with_std(dis_map, layers, "source_distortion_mean")
    y_t, s_t = _extract_with_std(dis_map, layers, "target_distortion_mean")
    ax[1, 1].plot(layers, y_s, marker="o", color="tab:brown", label="source distortion")
    ax[1, 1].fill_between(layers, [a - b for a, b in zip(y_s, s_s)], [a + b for a, b in zip(y_s, s_s)], alpha=0.2, color="tab:brown")
    ax[1, 1].plot(layers, y_t, marker="s", color="tab:gray", label="target distortion")
    ax[1, 1].fill_between(layers, [a - b for a, b in zip(y_t, s_t)], [a + b for a, b in zip(y_t, s_t)], alpha=0.2, color="tab:gray")
    ax[1, 1].set(title="Distortion by layer", xlabel="Propagation step", ylabel="changed-label ratio")
    ax[1, 1].grid(alpha=0.3)
    ax[1, 1].legend(loc="best")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True
