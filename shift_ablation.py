"""
Simple layer-wise shift experiment (no argparse).

For each dataset/source->target scenario:
1) Load source and target graphs.
2) Propagate real features for layers 0..max_layers.
3) Build Gaussian probe features from combined mean/std, then propagate the same way.
4) Compute MMD and conditional MMD per layer for real and probe features.
5) Save CSV + plot.
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from data.build_dataset import build_dataset
from utils.config_utils import build_config
from utils.expt_utils import set_seed


class Propagation(MessagePassing):
    """GCN-style normalized one-hop propagation."""

    def __init__(self, aggr: str = "add"):
        super().__init__(aggr=aggr)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.0)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j


def resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def load_pair(dataset: str, source: str, target: str, device: str, seed: int):
    config = build_config(
        {"data": dataset, "expt": "default", "model": "test"},
        {
            "expt": {
                "source": source,
                "target": target,
                "device": device,
                "seed": seed,
                "verbose": 0,
                "wandb_enabled": False,
            }
        },
    )
    source_dataset, target_dataset = build_dataset(config)
    source_data = source_dataset[0].to(device)
    target_data = target_dataset[0].to(device)

    if source_data.edge_index is not None:
        source_data.edge_index = source_data.edge_index.contiguous()
    if target_data.edge_index is not None:
        target_data.edge_index = target_data.edge_index.contiguous()

    if source_data.x is None or target_data.x is None:
        raise ValueError("Missing node features.")
    if source_data.edge_index is None or target_data.edge_index is None:
        raise ValueError("Missing edge_index.")
    if source_data.y is None or target_data.y is None:
        raise ValueError("Missing node labels.")
    if int(source_data.x.size(1)) != int(target_data.x.size(1)):
        raise ValueError(
            f"Feature dim mismatch: source={source_data.x.size(1)}, target={target_data.x.size(1)}"
        )
    source_data.y = source_data.y.view(-1).long()
    target_data.y = target_data.y.view(-1).long()
    return source_data, target_data


def pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True).transpose(0, 1)
    return torch.clamp(x_norm + y_norm - 2.0 * (x @ y.transpose(0, 1)), min=0.0)


def median_bandwidth(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    all_feat = torch.cat([x, y], dim=0)
    dists = pairwise_sq_dist(all_feat, all_feat).detach()
    n = dists.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=dists.device)
    vals = dists[mask]
    if vals.numel() == 0:
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)
    return vals.median().clamp_min(eps)


def mmd_rbf(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float,
    kernel_num: int,
    fix_sigma: float | None,
    eps: float = 1e-6,
) -> torch.Tensor:
    bandwidth = (
        torch.as_tensor(fix_sigma, device=source.device, dtype=source.dtype)
        if fix_sigma is not None
        else median_bandwidth(source, target, eps=eps)
    )
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))

    xx = pairwise_sq_dist(source, source)
    yy = pairwise_sq_dist(target, target)
    xy = pairwise_sq_dist(source, target)

    k_xx = 0.0
    k_yy = 0.0
    k_xy = 0.0
    for i in range(kernel_num):
        bw = (bandwidth * (kernel_mul**i)).clamp_min(eps)
        k_xx = k_xx + torch.exp(-xx / bw)
        k_yy = k_yy + torch.exp(-yy / bw)
        k_xy = k_xy + torch.exp(-xy / bw)
    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


def conditional_mmd(
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    source_y: torch.Tensor,
    target_y: torch.Tensor,
    kernel_mul: float,
    kernel_num: int,
    fix_sigma: float | None,
) -> torch.Tensor:
    classes = torch.unique(torch.cat([source_y, target_y], dim=0))
    vals = []
    for cls in classes:
        src_mask = source_y == cls
        tgt_mask = target_y == cls
        if int(src_mask.sum().item()) == 0 or int(tgt_mask.sum().item()) == 0:
            continue
        vals.append(
            mmd_rbf(
                source_feat[src_mask],
                target_feat[tgt_mask],
                kernel_mul=kernel_mul,
                kernel_num=kernel_num,
                fix_sigma=fix_sigma,
            )
        )
    if not vals:
        return torch.tensor(0.0, device=source_feat.device, dtype=source_feat.dtype)
    return torch.stack(vals).mean()


def propagate_layers(x: torch.Tensor, edge_index: torch.Tensor, max_layers: int, prop: Propagation):
    layers = [x]
    cur = x
    for _ in range(max_layers):
        cur = prop(cur, edge_index)
        layers.append(cur)
    return layers


def sample_idx(n: int, max_samples: int, device: torch.device) -> torch.Tensor:
    if max_samples <= 0 or n <= max_samples:
        return torch.arange(n, device=device)
    return torch.randperm(n, device=device)[:max_samples]


def as_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def compute_table(source_data, target_data, cfg: dict):
    source_x = source_data.x.detach().float()
    target_x = target_data.x.detach().float()
    source_y = source_data.y.detach()
    target_y = target_data.y.detach()

    all_real = torch.cat([source_x, target_x], dim=0)
    probe_mean = all_real.mean(dim=0, keepdim=True)
    probe_std = all_real.std(dim=0, keepdim=True).clamp_min(1e-6)
    source_probe0 = torch.randn_like(source_x) * probe_std + probe_mean
    target_probe0 = torch.randn_like(target_x) * probe_std + probe_mean

    prop = Propagation().to(source_x.device)
    src_real_layers = propagate_layers(source_x, source_data.edge_index, cfg["max_layers"], prop)
    tgt_real_layers = propagate_layers(target_x, target_data.edge_index, cfg["max_layers"], prop)
    src_probe_layers = propagate_layers(source_probe0, source_data.edge_index, cfg["max_layers"], prop)
    tgt_probe_layers = propagate_layers(target_probe0, target_data.edge_index, cfg["max_layers"], prop)

    idx_s = sample_idx(source_x.size(0), cfg["metric_sample_size"], source_x.device)
    idx_t = sample_idx(target_x.size(0), cfg["metric_sample_size"], target_x.device)

    rows = []
    for layer in range(cfg["max_layers"] + 1):
        xs = src_real_layers[layer][idx_s]
        xt = tgt_real_layers[layer][idx_t]
        ys = source_y[idx_s]
        yt = target_y[idx_t]

        real_mmd = mmd_rbf(xs, xt, cfg["kernel_mul"], cfg["kernel_num"], cfg["fix_sigma"])
        real_cmmd = conditional_mmd(xs, xt, ys, yt, cfg["kernel_mul"], cfg["kernel_num"], cfg["fix_sigma"])

        ps = src_probe_layers[layer][idx_s]
        pt = tgt_probe_layers[layer][idx_t]
        probe_mmd = mmd_rbf(ps, pt, cfg["kernel_mul"], cfg["kernel_num"], cfg["fix_sigma"])
        probe_cmmd = conditional_mmd(ps, pt, ys, yt, cfg["kernel_mul"], cfg["kernel_num"], cfg["fix_sigma"])

        rows.append(
            {
                "layer": layer,
                "real_mmd": as_float(real_mmd),
                "real_cmmd": as_float(real_cmmd),
                "probe_mmd": as_float(probe_mmd),
                "probe_cmmd": as_float(probe_cmmd),
            }
        )
    return rows


def save_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "real_mmd", "real_cmmd", "probe_mmd", "probe_cmmd"])
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path: Path, rows: list[dict], title: str) -> None:
    x = [int(r["layer"]) for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(x, [r["real_mmd"] for r in rows], marker="o", label="Real MMD")
    axes[0].plot(x, [r["probe_mmd"] for r in rows], marker="o", label="Probe MMD")
    axes[0].set_ylabel("MMD")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, [r["real_cmmd"] for r in rows], marker="o", label="Real cMMD")
    axes[1].plot(x, [r["probe_cmmd"] for r in rows], marker="o", label="Probe cMMD")
    axes[1].set_ylabel("Conditional MMD")
    axes[1].set_xlabel("Layer (0 = initial)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_average_rows(all_scenario_rows: list[list[dict]], max_layers: int) -> list[dict]:
    avg_rows = []
    if not all_scenario_rows:
        return avg_rows

    for layer in range(max_layers + 1):
        layer_rows = [rows[layer] for rows in all_scenario_rows if len(rows) > layer]
        if not layer_rows:
            continue

        n = float(len(layer_rows))
        avg_rows.append(
            {
                "layer": layer,
                "real_mmd": sum(float(r["real_mmd"]) for r in layer_rows) / n,
                "real_cmmd": sum(float(r["real_cmmd"]) for r in layer_rows) / n,
                "probe_mmd": sum(float(r["probe_mmd"]) for r in layer_rows) / n,
                "probe_cmmd": sum(float(r["probe_cmmd"]) for r in layer_rows) / n,
                "n_scenarios": int(n),
            }
        )
    return avg_rows


def save_average_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "real_mmd", "real_cmmd", "probe_mmd", "probe_cmmd", "n_scenarios"],
        )
        writer.writeheader()
        writer.writerows(rows)


def experiment(dataset: str, source: str, target: str, cfg: dict) -> tuple[dict, list[dict]]:
    set_seed(cfg["seed"])
    source_data, target_data = load_pair(dataset, source, target, cfg["device"], cfg["seed"])
    rows = compute_table(source_data, target_data, cfg)

    out_dir = Path(cfg["out_dir"]) / dataset / f"{source}_{target}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "layer_metrics.csv"
    png_path = out_dir / "layer_metrics.png"
    save_table(csv_path, rows)
    save_plot(png_path, rows, f"{dataset}: {source}->{target}")

    print(f"[saved] {csv_path}")
    print(f"[saved] {png_path}")

    last = rows[-1]
    summary = {
        "dataset": dataset,
        "source": source,
        "target": target,
        "final_layer": int(last["layer"]),
        "real_mmd": float(last["real_mmd"]),
        "real_cmmd": float(last["real_cmmd"]),
        "probe_mmd": float(last["probe_mmd"]),
        "probe_cmmd": float(last["probe_cmmd"]),
    }
    return summary, rows


if __name__ == "__main__":
    test_config = {
        "citation": [
            ("ACMv9", "Citationv1"), ("Citationv1", "DBLPv7"), ("DBLPv7", "ACMv9"),
            ("ACMv9", "DBLPv7"), ("Citationv1", "ACMv9"), ("DBLPv7", "Citationv1"),
            ],
        "blog": [("Blog1", "Blog2"), ("Blog2", "Blog1")],
        "airport": [
            ("BRAZIL", "USA"), ("USA", "EUROPE"), ("EUROPE", "BRAZIL"),
            ("BRAZIL", "EUROPE"), ("USA", "BRAZIL"), ("EUROPE", "USA"),
            ],
        "twitch": [("DE", "EN"), ("EN", "DE")],
    }

    expt_config = {
        "seed": 0,
        "device": resolve_device("cuda:7"),
        "max_layers": 5,
        "metric_sample_size": 2000,
        "kernel_mul": 2.0,
        "kernel_num": 5,
        "fix_sigma": None,
        "out_dir": "__saved__/analysis/cusom_shifts",
    }

    summary_rows = []
    all_scenario_rows = []
    dataset_to_rows = {dataset: [] for dataset in test_config.keys()}
    failed_rows = []
    for dataset, scenarios in test_config.items():
        for source, target in scenarios:
            try:
                summary, rows = experiment(dataset, source, target, expt_config)
                summary_rows.append(summary)
                all_scenario_rows.append(rows)
                dataset_to_rows[dataset].append(rows)
            except Exception as exc:
                failed_rows.append(
                    {
                        "dataset": dataset,
                        "source": source,
                        "target": target,
                        "error": str(exc),
                    }
                )
                print(f"[failed] {dataset} {source}->{target}: {exc}")

    summary_path = Path(expt_config["out_dir"]) / "summary_final_layer.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "source", "target", "final_layer", "real_mmd", "real_cmmd", "probe_mmd", "probe_cmmd"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[saved] {summary_path}")

    average_rows = build_average_rows(all_scenario_rows, expt_config["max_layers"])
    if average_rows:
        average_csv_path = Path(expt_config["out_dir"]) / "average_layer_metrics.csv"
        average_png_path = Path(expt_config["out_dir"]) / "average_layer_metrics.png"
        save_average_table(average_csv_path, average_rows)
        save_plot(average_png_path, average_rows, "Average shift evolution across all successful scenarios")
        print(f"[saved] {average_csv_path}")
        print(f"[saved] {average_png_path}")
    else:
        print("[warning] No successful scenarios, skipping average plot.")

    for dataset, dataset_rows in dataset_to_rows.items():
        dataset_avg_rows = build_average_rows(dataset_rows, expt_config["max_layers"])
        if not dataset_avg_rows:
            print(f"[warning] No successful scenarios for dataset={dataset}, skipping dataset average plot.")
            continue

        dataset_dir = Path(expt_config["out_dir"]) / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_avg_csv_path = dataset_dir / "average_layer_metrics.csv"
        dataset_avg_png_path = dataset_dir / "average_layer_metrics.png"

        save_average_table(dataset_avg_csv_path, dataset_avg_rows)
        save_plot(
            dataset_avg_png_path,
            dataset_avg_rows,
            f"Average shift evolution for dataset={dataset}",
        )
        print(f"[saved] {dataset_avg_csv_path}")
        print(f"[saved] {dataset_avg_png_path}")

    if failed_rows:
        failed_path = Path(expt_config["out_dir"]) / "failed_scenarios.csv"
        with failed_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "source", "target", "error"])
            writer.writeheader()
            writer.writerows(failed_rows)
        print(f"[saved] {failed_path}")
