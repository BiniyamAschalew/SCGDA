"""
Layer-wise shift experiment for blog/airport (no argparse).

For each dataset/source->target scenario:
1) Load source and target graphs.
2) Propagate real features for layers 0..max_layers.
3) Build Gaussian probe features from combined mean/std, then propagate the same way.
4) For each layer, train a MLP on propagated source features and evaluate on propagated target features.
5) Compute MMD and conditional MMD per layer for real and probe features.
6) Save CSV + plot.
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from data.build_dataset import build_dataset
from utils.config_utils import build_config
from utils.expt_utils import set_seed
from utils.filter_utils import conditional_mmd, make_gaussian_probe, mmd_rbf

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    }
)


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


class TransferMLP(nn.Module):
    """Small MLP for source-train / target-test transfer evaluation."""

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers not in (2, 3):
            raise ValueError(f"num_layers must be 2 or 3, got {num_layers}")

        layers = [nn.Linear(in_dim, hid_dim), nn.ReLU()]
        if num_layers == 3:
            layers += [nn.Dropout(float(dropout)), nn.Linear(hid_dim, hid_dim), nn.ReLU()]
        layers += [nn.Dropout(float(dropout)), nn.Linear(hid_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_source_train_mask(data) -> torch.Tensor:
    if hasattr(data, "train_mask") and data.train_mask is not None:
        mask = data.train_mask.view(-1).bool()
        if int(mask.sum().item()) > 0:
            return mask
    return torch.ones_like(data.y, dtype=torch.bool)


def macro_f1_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    classes = torch.unique(torch.cat([pred, labels], dim=0))
    f1_vals = []
    for cls in classes:
        tp = ((pred == cls) & (labels == cls)).sum().float()
        fp = ((pred == cls) & (labels != cls)).sum().float()
        fn = ((pred != cls) & (labels == cls)).sum().float()
        denom = 2.0 * tp + fp + fn
        f1 = torch.where(denom > 0, (2.0 * tp) / denom, torch.tensor(0.0, device=labels.device))
        f1_vals.append(f1)
    if not f1_vals:
        return 0.0
    return float(torch.stack(f1_vals).mean().item())


def train_eval_transfer_once(
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    source_y: torch.Tensor,
    target_y: torch.Tensor,
    source_train_mask: torch.Tensor,
    cfg: dict,
    seed: int,
) -> dict[str, float]:
    set_seed(int(seed))
    in_dim = int(source_feat.size(1))
    num_classes = int(torch.max(torch.cat([source_y, target_y], dim=0)).item()) + 1
    model = TransferMLP(
        in_dim=in_dim,
        hid_dim=int(cfg["mlp_hid_dim"]),
        out_dim=num_classes,
        num_layers=int(cfg["mlp_layers"]),
        dropout=float(cfg["mlp_dropout"]),
    ).to(source_feat.device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["mlp_lr"]),
        weight_decay=float(cfg["mlp_weight_decay"]),
    )

    for _ in range(int(cfg["mlp_epochs"])):
        model.train()
        opt.zero_grad()
        logits = model(source_feat)
        loss = F.cross_entropy(logits[source_train_mask], source_y[source_train_mask])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        src_logits = model(source_feat)
        tgt_logits = model(target_feat)
        src_acc = float((src_logits.argmax(dim=1) == source_y).float().mean().item())
        tgt_acc = float((tgt_logits.argmax(dim=1) == target_y).float().mean().item())
        src_macro = macro_f1_from_logits(src_logits, source_y)
        tgt_macro = macro_f1_from_logits(tgt_logits, target_y)

    return {
        "source_acc": src_acc,
        "target_acc": tgt_acc,
        "source_macro_f1": src_macro,
        "target_macro_f1": tgt_macro,
    }


def compute_table(source_data, target_data, cfg: dict):
    source_x = source_data.x.detach().float()
    target_x = target_data.x.detach().float()
    source_y = source_data.y.detach()
    target_y = target_data.y.detach()
    source_train_mask = get_source_train_mask(source_data).to(source_x.device)

    source_probe0, target_probe0 = make_gaussian_probe(source_x, target_x)

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

        transfer_runs = []
        n_repeats = max(1, int(cfg["mlp_repeats"]))
        for rep in range(n_repeats):
            rep_seed = int(cfg["seed"]) + 1000 * layer + 100 * rep
            metrics = train_eval_transfer_once(
                source_feat=src_real_layers[layer].detach(),
                target_feat=tgt_real_layers[layer].detach(),
                source_y=source_y,
                target_y=target_y,
                source_train_mask=source_train_mask,
                cfg=cfg,
                seed=rep_seed,
            )
            transfer_runs.append(metrics)

        src_acc_vals = torch.tensor([float(r["source_acc"]) for r in transfer_runs], dtype=torch.float32)
        tgt_acc_vals = torch.tensor([float(r["target_acc"]) for r in transfer_runs], dtype=torch.float32)
        src_f1_vals = torch.tensor([float(r["source_macro_f1"]) for r in transfer_runs], dtype=torch.float32)
        tgt_f1_vals = torch.tensor([float(r["target_macro_f1"]) for r in transfer_runs], dtype=torch.float32)

        rows.append(
            {
                "layer": layer,
                "real_mmd": as_float(real_mmd),
                "real_cmmd": as_float(real_cmmd),
                "probe_mmd": as_float(probe_mmd),
                "probe_cmmd": as_float(probe_cmmd),
                "mlp_n_runs": int(len(transfer_runs)),
                "mlp_source_acc_mean": float(src_acc_vals.mean().item()),
                "mlp_source_acc_std": float(src_acc_vals.std(unbiased=False).item()),
                "mlp_target_acc_mean": float(tgt_acc_vals.mean().item()),
                "mlp_target_acc_std": float(tgt_acc_vals.std(unbiased=False).item()),
                "mlp_source_macro_f1_mean": float(src_f1_vals.mean().item()),
                "mlp_source_macro_f1_std": float(src_f1_vals.std(unbiased=False).item()),
                "mlp_target_macro_f1_mean": float(tgt_f1_vals.mean().item()),
                "mlp_target_macro_f1_std": float(tgt_f1_vals.std(unbiased=False).item()),
            }
        )

    if rows:
        base_target_acc = float(rows[0]["mlp_target_acc_mean"])
        base_target_f1 = float(rows[0]["mlp_target_macro_f1_mean"])
        for row in rows:
            row["mlp_target_acc_drop_vs_l0"] = base_target_acc - float(row["mlp_target_acc_mean"])
            row["mlp_target_macro_f1_drop_vs_l0"] = base_target_f1 - float(row["mlp_target_macro_f1_mean"])
    return rows


def save_table(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "layer",
        "real_mmd",
        "real_cmmd",
        "probe_mmd",
        "probe_cmmd",
        "mlp_n_runs",
        "mlp_source_acc_mean",
        "mlp_source_acc_std",
        "mlp_target_acc_mean",
        "mlp_target_acc_std",
        "mlp_source_macro_f1_mean",
        "mlp_source_macro_f1_std",
        "mlp_target_macro_f1_mean",
        "mlp_target_macro_f1_std",
        "mlp_target_acc_drop_vs_l0",
        "mlp_target_macro_f1_drop_vs_l0",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_combined_on_axes(ax_mmd, rows: list[dict]) -> None:
    x = [int(r["layer"]) for r in rows]
    ax_perf = ax_mmd.twinx()

    real_mmd = [float(r["real_mmd"]) for r in rows]
    real_cmmd = [float(r["real_cmmd"]) for r in rows]
    tgt_acc = [float(r.get("mlp_target_acc_mean", float("nan"))) for r in rows]
    tgt_acc_std = [float(r.get("mlp_target_acc_std", 0.0)) for r in rows]

    ax_mmd.plot(x, real_mmd, marker="o", linewidth=2.0, label="Real MMD", color="tab:blue")
    ax_mmd.plot(x, real_cmmd, marker="o", linewidth=2.0, label="Real cMMD", color="tab:orange")
    ax_mmd.set_ylabel("MMD / cMMD")
    ax_mmd.grid(alpha=0.3)

    ax_perf.plot(x, tgt_acc, marker="o", linewidth=2.0, label="MLP target acc", color="tab:green")
    ax_perf.fill_between(
        x,
        [m - s for m, s in zip(tgt_acc, tgt_acc_std)],
        [m + s for m, s in zip(tgt_acc, tgt_acc_std)],
        alpha=0.18,
        color="tab:green",
        linewidth=0.0,
    )
    ax_perf.set_ylabel("Transfer metric")

    ax_mmd.set_xlabel("Layer (0 = initial)")

    h_mmd, l_mmd = ax_mmd.get_legend_handles_labels()
    h_perf, l_perf = ax_perf.get_legend_handles_labels()
    ax_mmd.legend(h_mmd + h_perf, l_mmd + l_perf, loc="best")


def save_plot(path: Path, rows: list[dict], title: str) -> None:
    _ = title
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.6))
    _plot_combined_on_axes(ax, rows)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_blog_airport_side_by_side_plot(path: Path, dataset_avg_rows: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharex=True)
    order = ["airport", "blog"]
    labels = {"airport": "Airport Average", "blog": "Blog Average"}

    for i, dataset in enumerate(order):
        ax = axes[i]
        rows = dataset_avg_rows.get(dataset, [])
        if not rows:
            ax.text(0.5, 0.5, f"No rows for {dataset}", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("Layer (0 = initial)")
            continue
        _plot_combined_on_axes(ax, rows)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_average_rows(all_scenario_rows: list[list[dict]], max_layers: int) -> list[dict]:
    def _mean(layer_rows: list[dict], key: str) -> float:
        vals = [float(r[key]) for r in layer_rows if key in r]
        if not vals:
            return float("nan")
        return float(sum(vals) / len(vals))

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
                "real_mmd": _mean(layer_rows, "real_mmd"),
                "real_cmmd": _mean(layer_rows, "real_cmmd"),
                "probe_mmd": _mean(layer_rows, "probe_mmd"),
                "probe_cmmd": _mean(layer_rows, "probe_cmmd"),
                "mlp_n_runs": _mean(layer_rows, "mlp_n_runs"),
                "mlp_source_acc_mean": _mean(layer_rows, "mlp_source_acc_mean"),
                "mlp_source_acc_std": _mean(layer_rows, "mlp_source_acc_std"),
                "mlp_target_acc_mean": _mean(layer_rows, "mlp_target_acc_mean"),
                "mlp_target_acc_std": _mean(layer_rows, "mlp_target_acc_std"),
                "mlp_source_macro_f1_mean": _mean(layer_rows, "mlp_source_macro_f1_mean"),
                "mlp_source_macro_f1_std": _mean(layer_rows, "mlp_source_macro_f1_std"),
                "mlp_target_macro_f1_mean": _mean(layer_rows, "mlp_target_macro_f1_mean"),
                "mlp_target_macro_f1_std": _mean(layer_rows, "mlp_target_macro_f1_std"),
                "mlp_target_acc_drop_vs_l0": _mean(layer_rows, "mlp_target_acc_drop_vs_l0"),
                "mlp_target_macro_f1_drop_vs_l0": _mean(layer_rows, "mlp_target_macro_f1_drop_vs_l0"),
                "n_scenarios": int(n),
            }
        )
    return avg_rows


def save_average_table(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "layer",
        "real_mmd",
        "real_cmmd",
        "probe_mmd",
        "probe_cmmd",
        "mlp_n_runs",
        "mlp_source_acc_mean",
        "mlp_source_acc_std",
        "mlp_target_acc_mean",
        "mlp_target_acc_std",
        "mlp_source_macro_f1_mean",
        "mlp_source_macro_f1_std",
        "mlp_target_macro_f1_mean",
        "mlp_target_macro_f1_std",
        "mlp_target_acc_drop_vs_l0",
        "mlp_target_macro_f1_drop_vs_l0",
        "n_scenarios",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
        "mlp_target_acc_mean": float(last["mlp_target_acc_mean"]),
        "mlp_target_macro_f1_mean": float(last["mlp_target_macro_f1_mean"]),
        "mlp_target_acc_drop_vs_l0": float(last["mlp_target_acc_drop_vs_l0"]),
        "mlp_target_macro_f1_drop_vs_l0": float(last["mlp_target_macro_f1_drop_vs_l0"]),
    }
    return summary, rows


if __name__ == "__main__":
    test_config = {
        "blog": [("Blog1", "Blog2"), ("Blog2", "Blog1")],
        "airport": [
            ("BRAZIL", "USA"), ("USA", "EUROPE"), ("EUROPE", "BRAZIL"),
            ("BRAZIL", "EUROPE"), ("USA", "BRAZIL"), ("EUROPE", "USA"),
            ],
    }

    expt_config = {
        "seed": 0,
        "device": resolve_device("cuda:7"),
        "max_layers": 5,
        "metric_sample_size": 2000,
        "kernel_mul": 2.0,
        "kernel_num": 5,
        "fix_sigma": None,
        "mlp_repeats": 3,
        "mlp_hid_dim": 64,
        "mlp_layers": 2,
        "mlp_dropout": 0.0,
        "mlp_lr": 1e-2,
        "mlp_weight_decay": 5e-4,
        "mlp_epochs": 100,
        "out_dir": "../__saved__/analysis/custom_shifts_blog_airport",
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
        summary_fieldnames = [
            "dataset",
            "source",
            "target",
            "final_layer",
            "real_mmd",
            "real_cmmd",
            "probe_mmd",
            "probe_cmmd",
            "mlp_target_acc_mean",
            "mlp_target_macro_f1_mean",
            "mlp_target_acc_drop_vs_l0",
            "mlp_target_macro_f1_drop_vs_l0",
        ]
        writer = csv.DictWriter(
            f,
            fieldnames=summary_fieldnames,
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

    dataset_average_rows = {}
    for dataset, dataset_rows in dataset_to_rows.items():
        dataset_avg_rows = build_average_rows(dataset_rows, expt_config["max_layers"])
        if not dataset_avg_rows:
            print(f"[warning] No successful scenarios for dataset={dataset}, skipping dataset average plot.")
            continue

        dataset_average_rows[dataset] = dataset_avg_rows
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

    side_by_side_png_path = Path(expt_config["out_dir"]) / "airport_blog_average_side_by_side.png"
    save_blog_airport_side_by_side_plot(side_by_side_png_path, dataset_average_rows)
    print(f"[saved] {side_by_side_png_path}")

    if failed_rows:
        failed_path = Path(expt_config["out_dir"]) / "failed_scenarios.csv"
        with failed_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "source", "target", "error"])
            writer.writeheader()
            writer.writerows(failed_rows)
        print(f"[saved] {failed_path}")
