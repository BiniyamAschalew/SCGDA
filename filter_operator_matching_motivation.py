"""Motivation experiment: fixed vs pre-aligned trainable shared monomial filter.

Case i:
- Shared monomial hop filter is frozen to [0, 0, 1] (hop-space), i.e. A^3-like propagation
  under `FilterGCNConv` mono hop interpretation.

Case ii:
- Shared monomial hop filter is trainable.
- Before supervised source training, do filter-only pre-alignment for N epochs by
  minimizing MMD between source/target probe pushforwards under the shared filter.

Both cases:
- Train on source labels only.
- Evaluate on target labels (transferability, no DA objective during supervised stage).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from models.__filters.mono import MonoProp
from models.__layers.filter_gcn_conv import FilterGCNConv
from utils.ablation_utils.alignment import build_struct_positive_edges, edge_bpr_structure_loss
from utils.ablation_utils.common import load_pair, save_table
from utils.expt_utils import set_seed
from utils.filter_utils import make_gaussian_probe, mmd_rbf, tensor_to_float_list

CFG = {
    "dataset": ["airport", "citation", "blog"],
    # Used only when all_pairs=False:
    "source": ["USA", "EUROPE"],
    "target": ["EUROPE", "USA"],
    # If True, source/target are ignored and all ordered domain pairs are evaluated.
    "all_pairs": True,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "seed": 7,
    # Shared monomial hop basis size (hop-space: A, A^2, ..., A^K)
    "k_hops": 3,
    # Filter coefficient parameterization:
    # - "softmax": unconstrained simplex
    # - "monotone_cumsum": c1<=c2<=... by construction (recommended with monotonicity)
    "hop_parametrization": "monotone_cumsum",
    # Case-i fixed shared hop weights
    "fixed_hop_coeffs": [0.0, 0.0, 1.0],
    # Filter pre-alignment (case-ii only)
    "align_epochs": 20,
    "align_lr_mult": 1.25,  # alignment lr = lr * align_lr_mult (slightly higher)
    "align_optimizer": "sgd",  # sgd is less prone to near-identical Adam logit trajectories
    "align_momentum": 0.0,
    "freeze_after_align": True,  # freeze aligned operator during semantic training
    "align_sample_size": 4096,
    "kernel_mul": 2.0,
    "kernel_num": 5,
    "fix_sigma": None,
    # Structural filter learning (BPR on graph positives; supports adjacency/ppmi)
    "struct_loss_mode": "ppmi",  # "adjacency" or "ppmi"
    "struct_bpr_weight": 0.1,
    "struct_bpr_samples": 2048,
    "struct_bpr_margin": 0.0,
    "ppmi_path_len": 5,
    "ppmi_pos_threshold": 0.0,
    # Encourage incremental hop coefficients: c1 <= c2 <= ... <= cK
    "monotonicity_weight": 1.0,
    "monotonicity_projection": False,  # only used in softmax mode
    # Supervised training
    "epochs": 200,
    "lr": 1e-2,
    "weight_decay": 5e-4,
    "dropout": 0.2,
    "hid_dim": 64,
    # L1 on trainable filter logits (raw_hops).
    "filter_l1_weight": 0.0,
    # Outputs
    "out_root": "__saved__/analysis/filter_operator_matching_motivation",
    "tag": "",
}

DATASET_DOMAINS = {
    "airport": ["BRAZIL", "EUROPE", "USA"],
    "citation": ["ACMv9", "Citationv1", "DBLPv7"],
    "blog": ["Blog1", "Blog2"],
}


class SharedFilterGCN(nn.Module):
    """Small node classifier with one runtime monomial-filtered GCN layer."""

    def __init__(self, in_dim: int, hid_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.conv = FilterGCNConv(
            in_channels=in_dim,
            out_channels=hid_dim,
            filter_type="mono",
            prepend_zero_to_filter=True,  # hop coeffs: [c1,c2,...] => c1*A + c2*A^2 + ...
        )
        if not bool(getattr(self.conv, "prepend_zero_to_filter", False)):
            raise RuntimeError(
                "SharedFilterGCN requires hop semantics (coeff[0] -> A X). "
                "Expected prepend_zero_to_filter=True."
            )
        self.cls = nn.Linear(hid_dim, num_classes)
        self.dropout = float(dropout)

    def forward(self, data, hop_coeffs: torch.Tensor) -> torch.Tensor:
        x = self.conv(data.x, data.edge_index, filter_param=hop_coeffs)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.cls(x)


def _as_mask(data, names: tuple[str, ...]) -> torch.Tensor:
    num_nodes = int(data.y.size(0))
    out = torch.zeros(num_nodes, device=data.y.device, dtype=torch.bool)
    any_nonempty = False
    for name in names:
        if hasattr(data, name):
            mask = getattr(data, name)
            if mask is None:
                continue
            if mask.dim() > 1:
                mask = mask[:, 0]
            mask = mask.view(-1).bool()
            if int(mask.sum().item()) > 0:
                out = out | mask
                any_nonempty = True
    if any_nonempty:
        return out
    return torch.ones(num_nodes, device=data.y.device, dtype=torch.bool)


def _micro_f1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits[mask].argmax(dim=1)
    y = labels[mask]
    return float((pred == y).float().mean().item())


def _macro_f1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits[mask].argmax(dim=1)
    y = labels[mask]
    classes = torch.unique(torch.cat([pred, y], dim=0))
    f1_vals = []
    for cls in classes:
        tp = ((pred == cls) & (y == cls)).sum().float()
        fp = ((pred == cls) & (y != cls)).sum().float()
        fn = ((pred != cls) & (y == cls)).sum().float()
        denom = 2.0 * tp + fp + fn
        f1 = torch.where(denom > 0, (2.0 * tp) / denom, torch.tensor(0.0, device=y.device))
        f1_vals.append(f1)
    return float(torch.stack(f1_vals).mean().item()) if f1_vals else 0.0


def _subsample_rows(x: torch.Tensor, n: int) -> torch.Tensor:
    if n <= 0 or x.size(0) <= n:
        return x
    idx = torch.randperm(x.size(0), device=x.device)[:n]
    return x[idx]


def _hop_to_poly(hop_coeffs: torch.Tensor) -> torch.Tensor:
    """Convert hop-space [A, A^2, ...] coeffs to monomial [I, A, A^2, ...]."""
    poly = torch.cat([hop_coeffs.new_zeros(1), hop_coeffs], dim=0)
    # Enforce no identity term; hop coeff index 0 is always A X.
    poly[0] = 0.0
    return poly


def _effective_hops(raw: torch.Tensor, cfg: dict | None = None) -> torch.Tensor:
    mode = str((cfg or {}).get("hop_parametrization", "softmax")).lower()
    if mode == "softmax":
        return F.softmax(raw, dim=0)
    if mode == "monotone_cumsum":
        inc = F.softplus(raw) + 1e-8
        hop = torch.cumsum(inc, dim=0)
        return hop / hop.sum().clamp_min(1e-8)
    raise ValueError(f"Unsupported hop_parametrization: {mode}")


def _monotonicity_loss_increasing(hop_coeffs: torch.Tensor) -> torch.Tensor:
    """Penalty for violating c1 <= c2 <= ... <= cK."""
    if hop_coeffs.numel() <= 1:
        return hop_coeffs.new_tensor(0.0)
    return F.relu(hop_coeffs[:-1] - hop_coeffs[1:]).mean()


def _filter_l1_penalty(raw_hops: nn.Parameter) -> torch.Tensor:
    return raw_hops.abs().mean()


def _project_hops_non_decreasing(hop_coeffs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    proj = torch.cummax(hop_coeffs, dim=0).values
    proj = proj.clamp_min(eps)
    proj = proj / proj.sum().clamp_min(eps)
    return proj


def _apply_monotonic_projection(raw_hops: nn.Parameter, cfg: dict) -> None:
    if str(cfg.get("hop_parametrization", "softmax")).lower() != "softmax":
        return
    with torch.no_grad():
        hop = _effective_hops(raw_hops, cfg)
        hop_proj = _project_hops_non_decreasing(hop)
        raw_hops.copy_(torch.log(hop_proj))


def _get_align_lr(cfg: dict) -> float:
    if "align_lr" in cfg and cfg["align_lr"] is not None:
        return float(cfg["align_lr"])
    return float(cfg["lr"]) * float(cfg.get("align_lr_mult", 1.0))


def prealign_filter(
    raw_hops: nn.Parameter,
    source_data,
    target_data,
    cfg: dict,
) -> list[dict]:
    mono = MonoProp()
    align_lr = _get_align_lr(cfg)
    opt_name = str(cfg.get("align_optimizer", "sgd")).lower()
    if opt_name == "adam":
        opt = torch.optim.Adam([raw_hops], lr=align_lr)
    elif opt_name == "sgd":
        opt = torch.optim.SGD([raw_hops], lr=align_lr, momentum=float(cfg.get("align_momentum", 0.0)))
    else:
        raise ValueError(f"Unsupported align_optimizer: {opt_name}")
    history = []
    struct_weight = float(cfg.get("struct_bpr_weight", 0.0))
    struct_samples = int(cfg.get("struct_bpr_samples", 2048))
    struct_margin = float(cfg.get("struct_bpr_margin", 0.0))
    source_pos_edges = build_struct_positive_edges(source_data, cfg) if struct_weight > 0 else None
    target_pos_edges = build_struct_positive_edges(target_data, cfg) if struct_weight > 0 else None

    for epoch in tqdm(range(1, int(cfg["align_epochs"]) + 1), desc="Pre-align", leave=False):
        hop = _effective_hops(raw_hops, cfg)
        poly = _hop_to_poly(hop)

        source_probe, target_probe = make_gaussian_probe(source_data.x, target_data.x)
        source_push_full = mono(source_probe, source_data.edge_index, poly)
        target_push_full = mono(target_probe, target_data.edge_index, poly)

        source_push = _subsample_rows(source_push_full, int(cfg["align_sample_size"]))
        target_push = _subsample_rows(target_push_full, int(cfg["align_sample_size"]))
        n = min(int(source_push.size(0)), int(target_push.size(0)))
        source_push = source_push[:n]
        target_push = target_push[:n]

        align_mmd = mmd_rbf(
            source_push,
            target_push,
            kernel_mul=float(cfg["kernel_mul"]),
            kernel_num=int(cfg["kernel_num"]),
            fix_sigma=cfg["fix_sigma"],
        )
        monotonicity_loss = _monotonicity_loss_increasing(hop)
        l1_reg = _filter_l1_penalty(raw_hops)
        if struct_weight > 0 and source_pos_edges is not None and target_pos_edges is not None:
            source_struct_bpr = edge_bpr_structure_loss(
                source_push_full,
                source_pos_edges,
                num_samples=struct_samples,
                margin=struct_margin,
            )
            target_struct_bpr = edge_bpr_structure_loss(
                target_push_full,
                target_pos_edges,
                num_samples=struct_samples,
                margin=struct_margin,
            )
            struct_loss = struct_weight * (source_struct_bpr + target_struct_bpr)
        else:
            source_struct_bpr = align_mmd.new_tensor(0.0)
            target_struct_bpr = align_mmd.new_tensor(0.0)
            struct_loss = align_mmd.new_tensor(0.0)
        loss = (
            align_mmd
            + float(cfg.get("monotonicity_weight", 0.0)) * monotonicity_loss
            + float(cfg.get("filter_l1_weight", 0.0)) * l1_reg
            + struct_loss
        )

        opt.zero_grad()
        loss.backward()
        opt.step()
        if bool(cfg.get("monotonicity_projection", False)):
            _apply_monotonic_projection(raw_hops, cfg)

        history.append(
            {
                "align_epoch": epoch,
                "align_mmd": float(align_mmd.item()),
                "monotonicity_loss": float(monotonicity_loss.item()),
                "l1_reg": float(l1_reg.item()),
                "source_struct_bpr": float(source_struct_bpr.item()),
                "target_struct_bpr": float(target_struct_bpr.item()),
                "struct_loss": float(struct_loss.item()),
                "align_total_loss": float(loss.item()),
                **{f"hop_c{i+1}": float(v) for i, v in enumerate(hop.detach().cpu().tolist())},
            }
        )
    return history


def run_case(
    case_name: str,
    source_data,
    target_data,
    cfg: dict,
    trainable_filter: bool,
    do_prealign: bool,
) -> tuple[list[dict], list[dict], list[float], list[float]]:
    in_dim = int(source_data.x.size(1))
    num_classes = int(torch.max(torch.cat([source_data.y, target_data.y], dim=0)).item()) + 1

    model = SharedFilterGCN(
        in_dim=in_dim,
        hid_dim=int(cfg["hid_dim"]),
        num_classes=num_classes,
        dropout=float(cfg["dropout"]),
    ).to(source_data.x.device)

    k_hops = int(cfg["k_hops"])
    fixed = torch.tensor(cfg["fixed_hop_coeffs"], dtype=source_data.x.dtype, device=source_data.x.device).view(-1)
    if int(fixed.numel()) != k_hops:
        raise ValueError(f"fixed_hop_coeffs length mismatch: expected {k_hops}, got {int(fixed.numel())}")

    align_history: list[dict] = []
    if trainable_filter:
        raw_hops = nn.Parameter(torch.empty(k_hops, device=source_data.x.device, dtype=source_data.x.dtype))
        with torch.no_grad():
            # Slight symmetry breaking; softmax(raw_hops) still starts near-uniform.
            raw_hops.normal_(mean=0.0, std=1e-3)
        start_hops = _effective_hops(raw_hops, cfg).detach().cpu().tolist()
        if do_prealign:
            align_history = prealign_filter(raw_hops, source_data, target_data, cfg)
            if bool(cfg.get("freeze_after_align", True)):
                with torch.no_grad():
                    fixed = _effective_hops(raw_hops, cfg).detach()
                raw_hops = None
        opt_params = list(model.parameters()) + ([] if raw_hops is None else [raw_hops])
    else:
        raw_hops = None
        opt_params = list(model.parameters())
        start_hops = fixed.detach().cpu().tolist()

    optimizer = torch.optim.Adam(
        opt_params,
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    source_train_mask = _as_mask(source_data, ("train_mask",))
    source_eval_mask = _as_mask(source_data, ("test_mask", "val_mask"))
    target_eval_mask = _as_mask(target_data, ("test_mask", "val_mask"))

    def current_hops() -> torch.Tensor:
        if raw_hops is None:
            return fixed
        return _effective_hops(raw_hops, cfg)

    train_history = []

    for epoch in tqdm(range(int(cfg["epochs"]) + 1), desc=f"{case_name} train", leave=False):
        hop = current_hops()
        if epoch > 0:
            model.train()
            src_logits = model(source_data, hop)
            sup_loss = F.cross_entropy(src_logits[source_train_mask], source_data.y[source_train_mask])
            l1_reg = _filter_l1_penalty(raw_hops) if raw_hops is not None else hop.new_tensor(0.0)
            mono_reg = _monotonicity_loss_increasing(hop)
            loss = (
                sup_loss
                + float(cfg["filter_l1_weight"]) * l1_reg
                + float(cfg.get("monotonicity_weight", 0.0)) * mono_reg
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if raw_hops is not None and bool(cfg.get("monotonicity_projection", False)):
                _apply_monotonic_projection(raw_hops, cfg)
        else:
            loss = torch.tensor(0.0, device=source_data.x.device)

        model.eval()
        with torch.no_grad():
            hop_eval = current_hops()
            src_logits_eval = model(source_data, hop_eval)
            tgt_logits_eval = model(target_data, hop_eval)

        row = {
            "case": case_name,
            "epoch": int(epoch),
            "source_loss": float(loss.item()),
            "src_micro_f1": _micro_f1(src_logits_eval, source_data.y, source_eval_mask),
            "src_macro_f1": _macro_f1(src_logits_eval, source_data.y, source_eval_mask),
            "tgt_micro_f1": _micro_f1(tgt_logits_eval, target_data.y, target_eval_mask),
            "tgt_macro_f1": _macro_f1(tgt_logits_eval, target_data.y, target_eval_mask),
        }
        row.update({f"hop_c{i+1}": float(v) for i, v in enumerate(hop_eval.detach().cpu().tolist())})
        train_history.append(row)

    end_hops = current_hops().detach().cpu().tolist()
    return train_history, align_history, start_hops, end_hops


def plot_target_transfer(case_rows: dict[str, list[dict]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for case, rows in case_rows.items():
        epochs = [int(r["epoch"]) for r in rows]
        vals = [float(r["tgt_micro_f1"]) for r in rows]
        ax.plot(epochs, vals, linewidth=2.0, label=case)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Target micro-F1")
    ax.set_title("Target Transferability (No DA during supervised stage)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_trainable_hops(align_rows: list[dict], train_rows: list[dict], out_path: Path) -> None:
    if not train_rows:
        return
    hop_keys = [k for k in train_rows[0].keys() if k.startswith("hop_c")]
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=False)

    for hk in hop_keys:
        if align_rows:
            axes[0].plot([int(r["align_epoch"]) for r in align_rows], [float(r[hk]) for r in align_rows], label=hk)
        axes[1].plot([int(r["epoch"]) for r in train_rows], [float(r[hk]) for r in train_rows], label=hk)

    axes[0].set_title("Case-ii Hop Coefficients During Pre-alignment")
    axes[0].set_ylabel("Coefficient")
    axes[0].grid(alpha=0.25)
    if align_rows:
        axes[0].legend(ncol=min(4, len(hop_keys)))

    axes[1].set_title("Case-ii Hop Coefficients During Source Supervised Training")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Coefficient")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=min(4, len(hop_keys)))

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_align_mmd(align_rows: list[dict], out_path: Path) -> None:
    if not align_rows:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(
        [int(r["align_epoch"]) for r in align_rows],
        [float(r["align_mmd"]) for r in align_rows],
        linewidth=2.0,
    )
    ax.set_xlabel("Pre-alignment Epoch")
    ax.set_ylabel("Probe MMD")
    ax.set_title("Case-ii Pre-alignment Objective")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_final_summary_rows(
    case_histories: list[list[dict]],
    source_name: str,
    target_name: str,
) -> list[dict]:
    summary_rows: list[dict] = []
    for idx, history in enumerate(case_histories, start=1):
        if not history:
            continue
        last = history[-1]
        row = {
            "case": f"case {idx}",
            "src": source_name,
            "tgt": target_name,
            "src_micro_f1": float(last["src_micro_f1"]),
            "src_macro_f1": float(last["src_macro_f1"]),
            "tgt_micro_f1": float(last["tgt_micro_f1"]),
            "tgt_macro_f1": float(last["tgt_macro_f1"]),
        }
        summary_rows.append(row)
    return summary_rows


def _build_manual_scenarios(cfg: dict) -> list[tuple[str, str]]:
    src = cfg["source"]
    tgt = cfg["target"]

    src_is_seq = isinstance(src, (list, tuple))
    tgt_is_seq = isinstance(tgt, (list, tuple))

    if src_is_seq or tgt_is_seq:
        if not (src_is_seq and tgt_is_seq):
            raise ValueError("source and target must both be lists/tuples when using multi-scenario mode.")
        if len(src) != len(tgt):
            raise ValueError(f"source/target length mismatch: {len(src)} vs {len(tgt)}")
        if len(src) == 0:
            raise ValueError("source/target lists must not be empty.")
        return [(str(s), str(t)) for s, t in zip(src, tgt)]

    return [(str(src), str(tgt))]


def _build_datasets(cfg: dict) -> list[str]:
    ds = cfg["dataset"]
    if isinstance(ds, (list, tuple)):
        if len(ds) == 0:
            raise ValueError("dataset list must not be empty.")
        return [str(d).lower() for d in ds]
    return [str(ds).lower()]


def _build_scenarios_for_dataset(cfg: dict, dataset_name: str) -> list[tuple[str, str]]:
    if bool(cfg.get("all_pairs", False)):
        domains = DATASET_DOMAINS.get(dataset_name)
        if domains is None:
            raise ValueError(f"No domain list found for dataset='{dataset_name}'.")
        return [(s, t) for s in domains for t in domains if s != t]
    return _build_manual_scenarios(cfg)


def _metric_mean(rows: list[dict], key: str) -> float:
    return float(sum(float(r[key]) for r in rows) / float(len(rows)))


def build_dataset_average_rows(dataset_name: str, final_rows: list[dict]) -> list[dict]:
    out = []
    for case in ("case 1", "case 2"):
        subset = [r for r in final_rows if r["case"] == case]
        if not subset:
            continue
        out.append(
            {
                "dataset": dataset_name,
                "case": case,
                "n_pairs": int(len(subset)),
                "src_micro_f1": _metric_mean(subset, "src_micro_f1"),
                "src_macro_f1": _metric_mean(subset, "src_macro_f1"),
                "tgt_micro_f1": _metric_mean(subset, "tgt_micro_f1"),
                "tgt_macro_f1": _metric_mean(subset, "tgt_macro_f1"),
            }
        )
    return out


def build_overall_average_rows(dataset_avg_rows: list[dict]) -> list[dict]:
    out = []
    for case in ("case 1", "case 2"):
        subset = [r for r in dataset_avg_rows if r["case"] == case]
        if not subset:
            continue
        out.append(
            {
                "case": case,
                "n_datasets": int(len(subset)),
                "src_micro_f1": _metric_mean(subset, "src_micro_f1"),
                "src_macro_f1": _metric_mean(subset, "src_macro_f1"),
                "tgt_micro_f1": _metric_mean(subset, "tgt_micro_f1"),
                "tgt_macro_f1": _metric_mean(subset, "tgt_macro_f1"),
            }
        )
    return out


def main():
    base_seed = int(CFG["seed"])
    set_seed(base_seed)
    datasets = _build_datasets(CFG)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    tag = str(CFG["tag"]).strip()
    run_name = timestamp if not tag else f"{timestamp}_{tag}"
    run_root = Path(CFG["out_root"]) / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    all_final_rows: list[dict] = []
    all_dataset_avg_rows: list[dict] = []

    for d_idx, dataset_name in enumerate(datasets):
        dataset_dir = run_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        scenarios = _build_scenarios_for_dataset(CFG, dataset_name)
        dataset_final_rows: list[dict] = []

        print(f"\n##### Dataset: {dataset_name} | scenarios={len(scenarios)} #####")
        for idx, (source_name, target_name) in enumerate(scenarios):
            scenario_seed = base_seed + d_idx * 10_000 + idx
            set_seed(scenario_seed)
            scenario_dir = dataset_dir / f"{idx:02d}_{source_name}_{target_name}"
            scenario_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n=== Scenario {idx + 1}/{len(scenarios)}: {source_name} -> {target_name} ===")
            source_data, target_data = load_pair(
                dataset=dataset_name,
                source=source_name,
                target=target_name,
                device=str(CFG["device"]),
                seed=scenario_seed,
            )

            fixed_rows, fixed_align_rows, fixed_start, fixed_end = run_case(
                case_name="case 1",
                source_data=source_data,
                target_data=target_data,
                cfg=CFG,
                trainable_filter=False,
                do_prealign=False,
            )
            train_rows, train_align_rows, train_start, train_end = run_case(
                case_name="case 2",
                source_data=source_data,
                target_data=target_data,
                cfg=CFG,
                trainable_filter=True,
                do_prealign=True,
            )

            save_table(scenario_dir / "case_1_train_history.csv", fixed_rows)
            save_table(scenario_dir / "case_2_train_history.csv", train_rows)
            if train_align_rows:
                save_table(scenario_dir / "case_2_align_history.csv", train_align_rows)

            final_rows = build_final_summary_rows(
                [fixed_rows, train_rows],
                source_name=source_name,
                target_name=target_name,
            )
            for row in final_rows:
                row["dataset"] = dataset_name
            save_table(scenario_dir / "final_metrics.csv", final_rows)
            dataset_final_rows.extend(final_rows)
            all_final_rows.extend(final_rows)

            case_rows = {
                "case 1": fixed_rows,
                "case 2": train_rows,
            }
            plot_target_transfer(case_rows, scenario_dir / "target_micro_f1_comparison.png")
            plot_trainable_hops(train_align_rows, train_rows, scenario_dir / "case_2_hop_coeff_trajectory.png")
            plot_align_mmd(train_align_rows, scenario_dir / "case_2_align_mmd.png")

            print(f"Results saved to: {scenario_dir}")
            print(
                f"Case-1 start/end hop coeffs: "
                f"{tensor_to_float_list(torch.tensor(fixed_start))} -> {tensor_to_float_list(torch.tensor(fixed_end))}"
            )
            print(
                f"Case-2 start/end hop coeffs: "
                f"{tensor_to_float_list(torch.tensor(train_start))} -> {tensor_to_float_list(torch.tensor(train_end))}"
            )
            print(f"Case-1 final tgt micro-F1: {fixed_rows[-1]['tgt_micro_f1']:.4f}")
            print(f"Case-2 final tgt micro-F1: {train_rows[-1]['tgt_micro_f1']:.4f}")

        if dataset_final_rows:
            save_table(dataset_dir / "combined_final_metrics.csv", dataset_final_rows)
            dataset_avg_rows = build_dataset_average_rows(dataset_name, dataset_final_rows)
            save_table(dataset_dir / "dataset_average_metrics.csv", dataset_avg_rows)
            all_dataset_avg_rows.extend(dataset_avg_rows)
            print(f"\n{dataset_name}: combined + average saved to {dataset_dir}")

    if all_final_rows:
        save_table(run_root / "all_datasets_combined_final_metrics.csv", all_final_rows)
    if all_dataset_avg_rows:
        save_table(run_root / "overall_average_metrics.csv", build_overall_average_rows(all_dataset_avg_rows))
    if all_final_rows or all_dataset_avg_rows:
        print(f"\nRun root: {run_root}")


if __name__ == "__main__":
    main()
