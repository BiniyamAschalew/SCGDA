from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from .config import CASE_SPECS, ExperimentConfig, dataset_triplets, resolve_case_hparams
from .data import load_dataset_graphs, triplet_graphs
from .plotting import (
    plot_bound_proxy,
    plot_component_influence,
    plot_component_summary,
    plot_domain_performance,
    plot_lipschitz_scatter,
    plot_pairwise_mmd,
    plot_shift_summary,
    plot_target_heatmap,
    plot_training_curves,
)
from .train import train_case


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run the generalized 3-domain triplet bound experiment.",
    )
    parser.add_argument("--dataset", type=str, default="airport")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hid-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout-ratio", type=float, default=0.1)
    parser.add_argument("--gnn", type=str, default="gcn")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--mmd-weight", type=float, default=0.1)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--reuse-saved-masks", action="store_true")
    parser.add_argument("--disable-pair-tuned-hparams", action="store_true")
    parser.add_argument("--metric-sample-size", type=int, default=0)
    parser.add_argument("--grad-nodes-cap", type=int, default=0)
    parser.add_argument("--max-triplets", type=int, default=0)
    parser.add_argument("--out-root", type=str, default="./__saved__/analysis/triplet_bound")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    return ExperimentConfig(
        dataset=args.dataset,
        seeds=tuple(args.seeds),
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hid_dim=args.hid_dim,
        num_layers=args.num_layers,
        dropout_ratio=args.dropout_ratio,
        gnn=args.gnn,
        activation=args.activation,
        mmd_weight=args.mmd_weight,
        train_split=args.train_split,
        val_split=args.val_split,
        resample_masks=not args.reuse_saved_masks,
        use_pair_tuned_hparams=not args.disable_pair_tuned_hparams,
        metric_sample_size=args.metric_sample_size,
        grad_nodes_cap=args.grad_nodes_cap,
        max_triplets=args.max_triplets,
        out_root=args.out_root,
        run_name=args.run_name,
        verbose=args.verbose,
    )


def _flatten_summary(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    summary = df.groupby(group_cols, as_index=False, sort=False)[metric_cols].agg(["mean", "std"])
    summary.columns = [
        "_".join([part for part in col if part]).rstrip("_")
        for col in summary.columns.to_flat_index()
    ]
    return summary


def _attach_bound_proxy(df: pd.DataFrame) -> pd.DataFrame:
    oracle = df[df["case_key"] == "oracle"][
        [
            "seed",
            "triplet_key",
            "source_micro_f1",
            "source_error",
            "target_micro_f1",
            "target_error",
            "reference_micro_f1",
            "reference_error",
        ]
    ].rename(
        columns={
            "source_micro_f1": "oracle_source_micro_f1",
            "source_error": "oracle_source_error",
            "target_micro_f1": "oracle_target_micro_f1",
            "target_error": "oracle_target_error",
            "reference_micro_f1": "oracle_reference_micro_f1",
            "reference_error": "oracle_reference_error",
        }
    )
    merged = df.merge(oracle, on=["seed", "triplet_key"], how="left")
    merged["ideal_joint_error_proxy"] = 0.5 * (
        merged["oracle_source_error"] + merged["oracle_target_error"]
    )
    merged["oracle_target_gap"] = (
        merged["oracle_target_micro_f1"] - merged["target_micro_f1"]
    )

    merged["adaptation_proxy"] = (
        merged["global_grad_input_max"] * merged["val_mmd_source_target"]
    )
    merged["bound_proxy_rhs_joint"] = (
        merged["source_error"]
        + merged["adaptation_proxy"]
        + merged["ideal_joint_error_proxy"]
    )
    merged["bound_proxy_gap_joint"] = (
        merged["bound_proxy_rhs_joint"] - merged["target_error"]
    )

    merged["adaptation_proxy_train_target"] = (
        merged["global_grad_input_max"] * merged["shift_mmd_source_train_target_val"]
    )
    merged["bound_proxy_rhs_train_target"] = (
        merged["source_error"]
        + merged["adaptation_proxy_train_target"]
        + merged["ideal_joint_error_proxy"]
    )
    merged["bound_proxy_gap_train_target"] = (
        merged["bound_proxy_rhs_train_target"] - merged["target_error"]
    )
    return merged


def _component_correlations(raw_df: pd.DataFrame) -> pd.DataFrame:
    view = raw_df[raw_df["case_key"] != "oracle"].copy()
    components = [
        "source_error",
        "shift_mmd_source_train_source_val",
        "shift_mmd_source_train_target_val",
        "global_grad_input_max",
        "oracle_target_micro_f1",
        "bound_proxy_rhs_train_target",
    ]
    rows = []
    for case_label, frame in view.groupby("case_label", sort=False):
        for component in components:
            if len(frame) < 2 or frame[component].nunique() <= 1 or frame["target_micro_f1"].nunique() <= 1:
                corr = float("nan")
            else:
                corr = frame[component].corr(frame["target_micro_f1"])
            rows.append(
                {
                    "case_label": case_label,
                    "component": component,
                    "pearson_with_target_micro_f1": corr,
                }
            )
    return pd.DataFrame(rows)


def run_experiment(exp_cfg: ExperimentConfig) -> Path:
    out_dir = exp_cfg.output_dir()
    triplets = dataset_triplets(exp_cfg.dataset, exp_cfg.max_triplets)

    case_hparams_map = {
        triplet.key: {
            case.key: resolve_case_hparams(exp_cfg, triplet, case.key) for case in CASE_SPECS
        }
        for triplet in triplets
    }

    config_payload = exp_cfg.to_dict()
    config_payload["triplets"] = [triplet.label for triplet in triplets]
    config_payload["case_hparams_by_triplet"] = case_hparams_map
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    raw_rows: list[dict] = []
    history_rows: list[dict] = []
    total_runs = len(exp_cfg.seeds) * len(triplets) * len(CASE_SPECS)
    run_idx = 0

    for seed in exp_cfg.seeds:
        graphs, metadata = load_dataset_graphs(exp_cfg, seed)
        for triplet in triplets:
            run_graphs = triplet_graphs(graphs, triplet)
            for case in CASE_SPECS:
                case_hparams = case_hparams_map[triplet.key][case.key]
                case_to_run = case
                if case.key == "source_mmd_target":
                    case_to_run = case.__class__(
                        key=case.key,
                        label=case.label,
                        source_ce_weight=case.source_ce_weight,
                        target_ce_weight=case.target_ce_weight,
                        mmd_weight=float(case_hparams["mmd_weight"]),
                    )
                output = train_case(
                    exp_cfg=exp_cfg,
                    case=case_to_run,
                    triplet=triplet,
                    graphs=run_graphs,
                    metadata=metadata,
                    case_hparams=case_hparams,
                    seed=seed,
                )
                raw_rows.append(output.row)
                history_rows.extend(output.history)
                run_idx += 1
                if exp_cfg.verbose:
                    print(
                        f"[{run_idx}/{total_runs}] "
                        f"dataset={exp_cfg.dataset.lower()} seed={seed} "
                        f"triplet={triplet.label} case={case_to_run.label} "
                        f"src_micro_f1={output.row['source_micro_f1']:.4f} "
                        f"target_micro_f1={output.row['target_micro_f1']:.4f}"
                    )

    raw_df = pd.DataFrame(raw_rows)
    history_df = pd.DataFrame(history_rows)
    raw_df = _attach_bound_proxy(raw_df)

    raw_df.to_csv(out_dir / "raw_results.csv", index=False)
    history_df.to_csv(out_dir / "training_history.csv", index=False)

    summary_metrics = [
        "source_micro_f1",
        "source_macro_f1",
        "source_error",
        "target_micro_f1",
        "target_macro_f1",
        "target_error",
        "reference_micro_f1",
        "reference_macro_f1",
        "reference_error",
        "train_st_mmd",
        "shift_mmd_source_train_source_val",
        "shift_mmd_source_train_target_val",
        "shift_gap_target_minus_source",
        "val_mmd_source_target",
        "val_mmd_source_reference",
        "val_mmd_target_reference",
        "global_grad_input_max",
        "source_grad_input_max",
        "target_grad_input_max",
        "reference_grad_input_max",
        "oracle_target_micro_f1",
        "oracle_target_error",
        "oracle_target_gap",
        "bound_proxy_rhs_joint",
        "bound_proxy_gap_joint",
        "bound_proxy_rhs_train_target",
        "bound_proxy_gap_train_target",
        "train_time",
    ]
    summary_by_case = _flatten_summary(raw_df, ["case_key", "case_label"], summary_metrics)
    summary_by_triplet_case = _flatten_summary(
        raw_df,
        ["dataset", "triplet_key", "triplet_label", "source", "target", "reference", "case_key", "case_label"],
        summary_metrics,
    )
    correlation_df = _component_correlations(raw_df)

    summary_by_case.to_csv(out_dir / "summary_by_case.csv", index=False)
    summary_by_triplet_case.to_csv(out_dir / "summary_by_triplet_case.csv", index=False)
    correlation_df.to_csv(out_dir / "component_correlations.csv", index=False)

    dataset_name = exp_cfg.dataset.capitalize()
    plot_domain_performance(summary_by_case, out_dir, dataset_name)
    plot_target_heatmap(summary_by_triplet_case, out_dir, dataset_name)
    plot_pairwise_mmd(summary_by_case, out_dir, dataset_name)
    plot_shift_summary(summary_by_case, out_dir, dataset_name)
    plot_component_summary(summary_by_case, out_dir, dataset_name)
    plot_lipschitz_scatter(raw_df, out_dir, dataset_name)
    plot_component_influence(raw_df, out_dir, dataset_name)
    plot_bound_proxy(
        raw_df,
        out_dir,
        dataset_name,
        column="bound_proxy_rhs_joint",
        filename="bound_proxy_scatter.png",
        title="Bound Proxy (val-val shift)",
    )
    plot_bound_proxy(
        raw_df,
        out_dir,
        dataset_name,
        column="bound_proxy_rhs_train_target",
        filename="bound_proxy_train_target_scatter.png",
        title="Bound Proxy (source-train to target-val shift)",
    )
    plot_training_curves(history_df, out_dir, dataset_name)

    readme_lines = [
        f"# {dataset_name} Triplet Bound Experiment",
        "",
        "Generated files:",
        "- `config.json`: full run configuration and pair-specific tuned hyperparameters.",
        "- `raw_results.csv`: one row per seed x triplet x case.",
        "- `training_history.csv`: per-epoch loss and train-mask MMD.",
        "- `summary_by_case.csv`: averages over all seeds and triplets.",
        "- `summary_by_triplet_case.csv`: averages per ordered triplet and case.",
        "- `component_correlations.csv`: Pearson correlations between target micro-F1 and the main components.",
        "- `*.png`: performance, shift, sensitivity, and bound-proxy plots.",
        "",
        "Notes:",
        "- Training uses only `train_mask`.",
        "- Evaluation uses only `val_mask`.",
        "- `shift_mmd_source_train_source_val` compares source train vs source val embeddings.",
        "- `shift_mmd_source_train_target_val` compares source train vs target val embeddings.",
        "- `bound_proxy_rhs_train_target` uses source error + sensitivity x source-train/target-val shift + oracle joint error proxy.",
    ]
    with open(out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines) + "\n")

    return out_dir


def main() -> None:
    exp_cfg = parse_args()
    out_dir = run_experiment(exp_cfg)
    print(f"results_dir={out_dir}")


if __name__ == "__main__":
    main()
