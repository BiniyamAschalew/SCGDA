from pathlib import Path

import torch

from utils.ablation_utils.common import build_average_rows, save_table
from utils.ablation_utils.comparison import COMPARE_METRIC_KEYS
from utils.ablation_utils.plots import (
    save_align_history_plot,
    save_average_target_transfer_plot,
    save_compare_plot,
    save_transferability_plot,
)
from utils.ablation_utils.structure import save_structural_analysis
from utils.ablation_utils.transfer import (
    build_average_target_transfer_rows,
    write_transferability_protocol,
)


SUMMARY_FIELDNAMES = [
    "dataset",
    "source",
    "target",
    "final_layer",
    "normal_real_mmd",
    "normal_real_cmmd",
    "aligned_real_mmd",
    "aligned_real_cmmd",
    "delta_real_mmd",
    "delta_real_cmmd",
    "source_edges_normal",
    "source_edges_aligned",
    "target_edges_normal",
    "target_edges_aligned",
    "source_edge_l1_distance",
    "source_edge_l2_distance",
    "source_edge_jaccard",
    "target_edge_l1_distance",
    "target_edge_l2_distance",
    "target_edge_jaccard",
    "avg_edge_l1_distance",
    "avg_edge_jaccard",
    "normal_transfer_target_acc_mean_lK",
    "aligned_transfer_target_acc_mean_lK",
    "delta_transfer_target_acc_mean_lK",
    "normal_transfer_target_macro_f1_mean_lK",
    "aligned_transfer_target_macro_f1_mean_lK",
    "delta_transfer_target_macro_f1_mean_lK",
]

AVERAGE_TRANSFER_FIELDNAMES = [
    "layer",
    "n_scenarios",
    "normal_target_acc_mean",
    "normal_target_acc_std",
    "aligned_target_acc_mean",
    "aligned_target_acc_std",
    "delta_target_acc_mean",
]


def _scenario_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "compare_csv": out_dir / "compare_layer_metrics.csv",
        "compare_png": out_dir / "compare_layer_metrics.png",
        "align_hist_csv": out_dir / "align_history.csv",
        "align_hist_png": out_dir / "align_history.png",
        "aligned_edges_pt": out_dir / "aligned_edges.pt",
        "struct_png": out_dir / "structural_distributions.png",
        "struct_csv": out_dir / "structural_stats.csv",
        "transfer_runs_csv": out_dir / "transferability_runs.csv",
        "transfer_summary_csv": out_dir / "transferability_summary.csv",
        "transfer_plot_png": out_dir / "transferability_plot.png",
        "transfer_protocol_txt": out_dir / "transferability_protocol.txt",
    }


def _save_average_shift_bundle(out_dir: Path, rows: list[dict], title: str) -> None:
    avg_csv = out_dir / "average_layer_metrics.csv"
    avg_png = out_dir / "average_layer_metrics.png"
    save_table(avg_csv, rows)
    save_compare_plot(avg_png, rows, title)
    print(f"[saved] {avg_csv}")
    print(f"[saved] {avg_png}")


def _save_average_transfer_bundle(out_dir: Path, rows: list[dict], title: str) -> None:
    avg_csv = out_dir / "average_target_transfer_accuracy.csv"
    avg_png = out_dir / "average_target_transfer_accuracy.png"
    save_table(avg_csv, rows, fieldnames=AVERAGE_TRANSFER_FIELDNAMES)
    save_average_target_transfer_plot(avg_png, rows, title)
    print(f"[saved] {avg_csv}")
    print(f"[saved] {avg_png}")


def save_scenario_artifacts(dataset: str, source: str, target: str, payload: dict, cfg: dict) -> None:
    out_dir = Path(cfg["out_dir"]) / dataset / f"{source}_{target}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _scenario_paths(out_dir)

    layer_rows = payload["layer_rows"]
    transfer_run_rows = payload["transfer_run_rows"]
    transfer_summary_rows = payload["transfer_summary_rows"]
    source_data = payload["source_data"]
    target_data = payload["target_data"]
    align_state = payload["align_state"]

    save_table(paths["compare_csv"], layer_rows)
    save_compare_plot(paths["compare_png"], layer_rows, f"{dataset}: {source}->{target}")

    if align_state["history"]:
        save_table(paths["align_hist_csv"], align_state["history"])
        save_align_history_plot(
            paths["align_hist_png"],
            align_state["history"],
            f"Aligner history {dataset}: {source}->{target}",
        )

    save_structural_analysis(paths["struct_png"], paths["struct_csv"], source_data, target_data, align_state, cfg)

    save_table(paths["transfer_runs_csv"], transfer_run_rows)
    save_table(paths["transfer_summary_csv"], transfer_summary_rows)
    save_transferability_plot(
        paths["transfer_plot_png"],
        transfer_summary_rows,
        title=f"Transferability ({dataset}: {source}->{target})",
    )
    write_transferability_protocol(paths["transfer_protocol_txt"])

    torch.save(
        {
            "source_edge_index": align_state["source_aligned_edge_index"].detach().cpu(),
            "source_edge_weight": align_state["source_aligned_edge_weight"].detach().cpu(),
            "target_edge_index": align_state["target_aligned_edge_index"].detach().cpu(),
            "target_edge_weight": align_state["target_aligned_edge_weight"].detach().cpu(),
            "source_temp": align_state["source_temp"].detach().cpu(),
            "target_temp": align_state["target_temp"].detach().cpu(),
        },
        paths["aligned_edges_pt"],
    )

    print(f"[saved] {paths['compare_csv']}")
    print(f"[saved] {paths['compare_png']}")
    if align_state["history"]:
        print(f"[saved] {paths['align_hist_csv']}")
        print(f"[saved] {paths['align_hist_png']}")
    print(f"[saved] {paths['struct_csv']}")
    print(f"[saved] {paths['struct_png']}")
    print(f"[saved] {paths['transfer_runs_csv']}")
    print(f"[saved] {paths['transfer_summary_csv']}")
    print(f"[saved] {paths['transfer_plot_png']}")
    print(f"[saved] {paths['transfer_protocol_txt']}")
    print(f"[saved] {paths['aligned_edges_pt']}")


def save_aggregate_reports(
    test_config: dict,
    cfg: dict,
    summary_rows: list[dict],
    all_layer_rows: list[list[dict]],
    all_transfer_summary_rows: list[list[dict]],
    dataset_to_rows: dict[str, list[list[dict]]],
    dataset_to_transfer: dict[str, list[list[dict]]],
    failed_rows: list[dict],
) -> None:
    out_root = Path(cfg["out_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    summary_path = out_root / "summary_final_layer.csv"
    save_table(summary_path, summary_rows, fieldnames=SUMMARY_FIELDNAMES)
    print(f"[saved] {summary_path}")

    avg_rows = build_average_rows(all_layer_rows, cfg["max_layers"], COMPARE_METRIC_KEYS)
    if avg_rows:
        _save_average_shift_bundle(
            out_root,
            avg_rows,
            "Average normal vs aligned shifts (all successful scenarios)",
        )
    else:
        print("[warning] No successful scenarios, skipping global average plot.")

    avg_transfer_rows = build_average_target_transfer_rows(
        all_transfer_summary_rows,
        cfg["max_layers"],
    )
    if avg_transfer_rows:
        _save_average_transfer_bundle(
            out_root,
            avg_transfer_rows,
            "Average target transfer accuracy (all successful scenarios)",
        )
    else:
        print("[warning] No successful scenarios, skipping global transfer average plot.")

    for dataset in test_config.keys():
        dataset_rows = dataset_to_rows[dataset]
        dataset_avg_rows = build_average_rows(dataset_rows, cfg["max_layers"], COMPARE_METRIC_KEYS)
        if not dataset_avg_rows:
            print(f"[warning] No successful scenarios for dataset={dataset}, skipping dataset averages.")
            continue

        dataset_dir = out_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _save_average_shift_bundle(
            dataset_dir,
            dataset_avg_rows,
            f"Average normal vs aligned shifts for dataset={dataset}",
        )

        dataset_transfer_rows = build_average_target_transfer_rows(
            dataset_to_transfer[dataset],
            cfg["max_layers"],
        )
        if dataset_transfer_rows:
            _save_average_transfer_bundle(
                dataset_dir,
                dataset_transfer_rows,
                f"Average target transfer accuracy for dataset={dataset}",
            )
        else:
            print(f"[warning] No transfer rows for dataset={dataset}, skipping transfer averages.")

    if failed_rows:
        failed_path = out_root / "failed_scenarios.csv"
        save_table(failed_path, failed_rows, fieldnames=["dataset", "source", "target", "error"])
        print(f"[saved] {failed_path}")
