import torch

from utils.filter_utils import make_gaussian_probe
from utils.ablation_utils.common import sample_idx
from utils.ablation_utils.metrics import compute_shift_metrics
from utils.ablation_utils.propagation import (
    WeightedPropagation,
    propagate_fixed_cheb_layers,
    propagate_weighted_layers,
)
from utils.ablation_utils.structure import edge_distance_metrics


COMPARE_METRIC_KEYS = [
    "normal_real_mmd",
    "normal_real_cmmd",
    "normal_probe_mmd",
    "normal_probe_cmmd",
    "aligned_real_mmd",
    "aligned_real_cmmd",
    "aligned_probe_mmd",
    "aligned_probe_cmmd",
    "delta_real_mmd",
    "delta_real_cmmd",
    "delta_probe_mmd",
    "delta_probe_cmmd",
    "source_edge_l1_distance",
    "source_edge_l2_distance",
    "source_edge_jaccard",
    "target_edge_l1_distance",
    "target_edge_l2_distance",
    "target_edge_jaccard",
    "avg_edge_l1_distance",
    "avg_edge_jaccard",
]


def build_compare_table(source_data, target_data, align_state: dict, cfg: dict) -> list[dict]:
    source_x = source_data.x.detach().float()
    target_x = target_data.x.detach().float()
    source_y = source_data.y.detach()
    target_y = target_data.y.detach()

    source_probe, target_probe = make_gaussian_probe(source_x, target_x)
    idx_s = sample_idx(source_x.size(0), cfg["metric_sample_size"], source_x.device)
    idx_t = sample_idx(target_x.size(0), cfg["metric_sample_size"], target_x.device)

    normal_cheb_k = int(cfg.get("normal_cheb_k", cfg["cheb_k"]))
    normal_src_real = propagate_fixed_cheb_layers(
        source_x,
        source_data,
        cfg["max_layers"],
        cheb_k=normal_cheb_k,
        lambda_max=cfg["cheb_lambda_max"],
    )
    normal_tgt_real = propagate_fixed_cheb_layers(
        target_x,
        target_data,
        cfg["max_layers"],
        cheb_k=normal_cheb_k,
        lambda_max=cfg["cheb_lambda_max"],
    )
    normal_src_probe = propagate_fixed_cheb_layers(
        source_probe,
        source_data,
        cfg["max_layers"],
        cheb_k=normal_cheb_k,
        lambda_max=cfg["cheb_lambda_max"],
    )
    normal_tgt_probe = propagate_fixed_cheb_layers(
        target_probe,
        target_data,
        cfg["max_layers"],
        cheb_k=normal_cheb_k,
        lambda_max=cfg["cheb_lambda_max"],
    )

    aligned_prop = WeightedPropagation().to(source_x.device)
    aligned_src_real = propagate_weighted_layers(
        source_x,
        align_state["source_aligned_edge_index"],
        align_state["source_aligned_edge_weight"],
        cfg["max_layers"],
        aligned_prop,
    )
    aligned_tgt_real = propagate_weighted_layers(
        target_x,
        align_state["target_aligned_edge_index"],
        align_state["target_aligned_edge_weight"],
        cfg["max_layers"],
        aligned_prop,
    )
    aligned_src_probe = propagate_weighted_layers(
        source_probe,
        align_state["source_aligned_edge_index"],
        align_state["source_aligned_edge_weight"],
        cfg["max_layers"],
        aligned_prop,
    )
    aligned_tgt_probe = propagate_weighted_layers(
        target_probe,
        align_state["target_aligned_edge_index"],
        align_state["target_aligned_edge_weight"],
        cfg["max_layers"],
        aligned_prop,
    )

    src_struct = edge_distance_metrics(
        source_data.edge_index,
        align_state["source_aligned_edge_index"],
        align_state["source_aligned_edge_weight"],
        num_nodes=int(source_x.size(0)),
    )
    tgt_struct = edge_distance_metrics(
        target_data.edge_index,
        align_state["target_aligned_edge_index"],
        align_state["target_aligned_edge_weight"],
        num_nodes=int(target_x.size(0)),
    )

    rows = []
    for layer in range(cfg["max_layers"] + 1):
        normal_real_mmd, normal_real_cmmd = compute_shift_metrics(
            normal_src_real[layer],
            normal_tgt_real[layer],
            source_y,
            target_y,
            idx_s,
            idx_t,
            cfg,
        )
        normal_probe_mmd, normal_probe_cmmd = compute_shift_metrics(
            normal_src_probe[layer],
            normal_tgt_probe[layer],
            source_y,
            target_y,
            idx_s,
            idx_t,
            cfg,
        )
        aligned_real_mmd, aligned_real_cmmd = compute_shift_metrics(
            aligned_src_real[layer],
            aligned_tgt_real[layer],
            source_y,
            target_y,
            idx_s,
            idx_t,
            cfg,
        )
        aligned_probe_mmd, aligned_probe_cmmd = compute_shift_metrics(
            aligned_src_probe[layer],
            aligned_tgt_probe[layer],
            source_y,
            target_y,
            idx_s,
            idx_t,
            cfg,
        )

        rows.append(
            {
                "layer": layer,
                "normal_real_mmd": normal_real_mmd,
                "normal_real_cmmd": normal_real_cmmd,
                "normal_probe_mmd": normal_probe_mmd,
                "normal_probe_cmmd": normal_probe_cmmd,
                "aligned_real_mmd": aligned_real_mmd,
                "aligned_real_cmmd": aligned_real_cmmd,
                "aligned_probe_mmd": aligned_probe_mmd,
                "aligned_probe_cmmd": aligned_probe_cmmd,
                "delta_real_mmd": aligned_real_mmd - normal_real_mmd,
                "delta_real_cmmd": aligned_real_cmmd - normal_real_cmmd,
                "delta_probe_mmd": aligned_probe_mmd - normal_probe_mmd,
                "delta_probe_cmmd": aligned_probe_cmmd - normal_probe_cmmd,
                "source_edges_normal": int(source_data.edge_index.size(1)),
                "source_edges_aligned": int(align_state["source_aligned_edge_index"].size(1)),
                "target_edges_normal": int(target_data.edge_index.size(1)),
                "target_edges_aligned": int(align_state["target_aligned_edge_index"].size(1)),
                "source_edge_l1_distance": float(src_struct["l1"]),
                "source_edge_l2_distance": float(src_struct["l2"]),
                "source_edge_jaccard": float(src_struct["jaccard"]),
                "target_edge_l1_distance": float(tgt_struct["l1"]),
                "target_edge_l2_distance": float(tgt_struct["l2"]),
                "target_edge_jaccard": float(tgt_struct["jaccard"]),
                "avg_edge_l1_distance": float((src_struct["l1"] + tgt_struct["l1"]) * 0.5),
                "avg_edge_jaccard": float((src_struct["jaccard"] + tgt_struct["jaccard"]) * 0.5),
            }
        )

    return rows


def build_final_summary(
    dataset: str,
    source: str,
    target: str,
    layer_rows: list[dict],
    transfer_summary_rows: list[dict],
    max_layers: int,
) -> dict[str, float | int | str]:
    final_layer = int(max_layers)
    normal_final = next(
        (row for row in transfer_summary_rows if row["case"] == "normal" and int(row["layer"]) == final_layer),
        None,
    )
    aligned_final = next(
        (row for row in transfer_summary_rows if row["case"] == "aligned" and int(row["layer"]) == final_layer),
        None,
    )

    normal_acc = float(normal_final["target_acc_mean"]) if normal_final else float("nan")
    aligned_acc = float(aligned_final["target_acc_mean"]) if aligned_final else float("nan")
    normal_f1 = float(normal_final["target_macro_f1_mean"]) if normal_final else float("nan")
    aligned_f1 = float(aligned_final["target_macro_f1_mean"]) if aligned_final else float("nan")

    last = layer_rows[-1]
    return {
        "dataset": dataset,
        "source": source,
        "target": target,
        "final_layer": int(last["layer"]),
        "normal_real_mmd": float(last["normal_real_mmd"]),
        "normal_real_cmmd": float(last["normal_real_cmmd"]),
        "aligned_real_mmd": float(last["aligned_real_mmd"]),
        "aligned_real_cmmd": float(last["aligned_real_cmmd"]),
        "delta_real_mmd": float(last["delta_real_mmd"]),
        "delta_real_cmmd": float(last["delta_real_cmmd"]),
        "source_edges_normal": int(last["source_edges_normal"]),
        "source_edges_aligned": int(last["source_edges_aligned"]),
        "target_edges_normal": int(last["target_edges_normal"]),
        "target_edges_aligned": int(last["target_edges_aligned"]),
        "source_edge_l1_distance": float(last["source_edge_l1_distance"]),
        "source_edge_l2_distance": float(last["source_edge_l2_distance"]),
        "source_edge_jaccard": float(last["source_edge_jaccard"]),
        "target_edge_l1_distance": float(last["target_edge_l1_distance"]),
        "target_edge_l2_distance": float(last["target_edge_l2_distance"]),
        "target_edge_jaccard": float(last["target_edge_jaccard"]),
        "avg_edge_l1_distance": float(last["avg_edge_l1_distance"]),
        "avg_edge_jaccard": float(last["avg_edge_jaccard"]),
        "normal_transfer_target_acc_mean_lK": normal_acc,
        "aligned_transfer_target_acc_mean_lK": aligned_acc,
        "delta_transfer_target_acc_mean_lK": aligned_acc - normal_acc,
        "normal_transfer_target_macro_f1_mean_lK": normal_f1,
        "aligned_transfer_target_macro_f1_mean_lK": aligned_f1,
        "delta_transfer_target_macro_f1_mean_lK": aligned_f1 - normal_f1,
    }
