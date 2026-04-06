from Learn.Clean_SCGDA.utils.expt_utils import set_seed
from Learn.Clean_SCGDA.utils.ablation_utils.alignment import train_probe_aligner
from Learn.Clean_SCGDA.utils.ablation_utils.common import load_pair
from Learn.Clean_SCGDA.utils.ablation_utils.comparison import build_compare_table, build_final_summary
from Learn.Clean_SCGDA.utils.ablation_utils.transfer import evaluate_transferability


def compute_scenario_payload(dataset: str, source: str, target: str, cfg: dict) -> dict:
    """Run one scenario end-to-end and return in-memory outputs."""
    set_seed(cfg["seed"])
    source_data, target_data = load_pair(dataset, source, target, cfg["device"], cfg["seed"])
    align_state = train_probe_aligner(source_data, target_data, cfg)

    layer_rows = build_compare_table(source_data, target_data, align_state, cfg)
    transfer_run_rows, transfer_summary_rows = evaluate_transferability(
        source_data,
        target_data,
        align_state,
        cfg,
    )
    summary = build_final_summary(
        dataset,
        source,
        target,
        layer_rows,
        transfer_summary_rows,
        max_layers=int(cfg["max_layers"]),
    )

    return {
        "source_data": source_data,
        "target_data": target_data,
        "align_state": align_state,
        "layer_rows": layer_rows,
        "transfer_run_rows": transfer_run_rows,
        "transfer_summary_rows": transfer_summary_rows,
        "summary": summary,
    }
