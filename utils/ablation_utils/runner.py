from utils.ablation_utils.reporting import save_aggregate_reports, save_scenario_artifacts
from utils.ablation_utils.scenario_pipeline import compute_scenario_payload


def _init_accumulators(test_config: dict) -> dict:
    return {
        "summary_rows": [],
        "all_layer_rows": [],
        "all_transfer_summary_rows": [],
        "dataset_to_rows": {dataset: [] for dataset in test_config.keys()},
        "dataset_to_transfer": {dataset: [] for dataset in test_config.keys()},
        "failed_rows": [],
    }


def _record_success(store: dict, dataset: str, payload: dict) -> None:
    store["summary_rows"].append(payload["summary"])
    store["all_layer_rows"].append(payload["layer_rows"])
    store["all_transfer_summary_rows"].append(payload["transfer_summary_rows"])
    store["dataset_to_rows"][dataset].append(payload["layer_rows"])
    store["dataset_to_transfer"][dataset].append(payload["transfer_summary_rows"])


def _record_failure(store: dict, dataset: str, source: str, target: str, exc: Exception) -> None:
    store["failed_rows"].append(
        {
            "dataset": dataset,
            "source": source,
            "target": target,
            "error": str(exc),
        }
    )
    print(f"[failed] {dataset} {source}->{target}: {exc}")


def run_all(test_config: dict, expt_config: dict) -> None:
    store = _init_accumulators(test_config)

    for dataset, scenarios in test_config.items():
        for source, target in scenarios:
            try:
                payload = compute_scenario_payload(dataset, source, target, expt_config)
                save_scenario_artifacts(dataset, source, target, payload, expt_config)
                _record_success(store, dataset, payload)
            except Exception as exc:
                _record_failure(store, dataset, source, target, exc)

    save_aggregate_reports(
        test_config=test_config,
        cfg=expt_config,
        summary_rows=store["summary_rows"],
        all_layer_rows=store["all_layer_rows"],
        all_transfer_summary_rows=store["all_transfer_summary_rows"],
        dataset_to_rows=store["dataset_to_rows"],
        dataset_to_transfer=store["dataset_to_transfer"],
        failed_rows=store["failed_rows"],
    )
