"""Thin entrypoint for filter-learning inspection experiment."""

from __future__ import annotations

from utils.ablation_utils.filter_inspect_defaults import (
    DEFAULT_FILTER_INSPECT_EXPT_CONFIG,
    DEFAULT_FILTER_INSPECT_TEST_CONFIG,
)
from utils.ablation_utils.filter_inspect_pipeline import build_scenario_payload
from utils.ablation_utils.filter_inspect_reporting import (
    save_aggregate_reports,
    save_scenario_artifacts,
)


def run_scenario(dataset: str, source: str, target: str, cfg: dict) -> dict:
    """Execute one scenario and return a summary row."""
    payload = build_scenario_payload(dataset, source, target, cfg)
    save_scenario_artifacts(dataset, source, target, payload, cfg)
    return payload["summary"]


def run_all(test_config: dict, expt_config: dict) -> None:
    """Run all scenarios for filter-learning inspection experiment."""
    summary_rows = []
    failed_rows = []

    for dataset, scenarios in test_config.items():
        for source, target in scenarios:
            try:
                summary = run_scenario(dataset, source, target, expt_config)
                summary_rows.append(summary)
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

    save_aggregate_reports(
        test_config=test_config,
        cfg=expt_config,
        summary_rows=summary_rows,
        failed_rows=failed_rows,
    )


def main() -> None:
    """Create local config copies and launch the experiment."""
    expt_config = dict(DEFAULT_FILTER_INSPECT_EXPT_CONFIG)
    test_config = dict(DEFAULT_FILTER_INSPECT_TEST_CONFIG)

    # Example local override:
    expt_config["device"] = "cuda:7"

    run_all(test_config=test_config, expt_config=expt_config)


if __name__ == "__main__":
    main()
