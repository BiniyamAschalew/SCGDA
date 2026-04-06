"""Thin entrypoint for aligner shift ablation."""

# from utils.ablation_utils.defaults import DEFAULT_EXPT_CONFIG, DEFAULT_TEST_CONFIG
# from utils.ablation_utils.runner import run_all

from Learn.Clean_SCGDA.utils.ablation_utils.defaults import DEFAULT_EXPT_CONFIG, DEFAULT_TEST_CONFIG
from Learn.Clean_SCGDA.utils.ablation_utils.runner import run_all

def main() -> None:
    # Keep local overrides minimal; move durable config to utils/ablation_utils/defaults.py.
    expt_config = dict(DEFAULT_EXPT_CONFIG)
    test_config = dict(DEFAULT_TEST_CONFIG)

    # Example local override:
    # expt_config["device"] = "cuda:0"

    run_all(test_config=test_config, expt_config=expt_config)


if __name__ == "__main__":
    main()
