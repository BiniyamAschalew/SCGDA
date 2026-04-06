""" Adversarial alignment of filters """
import time
import torch

from pathlib import Path
from  utils.config_utils import load_config

def _build_scenarios(transfer_settings):
    # unpack the transfer scenarios to dataset: (source, target) pairs
    scenarios = {}
    for dataset, pairs in transfer_settings.items():
        for source, targets in pairs.items():
            for target in targets:
                scenarios[dataset] = scenarios.get(dataset, []) + [(source, target)]
    return scenarios

def run(config):
    pass


def run_all(scenarios, sconfig):

    all_shift_summary = []
    all_transfer_summary = []
    dataset_to_shift = {dataset: [] for dataset in scenarios.keys()}
    dataset_to_transfer = {dataset: [] for dataset in scenarios.keys()}


    for dataset, pairs in scenarios.items():
        for source, target in pairs:
            print(f"Running scenario: {dataset} - {source} -> {target}")

            config = sconfig.copy()
            config.dataset = dataset
            config.source = source
            config.target = target

            shift_summary, transfer_summary, out = run(config)

            all_shift_summary.append(shift_summary)
            all_transfer_summary.append(transfer_summary)

            dataset_to_shift[dataset].append(shift_summary)
            dataset_to_transfer[dataset].append(transfer_summary)



if __name__ == "__main__":
    config_path = Path("configs/ablation_configs/motivation.yaml")
    config = load_config(config_path)
    config.device = torch.device(config.device)

    timestamp = time.strftime("%d%H%M%S")
    config.out_dir = f"results/motivation/{timestamp}"

    scenarios = _build_scenarios(config.transfer_settings)
    run_all(scenarios, config)
    print(f"Results saved to {config.out_dir}")