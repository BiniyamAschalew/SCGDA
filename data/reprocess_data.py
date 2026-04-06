"""Reprocess selected datasets with SpecReg (SVD pre_transform)."""

import os

from Learn.Clean_SCGDA.data.build_dataset import get_dataset
from Learn.Clean_SCGDA.utils.config_utils import build_config


# Edit this dictionary directly.
# value = None -> process all domains in that dataset config
# value = [domain1, domain2] -> process only selected domains
DATASETS = {

    # "airport": None,
    # "blog": None,
    # "citation": None,
    # "twitch": None,
    "mag": None,

}


if __name__ == "__main__":
    for dataset_name, selected_domains in DATASETS.items():
        config_setup = {
            "data": dataset_name,
            "expt": "default",
            "model": "specreg",
        }
        config = build_config(config_setup, update_config={"expt": {"verbose": 0}})

        domains = selected_domains or config["data"]["domains"]
        print(f"\n{dataset_name}: {domains}")

        for domain in domains:

            get_dataset(domain, config)
            print(f"  done -> {domain}")




