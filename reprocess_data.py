"""Applying the svd transform preprocessing to allow spec reg to run on the datasets (creates eval and evec embeddings)"""

import os.path as osp
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import torch
torch.set_num_threads(4)

from pygda.datasets import CitationDataset, TwitchDataset
from pygda.datasets import AirportDataset, MAGDataset, BlogDataset
from pygda.utils import svd_transform

root_path = "/data/biniyam/datasets/pygda_data/raw/"

dataset_domain = {
    # "Airport": ["BRAZIL", "EUROPE", "USA"],
    # "Blog": ["Blog1", "Blog2"],
    # "Citation": ["ACMv9", "Citationv1", "DBLPv7"],
    "MAG": ["MAG_FR", "MAG_CN", "MAG_DE", "MAG_RU", "MAG_JP", "MAG_US"],
    # "Twitch": ["DE", "EN", "ES", "FR", "PT", "RU"],
    # "Twitch": ["ES", "FR", "PT", "RU"],
}

dataset_class = {
    "Airport": AirportDataset,
    "Blog": BlogDataset,
    "Citation": CitationDataset,
    "MAG": MAGDataset,
    "Twitch": TwitchDataset,
}

for dataset, domain_list in dataset_domain.items():
    print(f"Processing dataset: {dataset}")

    for domain in domain_list:
        print(f"  Processing domain: {domain}")
        DatasetClass = dataset_class[dataset]

        domain_root = osp.join(root_path, dataset, domain)

        data = DatasetClass(
            root=domain_root, 
            name=domain,
            pre_transform=svd_transform,
            force_reload=True,
            )

        data.process()
        print(f"    Processed data saved at: {data.processed_dir}")


