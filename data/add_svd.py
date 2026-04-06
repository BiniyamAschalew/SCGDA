
import os
import os.path as osp
import argparse

import torch
import numpy as np

from Learn.Clean_SCGDA.utils.config_utils import load_config
from Learn.Clean_SCGDA.utils.data_utils.svd_transform import svd_transform



def add_svd_to_domain(data, processed_path, force=False):

    eival_path = processed_path + 'eival.pt'
    eivec_path = processed_path + 'eivec.pt'
    
    # Check if SVD files already exist
    if not force and os.path.exists(eival_path) and os.path.exists(eivec_path):
        print(f"  SVD files already exist at {processed_path}, skipping...")
        return False
    
    print(f"  Computing SVD for data at {processed_path}...")
    
    # Use svd_transform to compute and save SVD files
    svd_transform(data, processed_path)
    
    print(f"  Saved: {eival_path}")
    print(f"  Saved: {eivec_path}")
    
    return True


def add_svd_to_datasets(config, domains=None, force=False):

    if domains is None:
        domains = config["data"]["domains"]
    
    data_root = config["data"]["root"]
    processed_file = config["data"]["processed_file_names"][0]  # "data.pt"

    print(f"\nProcessing datasets in: {data_root}")
    print(f"Domains to process: {domains}")
    print(f"Force recompute: {force}\n")

    for domain in domains:
        processed_dir = osp.join(data_root, domain, "processed")
        processed_path = osp.join(processed_dir, processed_file)
        
        if not os.path.exists(processed_path):
            print(f"[{domain}] Processed data not found at {processed_path}, skipping...")
            continue
        
        print(f"[{domain}] Loading data from {processed_path}...")
        
        try:
            data, slices = torch.load(processed_path)
            add_svd_to_domain(data, processed_path, force=force)
        except Exception as e:
            print(f"[{domain}] Error processing: {e}")
            continue
    
    print("\nDone!")   

    
if __name__ == "__main__":
    # DATASET = "citation"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="citation",
                        help="Name of the dataset to process (e.g., citation, blog, airport, twitch, mag)")
    
    parser.add_argument("--force", action="store_true",
                        help="Force recomputation of SVD even if files exist")
    args = parser.parse_args()

    dataset = args.dataset.lower()
    data_config = load_config(f"./configs/data_configs/{dataset}.yaml")
    config = {"data": data_config}
    add_svd_to_datasets(config, force=False)

