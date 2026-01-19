import time

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import torch
torch.set_num_threads(4)

from data.build_dataset import build_dataset
from models.build_model import build_model
from utils.train_utils.metrics import BaseMetric


def run(config: dict):

    # print("\n=== Experiment Configuration ===")
    # print(f"Dataset: {config['data']['name']}, epochs: {config['expt']['epochs']}, model: {config['model']['name']}")

    device = config["expt"]["device"]
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    source_dataset = target_dataset = None
    source_data = target_data = None
    model = None
    logits = labels = None

    source_dataset, target_dataset = build_dataset(config)
    source_data = source_dataset[0].to(device)
    target_data = target_dataset[0].to(device)

    num_features = source_data.x.shape[1]
    num_classes = len(source_data.y.unique())

    # updating the config according to the dataset
    config["model"]["in_dim"] = num_features
    config["model"]["num_classes"] = num_classes
    # build and train the model
    model = build_model(config)
    start_time = time.time()
    model.fit(source_data, target_data)
    end_time = time.time()

    # evaluate the model
    metrics = BaseMetric(config)
    logits, labels = model.predict(target_data)
    result = metrics(logits, labels)

    result["source"] = config["expt"]["source"]
    result["target"] = config["expt"]["target"]
    result["model"] = config["model"]["name"]
    result["seed"] = config["expt"]["seed"]
    result["train_time"] = end_time - start_time

    # print(f"\n=== Result ===\n{result}\n")

    return result