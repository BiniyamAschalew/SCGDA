import time
import gc

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
from utils.pygda_utils import build_pygda_model
from utils.train_utils.metrics import BaseMetric
from utils.expt_utils import set_seed

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _build_error_result(config: dict, error: Exception, error_type: str, stage: str, elapsed: float):
    result = {
        "status": "error",
        "error_type": error_type,
        "error_stage": stage,
        "error": str(error),
        "source": config["expt"].get("source"),
        "target": config["expt"].get("target"),
        "model": config["model"].get("name"),
        "seed": config["expt"].get("seed"),
        "train_time": elapsed,
        "device": config["expt"].get("device"),
    }

    for metric in config["expt"].get("metrics", []):
        result[metric] = float("nan")

    return result


def _cleanup_cuda():
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def run(config: dict, from_pygda: bool = False) -> dict:

    # print("\n=== Experiment Configuration ===")
    # print(f"Dataset: {config['data']['name']}, epochs: {config['expt']['epochs']}, model: {config['model']['name']}")

    set_seed(config["expt"]["seed"])
    device = config["expt"]["device"]
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    source_dataset = target_dataset = None
    source_data = target_data = None
    model = None
    logits = labels = None

    stage = "init"
    start_time = time.time()
    try:
        stage = "load_data"
        source_dataset, target_dataset = build_dataset(config)
        source_data = source_dataset[0].to(device)
        target_data = target_dataset[0].to(device)

        # Some pyg sampling ops require contiguous edge indices.
        if source_data.edge_index is not None:
            source_data.edge_index = source_data.edge_index.contiguous()
        if target_data.edge_index is not None:
            target_data.edge_index = target_data.edge_index.contiguous()

        num_features = source_data.x.shape[1]
        num_classes = len(source_data.y.unique())

        # updating the config according to the dataset
        config["model"]["in_dim"] = num_features
        config["model"]["num_classes"] = num_classes

        # build and train the model
        stage = "build_model"
        model = build_model(config, from_pygda=from_pygda)

        # if the model is from pygda module
        # if from_pygda:
        #     model = build_pygda_model(config)

        stage = "fit"
        model.fit(source_data, target_data)
        end_time = time.time()

        # evaluate the model
        stage = "predict"
        metrics = BaseMetric(config)
        logits, labels = model.predict(target_data)
        result = metrics(logits, labels)

        result["source"] = config["expt"]["source"]
        result["target"] = config["expt"]["target"]
        result["model"] = config["model"]["name"]
        result["seed"] = config["expt"]["seed"]
        result["train_time"] = end_time - start_time
        result["status"] = "ok"

        return result
    except torch.cuda.OutOfMemoryError as error:
        elapsed = time.time() - start_time
        result = _build_error_result(config, error, "cuda_oom", stage, elapsed)
        result["status"] = "oom"
        return result
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            elapsed = time.time() - start_time
            result = _build_error_result(config, error, "cuda_oom", stage, elapsed)
            result["status"] = "oom"
            return result
        raise
    finally:
        _cleanup_cuda()
