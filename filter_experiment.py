import gc
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from data.build_dataset import build_dataset
from models.__filters.build_filter import build_filter
from models.build_model import build_model
from utils.config_utils import build_config, load_config
from utils.expt_utils import set_seed, to_valid_dir
from utils.train_utils.metrics import BaseMetric


BASELINES = {
    "gnn",
    "dane",
    "simgda",
    "grade",
    "a2gnn",
    "strurw",
    "dgsda",
    "specreg",
    "kbl",
    "pairalign",
}
MODELS = {
    0: "gnn",
    1: "dane",
    2: "simgda",
    3: "grade",
    4: "a2gnn",
    5: "strurw",
    6: "dgsda",
    7: "specreg",
    8: "simgda_role",
    9: "simgda_spectral",
    10: "structalign2",
    11: "mlp",
    12: "simmlp",
    13: "acdne",
    14: "asn",
    15: "adagcn",
    16: "dlit",
    17: "simgda_cheb",
    18: "scgda",
    19: "test",
    20: "bdlite",
    21: "kbl",
    22: "pairalign",
    23: "simgda_filter",
}

DATASETS = {0: "citation", 1: "blog", 2: "airport", 3: "twitch", 4: "mag"}

ROLE_TYPES = {0: "random_role", 1: "graphwave", 2: "signal_role"}

REPEATS = 1
role_type = ROLE_TYPES[2]

SEED = 0
DEVICE = "cuda:4"

USE_TUNED = 2
BORROW = "a2gnn"
USE_DEFAULT = False
WANDB = False
FROM_PYGDA = False

id = {
    "model": [23],
    "dataset": [0, 1],
    "source": [1],
    "target": [0],
}
notes = "filter_evaluation"


def _cuda_device_index(device):
    try:
        dev = torch.device(device)
    except Exception:
        return None
    if dev.type != "cuda":
        return None
    return dev.index if dev.index is not None else torch.cuda.current_device()


def _collect_cuda_memory(device):
    if not torch.cuda.is_available():
        return {}
    idx = _cuda_device_index(device)
    if idx is None:
        return {}
    mb = 1024.0 * 1024.0
    try:
        return {
            "cuda_device_index": idx,
            "cuda_allocated_mb": torch.cuda.memory_allocated(idx) / mb,
            "cuda_reserved_mb": torch.cuda.memory_reserved(idx) / mb,
            "cuda_max_allocated_mb": torch.cuda.max_memory_allocated(idx) / mb,
            "cuda_max_reserved_mb": torch.cuda.max_memory_reserved(idx) / mb,
        }
    except Exception:
        return {}


def _cleanup_cuda():
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _run_with_model(config: dict, from_pygda: bool = False):
    set_seed(config["expt"]["seed"])
    device = config["expt"]["device"]

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    cuda_idx = _cuda_device_index(device)
    if torch.cuda.is_available() and cuda_idx is not None:
        try:
            torch.cuda.reset_peak_memory_stats(cuda_idx)
        except Exception:
            pass

    source_data = target_data = None
    model = None
    start_time = time.time()

    try:
        source_dataset, target_dataset = build_dataset(config)
        source_data = source_dataset[0].to(device)
        target_data = target_dataset[0].to(device)

        if source_data.edge_index is not None:
            source_data.edge_index = source_data.edge_index.contiguous()
        if target_data.edge_index is not None:
            target_data.edge_index = target_data.edge_index.contiguous()

        config["model"]["in_dim"] = source_data.x.shape[1]
        config["model"]["num_classes"] = len(source_data.y.unique())

        model = build_model(config, from_pygda=from_pygda)
        model.fit(source_data, target_data)

        metrics = BaseMetric(config)
        logits, labels = model.predict(target_data)
        result = metrics(logits, labels)
        result["status"] = "ok"

    except torch.cuda.OutOfMemoryError as error:
        result = {metric: float("nan") for metric in config["expt"].get("metrics", [])}
        result.update(
            {
                "status": "oom",
                "error_type": "cuda_oom",
                "error": str(error),
            }
        )
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            result = {metric: float("nan") for metric in config["expt"].get("metrics", [])}
            result.update(
                {
                    "status": "oom",
                    "error_type": "cuda_oom",
                    "error": str(error),
                }
            )
        else:
            raise
    finally:
        elapsed = time.time() - start_time
        if "result" not in locals():
            result = {metric: float("nan") for metric in config["expt"].get("metrics", [])}
            result["status"] = "error"
        result["train_time"] = elapsed
        result["device"] = device
        result.update(_collect_cuda_memory(device))
        _cleanup_cuda()

    return result, model


def _infer_filter_type(config: dict) -> str:
    gnn_type = str(config["model"].get("gnn", "filter_mono")).lower()
    if gnn_type.startswith("filter_"):
        return gnn_type.split("_", 1)[1]
    return str(config["model"].get("filter_type", "mono")).lower()


def _to_polynomial(raw_params: torch.Tensor, filter_type: str) -> torch.Tensor:
    raw = torch.as_tensor(raw_params).detach().cpu().flatten()

    if filter_type == "mono":
        return raw

    filter_cls = build_filter(filter_type)
    if filter_type == "bern":
        return filter_cls.to_polynomial(raw, enforce_nonneg=False).detach().cpu().flatten()
    return filter_cls.to_polynomial(raw).detach().cpu().flatten()


def _list_to_str(values: torch.Tensor) -> str:
    flat = torch.as_tensor(values).flatten().tolist()
    return "[" + ", ".join(f"{float(v):.6f}" for v in flat) + "]"


def _save_filter_artifacts(model, config: dict, scenario_dir: Path):
    if model is None:
        return {}
    if not hasattr(model, "source_filter_param") or not hasattr(model, "target_filter_param"):
        return {}

    scenario_dir.mkdir(parents=True, exist_ok=True)
    filter_type = _infer_filter_type(config)

    source_raw = model.source_filter_param.detach().cpu().flatten()
    if hasattr(model, "_target_param"):
        target_raw = model._target_param().detach().cpu().flatten()
    else:
        target_raw = model.target_filter_param.detach().cpu().flatten()

    source_poly = _to_polynomial(source_raw, filter_type)
    target_poly = _to_polynomial(target_raw, filter_type)

    final_df = pd.DataFrame(
        [
            {
                "domain": "source",
                "filter_type": filter_type,
                "raw_coeffs": _list_to_str(source_raw),
                "poly_coeffs": _list_to_str(source_poly),
            },
            {
                "domain": "target",
                "filter_type": filter_type,
                "raw_coeffs": _list_to_str(target_raw),
                "poly_coeffs": _list_to_str(target_poly),
            },
        ]
    )
    final_path = scenario_dir / "learned_filter_polynomial.csv"
    final_df.to_csv(final_path, index=False)

    print(f"Source polynomial ({filter_type}): {_list_to_str(source_poly)}")
    print(f"Target polynomial ({filter_type}): {_list_to_str(target_poly)}")

    history = getattr(model, "filter_history", None)
    if history:
        rows = []
        target_delta_trace = []
        target_grad_trace = []
        for item in history:
            epoch = int(item["epoch"])
            src_raw = torch.as_tensor(item["source_raw"]).detach().cpu().flatten()
            tgt_raw = torch.as_tensor(item["target_raw"]).detach().cpu().flatten()
            src_poly = _to_polynomial(src_raw, filter_type)
            tgt_poly = _to_polynomial(tgt_raw, filter_type)
            if "target_delta_l2" in item:
                target_delta_trace.append(float(item["target_delta_l2"]))
            if "target_grad_norm" in item:
                target_grad_trace.append(float(item["target_grad_norm"]))

            for i, v in enumerate(src_raw.tolist()):
                rows.append({"epoch": epoch, "domain": "source", "space": "raw", "index": i, "value": float(v)})
            for i, v in enumerate(tgt_raw.tolist()):
                rows.append({"epoch": epoch, "domain": "target", "space": "raw", "index": i, "value": float(v)})
            for i, v in enumerate(src_poly.tolist()):
                rows.append({"epoch": epoch, "domain": "source", "space": "poly", "index": i, "value": float(v)})
            for i, v in enumerate(tgt_poly.tolist()):
                rows.append({"epoch": epoch, "domain": "target", "space": "poly", "index": i, "value": float(v)})

        hist_df = pd.DataFrame(rows)
        hist_path = scenario_dir / "filter_history.csv"
        hist_df.to_csv(hist_path, index=False)

        poly_df = hist_df[hist_df["space"] == "poly"]
        if not poly_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
            for ax, domain in zip(axes, ["source", "target"]):
                domain_df = poly_df[poly_df["domain"] == domain]
                for idx in sorted(domain_df["index"].unique()):
                    coeff_df = domain_df[domain_df["index"] == idx]
                    ax.plot(coeff_df["epoch"], coeff_df["value"], label=f"A^{idx}", linewidth=1.8)
                ax.set_title(f"{domain.capitalize()} Polynomial Coefficients")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Coefficient")
                ax.grid(True, alpha=0.2)
                ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig_path = scenario_dir / "filter_polynomial_evolution.png"
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)

        extra = {}
        if target_delta_trace:
            extra["target_delta_l2_max"] = float(max(target_delta_trace))
            extra["target_delta_l2_mean"] = float(sum(target_delta_trace) / len(target_delta_trace))
            extra["target_delta_l2_last"] = float(target_delta_trace[-1])
        if target_grad_trace:
            extra["target_grad_norm_mean"] = float(sum(target_grad_trace) / len(target_grad_trace))
            extra["target_grad_norm_last"] = float(target_grad_trace[-1])
    else:
        extra = {}

    out = {
        "filter_type": filter_type,
        "source_raw_coeffs": _list_to_str(source_raw),
        "target_raw_coeffs": _list_to_str(target_raw),
        "source_poly_coeffs": _list_to_str(source_poly),
        "target_poly_coeffs": _list_to_str(target_poly),
    }
    out.update(extra)
    return out


def main():
    combined_df = pd.DataFrame()
    cur_time = time.strftime("%m%d%H%M%S")
    analysis_root = Path(f"./__saved__/analysis/filter_experiment/{cur_time}_{notes}")
    analysis_root.mkdir(parents=True, exist_ok=True)

    for repeat in range(REPEATS):
        for mid in id["model"]:
            for did in id["dataset"]:
                model_name = MODELS[mid][:]
                dataset = DATASETS[did]

                data_config = load_config(f"./configs/data_configs/{dataset}.yaml")
                domains = data_config["domains"]

                src_ids = range(len(domains)) if len(id["source"]) == 0 else id["source"]
                tgt_ids = range(len(domains)) if len(id["target"]) == 0 else id["target"]

                for sid in src_ids:
                    for tid in tgt_ids:
                        if sid == tid:
                            continue
                        if len(domains) <= max(sid, tid):
                            raise ValueError(
                                f"selected domain id {sid} or {tid} exceeds available domains: len={len(domains)}"
                            )

                        source = domains[sid]
                        target = domains[tid]

                        config_setup = {"data": dataset, "expt": "default", "model": model_name}
                        update_config = {
                            "expt": {
                                "source": source,
                                "target": target,
                                "device": DEVICE,
                                "seed": SEED + repeat,
                                "wandb_enabled": WANDB,
                                "project": "SCGDA",
                            },
                            "model": {
                                "adv": False,
                                "role_type": role_type,
                            },
                        }
                        config = build_config(
                            config_setup,
                            update_config,
                            borrow=BORROW,
                            use_tuned=USE_TUNED,
                            use_default=USE_DEFAULT,
                        )

                        result, model = _run_with_model(config, from_pygda=FROM_PYGDA)

                        scenario_dir = analysis_root / dataset / f"{source}_{target}"
                        filter_info = _save_filter_artifacts(model, config, scenario_dir)

                        result["repeat"] = repeat
                        result["source"] = source
                        result["target"] = target
                        result["cur_time"] = time.strftime("%d%H%M")
                        result["use_tuned"] = USE_TUNED
                        result["model"] = model_name
                        result["dataset"] = dataset
                        result.update(filter_info)

                        result_df = pd.DataFrame([result])
                        combined_df = pd.concat([combined_df, result_df], ignore_index=True)

    result_dir = f"./__saved__/results/evaluation/{cur_time}_{notes}.csv"
    if os.path.exists(result_dir):
        result_dir = to_valid_dir(result_dir)
    combined_df.to_csv(result_dir, index=False)
    combined_df.to_csv(analysis_root / "combined_results.csv", index=False)

    if not combined_df.empty:
        print(combined_df.tail(1).to_dict(orient="records")[0])
    print(f"Combined results saved to {result_dir}")
    print(f"Filter artifacts saved to {analysis_root}")


if __name__ == "__main__":
    main()
