from __future__ import annotations

import gc
import json
from pathlib import Path
import time

import pandas as pd
import torch

from Learn.Clean_SCGDA.data.build_dataset import build_dataset
from Learn.Clean_SCGDA.models.build_model import build_model
from Learn.Clean_SCGDA.utils.config_utils import build_config
from Learn.Clean_SCGDA.utils.expt_utils import set_seed

from Learn.Clean_SCGDA.experiments.analysis.plain_gnn_cross_domain.log_plot import make_output_dir


DEFAULT_NUM_LAYERS = 2
DEFAULT_MMD_WEIGHT = 0.1


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _build_experiment_config(
    *,
    dataset: str,
    source: str,
    target: str,
    seed: int,
    device: str,
    epochs: int,
    verbose: int,
    num_layers: int,
    mmd_weight: float,
):
    config_setup = {
        "model": "tracked_simgda",
        "data": dataset,
        "expt": "default",
    }
    update_config = {
        "expt": {
            "source": source,
            "target": target,
            "device": device,
            "seed": seed,
            "wandb_enabled": False,
            "project": "simgda_training_tracking",
            "verbose": verbose,
            "metrics": ["micro_f1"],
        },
        "model": {
            "epochs": epochs,
            "num_layers": num_layers,
            "mmd_weight": float(mmd_weight),
            "use_mask": True,
        },
    }
    config = build_config(config_setup, update_config=update_config, use_tuned=0)
    config["model"]["epochs"] = epochs
    config["model"]["num_layers"] = num_layers
    config["model"]["mmd_weight"] = float(mmd_weight)
    config["model"]["use_mask"] = True
    config["expt"]["metrics"] = ["micro_f1"]
    return config


def _load_pair(config: dict):
    source_dataset, target_dataset = build_dataset(config)
    source_data = source_dataset[0].to(config["expt"]["device"])
    target_data = target_dataset[0].to(config["expt"]["device"])

    if source_data.edge_index is not None:
        source_data.edge_index = source_data.edge_index.contiguous()
    if target_data.edge_index is not None:
        target_data.edge_index = target_data.edge_index.contiguous()

    num_features = int(source_data.x.size(1))
    num_classes = int(torch.unique(torch.cat([source_data.y, target_data.y])).numel())
    config["model"]["in_dim"] = num_features
    config["model"]["num_classes"] = num_classes
    return source_data, target_data


def _write_summary(
    out_dir: Path,
    *,
    config: dict,
    history_df: pd.DataFrame,
):
    final_row = history_df.iloc[-1]
    best_target_row = history_df.iloc[int(history_df["target_micro_f1"].idxmax())]
    best_source_target_row = history_df.iloc[int(history_df["source_micro_f1"].idxmax())]

    lines = [
        "Tracked SimGDA training diagnostics",
        "",
        f"Dataset: {config['data']['name']}",
        f"Source -> Target: {config['expt']['source']} -> {config['expt']['target']}",
        f"Device: {config['expt']['device']}",
        f"Seed: {config['expt']['seed']}",
        f"Epochs: {config['model']['epochs']}",
        f"GNN layers: {config['model']['num_layers']}",
        f"MMD weight: {float(config['model']['mmd_weight']):.4f}",
        "",
        "Training protocol:",
        "- The model is a GNN backbone with source supervised loss plus source-target MMD alignment.",
        "- Source labels are used for the classification loss.",
        "- Target labels are never used in the training loss.",
        "- Source and target micro-F1 are evaluated on the validation masks only.",
        "- The MMD columns use full-graph bottleneck features.",
        "- Source-source and target-target MMD are computed by randomly splitting that domain in half each epoch with deterministic epoch-dependent seeds.",
        "",
        "Final epoch:",
        f"- epoch: {int(final_row['epoch'])}",
        f"- loss: {float(final_row['loss']):.4f}",
        f"- source_micro_f1: {float(final_row['source_micro_f1']):.4f}",
        f"- target_micro_f1: {float(final_row['target_micro_f1']):.4f}",
        f"- source_target_mmd: {float(final_row['source_target_mmd']):.4f}",
        f"- source_source_mmd: {float(final_row['source_source_mmd']):.4f}",
        f"- target_target_mmd: {float(final_row['target_target_mmd']):.4f}",
        "",
        "Best target epoch:",
        f"- epoch: {int(best_target_row['epoch'])}",
        f"- target_micro_f1: {float(best_target_row['target_micro_f1']):.4f}",
        f"- source_micro_f1: {float(best_target_row['source_micro_f1']):.4f}",
        f"- source_target_mmd: {float(best_target_row['source_target_mmd']):.4f}",
        "",
        "Best source epoch:",
        f"- epoch: {int(best_source_target_row['epoch'])}",
        f"- source_micro_f1: {float(best_source_target_row['source_micro_f1']):.4f}",
        f"- target_micro_f1: {float(best_source_target_row['target_micro_f1']):.4f}",
        f"- source_target_mmd: {float(best_source_target_row['source_target_mmd']):.4f}",
        "",
        "Diagnostics table columns:",
        "- epoch: 1-based training epoch.",
        "- loss: total optimization loss accumulated across the epoch.",
        "- source_micro_f1: source micro-F1 on the source val mask.",
        "- target_micro_f1: target micro-F1 on the target val mask.",
        "- source_target_mmd: MMD between all source and target bottleneck features.",
        "- source_source_mmd: MMD between two random halves of the source bottleneck features.",
        "- target_target_mmd: MMD between two random halves of the target bottleneck features.",
    ]

    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    *,
    dataset: str = "airport",
    source: str = "USA",
    target: str = "BRAZIL",
    seed: int = 0,
    device: str = "cpu",
    epochs: int = 200,
    verbose: int = 0,
    num_layers: int = DEFAULT_NUM_LAYERS,
    mmd_weight: float = DEFAULT_MMD_WEIGHT,
    out_root: str = "./__saved__/analysis/simgda_training_tracking",
):
    start_time = time.time()
    set_seed(seed)

    config = _build_experiment_config(
        dataset=dataset,
        source=source,
        target=target,
        seed=seed,
        device=device,
        epochs=epochs,
        verbose=verbose,
        num_layers=num_layers,
        mmd_weight=mmd_weight,
    )

    dataset_out_root = Path(out_root) / str(dataset).lower()
    out_dir = make_output_dir(
        out_root=str(dataset_out_root),
        run_name=(
            f"simgda_tracking_{str(dataset).lower()}_{source}_{target}"
            f"_layers{num_layers}_mmd{str(mmd_weight).replace('.', 'p')}_seed{seed}_{time.strftime('%m%d_%H%M%S')}"
        ),
    )

    model = None
    try:
        source_data, target_data = _load_pair(config)
        model = build_model(config, from_pygda=False)
        model.fit(source_data, target_data, use_mask=True)

        history_df = pd.DataFrame(model.history_rows)
        if history_df.empty:
            raise RuntimeError("TrackedSimGDA did not record any epoch history.")

        history_df.to_csv(out_dir / "training_diagnostics.csv", index=False)

        metadata = {
            "config": config,
            "elapsed_seconds": float(time.time() - start_time),
            "artifacts": {
                "history": "training_diagnostics.csv",
                "summary": "summary.txt",
            },
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _write_summary(out_dir, config=config, history_df=history_df)

        print(f"[saved] {out_dir / 'training_diagnostics.csv'}")
        print(f"[saved] {out_dir / 'summary.txt'}")
        print(f"results_dir={out_dir}")
        return out_dir
    finally:
        del model
        _cleanup_cuda()
