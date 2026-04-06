from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from  experiments.analysis.airport_domain_generalization_tracking.experiment import (
    _add_derived_metrics,
    _aggregate_histories,
    _build_experiment_config,
    _cleanup_cuda,
    _dirichlet_energy,
    _epoch_domain_metrics,
    _load_dataset_graphs,
    _write_summary,
    build_dataset_rollup,
)
from  experiments.analysis.airport_domain_generalization_tracking.plotting import (
    plot_target_quality_vs_performance,
    plot_training_overview,
)
from  models.build_model import build_model
from  utils.expt_utils import set_seed
from  utils.train_utils.mmd import MMD


DEFAULT_NUM_LAYERS = 3
DEFAULT_HID_DIM = 64
DEFAULT_DROPOUT = 0.1

MODEL_VARIANTS = {
    "source_only": {
        "label": "Source-Only GNN",
        "alignment_kind": "none",
        "alignment_weight": 0.0,
    },
    "multi_target_mmd": {
        "label": "Multi-Target MMD",
        "alignment_kind": "mmd",
        "alignment_weight": 0.1,
    },
    "multi_target_energy_match": {
        "label": "Multi-Target Energy Match",
        "alignment_kind": "energy",
        "alignment_weight": 0.01,
    },
}


def _torch_dirichlet_energy(data, features: torch.Tensor) -> torch.Tensor:
    edge_index = data.edge_index
    if int(edge_index.numel()) == 0:
        return torch.zeros((), device=features.device, dtype=features.dtype)

    row, col = edge_index[0], edge_index[1]
    diff = features[row] - features[col]
    edge_scores = diff.pow(2).sum(dim=1)
    edge_weight = getattr(data, "edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device=features.device, dtype=features.dtype)
        edge_scores = edge_scores * edge_weight
    return edge_scores.mean()


def _train_masked_nll(model, logits, data, *, use_mask: bool) -> torch.Tensor:
    train_mask = model.get_mask(data, use_mask=use_mask, mask_name="train_mask")
    if int(train_mask.sum().item()) <= 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return F.nll_loss(logits[train_mask], data.y[train_mask])


def _alignment_loss(
    model,
    *,
    alignment_kind: str,
    source_data,
    source_features: torch.Tensor,
    target_datas: list[object],
) -> torch.Tensor:
    if alignment_kind == "none" or not target_datas:
        return torch.zeros((), device=source_features.device, dtype=source_features.dtype)

    losses = []
    if alignment_kind == "mmd":
        for target_data in target_datas:
            target_features = model.gnn.feat_bottleneck(target_data.x, target_data.edge_index)
            losses.append(MMD(source_features, target_features))
    elif alignment_kind == "energy":
        source_energy = _torch_dirichlet_energy(source_data, source_features)
        for target_data in target_datas:
            target_features = model.gnn.feat_bottleneck(target_data.x, target_data.edge_index)
            target_energy = _torch_dirichlet_energy(target_data, target_features)
            losses.append(torch.abs(source_energy - target_energy))
    else:
        raise ValueError(f"Unknown alignment_kind '{alignment_kind}'.")

    return torch.stack(losses).mean()


def _train_variant_and_track(
    model,
    source_data,
    graphs: dict[str, object],
    *,
    source_domain: str,
    eval_domains: list[str],
    variant_name: str,
    alignment_kind: str,
    alignment_weight: float,
    use_mask: bool,
    seed: int,
):
    source_loader = model.get_loader(source_data)
    model.gnn = model.init_model(**model.kwargs)
    optimizer = torch.optim.Adam(
        model.gnn.parameters(),
        lr=model.lr,
        weight_decay=model.weight_decay,
    )

    target_domains = [domain for domain in eval_domains if domain != source_domain]
    target_datas = [graphs[domain] for domain in target_domains]

    rows = []
    start_time = time.time()

    from tqdm import tqdm

    for epoch in tqdm(range(model.epoch), desc=f"Training {variant_name}:{source_domain}"):
        epoch_loss = 0.0
        epoch_supervised_loss = 0.0
        epoch_alignment_loss = 0.0
        epoch_source_logits = None
        epoch_source_labels = None

        for sampled_source_data in source_loader:
            sampled_source_data = sampled_source_data.to(model.device)
            model.gnn.train()

            source_features = model.gnn.feat_bottleneck(
                sampled_source_data.x,
                sampled_source_data.edge_index,
            )
            source_logits = model.gnn.feat_classifier(source_features, sampled_source_data.edge_index)
            source_logits = F.log_softmax(source_logits, dim=1)

            supervised_loss = _train_masked_nll(
                model,
                source_logits,
                sampled_source_data,
                use_mask=use_mask,
            )
            aux_loss = _alignment_loss(
                model,
                alignment_kind=alignment_kind,
                source_data=sampled_source_data,
                source_features=source_features,
                target_datas=target_datas,
            )
            loss = supervised_loss + float(alignment_weight) * aux_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_supervised_loss += float(supervised_loss.item())
            epoch_alignment_loss += float(aux_loss.item())

            train_source_logits, train_source_labels, train_source_mask = model.mask_logits_and_labels(
                source_logits,
                sampled_source_data,
                use_mask=use_mask,
                mask_name="train_mask",
            )
            if int(train_source_mask.sum().item()) > 0:
                if epoch_source_logits is None:
                    epoch_source_logits = train_source_logits
                    epoch_source_labels = train_source_labels
                else:
                    epoch_source_logits = torch.cat((epoch_source_logits, train_source_logits))
                    epoch_source_labels = torch.cat((epoch_source_labels, train_source_labels))

        if epoch_source_logits is None or epoch_source_labels is None:
            raise RuntimeError("No source training nodes were available for loss/metric computation.")

        epoch_row = {
            "epoch": int(epoch + 1),
            "loss": float(epoch_loss),
            "supervised_loss": float(epoch_supervised_loss),
            "alignment_loss": float(epoch_alignment_loss),
            "train_micro_f1_source": float(model.metrics(epoch_source_logits, epoch_source_labels)["micro_f1"]),
        }
        epoch_row.update(
            _epoch_domain_metrics(
                model,
                graphs,
                source_domain=source_domain,
                eval_domains=eval_domains,
                epoch_index=epoch,
                seed=seed,
            )
        )
        rows.append(epoch_row)

    model.train_time = time.time() - start_time
    model.finish()
    return pd.DataFrame(rows)


def _build_variant_summary(
    out_dir: Path,
    *,
    config: dict,
    history_df: pd.DataFrame,
    std_df: pd.DataFrame | None,
    source_domain: str,
    target_domains: list[str],
    eval_domains: list[str],
    seed_list: list[int] | None,
    variant_name: str,
    variant_label: str,
    alignment_kind: str,
    alignment_weight: float,
):
    _write_summary(
        out_dir,
        config=config,
        history_df=history_df,
        std_df=std_df,
        source_domain=source_domain,
        target_domains=target_domains,
        eval_domains=eval_domains,
        seed_list=seed_list,
    )
    summary_path = out_dir / "summary.txt"
    lines = summary_path.read_text(encoding="utf-8").splitlines()
    insert_block = [
        f"Model variant: {variant_name}",
        f"Model label: {variant_label}",
        f"Alignment kind: {alignment_kind}",
        f"Alignment weight: {float(alignment_weight)}",
        "",
    ]
    lines = [lines[0], ""] + insert_block + lines[1:]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def run_variant_experiment(
    *,
    dataset: str,
    source: str,
    eval_domains: list[str] | None,
    variant_name: str,
    seed: int,
    device: str,
    epochs: int,
    verbose: int,
    num_layers: int,
    hid_dim: int,
    dropout_ratio: float,
    out_dir: str | None = None,
):
    if variant_name not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant '{variant_name}'.")

    set_seed(seed)
    variant_cfg = MODEL_VARIANTS[variant_name]

    config = _build_experiment_config(
        dataset=dataset,
        source=source,
        seed=seed,
        device=device,
        epochs=epochs,
        verbose=verbose,
        num_layers=num_layers,
        hid_dim=hid_dim,
        dropout_ratio=dropout_ratio,
    )
    config["expt"]["project"] = "multi_model_domain_generalization_tracking"

    graphs, metadata = _load_dataset_graphs(dataset, config, device=device)
    if eval_domains is None:
        eval_domains = list(metadata["domains"])
    eval_domains = list(eval_domains)
    target_domains = [domain for domain in eval_domains if domain != source]

    config["model"]["in_dim"] = metadata["num_features"]
    config["model"]["num_classes"] = metadata["num_classes"]

    if out_dir is None:
        raise ValueError("out_dir must be provided for run_variant_experiment.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    model = None
    try:
        model = build_model(config, from_pygda=False)
        history_df = _train_variant_and_track(
            model,
            graphs[source],
            graphs,
            source_domain=source,
            eval_domains=eval_domains,
            variant_name=variant_name,
            alignment_kind=variant_cfg["alignment_kind"],
            alignment_weight=float(variant_cfg["alignment_weight"]),
            use_mask=True,
            seed=seed,
        )
        history_df = _add_derived_metrics(history_df, eval_domains=eval_domains)
        history_df.to_csv(out_dir / "training_diagnostics.csv", index=False)

        overview_plot = plot_training_overview(
            history_df,
            out_dir=out_dir,
            source_domain=source,
            eval_domains=eval_domains,
        )
        target_quality_plot = plot_target_quality_vs_performance(
            history_df,
            out_dir=out_dir,
            source_domain=source,
            target_domains=target_domains,
        )

        metadata_json = {
            "config": config,
            "dataset_metadata": metadata,
            "eval_domains": eval_domains,
            "target_domains": target_domains,
            "source_domain": source,
            "model_variant": variant_name,
            "model_label": variant_cfg["label"],
            "alignment_kind": variant_cfg["alignment_kind"],
            "alignment_weight": float(variant_cfg["alignment_weight"]),
            "artifacts": {
                "history": "training_diagnostics.csv",
                "summary": "summary.txt",
                "training_overview_plot": Path(overview_plot).name,
                "target_quality_plot": None if target_quality_plot is None else Path(target_quality_plot).name,
            },
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata_json, indent=2), encoding="utf-8")
        _build_variant_summary(
            out_dir,
            config=config,
            history_df=history_df,
            std_df=None,
            source_domain=source,
            target_domains=target_domains,
            eval_domains=eval_domains,
            seed_list=None,
            variant_name=variant_name,
            variant_label=variant_cfg["label"],
            alignment_kind=variant_cfg["alignment_kind"],
            alignment_weight=float(variant_cfg["alignment_weight"]),
        )

        print(f"[saved] {out_dir / 'training_diagnostics.csv'}")
        print(f"[saved] {out_dir / 'summary.txt'}")
        print(f"[saved] {overview_plot}")
        if target_quality_plot is not None:
            print(f"[saved] {target_quality_plot}")
        print(f"results_dir={out_dir}")
        return out_dir
    finally:
        del model
        _cleanup_cuda()


def run_multi_seed_variant_experiment(
    *,
    dataset: str,
    source: str,
    eval_domains: list[str] | None,
    variant_name: str,
    seeds: list[int],
    device: str,
    epochs: int,
    verbose: int,
    num_layers: int,
    hid_dim: int,
    dropout_ratio: float,
    out_dir: str,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    variant_cfg = MODEL_VARIANTS[variant_name]
    seed_frames = []
    seed_result_dirs = []
    config_for_summary = None
    eval_domains = list(eval_domains) if eval_domains is not None else None

    try:
        for seed in seeds:
            seed_dir = out_dir / f"seed_{int(seed)}"
            seed_result_dir = run_variant_experiment(
                dataset=dataset,
                source=source,
                eval_domains=eval_domains,
                variant_name=variant_name,
                seed=int(seed),
                device=device,
                epochs=epochs,
                verbose=verbose,
                num_layers=num_layers,
                hid_dim=hid_dim,
                dropout_ratio=dropout_ratio,
                out_dir=str(seed_dir),
            )
            seed_history = pd.read_csv(Path(seed_result_dir) / "training_diagnostics.csv")
            seed_history["seed"] = int(seed)
            seed_frames.append(seed_history)
            seed_result_dirs.append(str(seed_result_dir))
            if config_for_summary is None:
                config_for_summary = json.loads((Path(seed_result_dir) / "metadata.json").read_text(encoding="utf-8"))["config"]
                if eval_domains is None:
                    eval_domains = json.loads((Path(seed_result_dir) / "metadata.json").read_text(encoding="utf-8"))["eval_domains"]

        all_df, mean_df, std_df = _aggregate_histories(seed_frames)
        all_df.to_csv(out_dir / "training_diagnostics_all_seeds.csv", index=False)
        mean_df.to_csv(out_dir / "training_diagnostics_mean.csv", index=False)
        std_df.to_csv(out_dir / "training_diagnostics_std.csv", index=False)

        target_domains = [domain for domain in eval_domains if domain != source]
        overview_plot = plot_training_overview(
            mean_df,
            out_dir=out_dir,
            source_domain=source,
            eval_domains=eval_domains,
            std_df=std_df,
        )
        target_quality_plot = plot_target_quality_vs_performance(
            mean_df,
            out_dir=out_dir,
            source_domain=source,
            target_domains=target_domains,
            std_df=std_df,
        )

        metadata_json = {
            "config": config_for_summary,
            "dataset": str(dataset).lower(),
            "source_domain": source,
            "eval_domains": eval_domains,
            "target_domains": target_domains,
            "seeds": list(seeds),
            "seed_result_dirs": seed_result_dirs,
            "model_variant": variant_name,
            "model_label": variant_cfg["label"],
            "alignment_kind": variant_cfg["alignment_kind"],
            "alignment_weight": float(variant_cfg["alignment_weight"]),
            "artifacts": {
                "all_seed_history": "training_diagnostics_all_seeds.csv",
                "mean_history": "training_diagnostics_mean.csv",
                "std_history": "training_diagnostics_std.csv",
                "summary": "summary.txt",
                "training_overview_plot": Path(overview_plot).name,
                "target_quality_plot": None if target_quality_plot is None else Path(target_quality_plot).name,
            },
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata_json, indent=2), encoding="utf-8")
        _build_variant_summary(
            out_dir,
            config=config_for_summary,
            history_df=mean_df,
            std_df=std_df,
            source_domain=source,
            target_domains=target_domains,
            eval_domains=eval_domains,
            seed_list=list(seeds),
            variant_name=variant_name,
            variant_label=variant_cfg["label"],
            alignment_kind=variant_cfg["alignment_kind"],
            alignment_weight=float(variant_cfg["alignment_weight"]),
        )

        print(f"[saved] {out_dir / 'training_diagnostics_all_seeds.csv'}")
        print(f"[saved] {out_dir / 'training_diagnostics_mean.csv'}")
        print(f"[saved] {out_dir / 'training_diagnostics_std.csv'}")
        print(f"[saved] {overview_plot}")
        if target_quality_plot is not None:
            print(f"[saved] {target_quality_plot}")
        print(f"[saved] {out_dir / 'summary.txt'}")
        print(f"results_dir={out_dir}")
        return out_dir
    finally:
        _cleanup_cuda()


def build_variant_dataset_rollup(
    *,
    dataset: str,
    dataset_out_dir: str | Path,
    variant_name: str,
    case_dirs: list[str | Path],
):
    dataset_out_dir = Path(dataset_out_dir)
    build_dataset_rollup(
        dataset=dataset,
        dataset_out_dir=dataset_out_dir,
        case_dirs=case_dirs,
    )

    summary_path = dataset_out_dir / "summary.txt"
    lines = summary_path.read_text(encoding="utf-8").splitlines()
    variant_cfg = MODEL_VARIANTS[variant_name]
    insert_block = [
        f"Model variant: {variant_name}",
        f"Model label: {variant_cfg['label']}",
        f"Alignment kind: {variant_cfg['alignment_kind']}",
        f"Alignment weight: {float(variant_cfg['alignment_weight'])}",
        "",
    ]
    lines = [lines[0], ""] + insert_block + lines[1:]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
