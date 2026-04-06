"""Projection-before-message-passing GNN comparison for transfer and held-out-domain generalization.

This script keeps the experiment self-contained in a single file:

1. `ProjectionMMDGNNBase` defines the backbone.
   - Two linear layers with a ReLU in between project raw features to `hid_dim`.
   - The projected features then enter the usual message-passing stack.
   - A small auxiliary classifier on the projection output is available for the
     V4 loss.

2. `ProjectionMMDGNN` defines the trainer.
   - The final source classification loss is always active.
   - The four variants are selected only by loss weights:
       V1: final CE only
       V2: final CE + final-feature MMD
       V3: final CE + projection-feature MMD
       V4: final CE + projection-feature MMD + projection-level CE

3. Pair-specific tuned `gnn` hyperparameters are reused through
   `build_config(..., use_tuned=2)`.
   The adaptation is architectural rather than re-tuning: the tuned hidden
   size, optimizer settings, dropout, activation, and message-passing depth are
   preserved, while the old input-to-hidden graph layer is replaced by the
   requested two-layer projection MLP.

4. For every ordered source->target pair, the trained model is evaluated on all
   domains of the dataset. Results include `delta_micro_f1_vs_v1`, so the
   target-domain and held-out-domain changes can be compared directly against
   the source-only baseline under the same transfer setup and repetition.

Default usage focuses on Citation and averages over 3 repetitions:

    python experiments/analysis/projection_mmd_transfer_variants.py

The script also works for other repo datasets with domain lists in
`configs/data_configs/*.yaml`.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.transforms import OneHotDegree

torch.set_num_threads(4)

from Learn.Clean_SCGDA.data.build_dataset import get_dataset, get_max_degree
from Learn.Clean_SCGDA.experiments.analysis.plain_gnn_cross_domain.log_plot import make_output_dir
from Learn.Clean_SCGDA.models.__layers.build_layer import build_activation, build_layer
from Learn.Clean_SCGDA.utils.config_utils import build_config, load_config
from Learn.Clean_SCGDA.utils.expt_utils import set_seed
from Learn.Clean_SCGDA.utils.train_utils.metrics import BaseMetric
from Learn.Clean_SCGDA.utils.train_utils.mmd import MMD


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message="The `pre_transform` argument differs from the one used in the pre-processed version of this dataset.*",
    category=UserWarning,
)


class ProjectionMMDGNNBase(nn.Module):

    def __init__(self, config: dict):
        super(ProjectionMMDGNNBase, self).__init__()

        model_cfg = config["model"]
        self.in_dim = int(model_cfg["in_dim"])
        self.hid_dim = int(model_cfg["hid_dim"])
        self.num_classes = int(model_cfg["num_classes"])
        self.num_layers = int(model_cfg["num_layers"])
        self.dropout = float(model_cfg["dropout_ratio"])
        self.gnn_type = model_cfg["gnn"]
        self.mode = model_cfg["mode"]

        self.act = build_activation(model_cfg["activation"])
        self.proj1 = nn.Linear(self.in_dim, self.hid_dim)
        self.proj2 = nn.Linear(self.hid_dim, self.hid_dim)
        self.proj_cls = nn.Linear(self.hid_dim, self.num_classes)

        self.convs = nn.ModuleList(
            [build_layer(self.hid_dim, self.hid_dim, self.gnn_type) for _ in range(max(1, self.num_layers))]
        )

        if self.mode == "node":
            self.cls = build_layer(self.hid_dim, self.num_classes, self.gnn_type)
        else:
            self.cls = nn.Linear(self.hid_dim, self.num_classes)

    def project_features(self, x):
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        return x

    def _pool_if_needed(self, x, batch):
        if self.mode == "graph":
            return global_mean_pool(x, batch)
        return x

    def feat_bottleneck(self, x, edge_index, edge_weight=None, batch=None, return_projection=False):
        projection = self.project_features(x)
        hidden = projection

        for idx, conv in enumerate(self.convs):
            hidden = conv(hidden, edge_index, edge_weight)
            if idx < len(self.convs) - 1:
                hidden = self.act(hidden)
                hidden = F.dropout(hidden, p=self.dropout, training=self.training)

        projection_out = self._pool_if_needed(projection, batch)
        hidden_out = self._pool_if_needed(hidden, batch)

        if return_projection:
            return hidden_out, projection_out
        return hidden_out

    def feat_classifier(self, x, edge_index, edge_weight=None):
        if self.mode == "node":
            return self.cls(x, edge_index, edge_weight)
        return self.cls(x)

    def projection_classifier(self, x):
        return self.proj_cls(x)

    def forward(self, x, edge_index, edge_weight=None, batch=None, return_all=False):
        final_features, projection_features = self.feat_bottleneck(
            x,
            edge_index,
            edge_weight=edge_weight,
            batch=batch,
            return_projection=True,
        )
        final_logits = F.log_softmax(
            self.feat_classifier(final_features, edge_index, edge_weight),
            dim=1,
        )
        projection_logits = F.log_softmax(self.projection_classifier(projection_features), dim=1)

        if return_all:
            return {
                "final_logits": final_logits,
                "projection_logits": projection_logits,
                "final_features": final_features,
                "projection_features": projection_features,
            }
        return final_logits


class ProjectionMMDGNN:

    def __init__(
        self,
        config: dict,
        *,
        variant_key: str,
        variant_label: str,
        final_mmd_weight: float,
        projection_mmd_weight: float,
        projection_cls_weight: float,
        mmd_sampling_num: int,
        mmd_times: int,
    ):
        self.config = config
        self.device = config["expt"]["device"]
        self.lr = float(config["model"]["lr"])
        self.weight_decay = float(config["model"]["weight_decay"])
        self.epochs = int(config["model"]["epochs"])
        self.use_mask = bool(config["model"]["use_mask"])

        self.variant_key = variant_key
        self.variant_label = variant_label
        self.final_mmd_weight = float(final_mmd_weight)
        self.projection_mmd_weight = float(projection_mmd_weight)
        self.projection_cls_weight = float(projection_cls_weight)
        self.mmd_sampling_num = int(mmd_sampling_num)
        self.mmd_times = int(mmd_times)

        self.metric = BaseMetric(config)
        self.model = ProjectionMMDGNNBase(config).to(self.device)
        self.train_time = 0.0
        self.final_loss = 0.0

    def _mask(self, data, mask_name: str):
        if self.config["model"]["mode"] != "node":
            return torch.ones(data.y.size(0), dtype=torch.bool, device=data.y.device)
        if not self.use_mask:
            return torch.ones_like(data.y, dtype=torch.bool, device=data.y.device)
        mask = getattr(data, mask_name, None)
        if mask is None:
            return torch.ones_like(data.y, dtype=torch.bool, device=data.y.device)
        return mask.bool()

    def _masked_nll(self, logits, labels, mask):
        if int(mask.sum().item()) <= 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        return F.nll_loss(logits[mask], labels[mask])

    def _masked_mmd(self, source_feat, target_feat, source_mask, target_mask):
        if int(source_mask.sum().item()) <= 0 or int(target_mask.sum().item()) <= 0:
            return torch.zeros((), device=source_feat.device, dtype=source_feat.dtype)
        return MMD(
            source_feat[source_mask],
            target_feat[target_mask],
            sampling_num=self.mmd_sampling_num,
            times=self.mmd_times,
        )

    def fit(self, source_data, target_data):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        source_train_mask = self._mask(source_data, "train_mask")
        target_train_mask = self._mask(target_data, "train_mask")
        start = time.time()

        for _ in range(self.epochs):
            self.model.train()

            source_out = self.model(
                source_data.x,
                source_data.edge_index,
                edge_weight=getattr(source_data, "edge_weight", None),
                batch=getattr(source_data, "batch", None),
                return_all=True,
            )
            target_out = self.model(
                target_data.x,
                target_data.edge_index,
                edge_weight=getattr(target_data, "edge_weight", None),
                batch=getattr(target_data, "batch", None),
                return_all=True,
            )

            loss = self._masked_nll(
                source_out["final_logits"],
                source_data.y,
                source_train_mask,
            )

            if self.final_mmd_weight != 0.0:
                loss = loss + self.final_mmd_weight * self._masked_mmd(
                    source_out["final_features"],
                    target_out["final_features"],
                    source_train_mask,
                    target_train_mask,
                )

            if self.projection_mmd_weight != 0.0:
                loss = loss + self.projection_mmd_weight * self._masked_mmd(
                    source_out["projection_features"],
                    target_out["projection_features"],
                    source_train_mask,
                    target_train_mask,
                )

            if self.projection_cls_weight != 0.0:
                loss = loss + self.projection_cls_weight * self._masked_nll(
                    source_out["projection_logits"],
                    source_data.y,
                    source_train_mask,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.final_loss = float(loss.detach().cpu().item())

        self.train_time = time.time() - start

    def predict(self, data, mask_name: str = "val_mask"):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                data.x,
                data.edge_index,
                edge_weight=getattr(data, "edge_weight", None),
                batch=getattr(data, "batch", None),
            )

        mask = self._mask(data, mask_name)
        return logits[mask], data.y[mask]

    def evaluate(self, data, mask_name: str = "val_mask"):
        logits, labels = self.predict(data, mask_name=mask_name)
        if int(labels.numel()) <= 0:
            return {"micro_f1": float("nan"), "macro_f1": float("nan")}
        return self.metric(logits, labels)


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _load_dataset_graphs(dataset_name: str, device: str):
    data_cfg = load_config(f"./configs/data_configs/{dataset_name}.yaml")
    config = {
        "data": {
            "name": data_cfg["name"],
            "root": data_cfg["root"],
            "domains": list(data_cfg["domains"]),
            "raw_file_names": list(data_cfg["raw_file_names"]),
            "processed_file_names": list(data_cfg["processed_file_names"]),
            "train_split": float(data_cfg["train_split"]),
            "val_split": float(data_cfg["val_split"]),
            "test_split": float(data_cfg["test_split"]),
            "metric": data_cfg["metric"],
        },
        "model": {"name": "gnn"},
        "expt": {"verbose": 0},
    }

    graphs = {}
    max_degree = 0
    if config["data"]["name"].lower() == "airport":
        max_degree = int(get_max_degree(config))

    for domain in config["data"]["domains"]:
        dataset = get_dataset(domain, config)
        if config["data"]["name"].lower() == "airport":
            dataset.transform = OneHotDegree(max_degree)
        data = dataset[0].to(device)
        if data.edge_index is not None:
            data.edge_index = data.edge_index.contiguous()
        graphs[domain] = data

    labels = torch.cat([graph.y for graph in graphs.values()])
    return graphs, {
        "domains": list(config["data"]["domains"]),
        "num_features": int(next(iter(graphs.values())).x.size(1)),
        "num_classes": int(torch.unique(labels).numel()),
        "max_degree": max_degree,
    }


def _build_adapted_config(
    *,
    dataset: str,
    source: str,
    target: str,
    seed: int,
    device: str,
    epochs: int | None,
    verbose: int,
    use_tuned: int,
):
    config_setup = {
        "model": "gnn",
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
            "project": "projection_mmd_transfer_variants",
            "verbose": verbose,
            "metrics": ["micro_f1", "macro_f1"],
        },
        "model": {
            "use_mask": True,
        },
    }

    config = build_config(
        config_setup,
        update_config=update_config,
        use_tuned=use_tuned,
    )

    config["model"]["name"] = "ProjectionMMDGNN"
    config["model"]["use_mask"] = True
    config["expt"]["metrics"] = ["micro_f1", "macro_f1"]
    if epochs is not None:
        config["model"]["epochs"] = int(epochs)
    return config


def _variant_specs(mmd_weight: float, projection_cls_weight: float):
    return (
        {
            "key": "v1_source_only",
            "label": "V1 Source Only",
            "final_mmd_weight": 0.0,
            "projection_mmd_weight": 0.0,
            "projection_cls_weight": 0.0,
        },
        {
            "key": "v2_final_mmd",
            "label": "V2 Final MMD",
            "final_mmd_weight": float(mmd_weight),
            "projection_mmd_weight": 0.0,
            "projection_cls_weight": 0.0,
        },
        {
            "key": "v3_projection_mmd",
            "label": "V3 Projection MMD",
            "final_mmd_weight": 0.0,
            "projection_mmd_weight": float(mmd_weight),
            "projection_cls_weight": 0.0,
        },
        {
            "key": "v4_projection_mmd_cls",
            "label": "V4 Projection MMD + CE",
            "final_mmd_weight": 0.0,
            "projection_mmd_weight": float(mmd_weight),
            "projection_cls_weight": float(projection_cls_weight),
        },
    )


def _evaluation_role(eval_domain: str, source: str, target: str):
    if eval_domain == source:
        return "source"
    if eval_domain == target:
        return "target"
    return "other"


def _flatten_summary(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]):
    summary = df.groupby(group_cols, as_index=False, sort=False)[value_cols].agg(["mean", "std"])
    summary.columns = [
        "_".join([part for part in col if part]).rstrip("_")
        for col in summary.columns.to_flat_index()
    ]
    return summary


def _build_summary_text(
    *,
    dataset: str,
    domains: list[str],
    repeats: int,
    seed_start: int,
    use_tuned: int,
    epochs: int,
    summary_eval: pd.DataFrame,
):
    lines = [
        "Projection MMD Transfer Variant Comparison",
        f"Dataset: {dataset}",
        f"Domains: {domains}",
        f"Repeats: {repeats}",
        f"Seed start: {seed_start}",
        f"Use tuned GNN config: {use_tuned}",
        f"Epochs: {epochs}",
        "",
        "Variant losses:",
        "- V1: final CE",
        "- V2: final CE + final-feature MMD",
        "- V3: final CE + projection-feature MMD",
        "- V4: final CE + projection-feature MMD + projection CE",
        "",
        "Mean delta micro-F1 vs V1 by transfer pair and evaluation domain:",
    ]

    for (source, target), pair_df in summary_eval.groupby(["source", "target"], sort=False):
        lines.append(f"{source} -> {target}")
        for eval_domain in domains:
            domain_df = pair_df[pair_df["evaluation_domain"] == eval_domain]
            if domain_df.empty:
                continue
            domain_role = domain_df["evaluation_role"].iloc[0]
            pieces = []
            for variant_key in ("v2_final_mmd", "v3_projection_mmd", "v4_projection_mmd_cls"):
                cur = domain_df[domain_df["variant_key"] == variant_key]
                if cur.empty:
                    continue
                pieces.append(
                    f"{cur['variant_label'].iloc[0]}={float(cur['delta_micro_f1_vs_v1_mean'].iloc[0]):+.4f}"
                )
            if pieces:
                lines.append(f"  {eval_domain} ({domain_role}): " + ", ".join(pieces))

    return "\n".join(lines) + "\n"


def run_experiment(
    *,
    dataset: str = "citation",
    repeats: int = 3,
    seed_start: int = 0,
    epochs: int | None = None,
    use_tuned: int = 2,
    mmd_weight: float | None = None,
    projection_cls_weight: float = 1.0,
    mmd_sampling_num: int = 256,
    mmd_times: int = 2,
    device: str = "cpu",
    verbose: int = 1,
    out_root: str = "./__saved__/analysis/projection_mmd_transfer_variants",
):
    graphs, metadata = _load_dataset_graphs(dataset, device=device)
    domains = metadata["domains"]
    ordered_pairs = [(src, tgt) for src in domains for tgt in domains if src != tgt]

    run_name = (
        f"{dataset}_projection_mmd_variants_r{repeats}_"
        f"{device.replace(':', '')}_{time.strftime('%m%d_%H%M%S')}"
    )
    out_dir = make_output_dir(out_root=out_root, run_name=run_name)

    rows = []
    total_runs = int(repeats) * len(ordered_pairs) * 4
    run_idx = 0
    effective_epochs = None

    metadata_payload = {
        "dataset": dataset,
        "domains": domains,
        "repeats": int(repeats),
        "seed_start": int(seed_start),
        "epochs_override": None if epochs is None else int(epochs),
        "use_tuned": int(use_tuned),
        "projection_cls_weight": float(projection_cls_weight),
        "mmd_sampling_num": int(mmd_sampling_num),
        "mmd_times": int(mmd_times),
        "device": device,
        "num_features": metadata["num_features"],
        "num_classes": metadata["num_classes"],
    }

    for repeat in range(int(repeats)):
        seed = int(seed_start) + int(repeat)
        for source, target in ordered_pairs:
            set_seed(seed)
            pair_config = _build_adapted_config(
                dataset=dataset,
                source=source,
                target=target,
                seed=seed,
                device=device,
                epochs=epochs,
                verbose=0,
                use_tuned=use_tuned,
            )
            pair_config["model"]["in_dim"] = metadata["num_features"]
            pair_config["model"]["num_classes"] = metadata["num_classes"]
            effective_epochs = int(pair_config["model"]["epochs"])

            pair_mmd_weight = (
                float(pair_config["model"].get("mmd_weight", 0.1))
                if mmd_weight is None
                else float(mmd_weight)
            )

            for spec in _variant_specs(pair_mmd_weight, projection_cls_weight):
                run_idx += 1
                variant_config = copy.deepcopy(pair_config)
                variant_config["model"]["name"] = spec["key"]

                trainer = ProjectionMMDGNN(
                    variant_config,
                    variant_key=spec["key"],
                    variant_label=spec["label"],
                    final_mmd_weight=spec["final_mmd_weight"],
                    projection_mmd_weight=spec["projection_mmd_weight"],
                    projection_cls_weight=spec["projection_cls_weight"],
                    mmd_sampling_num=mmd_sampling_num,
                    mmd_times=mmd_times,
                )
                trainer.fit(graphs[source], graphs[target])

                current_rows = []
                for eval_domain in domains:
                    metrics = trainer.evaluate(graphs[eval_domain], mask_name="val_mask")
                    row = {
                        "dataset": dataset,
                        "repeat": int(repeat),
                        "seed": int(seed),
                        "source": source,
                        "target": target,
                        "evaluation_domain": eval_domain,
                        "evaluation_role": _evaluation_role(eval_domain, source, target),
                        "variant_key": spec["key"],
                        "variant_label": spec["label"],
                        "micro_f1": float(metrics["micro_f1"]),
                        "macro_f1": float(metrics["macro_f1"]),
                        "train_time_sec": float(trainer.train_time),
                        "final_loss": float(trainer.final_loss),
                        "adapted_tuned_num_layers": int(pair_config["model"]["num_layers"]),
                        "adapted_tuned_hid_dim": int(pair_config["model"]["hid_dim"]),
                        "adapted_tuned_lr": float(pair_config["model"]["lr"]),
                        "adapted_tuned_weight_decay": float(pair_config["model"]["weight_decay"]),
                        "adapted_tuned_dropout_ratio": float(pair_config["model"]["dropout_ratio"]),
                        "variant_mmd_weight": float(pair_mmd_weight),
                        "variant_projection_cls_weight": float(spec["projection_cls_weight"]),
                    }
                    rows.append(row)
                    current_rows.append(row)

                if verbose:
                    role_to_score = {row["evaluation_role"]: row["micro_f1"] for row in current_rows}
                    print(
                        f"[{run_idx}/{total_runs}] repeat={repeat} {dataset} {source}->{target} "
                        f"{spec['label']} src={role_to_score.get('source', float('nan')):.4f} "
                        f"tgt={role_to_score.get('target', float('nan')):.4f} "
                        f"other={role_to_score.get('other', float('nan')):.4f}"
                    )

                del trainer
                _cleanup_cuda()

    raw_df = pd.DataFrame(rows)
    baseline_df = raw_df[raw_df["variant_key"] == "v1_source_only"][
        ["dataset", "repeat", "seed", "source", "target", "evaluation_domain", "micro_f1"]
    ].rename(columns={"micro_f1": "v1_micro_f1"})
    raw_df = raw_df.merge(
        baseline_df,
        on=["dataset", "repeat", "seed", "source", "target", "evaluation_domain"],
        how="left",
    )
    raw_df["delta_micro_f1_vs_v1"] = raw_df["micro_f1"] - raw_df["v1_micro_f1"]

    summary_eval = _flatten_summary(
        raw_df,
        [
            "dataset",
            "source",
            "target",
            "evaluation_domain",
            "evaluation_role",
            "variant_key",
            "variant_label",
        ],
        ["micro_f1", "macro_f1", "delta_micro_f1_vs_v1", "train_time_sec"],
    )
    summary_role = _flatten_summary(
        raw_df,
        ["dataset", "source", "target", "evaluation_role", "variant_key", "variant_label"],
        ["micro_f1", "delta_micro_f1_vs_v1"],
    )
    overall_role = _flatten_summary(
        raw_df,
        ["dataset", "evaluation_role", "variant_key", "variant_label"],
        ["micro_f1", "delta_micro_f1_vs_v1"],
    )

    raw_df.to_csv(out_dir / "raw_results.csv", index=False)
    summary_eval.to_csv(out_dir / "summary_by_eval_domain.csv", index=False)
    summary_role.to_csv(out_dir / "summary_by_role.csv", index=False)
    overall_role.to_csv(out_dir / "overall_role_summary.csv", index=False)

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    summary_text = _build_summary_text(
        dataset=dataset,
        domains=domains,
        repeats=int(repeats),
        seed_start=int(seed_start),
        use_tuned=int(use_tuned),
        epochs=int(effective_epochs if effective_epochs is not None else 0),
        summary_eval=summary_eval,
    )
    (out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    print(f"results_dir=\n{out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Compare projection-MMD GNN variants across all ordered domain pairs."
    )
    parser.add_argument("--dataset", type=str, default="citation")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--use-tuned", type=int, default=2)
    parser.add_argument("--mmd-weight", type=float, default=None)
    parser.add_argument("--projection-cls-weight", type=float, default=1.0)
    parser.add_argument("--mmd-sampling-num", type=int, default=256)
    parser.add_argument("--mmd-times", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--out-root",
        type=str,
        default="./__saved__/analysis/projection_mmd_transfer_variants",
    )
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    run_experiment(
        dataset=args.dataset,
        repeats=args.repeats,
        seed_start=args.seed_start,
        epochs=args.epochs,
        use_tuned=args.use_tuned,
        mmd_weight=args.mmd_weight,
        projection_cls_weight=args.projection_cls_weight,
        mmd_sampling_num=args.mmd_sampling_num,
        mmd_times=args.mmd_times,
        device=device,
        verbose=args.verbose,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
