from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
import time

from omegaconf import OmegaConf
import torch


ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_CONFIG_DIR = ROOT_DIR / "configs" / "data_configs"
PREV_TUNED_DIR = ROOT_DIR / "__hps__" / "prev_tuned"
DEFAULT_OUT_ROOT = "./__saved__/analysis/triplet_bound"


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    source_ce_weight: float
    target_ce_weight: float
    mmd_weight: float


CASE_SPECS = (
    CaseSpec(
        key="source_only",
        label="Source Only",
        source_ce_weight=1.0,
        target_ce_weight=0.0,
        mmd_weight=0.0,
    ),
    CaseSpec(
        key="source_mmd_target",
        label="Source + MMD",
        source_ce_weight=1.0,
        target_ce_weight=0.0,
        mmd_weight=0.1,
    ),
    CaseSpec(
        key="oracle",
        label="Oracle",
        source_ce_weight=1.0,
        target_ce_weight=1.0,
        mmd_weight=0.0,
    ),
)


@dataclass(frozen=True)
class TripletSpec:
    source: str
    target: str
    reference: str

    @property
    def key(self) -> str:
        return f"{self.source}__{self.target}__{self.reference}"

    @property
    def label(self) -> str:
        return f"{self.source}->{self.target} | ref={self.reference}"


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "airport"
    seeds: tuple[int, ...] = (0, 1, 2)
    device: str = "cuda:0"
    epochs: int = 200
    lr: float = 0.003
    weight_decay: float = 1e-4
    hid_dim: int = 128
    num_layers: int = 1
    dropout_ratio: float = 0.1
    gnn: str = "gcn"
    activation: str = "relu"
    mmd_weight: float = 0.1
    train_split: float = 0.8
    val_split: float = 0.2
    resample_masks: bool = True
    use_pair_tuned_hparams: bool = True
    metric_sample_size: int = 0
    grad_nodes_cap: int = 0
    max_triplets: int = 0
    out_root: str = DEFAULT_OUT_ROOT
    run_name: str = ""
    verbose: int = 1

    def resolved_device(self) -> str:
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return self.device

    def output_dir(self) -> Path:
        base = Path(self.out_root)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        dataset_name = self.dataset.lower()
        run_name = self.run_name.strip() or f"triplet_bound_{dataset_name}_{stamp}"
        out_dir = base / run_name
        suffix = 1
        while out_dir.exists():
            out_dir = base / f"{run_name}_{suffix}"
            suffix += 1
        out_dir.mkdir(parents=True, exist_ok=False)
        return out_dir

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["device"] = self.resolved_device()
        return payload


def _load_yaml(path: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def dataset_config(dataset_name: str, train_split: float, val_split: float) -> dict:
    path = DATA_CONFIG_DIR / f"{dataset_name.lower()}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")

    data_cfg = _load_yaml(path)
    data_cfg["train_split"] = float(train_split)
    data_cfg["val_split"] = float(val_split)
    data_cfg["test_split"] = max(0.0, 1.0 - float(train_split) - float(val_split))
    return {
        "data": data_cfg,
        "model": {"name": "simgda"},
        "expt": {"verbose": 0},
    }


def dataset_triplets(dataset_name: str, max_triplets: int = 0) -> list[TripletSpec]:
    cfg = dataset_config(dataset_name, train_split=0.8, val_split=0.2)
    domains = list(cfg["data"]["domains"])
    if len(domains) != 3:
        raise ValueError(
            f"Dataset '{dataset_name}' has {len(domains)} domains. "
            "This experiment expects exactly 3 domains."
        )
    triplets = [TripletSpec(*items) for items in permutations(domains, 3)]
    if max_triplets > 0:
        return triplets[:max_triplets]
    return triplets


def resolve_case_hparams(exp_cfg: ExperimentConfig, triplet: TripletSpec, case_key: str) -> dict:
    params = {
        "lr": float(exp_cfg.lr),
        "weight_decay": float(exp_cfg.weight_decay),
        "hid_dim": int(exp_cfg.hid_dim),
        "num_layers": int(exp_cfg.num_layers),
        "dropout_ratio": float(exp_cfg.dropout_ratio),
        "gnn": exp_cfg.gnn,
        "activation": exp_cfg.activation,
        "mmd_weight": float(exp_cfg.mmd_weight),
        "scheduler": "",
        "step_size": 10,
        "gamma": 0.1,
        "tuned_model_family": "manual",
    }
    if not exp_cfg.use_pair_tuned_hparams:
        return params

    tuned_family = "simgda" if case_key == "source_mmd_target" else "gnn"
    tuned_path = (
        PREV_TUNED_DIR
        / tuned_family
        / exp_cfg.dataset.lower()
        / f"{triplet.source}_{triplet.target}"
        / "best.yaml"
    )
    if not tuned_path.exists():
        return params

    tuned = _load_yaml(tuned_path)
    for key in params:
        if key in tuned:
            params[key] = tuned[key]
    params["tuned_model_family"] = tuned_family
    return params


def model_config(
    *,
    in_dim: int,
    num_classes: int,
    hid_dim: int,
    num_layers: int,
    dropout_ratio: float,
    gnn: str,
    activation: str,
) -> dict:
    return {
        "model": {
            "in_dim": int(in_dim),
            "hid_dim": int(hid_dim),
            "num_classes": int(num_classes),
            "num_layers": int(num_layers),
            "dropout_ratio": float(dropout_ratio),
            "gnn": gnn,
            "activation": activation,
            "mode": "node",
        }
    }
