import time

import torch
import torch.nn.functional as F
from torch import nn

from models.base_model import BaseGDA
from models.baselines.gnn.gnn_base import GNNBase
from models.__filters.mono import MonoProp
from utils.filter_utils import mmd_rbf


class LearnableProbeDistribution(nn.Module):
    """Simple learnable affine-noise perturbation for adversarial probes."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(1, feature_dim))
        self.log_scale = nn.Parameter(torch.zeros(1, feature_dim))

    def sample(self, base: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.log_scale) + 1e-4
        return base + self.shift + scale * torch.randn_like(base)

    def regularizer(self) -> torch.Tensor:
        return self.shift.pow(2).mean() + self.log_scale.pow(2).mean()


class ADAF(BaseGDA):
    """GNN + adversarially learned monomial filter alignment objective.

    Backbone/classifier path follows GNN. Additional losses:
    - adversarial probe pushforward alignment (min-max MMD)
    - source semantic supervision on filtered source embeddings
    """

    def __init__(self, config):
        super(ADAF, self).__init__(config)
        self.config = config

        self.verbose = config["expt"]["verbose"]
        self.batch_size = config["model"]["batch_size"]

        self.lr = float(config["model"]["lr"])
        self.weight_decay = float(config["model"]["weight_decay"])

        self.mono_degree = int(config["model"].get("mono_degree", 3))
        self.num_terms = self.mono_degree + 1
        self.mono_lambda_max = float(config["model"].get("mono_lambda_max", 2.0))
        self.mono_filter_temperature = float(config["model"].get("mono_filter_temperature", 1.0))
        self.mono_init = str(config["model"].get("mono_init", "uniform"))

        self.probe_mmd_weight = float(config["model"].get("probe_mmd_weight", 1.0))
        self.source_semantic_weight = float(config["model"].get("source_semantic_weight", 1.0))

        self.align_sample_size = int(config["model"].get("align_sample_size", 1024))
        self.kernel_mul = float(config["model"].get("kernel_mul", 2.0))
        self.kernel_num = int(config["model"].get("kernel_num", 5))
        self.fix_sigma = self._parse_fix_sigma(config["model"].get("fix_sigma", None))

        self.adv_steps = max(1, int(config["model"].get("adv_steps", 1)))
        self.adv_lr = float(config["model"].get("adv_lr", 3e-3))
        self.adv_weight_decay = float(config["model"].get("adv_weight_decay", 0.0))
        self.adv_distribution_reg_weight = float(
            config["model"].get("adv_distribution_reg_weight", 1e-3)
        )

        self.gnn = None
        self.mono_prop = MonoProp().to(self.device)
        self.source_filter_logits = None
        self.target_filter_logits = None
        self.semantic_head = None
        self.source_dist = None
        self.target_dist = None

    @staticmethod
    def _parse_fix_sigma(fix_sigma):
        if isinstance(fix_sigma, str):
            s = fix_sigma.strip().lower()
            if s in ("none", "null", ""):
                return None
            return float(s)
        return fix_sigma

    @staticmethod
    def _sample_idx(n: int, max_samples: int, device: torch.device) -> torch.Tensor:
        if max_samples <= 0 or n <= max_samples:
            return torch.arange(n, device=device)
        return torch.randperm(n, device=device)[:max_samples]

    def _mmd(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return mmd_rbf(
            source,
            target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma,
        )

    def _effective_filter(self, logits: torch.Tensor) -> torch.Tensor:
        t = max(float(self.mono_filter_temperature), 1e-6)
        return torch.softmax(logits / t, dim=0)

    def _init_filter_logits(self, init_mode: str, num_terms: int) -> torch.Tensor:
        mode = str(init_mode).lower()
        logits = torch.zeros(num_terms, device=self.device)
        if mode in {"adj", "adjacency", "a1"}:
            logits.fill_(-2.0)
            logits[1 if num_terms > 1 else 0] = 2.0
        elif mode in {"identity", "a0"}:
            logits.fill_(-2.0)
            logits[0] = 2.0
        return logits

    def _backbone_features(self, data) -> torch.Tensor:
        batch = getattr(data, "batch", None)
        edge_weight = getattr(data, "edge_weight", None)
        return self.gnn.feat_bottleneck(
            data.x,
            data.edge_index,
            edge_weight=edge_weight,
            batch=batch,
        )

    def _apply_filter(self, x: torch.Tensor, data, coeff: torch.Tensor) -> torch.Tensor:
        edge_weight = getattr(data, "edge_weight", None)
        return self.mono_prop(
            x,
            data.edge_index,
            parameters=coeff,
            edge_weight=edge_weight,
            lambda_max=self.mono_lambda_max,
        )

    def _domain_logits(self, features: torch.Tensor, data) -> torch.Tensor:
        edge_weight = getattr(data, "edge_weight", None)
        logits = self.gnn.feat_classifier(features, data.edge_index, edge_weight=edge_weight)
        return F.log_softmax(logits, dim=1)

    @staticmethod
    def _probe_stats(source_feat: torch.Tensor, target_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        all_feat = torch.cat([source_feat.detach(), target_feat.detach()], dim=0)
        mean = all_feat.mean(dim=0, keepdim=True)
        std = all_feat.std(dim=0, keepdim=True).clamp_min(1e-6)
        return mean, std

    @staticmethod
    def _sample_gaussian_probe_like(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(x) * std + mean

    def _sampled_probe_mmd(self, source_push: torch.Tensor, target_push: torch.Tensor) -> torch.Tensor:
        idx_s = self._sample_idx(source_push.size(0), self.align_sample_size, source_push.device)
        idx_t = self._sample_idx(target_push.size(0), self.align_sample_size, target_push.device)
        return self._mmd(source_push[idx_s], target_push[idx_t])

    def init_model(self, **kwargs):
        return GNNBase(self.config).to(self.device)

    def _init_train_state(self):
        self.source_filter_logits = nn.Parameter(
            self._init_filter_logits(self.mono_init, self.num_terms), requires_grad=True
        )
        self.target_filter_logits = nn.Parameter(
            self._init_filter_logits(self.mono_init, self.num_terms), requires_grad=True
        )

        self.semantic_head = nn.Linear(self.hid_dim, self.num_classes).to(self.device)
        self.source_dist = LearnableProbeDistribution(self.hid_dim).to(self.device)
        self.target_dist = LearnableProbeDistribution(self.hid_dim).to(self.device)

    def _adv_step(self, source_embed: torch.Tensor, target_embed: torch.Tensor, source_data, target_data):
        coeff_s = self._effective_filter(self.source_filter_logits.detach())
        coeff_t = self._effective_filter(self.target_filter_logits.detach())

        probe_mean, probe_std = self._probe_stats(source_embed, target_embed)
        source_probe_base = self._sample_gaussian_probe_like(source_embed, probe_mean, probe_std)
        target_probe_base = self._sample_gaussian_probe_like(target_embed, probe_mean, probe_std)

        source_probe_adv = self.source_dist.sample(source_probe_base)
        target_probe_adv = self.target_dist.sample(target_probe_base)

        source_push = self._apply_filter(source_probe_adv, source_data, coeff_s)
        target_push = self._apply_filter(target_probe_adv, target_data, coeff_t)

        adv_probe_mmd = self._sampled_probe_mmd(source_push, target_push)
        adv_reg = self.adv_distribution_reg_weight * (
            self.source_dist.regularizer() + self.target_dist.regularizer()
        )
        adv_objective = adv_probe_mmd - adv_reg
        return adv_objective, adv_probe_mmd, adv_reg

    def forward_model(self, source_data, target_data):
        source_embed = self._backbone_features(source_data)
        target_embed = self._backbone_features(target_data)

        coeff_s = self._effective_filter(self.source_filter_logits)
        coeff_t = self._effective_filter(self.target_filter_logits)

        source_filtered = self._apply_filter(source_embed, source_data, coeff_s)
        target_filtered = self._apply_filter(target_embed, target_data, coeff_t)

        source_logits = self._domain_logits(source_filtered, source_data)
        target_logits = self._domain_logits(target_filtered, target_data)

        cls_loss = F.nll_loss(source_logits, source_data.y)

        probe_mean, probe_std = self._probe_stats(source_embed, target_embed)
        source_probe_base = self._sample_gaussian_probe_like(source_embed, probe_mean, probe_std)
        target_probe_base = self._sample_gaussian_probe_like(target_embed, probe_mean, probe_std)
        with torch.no_grad():
            source_probe = self.source_dist.sample(source_probe_base)
            target_probe = self.target_dist.sample(target_probe_base)

        source_push = self._apply_filter(source_probe, source_data, coeff_s)
        target_push = self._apply_filter(target_probe, target_data, coeff_t)
        probe_mmd_loss = self._sampled_probe_mmd(source_push, target_push)

        semantic_logits = self.semantic_head(source_filtered)
        semantic_loss = F.cross_entropy(semantic_logits, source_data.y)

        loss = cls_loss
        loss = loss + self.probe_mmd_weight * probe_mmd_loss
        loss = loss + self.source_semantic_weight * semantic_loss

        logs = {
            "cls_loss": float(cls_loss.detach().cpu().item()),
            "probe_mmd_loss": float(probe_mmd_loss.detach().cpu().item()),
            "semantic_loss": float(semantic_loss.detach().cpu().item()),
            "source_coeff_identity": float(coeff_s[0].detach().cpu().item()),
            "target_coeff_identity": float(coeff_t[0].detach().cpu().item()),
        }

        return loss, source_logits, target_logits, logs

    def fit(self, source_data, target_data):
        source_loader = self.get_loader(source_data)
        target_loader = self.get_loader(target_data)

        self.gnn = self.init_model(**self.kwargs)
        self._init_train_state()

        main_params = list(self.gnn.parameters())
        main_params += [self.source_filter_logits, self.target_filter_logits]
        main_params += list(self.semantic_head.parameters())

        main_optimizer = torch.optim.Adam(
            main_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        adv_optimizer = torch.optim.Adam(
            list(self.source_dist.parameters()) + list(self.target_dist.parameters()),
            lr=self.adv_lr,
            weight_decay=self.adv_weight_decay,
        )

        start_time = time.time()

        from tqdm import tqdm

        for epoch in tqdm(range(self.epoch), desc="Training"):
            epoch_loss = 0.0
            epoch_source_logits = torch.empty((0, self.num_classes), device=self.device)
            epoch_source_labels = torch.empty(0, dtype=torch.long, device=self.device)

            loss_logs_acc = {
                "cls_loss": 0.0,
                "probe_mmd_loss": 0.0,
                "semantic_loss": 0.0,
                "source_coeff_identity": 0.0,
                "target_coeff_identity": 0.0,
                "adv_probe_mmd": 0.0,
                "adv_objective": 0.0,
                "adv_reg": 0.0,
            }
            n_batches = 0

            for sampled_source_data, sampled_target_data in zip(source_loader, target_loader):
                n_batches += 1
                sampled_source_data = sampled_source_data.to(self.device)
                sampled_target_data = sampled_target_data.to(self.device)

                self.gnn.train()
                self.semantic_head.train()
                self.source_dist.train()
                self.target_dist.train()

                source_embed_det = self._backbone_features(sampled_source_data).detach()
                target_embed_det = self._backbone_features(sampled_target_data).detach()

                adv_probe_val = 0.0
                adv_obj_val = 0.0
                adv_reg_val = 0.0
                for _ in range(self.adv_steps):
                    adv_optimizer.zero_grad()
                    adv_objective, adv_probe_mmd, adv_reg = self._adv_step(
                        source_embed_det,
                        target_embed_det,
                        sampled_source_data,
                        sampled_target_data,
                    )
                    (-adv_objective).backward()
                    adv_optimizer.step()
                    adv_probe_val = float(adv_probe_mmd.detach().cpu().item())
                    adv_obj_val = float(adv_objective.detach().cpu().item())
                    adv_reg_val = float(adv_reg.detach().cpu().item())

                loss, source_logits, _, logs = self.forward_model(sampled_source_data, sampled_target_data)
                epoch_loss += float(loss.item())

                main_optimizer.zero_grad()
                loss.backward()
                main_optimizer.step()

                epoch_source_logits = torch.cat((epoch_source_logits, source_logits.detach()), dim=0)
                epoch_source_labels = torch.cat((epoch_source_labels, sampled_source_data.y.detach()), dim=0)

                for key in logs:
                    loss_logs_acc[key] += float(logs[key])
                loss_logs_acc["adv_probe_mmd"] += adv_probe_val
                loss_logs_acc["adv_objective"] += adv_obj_val
                loss_logs_acc["adv_reg"] += adv_reg_val

            train_results = self.metrics(epoch_source_logits, epoch_source_labels)
            self.log(epoch, epoch_loss, train_results)

            if n_batches > 0:
                payload = {f"adaf_{k}": v / float(n_batches) for k, v in loss_logs_acc.items()}
                payload["epoch"] = epoch + 1
                self.wandb.log(payload)

        self.train_time = time.time() - start_time
        self.finish()

    def process_graph(self, data):
        pass

    def predict(self, data, domain: str = "target"):
        self.gnn.eval()
        self.semantic_head.eval()

        with torch.no_grad():
            embed = self._backbone_features(data)
            if str(domain).lower() == "source":
                coeff = self._effective_filter(self.source_filter_logits)
            else:
                coeff = self._effective_filter(self.target_filter_logits)
            feat = self._apply_filter(embed, data, coeff)
            logits = self._domain_logits(feat, data)

        return logits, data.y
