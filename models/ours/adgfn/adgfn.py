import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader, NeighborLoader

from models.__filters.mono import MonoProp
from models.base_model import BaseGDA
from models.baselines.adagcn.adagcn_base import AdaGCNBase
from utils.filter_utils import mmd_rbf


class LearnableProbeDistribution(nn.Module):
    """Learnable affine-noise perturbation for adversarial probes."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(1, feature_dim)) # the mean shift
        self.log_scale = nn.Parameter(torch.zeros(1, feature_dim)) # log of the noice scale, to ensure positivity

    def sample(self, base: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.log_scale) + 1e-4
        return base + self.shift + scale * torch.randn_like(base)

    def regularizer(self) -> torch.Tensor:
        return self.shift.pow(2).mean() + self.log_scale.pow(2).mean()


class ADGFN(BaseGDA):
    """AdaGCN with adversarially learned monomial filter coefficients."""

    def __init__(self, config):
        super(ADGFN, self).__init__(config)

        self.gnn_type = config["model"]["gnn"]
        self.adv_dim = config["model"]["adv_dim"]
        self.gp_weight = config["model"]["gp_weight"]
        self.domain_weight = config["model"]["domain_weight"]
        self.mode = config["model"]["mode"]

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

        self.adgfn = None
        self.discriminator = None
        self.c_optimizer = None
        self.source_loader = None
        self.target_loader = None

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

    def _apply_filter(self, x: torch.Tensor, data, coeff: torch.Tensor) -> torch.Tensor:
        if self.mode != "node":
            return x
        # MonoProp computes sum_k c_k A^k x. For (-A) basis we use c_k = coeff_k * (-1)^k.
        signed_coeff = coeff.clone()
        if signed_coeff.numel() > 1:
            signed_coeff[1::2] = -signed_coeff[1::2]
        edge_weight = getattr(data, "edge_weight", None)
        return self.mono_prop(
            x,
            data.edge_index,
            parameters=signed_coeff,
            edge_weight=edge_weight,
            lambda_max=self.mono_lambda_max,
        )

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

    def init_model(self):
        return AdaGCNBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            act=self.act,
            gnn_type=self.gnn_type,
            mode=self.mode,
        ).to(self.device)

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
        return adv_objective

    def forward_model(self, source_data, target_data):
        for _ in range(10):
            encoded_source = self.adgfn(source_data)
            encoded_target = self.adgfn(target_data)

            coeff_s = self._effective_filter(self.source_filter_logits.detach())
            coeff_t = self._effective_filter(self.target_filter_logits.detach())
            filtered_source = self._apply_filter(encoded_source, source_data, coeff_s)
            filtered_target = self._apply_filter(encoded_target, target_data, coeff_t)

            gp_loss = self.gradient_penalty(filtered_source, filtered_target)
            dis_s = torch.mean(self.discriminator(filtered_source).reshape(-1))
            dis_t = torch.mean(self.discriminator(filtered_target).reshape(-1))
            dis_loss = -torch.abs(dis_s - dis_t)

            loss = dis_loss + self.gp_weight * gp_loss

            self.c_optimizer.zero_grad()
            loss.backward()
            self.c_optimizer.step()

        encoded_source = self.adgfn(source_data)
        encoded_target = self.adgfn(target_data)

        coeff_s = self._effective_filter(self.source_filter_logits)
        coeff_t = self._effective_filter(self.target_filter_logits)
        filtered_source = self._apply_filter(encoded_source, source_data, coeff_s)
        filtered_target = self._apply_filter(encoded_target, target_data, coeff_t)

        source_logits = self.adgfn.cls_model(filtered_source)
        cls_loss = self.adgfn.loss_func(source_logits, source_data.y)

        dis_s = torch.mean(self.discriminator(filtered_source).reshape(-1))
        dis_t = torch.mean(self.discriminator(filtered_target).reshape(-1))
        dis_loss = torch.abs(dis_s - dis_t)

        target_logits = self.adgfn.cls_model(filtered_target)

        probe_mean, probe_std = self._probe_stats(encoded_source, encoded_target)
        source_probe_base = self._sample_gaussian_probe_like(encoded_source, probe_mean, probe_std)
        target_probe_base = self._sample_gaussian_probe_like(encoded_target, probe_mean, probe_std)
        with torch.no_grad():
            source_probe = self.source_dist.sample(source_probe_base)
            target_probe = self.target_dist.sample(target_probe_base)

        source_push = self._apply_filter(source_probe, source_data, coeff_s)
        target_push = self._apply_filter(target_probe, target_data, coeff_t)
        probe_mmd_loss = self._sampled_probe_mmd(source_push, target_push)

        semantic_logits = self.semantic_head(filtered_source)
        semantic_loss = F.cross_entropy(semantic_logits, source_data.y)

        loss = cls_loss + dis_loss * self.domain_weight
        loss = loss + self.probe_mmd_weight * probe_mmd_loss
        loss = loss + self.source_semantic_weight * semantic_loss

        return loss, source_logits, target_logits

    def fit(self, source_data, target_data):
        if self.mode == "node":
            self.num_source_nodes, _ = source_data.x.shape
            self.num_target_nodes, _ = target_data.x.shape

            if self.batch_size == 0:
                self.source_batch_size = source_data.x.shape[0]
                self.source_loader = NeighborLoader(
                    source_data,
                    self.num_neigh,
                    batch_size=self.source_batch_size,
                )
                self.target_batch_size = target_data.x.shape[0]
                self.target_loader = NeighborLoader(
                    target_data,
                    self.num_neigh,
                    batch_size=self.target_batch_size,
                )
            else:
                self.source_loader = NeighborLoader(
                    source_data,
                    self.num_neigh,
                    batch_size=self.batch_size,
                )
                self.target_loader = NeighborLoader(
                    target_data,
                    self.num_neigh,
                    batch_size=self.batch_size,
                )
        elif self.mode == "graph":
            if self.batch_size == 0:
                num_source_graphs = len(source_data)
                num_target_graphs = len(target_data)
                self.source_loader = DataLoader(source_data, batch_size=num_source_graphs, shuffle=True)
                self.target_loader = DataLoader(target_data, batch_size=num_target_graphs, shuffle=True)
            else:
                self.source_loader = DataLoader(source_data, batch_size=self.batch_size, shuffle=True)
                self.target_loader = DataLoader(target_data, batch_size=self.batch_size, shuffle=True)
        else:
            raise AssertionError("Invalid train mode")

        self.adgfn = self.init_model()
        self._init_train_state()

        optimizer = torch.optim.Adam(
            list(self.adgfn.parameters())
            + [self.source_filter_logits, self.target_filter_logits]
            + list(self.semantic_head.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.discriminator = nn.Sequential(
            nn.Linear(self.hid_dim, self.adv_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.adv_dim, 1),
            nn.Sigmoid(),
        ).to(self.device)

        self.c_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        adv_optimizer = torch.optim.Adam(
            list(self.source_dist.parameters()) + list(self.target_dist.parameters()),
            lr=self.adv_lr,
            weight_decay=self.adv_weight_decay,
        )

        start_time = time.time()

        for epoch in range(self.epoch):
            epoch_loss = 0.0
            epoch_source_logits = None
            epoch_source_labels = None

            for idx, (sampled_source_data, sampled_target_data) in enumerate(
                zip(self.source_loader, self.target_loader)
            ):
                self.adgfn.train()
                self.semantic_head.train()
                self.source_dist.train()
                self.target_dist.train()

                sampled_source_data = sampled_source_data.to(self.device)
                sampled_target_data = sampled_target_data.to(self.device)

                source_embed_det = self.adgfn(sampled_source_data).detach()
                target_embed_det = self.adgfn(sampled_target_data).detach()

                for _ in range(self.adv_steps):
                    adv_optimizer.zero_grad()
                    adv_objective = self._adv_step(
                        source_embed_det,
                        target_embed_det,
                        sampled_source_data,
                        sampled_target_data,
                    )
                    (-adv_objective).backward()
                    adv_optimizer.step()

                loss, source_logits, target_logits = self.forward_model(
                    sampled_source_data, sampled_target_data
                )
                epoch_loss += float(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if idx == 0:
                    epoch_source_logits, epoch_source_labels = source_logits, sampled_source_data.y
                else:
                    source_labels = sampled_source_data.y
                    epoch_source_logits = torch.cat((epoch_source_logits, source_logits), dim=0)
                    epoch_source_labels = torch.cat((epoch_source_labels, source_labels), dim=0)

            train_results = self.metrics(epoch_source_logits, epoch_source_labels)
            self.log(epoch, epoch_loss, train_results)

        self.train_time = time.time() - start_time
        self.finish()

    def predict(self, data, source=False):
        self.adgfn.eval()
        self.semantic_head.eval()

        if source:
            for idx, sampled_data in enumerate(self.source_loader):
                sampled_data = sampled_data.to(self.device)
                with torch.no_grad():
                    encoded_data = self.adgfn(sampled_data)
                    coeff = self._effective_filter(self.source_filter_logits)
                    encoded_data = self._apply_filter(encoded_data, sampled_data, coeff)
                    logits = self.adgfn.cls_model(encoded_data)

                    if idx == 0:
                        logits, labels = logits, sampled_data.y
                    else:
                        sampled_logits, sampled_labels = logits, sampled_data.y
                        logits = torch.cat((logits, sampled_logits))
                        labels = torch.cat((labels, sampled_labels))
        else:
            for idx, sampled_data in enumerate(self.target_loader):
                sampled_data = sampled_data.to(self.device)
                with torch.no_grad():
                    encoded_data = self.adgfn(sampled_data)
                    coeff = self._effective_filter(self.target_filter_logits)
                    encoded_data = self._apply_filter(encoded_data, sampled_data, coeff)
                    logits = self.adgfn.cls_model(encoded_data)

                    if idx == 0:
                        logits, labels = logits, sampled_data.y
                    else:
                        sampled_logits, sampled_labels = logits, sampled_data.y
                        logits = torch.cat((logits, sampled_logits))
                        labels = torch.cat((labels, sampled_labels))

        return logits, labels

    def gradient_penalty(self, encoded_source, encoded_target):
        num_s = encoded_source.shape[0]
        num_t = encoded_target.shape[0]

        if num_s < num_t:
            hidden = encoded_target[-num_s:,]
            hidden_s = torch.cat((encoded_source, encoded_source), dim=0)
            hidden_t = torch.cat((encoded_target[0:num_s,], hidden), dim=0)

            alpha = torch.rand((2 * num_s, 1)).to(self.device)

            difference = hidden_s - hidden_t
            interpolates = hidden_t + (alpha * difference)
        elif num_s > num_t:
            hidden = encoded_source[-num_t:,]
            hidden_s = torch.cat((encoded_source[0:num_t,], hidden), dim=0)
            hidden_t = torch.cat((encoded_target, encoded_target), dim=0)

            alpha = torch.rand((2 * num_t, 1)).to(self.device)

            difference = hidden_s - hidden_t
            interpolates = hidden_t + (alpha * difference)
        else:
            alpha = torch.rand((num_t, 1)).to(self.device)

            difference = encoded_source - encoded_target
            interpolates = encoded_target + (alpha * difference)

        inputs = torch.cat((encoded_source, encoded_target, interpolates), dim=0)
        scores = self.discriminator(inputs)

        gradient = torch.autograd.grad(
            inputs=inputs,
            outputs=scores,
            grad_outputs=torch.ones_like(scores).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

        return gradient_penalty
