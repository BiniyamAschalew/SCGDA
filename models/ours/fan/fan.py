import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from  models.base_model import BaseGDA
from  models.ours.fan.fan_base import FANBase
from  utils.train_utils.mmd import MMD


class LearnableProbeDistribution(nn.Module):
    """Shared affine-noise probe distribution for adversarial filter matching."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(1, feature_dim))
        self.log_scale = nn.Parameter(torch.zeros(1, feature_dim))

    def sample(self, base: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.log_scale) + 1e-4
        return base + self.shift + scale * torch.randn_like(base)

    def regularizer(self) -> torch.Tensor:
        return self.shift.pow(2).mean() + self.log_scale.pow(2).mean()


class FAN(BaseGDA):
    """A2GNN with learnable source/target filters and adversarial probe alignment."""

    def __init__(self, config: dict):
        super(FAN, self).__init__(config)

        self.config = config
        self.s_pnums = int(config["model"]["s_pnums"])
        self.t_pnums = int(config["model"]["t_pnums"])
        self.cls_pnums = int(config["model"].get("cls_pnums", 1))
        self.adv = bool(config["model"]["adv"])
        self.weight = float(config["model"]["weight"])
        self.mode = config["model"]["mode"]

        self.k = max(self.s_pnums, self.t_pnums, self.cls_pnums) + 1
        self.config["model"]["K"] = int(self.k)

        self.filter_temperature = float(config["model"].get("filter_temperature", 1.0))
        self.filter_l1_weight = float(config["model"].get("filter_l1_weight", 0.0))
        self.filter_init_scale = float(config["model"].get("filter_init_scale", 6.0))

        self.probe_mmd_weight = float(config["model"].get("probe_mmd_weight", 1.0))
        self.probe_sampling_num = int(config["model"].get("probe_sampling_num", 1000))
        self.probe_mmd_times = int(config["model"].get("probe_mmd_times", 5))

        self.adv_steps = max(1, int(config["model"].get("adv_steps", 1)))
        self.adv_lr = float(config["model"].get("adv_lr", self.lr))
        self.adv_weight_decay = float(config["model"].get("adv_weight_decay", 0.0))
        self.adv_distribution_reg_weight = float(
            config["model"].get("adv_distribution_reg_weight", 1e-3)
        )

        self.fan = None
        self.probe_dist = LearnableProbeDistribution(self.in_dim).to(self.device)
        self.source_filter_logits = nn.Parameter(
            self._init_filter_logits(self.s_pnums), requires_grad=True
        )
        self.target_filter_logits = nn.Parameter(
            self._init_filter_logits(self.t_pnums), requires_grad=True
        )

    def _init_filter_logits(self, idx: int) -> torch.Tensor:
        idx = int(idx)
        if idx < 0 or idx >= self.k:
            raise ValueError(f"basis index out of range: idx={idx}, K={self.k}")
        logits = torch.full((self.k,), -self.filter_init_scale, device=self.device)
        logits[idx] = self.filter_init_scale
        return logits

    def _effective_filter(self, logits: torch.Tensor) -> torch.Tensor:
        temp = max(self.filter_temperature, 1e-6)
        return F.softmax(logits / temp, dim=0)

    def init_model(self):
        return FANBase(self.config).to(self.device)

    @staticmethod
    def _probe_stats(source_x: torch.Tensor, target_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        all_x = torch.cat([source_x.detach(), target_x.detach()], dim=0)
        mean = all_x.mean(dim=0, keepdim=True)
        std = all_x.std(dim=0, keepdim=True).clamp_min(1e-6)
        return mean, std

    def _sample_probe_pair(
        self,
        source_x: torch.Tensor,
        target_x: torch.Tensor,
        detach_dist: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probe_mean, probe_std = self._probe_stats(source_x, target_x)
        source_base = torch.randn_like(source_x) * probe_std + probe_mean
        target_base = torch.randn_like(target_x) * probe_std + probe_mean
        if detach_dist:
            with torch.no_grad():
                return self.probe_dist.sample(source_base), self.probe_dist.sample(target_base)
        return self.probe_dist.sample(source_base), self.probe_dist.sample(target_base)

    def _source_target_batches(self, source_data, target_data):
        if self.mode == "node":
            return None, None
        return source_data.batch, target_data.batch

    def filter_alignment_loss(
        self,
        source_data,
        target_data,
        detach_probe: bool,
        detach_filters: bool,
    ) -> torch.Tensor:
        source_probe_x, target_probe_x = self._sample_probe_pair(
            source_data.x,
            target_data.x,
            detach_dist=detach_probe,
        )
        source_batch, target_batch = self._source_target_batches(source_data, target_data)

        source_logits = self.source_filter_logits.detach() if detach_filters else self.source_filter_logits
        target_logits = self.target_filter_logits.detach() if detach_filters else self.target_filter_logits
        source_filter = self._effective_filter(source_logits)
        target_filter = self._effective_filter(target_logits)

        source_probe_feat = self.fan.feat_bottleneck(
            source_probe_x,
            source_data.edge_index,
            source_batch,
            source_filter,
        )
        target_probe_feat = self.fan.feat_bottleneck(
            target_probe_x,
            target_data.edge_index,
            target_batch,
            target_filter,
        )
        return MMD(
            source_probe_feat,
            target_probe_feat,
            sampling_num=self.probe_sampling_num,
            times=self.probe_mmd_times,
        )

    def adversarial_probe_step(self, source_data, target_data):
        filter_loss = self.filter_alignment_loss(
            source_data,
            target_data,
            detach_probe=False,
            detach_filters=True,
        )
        return filter_loss - self.adv_distribution_reg_weight * self.probe_dist.regularizer()

    def forward_model(self, source_data, target_data, alpha):
        source_filter = self._effective_filter(self.source_filter_logits)
        target_filter = self._effective_filter(self.target_filter_logits)

        source_logits = self.fan(source_data, source_filter)
        target_logits = self.fan(target_data, target_filter)

        cls_loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
        loss = cls_loss

        source_batch, target_batch = self._source_target_batches(source_data, target_data)
        source_features = self.fan.feat_bottleneck(
            source_data.x,
            source_data.edge_index,
            source_batch,
            source_filter,
        )
        target_features = self.fan.feat_bottleneck(
            target_data.x,
            target_data.edge_index,
            target_batch,
            target_filter,
        )

        if self.adv:
            source_dlogits = self.fan.domain_classifier(source_features, alpha)
            target_dlogits = self.fan.domain_classifier(target_features, alpha)

            domain_label = torch.tensor(
                [0] * source_features.shape[0] + [1] * target_features.shape[0],
                device=self.device,
            )
            domain_loss = F.cross_entropy(torch.cat([source_dlogits, target_dlogits], 0), domain_label)
            loss = loss + self.weight * domain_loss
        else:
            mmd_loss = MMD(source_features, target_features)
            loss = loss + mmd_loss * self.weight

        probe_loss = self.filter_alignment_loss(
            source_data,
            target_data,
            detach_probe=True,
            detach_filters=False,
        )
        loss = loss + self.probe_mmd_weight * probe_loss

        if self.filter_l1_weight > 0.0:
            l1_penalty = self.source_filter_logits.abs().mean() + self.target_filter_logits.abs().mean()
            loss = loss + self.filter_l1_weight * l1_penalty

        return loss, source_logits, target_logits

    def fit(self, source_data, target_data):
        self.num_source_nodes = source_data.x.shape[0]
        self.num_target_nodes = target_data.x.shape[0]

        if self.mode == "node":
            self.source_loader = self.get_loader(source_data, batch_size=self.batch_size)
            self.target_loader = self.get_loader(target_data, batch_size=self.batch_size)
        elif self.mode == "graph":
            src_batch_size = self.batch_size
            tgt_batch_size = self.batch_size
            if self.batch_size == 0:
                src_batch_size = len(source_data)
                tgt_batch_size = len(target_data)
            self.source_loader = DataLoader(source_data, batch_size=src_batch_size, shuffle=True)
            self.target_loader = DataLoader(target_data, batch_size=tgt_batch_size, shuffle=True)
        else:
            raise AssertionError("Invalid train mode")

        self.fan = self.init_model()
        optimizer = torch.optim.Adam(
            list(self.fan.parameters()) + [self.source_filter_logits, self.target_filter_logits],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        adv_optimizer = torch.optim.Adam(
            self.probe_dist.parameters(),
            lr=self.adv_lr,
            weight_decay=self.adv_weight_decay,
        )

        from tqdm import tqdm
        start_time = time.time()

        for epoch in tqdm(range(self.epoch), desc="Training"):
            epoch_loss = 0.0
            epoch_source_logits = torch.empty((0, self.num_classes), device=self.device)
            epoch_source_labels = torch.empty(0, dtype=torch.long, device=self.device)

            p = float(epoch) / self.epoch
            alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

            for sampled_source_data, sampled_target_data in zip(self.source_loader, self.target_loader):
                self.fan.train()
                self.probe_dist.train()

                sampled_source_data = sampled_source_data.to(self.device)
                sampled_target_data = sampled_target_data.to(self.device)

                for _ in range(self.adv_steps):
                    adv_optimizer.zero_grad()
                    adv_objective = self.adversarial_probe_step(sampled_source_data, sampled_target_data)
                    (-adv_objective).backward()
                    adv_optimizer.step()
                    self.fan.zero_grad()

                loss, source_logits, _ = self.forward_model(
                    sampled_source_data,
                    sampled_target_data,
                    alpha,
                )
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_source_logits = torch.cat((epoch_source_logits, source_logits.detach()), dim=0)
                epoch_source_labels = torch.cat((epoch_source_labels, sampled_source_data.y.detach()), dim=0)

            train_results = self.metrics(epoch_source_logits, epoch_source_labels)
            self.log(epoch, epoch_loss, train_results)

        self.train_time = time.time() - start_time
        self.finish()

    def predict(self, data, source=False):
        self.fan.eval()

        loader = self.target_loader
        filter_param = self._effective_filter(self.target_filter_logits).detach()
        if source:
            loader = self.source_loader
            filter_param = self._effective_filter(self.source_filter_logits).detach()

        all_logits = torch.empty((0, self.num_classes), device=self.device)
        all_labels = torch.empty(0, dtype=torch.long, device=self.device)
        for sampled_data in loader:
            sampled_data = sampled_data.to(self.device)
            with torch.no_grad():
                logits = self.fan(sampled_data, filter_param)
                labels = sampled_data.y
                all_logits = torch.cat((all_logits, logits), dim=0)
                all_labels = torch.cat((all_labels, labels), dim=0)

        return all_logits, all_labels
