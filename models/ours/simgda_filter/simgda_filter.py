import time
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from models.base_model import BaseGDA
from models.ours.simgda_filter.simgda_filter_base import SimGDAFilterBase
from utils.train_utils.mmd import MMD


class SimGDAFilter(BaseGDA):
    """A2GNN-style training with monomial basis-index propagation controls."""

    def __init__(self, config: dict):
        super(SimGDAFilter, self).__init__(config)

        self.config = config
        self.s_pnums = int(config["model"].get("s_pnums", 1))
        self.t_pnums = int(config["model"].get("t_pnums", 1))
        self.cls_pnums = int(config["model"].get("cls_pnums", 1))
        self.adv = bool(config["model"].get("adv", False))
        # Auto-derive basis size from selected source/target indices.
        # Since indices are zero-based, size must be max_index + 1.
        self.k = max(self.s_pnums, self.t_pnums, self.cls_pnums) + 1
        self.config["model"]["K"] = int(self.k)

        # Keep A2GNN naming/behavior: one shared DA weight
        if "weight" in config["model"]:
            self.weight = float(config["model"]["weight"])
        else:
            # Backward compatibility with old SimGDA-Filter configs
            self.weight = float(config["model"].get("mmd_weight", 0.1))
        self.filter_l1_weight = float(config["model"].get("filter_l1_weight", 1e-4))

        self.mode = config["model"]["mode"]
        self.model = None

        # Trainable monomial basis vectors (initialized as one-hot).
        self.source_filter_param = torch.nn.Parameter(
            self._basis_one_hot(self.s_pnums), requires_grad=True
        )
        self.target_filter_param = torch.nn.Parameter(
            self._basis_one_hot(self.t_pnums), requires_grad=True
        )

    def init_model(self):
        return SimGDAFilterBase(self.config).to(self.device)

    def _basis_one_hot(self, idx: int) -> torch.Tensor:
        idx = int(idx)
        if idx < 0 or idx >= self.k:
            raise ValueError(f"basis index out of range: idx={idx}, K={self.k}")
        out = torch.zeros(self.k, device=self.device, dtype=torch.float32)
        out[idx] = 1.0
        return out

    def forward_model(self, source_data, target_data, alpha):
        source_logits = self.model(source_data, self.source_filter_param)
        target_logits = self.model(target_data, self.target_filter_param)

        cls_loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
        loss = cls_loss

        if self.mode == "node":
            source_batch = None
            target_batch = None
        else:
            source_batch = source_data.batch
            target_batch = target_data.batch

        source_features = self.model.feat_bottleneck(
            source_data.x,
            source_data.edge_index,
            source_batch,
            self.source_filter_param,
        )
        target_features = self.model.feat_bottleneck(
            target_data.x,
            target_data.edge_index,
            target_batch,
            self.target_filter_param,
        )

        if self.adv:
            source_dlogits = self.model.domain_classifier(source_features, alpha)
            target_dlogits = self.model.domain_classifier(target_features, alpha)

            domain_label = torch.tensor(
                [0] * source_data.x.shape[0] + [1] * target_data.x.shape[0],
                device=self.device,
            )

            domain_loss = F.cross_entropy(torch.cat([source_dlogits, target_dlogits], 0), domain_label)
            loss = loss + self.weight * domain_loss
        else:
            mmd_loss = MMD(source_features, target_features)
            loss = loss + mmd_loss * self.weight

        # Small L1 regularization on trainable filter coefficients.
        if self.filter_l1_weight > 0.0:
            l1_penalty = self.source_filter_param.abs().mean() + self.target_filter_param.abs().mean()
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

        self.model = self.init_model()

        src_start = self.source_filter_param.detach().cpu().clone()
        tgt_start = self.target_filter_param.detach().cpu().clone()
        print(f"[filters:start] source={src_start.tolist()} target={tgt_start.tolist()}")

        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + [self.source_filter_param, self.target_filter_param],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        from tqdm import tqdm

        for epoch in tqdm(range(self.epoch), desc="Training"):
            epoch_loss = 0.0
            epoch_source_logits = torch.empty(0).to(self.device)
            epoch_source_labels = torch.empty(0).to(self.device)

            p = float(epoch) / self.epoch
            alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

            for sampled_source_data, sampled_target_data in zip(self.source_loader, self.target_loader):
                self.model.train()

                sampled_source_data = sampled_source_data.to(self.device)
                sampled_target_data = sampled_target_data.to(self.device)

                loss, source_logits, _ = self.forward_model(sampled_source_data, sampled_target_data, alpha)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                source_labels = sampled_source_data.y
                epoch_source_logits = torch.cat((epoch_source_logits, source_logits))
                epoch_source_labels = torch.cat((epoch_source_labels, source_labels))

            train_results = self.metrics(epoch_source_logits, epoch_source_labels)
            self.log(epoch, epoch_loss, train_results)

        src_end = self.source_filter_param.detach().cpu().clone()
        tgt_end = self.target_filter_param.detach().cpu().clone()
        print(f"[filters:end] source={src_end.tolist()} target={tgt_end.tolist()}")

        self.finish()

    def predict(self, data, source=False):
        self.model.eval()

        loader, filter_param = self.target_loader, self.target_filter_param
        if source:
            loader, filter_param = self.source_loader, self.source_filter_param

        all_logits = torch.empty(0).to(self.device)
        all_labels = torch.empty(0).to(self.device)
        for sampled_data in loader:
            sampled_data = sampled_data.to(self.device)
            with torch.no_grad():
                logits = self.model(sampled_data, filter_param)
                labels = sampled_data.y
                all_logits = torch.cat((all_logits, logits), dim=0)
                all_labels = torch.cat((all_labels, labels), dim=0)

        return all_logits, all_labels
