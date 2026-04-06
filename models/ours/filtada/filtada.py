import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import NeighborLoader, DataLoader

from  models.base_model import BaseGDA
from  models.ours.filtada.filtada_base import FiltADABase
from  utils.filter_utils import make_gaussian_probe


class FiltADA(BaseGDA):
    """AdaGCN with learnable source/target filter coefficients."""

    def __init__(self, config):
        super(FiltADA, self).__init__(config)

        self.config = config
        gnn_type = str(config["model"].get("gnn", "filter_mono")).lower()
        self.gnn_type = gnn_type if gnn_type.startswith("filter") else "filter_mono"

        self.adv_dim = config["model"]["adv_dim"]
        self.gp_weight = config["model"]["gp_weight"]
        self.domain_weight = config["model"]["domain_weight"]
        self.probe_domain_weight = float(config["model"].get("probe_domain_weight", 0.0))
        self.mode = config["model"]["mode"]

        # Fixed depth for now as requested.
        self.filter_depth = int(config["model"].get("filter_depth", 4))
        self.filter_l1_weight = float(config["model"].get("filter_l1_weight", 0.0))
        self.filter_temperature = float(config["model"].get("filter_temperature", 1.0))

        if self.filter_depth != 4:
            raise ValueError(f"FiltADA currently supports filter_depth=4 only, got {self.filter_depth}.")

        # Trainable logits initialized to uniform 1/K.
        default_init = [1.0 / self.filter_depth] * self.filter_depth
        source_init = self.config["model"].get("source_filter_init", default_init)
        target_init = self.config["model"].get("target_filter_init", default_init)
        self.source_filter_param = nn.Parameter(
            torch.as_tensor(source_init, dtype=torch.float32, device=self.device).view(-1),
            requires_grad=True,
        )
        self.target_filter_param = nn.Parameter(
            torch.as_tensor(target_init, dtype=torch.float32, device=self.device).view(-1),
            requires_grad=True,
        )
        if int(self.source_filter_param.numel()) != self.filter_depth:
            raise ValueError(
                f"source_filter_init length mismatch: expected {self.filter_depth}, got {int(self.source_filter_param.numel())}."
            )
        if int(self.target_filter_param.numel()) != self.filter_depth:
            raise ValueError(
                f"target_filter_init length mismatch: expected {self.filter_depth}, got {int(self.target_filter_param.numel())}."
            )

    def _init_filter_param(self, key: str, one_hot_idx: int) -> torch.Tensor:
        raw = self.config["model"].get(key, None)
        if raw is not None:
            v = torch.as_tensor(raw, dtype=torch.float32, device=self.device).view(-1)
            if int(v.numel()) != self.filter_depth:
                raise ValueError(
                    f"{key} length mismatch: expected {self.filter_depth}, got {int(v.numel())}."
                )
            return v

        idx = int(one_hot_idx)
        if idx < 0 or idx >= self.filter_depth:
            raise ValueError(
                f"One-hot filter index out of range: idx={idx}, filter_depth={self.filter_depth}."
            )
        v = torch.zeros(self.filter_depth, dtype=torch.float32, device=self.device)
        v[idx] = 1.0
        return v

    def _effective_filter(self, filter_param: torch.Tensor) -> torch.Tensor:
        # Use softmax-normalized coefficients as attention-like filter weights.
        temp = max(self.filter_temperature, 1e-6)
        return F.softmax(filter_param / temp, dim=0)

    def init_model(self):
        return FiltADABase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            act=self.act,
            gnn_type=self.gnn_type,
            mode=self.mode,
            filter_depth=self.filter_depth,
        ).to(self.device)

    def forward_model(self, source_data, target_data):
        for _ in range(10):
            encoded_source = self.filtada(
                source_data, filter_param=self._effective_filter(self.source_filter_param)
            )
            encoded_target = self.filtada(
                target_data, filter_param=self._effective_filter(self.target_filter_param)
            )

            gp_loss = self.gradient_penalty(encoded_source, encoded_target)

            dis_s = torch.mean(self.discriminator(encoded_source).reshape(-1))
            dis_t = torch.mean(self.discriminator(encoded_target).reshape(-1))
            dis_loss = -torch.abs(dis_s - dis_t)

            loss = dis_loss + self.gp_weight * gp_loss

            if self.probe_domain_weight > 0.0:
                probe_source_x, probe_target_x = make_gaussian_probe(
                    source_data.x.detach(), target_data.x.detach()
                )
                probe_encoded_source = self.filtada.encoder(
                    probe_source_x,
                    source_data.edge_index,
                    None if self.mode == "node" else source_data.batch,
                    mode=self.mode,
                    filter_param=self._effective_filter(self.source_filter_param),
                )
                probe_encoded_target = self.filtada.encoder(
                    probe_target_x,
                    target_data.edge_index,
                    None if self.mode == "node" else target_data.batch,
                    mode=self.mode,
                    filter_param=self._effective_filter(self.target_filter_param),
                )
                probe_dis_s = torch.mean(self.discriminator(probe_encoded_source).reshape(-1))
                probe_dis_t = torch.mean(self.discriminator(probe_encoded_target).reshape(-1))
                probe_dis_loss = -torch.abs(probe_dis_s - probe_dis_t)
                loss = loss + self.probe_domain_weight * probe_dis_loss

            self.c_optimizer.zero_grad()
            loss.backward()
            self.c_optimizer.step()

        encoded_source = self.filtada(
            source_data, filter_param=self._effective_filter(self.source_filter_param)
        )
        encoded_target = self.filtada(
            target_data, filter_param=self._effective_filter(self.target_filter_param)
        )
        source_logits = self.filtada.cls_model(encoded_source)
        cls_loss = self.filtada.loss_func(source_logits, source_data.y)
        dis_s = torch.mean(self.discriminator(encoded_source).reshape(-1))
        dis_t = torch.mean(self.discriminator(encoded_target).reshape(-1))
        dis_loss = torch.abs(dis_s - dis_t)

        target_logits = self.filtada.cls_model(encoded_target)

        loss = cls_loss + dis_loss * self.domain_weight
        if self.probe_domain_weight > 0.0:
            probe_source_x, probe_target_x = make_gaussian_probe(
                source_data.x.detach(), target_data.x.detach()
            )
            probe_encoded_source = self.filtada.encoder(
                probe_source_x,
                source_data.edge_index,
                None if self.mode == "node" else source_data.batch,
                mode=self.mode,
                filter_param=self._effective_filter(self.source_filter_param),
            )
            probe_encoded_target = self.filtada.encoder(
                probe_target_x,
                target_data.edge_index,
                None if self.mode == "node" else target_data.batch,
                mode=self.mode,
                filter_param=self._effective_filter(self.target_filter_param),
            )
            probe_dis_s = torch.mean(self.discriminator(probe_encoded_source).reshape(-1))
            probe_dis_t = torch.mean(self.discriminator(probe_encoded_target).reshape(-1))
            probe_dis_loss = torch.abs(probe_dis_s - probe_dis_t)
            loss = loss + self.probe_domain_weight * probe_dis_loss

        if self.filter_l1_weight > 0.0:
            # Apply L1 on raw trainable logits (L1 on softmax weights is near-constant).
            l1 = self.source_filter_param.abs().mean() + self.target_filter_param.abs().mean()
            loss = loss + self.filter_l1_weight * l1

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

        self.filtada = self.init_model()
        self.filter_history = []
        self.epoch_history = []
        src_start = self.source_filter_param.detach().cpu().tolist()
        tgt_start = self.target_filter_param.detach().cpu().tolist()
        src_start_eff = self._effective_filter(self.source_filter_param).detach().cpu().tolist()
        tgt_start_eff = self._effective_filter(self.target_filter_param).detach().cpu().tolist()
        print(
            f"[filtada:start] source_logits={src_start} target_logits={tgt_start} "
            f"source_weights={src_start_eff} target_weights={tgt_start_eff}"
        )

        optimizer = torch.optim.Adam(
            list(self.filtada.parameters()) + [self.source_filter_param, self.target_filter_param],
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

        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            for idx, (sampled_source_data, sampled_target_data) in enumerate(
                zip(self.source_loader, self.target_loader)
            ):
                self.filtada.train()

                sampled_source_data = sampled_source_data.to(self.device)
                sampled_target_data = sampled_target_data.to(self.device)

                loss, source_logits, target_logits = self.forward_model(
                    sampled_source_data, sampled_target_data
                )
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if idx == 0:
                    epoch_source_logits, epoch_source_labels = source_logits, sampled_source_data.y
                else:
                    source_logits, source_labels = source_logits, sampled_source_data.y

                    epoch_source_logits = torch.cat((epoch_source_logits, source_logits))
                    epoch_source_labels = torch.cat((epoch_source_labels, source_labels))

            train_results = self.metrics(epoch_source_logits, epoch_source_labels)
            # Track target performance correctly via full eval-mode target inference.
            target_logits, target_labels = self.predict(target_data, source=False)
            test_results = self.metrics(target_logits, target_labels)
            self.log(epoch, epoch_loss, train_results, test_results)
            self.filter_history.append(
                {
                    "epoch": int(epoch),
                    "source_raw": self._effective_filter(self.source_filter_param).detach().cpu().clone(),
                    "target_raw": self._effective_filter(self.target_filter_param).detach().cpu().clone(),
                }
            )
            self.epoch_history.append(
                {
                    "epoch": int(epoch),
                    "train_metrics": train_results,
                    "target_metrics": test_results,
                }
            )

        print(
            f"[filtada:end] source_logits={self.source_filter_param.detach().cpu().tolist()} "
            f"target_logits={self.target_filter_param.detach().cpu().tolist()} "
            f"source_weights={self._effective_filter(self.source_filter_param).detach().cpu().tolist()} "
            f"target_weights={self._effective_filter(self.target_filter_param).detach().cpu().tolist()}"
        )
        self.finish()

    def predict(self, data, source=False):
        self.filtada.eval()

        if source:
            for idx, sampled_data in enumerate(self.source_loader):
                sampled_data = sampled_data.to(self.device)
                with torch.no_grad():
                    encoded_data = self.filtada(
                        sampled_data, filter_param=self._effective_filter(self.source_filter_param)
                    )
                    logits = self.filtada.cls_model(encoded_data)

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
                    encoded_data = self.filtada(
                        sampled_data, filter_param=self._effective_filter(self.target_filter_param)
                    )
                    logits = self.filtada.cls_model(encoded_data)

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
