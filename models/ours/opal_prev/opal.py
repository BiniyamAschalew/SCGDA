import time
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from models.__filters.mono import MonoProp
from models.__layers.build_layer import build_activation, build_layer
from models.base_model import BaseGDA
from utils.train_utils.mmd import MMD, Sinkhorn


class LearnableProbeGenerator(nn.Module):
    """Learnable Gaussian probe distribution with per-dimension mean/std."""

    def __init__(self, dim: int):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(1, dim))
        self.log_std = nn.Parameter(torch.zeros(1, dim))

    def sample(self, num_nodes: int, device, dtype, detach_params: bool = False) -> torch.Tensor:
        mean = self.mean
        log_std = self.log_std
        if detach_params:
            mean = mean.detach()
            log_std = log_std.detach()

        std = F.softplus(log_std) + 1e-6
        noise = torch.randn(num_nodes, mean.size(1), device=device, dtype=dtype)
        return mean.to(device=device, dtype=dtype) + std.to(device=device, dtype=dtype) * noise

    def regularizer(self) -> torch.Tensor:
        return self.mean.pow(2).mean() + self.log_std.pow(2).mean()


class OPALBase(nn.Module):
    """Backbone + classifier. Kept close to the original OPAL layout."""

    def __init__(self, config: dict):
        super(OPALBase, self).__init__()

        self.in_dim = config["model"]["in_dim"]
        self.hid_dim = config["model"]["hid_dim"]
        self.num_classes = config["model"]["num_classes"]
        self.num_layers = config["model"]["num_layers"]
        self.dropout = config["model"]["dropout_ratio"]
        self.mode = config["model"]["mode"]
        self.act = build_activation(config["model"]["activation"])
        self.gnn_type = config["model"]["gnn"].lower()
        self.K = int(config["model"].get("K", 1))

        self.convs = nn.ModuleList()
        self.convs.append(build_layer(self.in_dim, self.hid_dim, self.gnn_type))
        for _ in range(self.num_layers - 1):
            self.convs.append(build_layer(self.hid_dim, self.hid_dim, self.gnn_type))

        if self.mode == "node":
            self.cls = build_layer(self.hid_dim, self.num_classes, self.gnn_type)
        elif self.mode == "graph":
            self.cls = nn.Linear(self.hid_dim, self.num_classes)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _run_layer(self, layer, x, edge_index, params):
        if self.gnn_type.startswith("filter"):
            return layer(x, edge_index, filter_param=params)
        if self.gnn_type == "prop":
            return layer(x, edge_index, prop_nums=max(1, self.K))
        return layer(x, edge_index)

    def forward(self, data, params):
        if self.mode == "node":
            x, edge_index, batch = data.x, data.edge_index, None
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.feat_bottleneck(x, edge_index, batch, params=params)
        x = self.feat_classifier(x, edge_index, batch, params)
        return x

    def feat_bottleneck(self, x, edge_index, batch, params):
        for conv in self.convs:
            x = self._run_layer(conv, x, edge_index, params)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.mode == "graph":
            x = global_mean_pool(x, batch)
        return x

    def feat_classifier(self, x, edge_index, batch, params):
        if self.mode == "node":
            x = self._run_layer(self.cls, x, edge_index, params)
        else:
            x = self.cls(x)
        return x


class OPAL(BaseGDA):
    """
    OPAL (simplified):
    - supervised source classification
    - feature alignment (alpha)
    - probe-based filter alignment (beta)
    - semantic alignment for same labels (gamma)
    - structure matching on probe responses (delta)
    """

    def __init__(self, config: dict):
        super(OPAL, self).__init__(config)

        self.config = config
        self.K = int(config["model"]["K"])  # K + 1 coefficients including identity
        self.mode = config["model"]["mode"]

        # core weights (kept simple/readable)
        self.alpha = float(config["model"].get("alpha", 1.0))  # feature alignment
        self.beta = float(config["model"].get("beta", 1.0))  # filter alignment
        self.gamma = float(config["model"].get("gamma", 0.1))  # semantic alignment
        self.delta = float(config["model"].get("delta", 0.1))  # structure alignment

        # optional controls
        self.adv = bool(config["model"].get("adv", False))  # kept for compatibility
        self.use_mask = bool(config["model"].get("use_mask", False))
        self.filter_nonneg = bool(config["model"].get("filter_nonneg", False))
        self.filter_reg = bool(config["model"].get("filter_reg", False))
        self.prob_semantic = bool(config["model"].get("prob_semantic", True))
        self.filter_reg_weight = float(config["model"].get("filter_reg_weight", 0.01))
        self.corr_weight = float(config["model"].get("corr_weight", 0.5))
        self.semantic_sample_per_class = int(config["model"].get("semantic_sample_per_class", 128))

        self.probe_dim = int(config["model"].get("probe_dim", self.hid_dim))
        self.pseudo_label_threshold = float(config["model"].get("pseudo_label_threshold", 0.8))
        self.pseudo_label_temperature = float(config["model"].get("pseudo_label_temperature", 1.0))
        self.learnable_probe = bool(config["model"].get("learnable_probe", False))
        self.probe_adv_steps = int(config["model"].get("probe_adv_steps", 1))
        self.probe_adv_lr = float(config["model"].get("probe_adv_lr", self.lr))
        self.probe_adv_weight_decay = float(config["model"].get("probe_adv_weight_decay", 0.0))
        self.probe_reg_weight = float(config["model"].get("probe_reg_weight", 1e-3))

        self.lambda_max = float(config["model"].get("lambda_max", 2.0))

        # shared distance interface
        self.distance_name = str(config["model"].get("distance", "mmd")).lower()
        # Match A2GNN default sampling regime for MMD unless explicitly overridden.
        self.distance_sampling_num = int(config["model"].get("distance_sampling_num", 1000))
        self.distance_times = int(config["model"].get("distance_times", 5))
        self.sinkhorn_eps = float(config["model"].get("sinkhorn_eps", 0.05))
        self.sinkhorn_p = int(config["model"].get("sinkhorn_p", 2))
        self.sinkhorn_scaling = float(config["model"].get("sinkhorn_scaling", 0.9))
        self.sinkhorn_backend = str(config["model"].get("sinkhorn_backend", "auto"))
        self.sinkhorn_debias = bool(config["model"].get("sinkhorn_debias", True))

        init_val = 1.0 / (self.K + 1)
        self.s_params = nn.Parameter(torch.full((self.K + 1,), init_val, device=self.device))
        self.t_params = nn.Parameter(torch.full((self.K + 1,), init_val, device=self.device))
        self.propagator = MonoProp().to(self.device)
        self.probe_generator = None
        if self.learnable_probe:
            self.probe_generator = LearnableProbeGenerator(self.probe_dim).to(self.device)

        # Probe semantic matching is defined for node labels.
        if self.mode != "node":
            self.prob_semantic = False
        self.param_trace = []

    def init_model(self):
        return OPALBase(self.config).to(self.device)

    def _distance(self, source_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
        if source_feat.size(0) == 0 or target_feat.size(0) == 0:
            return source_feat.new_tensor(0.0)

        sampling_num = min(
            int(self.distance_sampling_num),
            int(source_feat.size(0)),
            int(target_feat.size(0)),
        )
        sampling_num = max(1, sampling_num)

        if self.distance_name == "sinkhorn":
            return Sinkhorn(
                source_feat,
                target_feat,
                sampling_num=sampling_num,
                times=max(1, self.distance_times),
                blur=self.sinkhorn_eps,
                p=self.sinkhorn_p,
                scaling=self.sinkhorn_scaling,
                debias=self.sinkhorn_debias,
                backend=self.sinkhorn_backend,
            )

        if self.distance_name == "mmd":
            # A2GNN uses MMD with random sub-sampling; we mirror that here.
            return MMD(
                source_feat,
                target_feat,
                sampling_num=sampling_num,
                times=max(1, self.distance_times),
            )

        raise ValueError(f"Unsupported distance: {self.distance_name}")

    def _source_mask(self, source_data):
        mask = torch.ones_like(source_data.y, dtype=torch.bool, device=source_data.y.device)
        if self.use_mask and hasattr(source_data, "train_mask") and source_data.train_mask is not None:
            mask = source_data.train_mask
        if int(mask.sum().item()) == 0:
            mask = torch.ones_like(source_data.y, dtype=torch.bool, device=source_data.y.device)
        return mask

    def _clamp_filter_params(self):
        if self.filter_nonneg:
            self.s_params.data.clamp_(min=0.0)
            self.t_params.data.clamp_(min=0.0)

    def _module_param_l2(self, module: nn.Module) -> float:
        total = 0.0
        for p in module.parameters(recurse=True):
            if p is None or p.numel() == 0:
                continue
            total += float(torch.norm(p.detach(), p=2).item())
        return total

    def _snapshot_epoch_params(self, epoch: int, epoch_loss: float, train_results: dict) -> None:
        row = {
            "epoch": int(epoch + 1),
            "loss": float(epoch_loss),
            "train_micro_f1": float(train_results.get("micro_f1", float("nan"))),
            "train_macro_f1": float(train_results.get("macro_f1", float("nan"))),
        }

        s = self.s_params.detach().cpu().view(-1)
        t = self.t_params.detach().cpu().view(-1)
        for i in range(s.numel()):
            row[f"s_param_{i}"] = float(s[i].item())
        for i in range(t.numel()):
            row[f"t_param_{i}"] = float(t[i].item())

        if self.learnable_probe and self.probe_generator is not None:
            pm = self.probe_generator.mean.detach().cpu().view(-1)
            ps = F.softplus(self.probe_generator.log_std.detach()).cpu().view(-1)
            row["probe_mean_l2"] = float(torch.norm(pm, p=2).item())
            row["probe_std_mean"] = float(ps.mean().item())
            row["probe_std_min"] = float(ps.min().item())
            row["probe_std_max"] = float(ps.max().item())

        for i, conv in enumerate(self.opal.convs):
            row[f"conv_{i}_param_l2"] = self._module_param_l2(conv)
        row["cls_param_l2"] = self._module_param_l2(self.opal.cls)

        self.param_trace.append(row)

    def _laplacian_energy(self, x: torch.Tensor, data) -> torch.Tensor:
        # Memory-light Dirichlet energy: avoids constructing sparse Laplacian matrix.
        edge_index = data.edge_index
        if edge_index.numel() == 0:
            return x.new_tensor(0.0)

        row, col = edge_index[0], edge_index[1]
        diff = x[row] - x[col]
        edge_weight = getattr(data, "edge_weight", None)
        if edge_weight is None:
            return diff.pow(2).sum(dim=1).mean()
        return (edge_weight * diff.pow(2).sum(dim=1)).mean()

    def generate_probe(self, data, probe_mean=None, probe_std=None, detach_generator=True):
        """Generate probes from a shared distribution for both domains."""
        num_nodes = data.x.size(0)

        if self.learnable_probe and self.probe_generator is not None:
            return self.probe_generator.sample(
                num_nodes=num_nodes,
                device=data.x.device,
                dtype=data.x.dtype,
                detach_params=detach_generator,
            )

        probe = torch.randn(
            num_nodes,
            self.probe_dim,
            device=data.x.device,
            dtype=data.x.dtype,
        )

        if (
            probe_mean is not None
            and probe_std is not None
            and int(probe_mean.size(-1)) == int(self.probe_dim)
            and int(probe_std.size(-1)) == int(self.probe_dim)
        ):
            probe = probe * probe_std + probe_mean
        return probe

    def _probe_projection_distance(
        self,
        source_data,
        target_data,
        source_embed=None,
        target_embed=None,
        detach_generator=True,
        detach_filter=False,
    ):
        probe_mean, probe_std = None, None
        if not self.learnable_probe:
            if source_embed is None or target_embed is None:
                raise ValueError("source_embed/target_embed are required when learnable_probe is False.")
            probe_stats = torch.cat([source_embed.detach(), target_embed.detach()], dim=0)
            probe_mean = probe_stats.mean(dim=0, keepdim=True)
            probe_std = probe_stats.std(dim=0, keepdim=True).clamp_min(1e-6)

        source_probe = self.generate_probe(
            source_data,
            probe_mean=probe_mean,
            probe_std=probe_std,
            detach_generator=detach_generator,
        )
        target_probe = self.generate_probe(
            target_data,
            probe_mean=probe_mean,
            probe_std=probe_std,
            detach_generator=detach_generator,
        )

        s_params = self.s_params.detach() if detach_filter else self.s_params
        t_params = self.t_params.detach() if detach_filter else self.t_params

        source_forward = self.propagator(
            source_probe,
            source_data.edge_index,
            s_params,
            edge_weight=getattr(source_data, "edge_weight", None),
            lambda_max=self.lambda_max,
        )
        target_forward = self.propagator(
            target_probe,
            target_data.edge_index,
            t_params,
            edge_weight=getattr(target_data, "edge_weight", None),
            lambda_max=self.lambda_max,
        )
        return self._distance(source_forward, target_forward), source_forward, target_forward

    def probe_adversarial_objective(self, source_data, target_data):
        if not self.learnable_probe or self.probe_generator is None:
            return None
        proj_dist, _, _ = self._probe_projection_distance(
            source_data,
            target_data,
            detach_generator=False,
            detach_filter=True,
        )
        reg = self.probe_generator.regularizer()
        return proj_dist - self.probe_reg_weight * reg

    def semantic_loss(self, features, labels):
        """Intra-class compactness."""
        if features.numel() == 0 or labels.numel() == 0:
            return features.new_tensor(0.0)

        classes = torch.unique(labels)
        if int(classes.numel()) == 0:
            return features.new_tensor(0.0)

        loss = features.new_tensor(0.0)
        valid = 0
        for c in classes:
            class_mask = labels == c
            if int(class_mask.sum().item()) > 1:
                class_feat = features[class_mask]
                max_k = max(2, int(self.semantic_sample_per_class))
                if class_feat.size(0) > max_k:
                    idx = torch.randperm(class_feat.size(0), device=class_feat.device)[:max_k]
                    class_feat = class_feat.index_select(0, idx)

                intra = F.cosine_similarity(
                    class_feat.unsqueeze(1),
                    class_feat.unsqueeze(0),
                    dim=-1,
                )
                loss = loss + (1.0 - intra).mean()
                valid += 1
        if valid == 0:
            return features.new_tensor(0.0)
        return loss / float(valid)

    def _cross_domain_semantic(self, source_feat, source_labels, target_feat, target_logits):
        """Match class centers across domains and their class-correlation matrix."""
        probs = torch.softmax(target_logits.detach() / max(1e-6, self.pseudo_label_temperature), dim=1)
        conf, pseudo = probs.max(dim=1)
        target_mask = conf >= self.pseudo_label_threshold

        proto_s, proto_t = [], []
        proto_loss = source_feat.new_tensor(0.0)

        for c in torch.unique(source_labels):
            s_mask = source_labels == c
            t_mask = target_mask & (pseudo == c)
            if int(s_mask.sum().item()) < 2 or int(t_mask.sum().item()) < 2:
                continue

            s_center = source_feat[s_mask].mean(dim=0)
            t_center = target_feat[t_mask].mean(dim=0)
            proto_s.append(s_center)
            proto_t.append(t_center)
            proto_loss = proto_loss + (
                1.0 - F.cosine_similarity(s_center.view(1, -1), t_center.view(1, -1)).mean()
            )

        pair_count = len(proto_s)
        if pair_count == 0:
            return source_feat.new_tensor(0.0)

        proto_loss = proto_loss / float(pair_count)
        corr_loss = source_feat.new_tensor(0.0)
        if pair_count > 1:
            s_proto = F.normalize(torch.stack(proto_s, dim=0), p=2, dim=1)
            t_proto = F.normalize(torch.stack(proto_t, dim=0), p=2, dim=1)
            s_corr = torch.matmul(s_proto, s_proto.t())
            t_corr = torch.matmul(t_proto, t_proto.t())
            corr_loss = F.mse_loss(s_corr, t_corr)

        return proto_loss + self.corr_weight * corr_loss

    def filter_alignment_loss(self, source_data, target_data, source_embed, target_embed, target_logits):
        alignment_loss, source_forward, target_forward = self._probe_projection_distance(
            source_data,
            target_data,
            source_embed=source_embed,
            target_embed=target_embed,
            detach_generator=True,
            detach_filter=False,
        )

        semantic_term = source_forward.new_tensor(0.0)
        if self.prob_semantic:
            semantic_term = semantic_term + self.semantic_loss(source_forward, source_data.y)

            probs = torch.softmax(target_logits.detach() / max(1e-6, self.pseudo_label_temperature), dim=1)
            conf, pseudo = probs.max(dim=1)
            pseudo_mask = conf >= self.pseudo_label_threshold
            if int(pseudo_mask.sum().item()) > 1:
                semantic_term = semantic_term + self.semantic_loss(
                    target_forward[pseudo_mask],
                    pseudo[pseudo_mask],
                )

            semantic_term = semantic_term + self._cross_domain_semantic(
                source_forward,
                source_data.y,
                target_forward,
                target_logits,
            )

        structure_term = torch.abs(
            self._laplacian_energy(source_forward, source_data)
            - self._laplacian_energy(target_forward, target_data)
        )

        filter_reg_loss = source_forward.new_tensor(0.0)
        if self.filter_reg:
            order = torch.arange(self.K + 1, device=self.s_params.device, dtype=self.s_params.dtype)
            filter_reg_loss = torch.sum(order * self.s_params.abs()) + torch.sum(order * self.t_params.abs())

        loss = alignment_loss
        loss = loss + self.gamma * semantic_term
        loss = loss + self.delta * structure_term
        loss = loss + self.filter_reg_weight * filter_reg_loss
        return loss

    def forward_model(self, source_data, target_data, alpha):
        # Source domain classification
        source_logits = self.opal(source_data, self.s_params)
        target_logits = self.opal(target_data, self.t_params)

        self._clamp_filter_params()

        source_mask = self._source_mask(source_data)
        cls_loss = F.cross_entropy(source_logits[source_mask], source_data.y[source_mask])
        loss = cls_loss

        if self.mode == "node":
            source_batch = None
            target_batch = None
        else:
            source_batch = source_data.batch
            target_batch = target_data.batch

        source_features = self.opal.feat_bottleneck(
            source_data.x, source_data.edge_index, source_batch, self.s_params
        )
        target_features = self.opal.feat_bottleneck(
            target_data.x, target_data.edge_index, target_batch, self.t_params
        )

        feat_align_loss = self._distance(source_features, target_features)
        loss = loss + self.alpha * feat_align_loss

        filt_align_loss = self.filter_alignment_loss(
            source_data,
            target_data,
            source_features,
            target_features,
            target_logits,
        )
        loss = loss + self.beta * filt_align_loss

        return loss, source_logits, target_logits

    def fit(self, source_data, target_data):
        self.num_source_nodes = source_data.x.shape[0]
        self.num_target_nodes = target_data.x.shape[0]

        if self.mode == "node":
            self.source_loader = self.get_loader(source_data, batch_size=self.batch_size)
            self.target_loader = self.get_loader(target_data, batch_size=self.batch_size)
        elif self.mode == "graph":
            src_batch_size = self.batch_size if self.batch_size > 0 else len(source_data)
            tgt_batch_size = self.batch_size if self.batch_size > 0 else len(target_data)
            self.source_loader = DataLoader(source_data, batch_size=src_batch_size, shuffle=True)
            self.target_loader = DataLoader(target_data, batch_size=tgt_batch_size, shuffle=True)
        else:
            raise AssertionError("Invalid train mode")

        self.opal = self.init_model()
        optimizer = torch.optim.Adam(
            list(self.opal.parameters()) + [self.s_params, self.t_params],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        probe_optimizer = None
        if self.learnable_probe and self.probe_generator is not None:
            probe_optimizer = torch.optim.Adam(
                self.probe_generator.parameters(),
                lr=self.probe_adv_lr,
                weight_decay=self.probe_adv_weight_decay,
            )

        start_time = time.time()

        for epoch in tqdm(range(self.epoch), desc="Training"):
            epoch_loss = 0.0
            epoch_source_logits = torch.empty((0, self.num_classes), device=self.device)
            epoch_source_labels = torch.empty(0, dtype=torch.long, device=self.device)

            p = float(epoch) / max(1, self.epoch)
            alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

            for sampled_source_data, sampled_target_data in zip(self.source_loader, self.target_loader):
                self.opal.train()

                sampled_source_data = sampled_source_data.to(self.device)
                sampled_target_data = sampled_target_data.to(self.device)

                if probe_optimizer is not None and self.probe_adv_steps > 0:
                    for _ in range(self.probe_adv_steps):
                        probe_optimizer.zero_grad()
                        adv_obj = self.probe_adversarial_objective(
                            sampled_source_data,
                            sampled_target_data,
                        )
                        if adv_obj is not None:
                            (-adv_obj).backward()
                            probe_optimizer.step()

                loss, source_logits, _ = self.forward_model(
                    sampled_source_data,
                    sampled_target_data,
                    alpha,
                )
                epoch_loss += float(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self._clamp_filter_params()

                source_mask = self._source_mask(sampled_source_data)
                epoch_source_logits = torch.cat(
                    (epoch_source_logits, source_logits[source_mask].detach()),
                    dim=0,
                )
                epoch_source_labels = torch.cat(
                    (epoch_source_labels, sampled_source_data.y[source_mask].detach()),
                    dim=0,
                )

            train_results = self.metrics(epoch_source_logits, epoch_source_labels)
            self.log(epoch, epoch_loss, train_results)
            self._snapshot_epoch_params(epoch, epoch_loss, train_results)

        self.train_time = time.time() - start_time
        self.finish()

    def predict(self, data, source=False):
        self.opal.eval()
        params = self.s_params if source else self.t_params
        logits = self.opal(data, params)
        labels = data.y
        return logits, labels
