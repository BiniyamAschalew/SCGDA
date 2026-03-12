import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian

from models.base_model import BaseGDA
from models.__filters.mono import MonoProp


class OPALBase(nn.Module):
    """Shared encoder + classifier used by OPAL."""

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        act=F.relu,
    ):
        super().__init__()

        num_layers = max(1, int(num_layers))
        self.dropout = float(dropout)
        self.act = act

        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(nn.Linear(in_dim, hid_dim))
        for _ in range(1, num_layers):
            self.encoder_layers.append(nn.Linear(hid_dim, hid_dim))

        self.classifier = nn.Linear(hid_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for idx, layer in enumerate(self.encoder_layers):
            h = layer(h)
            if idx < len(self.encoder_layers) - 1:
                h = self.act(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def classify(self, h: torch.Tensor) -> torch.Tensor:
        return self.classifier(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.encode(x))


class OPAL(BaseGDA):
    """Operator-level UGDA with adversarial graph-conditioned probe alignment."""

    def __init__(self, config: dict):
        super().__init__(config)

        model_cfg = config["model"]
        self.config = config

        self.mode = str(model_cfg.get("mode", "node")).lower()
        if self.mode != "node":
            raise AssertionError("OPAL currently supports node mode only.")

        self.use_mask = bool(model_cfg.get("use_mask", False))

        self.filter_order = int(model_cfg.get("K", 3))
        self.probe_order = int(model_cfg.get("Kp", 2))
        self.probe_dim = int(model_cfg.get("probe_dim", 32))
        self.lambda_max = float(model_cfg.get("lambda_max", 2.0))

        self.lambda_feat = float(model_cfg.get("lambda_feat", 1.0))
        self.lambda_filt = float(model_cfg.get("lambda_filt", 1.0))
        self.lambda_probe = float(model_cfg.get("lambda_probe", 0.2))
        self.lambda_spec = float(model_cfg.get("lambda_spec", 0.1))
        self.lambda_ilip = float(model_cfg.get("lambda_ilip", 0.01))

        self.probe_ilip_weight = float(model_cfg.get("probe_ilip_weight", 0.1))

        self.sinkhorn_eps = float(model_cfg.get("sinkhorn_eps", 0.05))
        self.sinkhorn_iters = int(model_cfg.get("sinkhorn_iters", 20))
        self.sinkhorn_tol = float(model_cfg.get("sinkhorn_tol", 1e-8))
        self.sinkhorn_debias = bool(model_cfg.get("sinkhorn_debias", True))
        self.align_sample_size = int(model_cfg.get("align_sample_size", 256))

        self.adv_steps = max(1, int(model_cfg.get("adv_steps", 1)))
        self.adv_lr = float(model_cfg.get("adv_lr", self.lr))
        self.adv_weight_decay = float(model_cfg.get("adv_weight_decay", 0.0))

        coeff_transform = str(model_cfg.get("coeff_transform", "identity")).lower()
        if coeff_transform not in {"identity", "tanh", "softmax"}:
            raise ValueError(
                f"Invalid coeff_transform='{coeff_transform}'. Expected one of: identity, tanh, softmax."
            )
        self.coeff_transform = coeff_transform
        self.coeff_bound = float(model_cfg.get("coeff_bound", 2.0))
        self.coeff_temperature = float(model_cfg.get("coeff_temperature", 1.0))

        num_filter_terms = self.filter_order + 1
        num_probe_terms = self.probe_order + 1

        self.source_theta = nn.Parameter(
            self._init_coeff(
                model_cfg,
                key="source_filter_init",
                num_terms=num_filter_terms,
                default_mode=str(model_cfg.get("source_filter_mode", model_cfg.get("filter_mode", "identity"))),
            ),
            requires_grad=True,
        )
        self.target_theta = nn.Parameter(
            self._init_coeff(
                model_cfg,
                key="target_filter_init",
                num_terms=num_filter_terms,
                default_mode=str(model_cfg.get("target_filter_mode", model_cfg.get("filter_mode", "identity"))),
            ),
            requires_grad=True,
        )
        self.probe_theta = nn.Parameter(
            self._init_coeff(
                model_cfg,
                key="probe_filter_init",
                num_terms=num_probe_terms,
                default_mode=str(model_cfg.get("probe_filter_mode", "identity")),
            ),
            requires_grad=True,
        )

        self.spec_alpha = nn.Parameter(
            torch.tensor(float(model_cfg.get("spec_alpha_init", 0.5)), device=self.device).clamp(1e-3, 1 - 1e-3).logit(),
            requires_grad=True,
        )

        self.mono_prop = MonoProp().to(self.device)

        self.opal = None
        self.source_loader = None
        self.target_loader = None

    def _init_coeff(self, model_cfg: dict, key: str, num_terms: int, default_mode: str) -> torch.Tensor:
        raw = model_cfg.get(key, None)
        if raw is not None:
            coeff = torch.as_tensor(raw, dtype=torch.float32, device=self.device).view(-1)
            if int(coeff.numel()) != int(num_terms):
                raise ValueError(
                    f"{key} length mismatch: expected {num_terms}, got {int(coeff.numel())}."
                )
            return coeff

        mode = str(default_mode).lower()
        coeff = torch.zeros(num_terms, dtype=torch.float32, device=self.device)

        if mode == "uniform":
            coeff.fill_(1.0 / float(num_terms))
        elif mode in {"identity", "a0"}:
            coeff[0] = 1.0
        elif mode in {"adj", "adjacency", "a1"}:
            coeff[1 if num_terms > 1 else 0] = 1.0
        elif mode == "random":
            coeff.normal_(mean=0.0, std=0.02)
        else:
            raise ValueError(
                f"Unsupported initialization mode '{mode}' for {key}. "
                "Expected one of: uniform, identity/a0, adjacency/a1, random."
            )

        return coeff

    def _effective_coeff(self, raw_coeff: torch.Tensor) -> torch.Tensor:
        if self.coeff_transform == "identity":
            return raw_coeff
        if self.coeff_transform == "tanh":
            return self.coeff_bound * torch.tanh(raw_coeff)
        temp = max(float(self.coeff_temperature), 1e-6)
        return torch.softmax(raw_coeff / temp, dim=0)

    def _apply_poly(self, x: torch.Tensor, data, coeff: torch.Tensor) -> torch.Tensor:
        edge_weight = getattr(data, "edge_weight", None)
        coeff = torch.as_tensor(coeff, dtype=x.dtype, device=x.device)
        return self.mono_prop(
            x,
            data.edge_index,
            parameters=coeff,
            edge_weight=edge_weight,
            lambda_max=self.lambda_max,
        )

    def _source_mask(self, source_data) -> torch.Tensor:
        mask = torch.ones_like(source_data.y, dtype=torch.bool, device=source_data.y.device)
        if self.use_mask and hasattr(source_data, "train_mask") and source_data.train_mask is not None:
            mask = source_data.train_mask
        return mask

    @staticmethod
    def _weighted_ilip(coeff: torch.Tensor) -> torch.Tensor:
        if coeff.numel() <= 1:
            return coeff.new_tensor(0.0)
        weights = torch.arange(1, coeff.numel(), device=coeff.device, dtype=coeff.dtype)
        return torch.sum(weights * coeff[1:].abs())

    def _sample_rows(self, x: torch.Tensor) -> torch.Tensor:
        max_rows = int(self.align_sample_size)
        if max_rows <= 0 or x.size(0) <= max_rows:
            return x
        idx = torch.randperm(x.size(0), device=x.device)[:max_rows]
        return x.index_select(0, idx)

    def _sinkhorn_cost(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n, m = int(x.size(0)), int(y.size(0))
        if n == 0 or m == 0:
            return x.new_tensor(0.0)

        # Normalize by feature dimension so epsilon is comparable across feature sizes.
        cost = torch.cdist(x, y, p=2).pow(2) / float(max(1, x.size(1)))

        eps = max(float(self.sinkhorn_eps), 1e-6)
        K = torch.exp(-cost / eps).clamp_min(self.sinkhorn_tol)

        a = x.new_full((n,), 1.0 / float(n))
        b = y.new_full((m,), 1.0 / float(m))

        u = x.new_ones(n)
        v = y.new_ones(m)

        for _ in range(self.sinkhorn_iters):
            Kv = torch.matmul(K, v).clamp_min(self.sinkhorn_tol)
            u = a / Kv
            KTu = torch.matmul(K.t(), u).clamp_min(self.sinkhorn_tol)
            v = b / KTu

        plan = (u.unsqueeze(1) * K) * v.unsqueeze(0)
        return torch.sum(plan * cost)

    def _sinkhorn_divergence(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source = self._sample_rows(source)
        target = self._sample_rows(target)

        st = self._sinkhorn_cost(source, target)
        if not self.sinkhorn_debias:
            return st

        ss = self._sinkhorn_cost(source, source)
        tt = self._sinkhorn_cost(target, target)
        return st - 0.5 * ss - 0.5 * tt

    def _probe_signals(self, source_data, target_data, probe_coeff, source_coeff, target_coeff):
        dtype = source_data.x.dtype

        source_noise = torch.randn(
            source_data.x.size(0),
            self.probe_dim,
            device=self.device,
            dtype=dtype,
        )
        target_noise = torch.randn(
            target_data.x.size(0),
            self.probe_dim,
            device=self.device,
            dtype=dtype,
        )

        source_z = self._apply_poly(source_noise, source_data, probe_coeff)
        target_z = self._apply_poly(target_noise, target_data, probe_coeff)

        source_v = self._apply_poly(source_z, source_data, source_coeff)
        target_v = self._apply_poly(target_z, target_data, target_coeff)

        return source_z, target_z, source_v, target_v

    @staticmethod
    def _energy_anchor(x: torch.Tensor) -> torch.Tensor:
        return (x.pow(2).mean() - 1.0).pow(2)

    def _probe_regularizer(self, source_z: torch.Tensor, target_z: torch.Tensor, probe_coeff: torch.Tensor):
        energy = self._energy_anchor(source_z) + self._energy_anchor(target_z)
        probe_ilip = self._weighted_ilip(probe_coeff).pow(2)
        total = energy + self.probe_ilip_weight * probe_ilip
        return total, energy, probe_ilip

    @staticmethod
    def _laplacian_energy(x: torch.Tensor, data) -> torch.Tensor:
        edge_weight = getattr(data, "edge_weight", None)
        lap_index, lap_weight = get_laplacian(
            data.edge_index,
            edge_weight=edge_weight,
            normalization="sym",
            dtype=x.dtype,
            num_nodes=x.size(0),
        )
        lap = torch.sparse_coo_tensor(
            lap_index,
            lap_weight,
            (x.size(0), x.size(0)),
            device=x.device,
            dtype=x.dtype,
        ).coalesce()
        lx = torch.sparse.mm(lap, x)
        return torch.sum(x * lx) / float(max(1, x.size(0) * x.size(1)))

    def _spectrum_regularizer(self, x: torch.Tensor, data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low = self._laplacian_energy(x, data)
        total_energy = x.pow(2).mean()
        high = 2.0 * total_energy - low

        alpha = torch.sigmoid(self.spec_alpha)
        mixed = alpha * low + (1.0 - alpha) * high
        return mixed, low, high

    def init_model(self, **kwargs):
        return OPALBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            act=self.act,
        ).to(self.device)

    def _forward_losses(self, source_data, target_data, probe_coeff):
        source_u = self.opal.encode(source_data.x)
        target_u = self.opal.encode(target_data.x)

        source_coeff = self._effective_coeff(self.source_theta)
        target_coeff = self._effective_coeff(self.target_theta)

        source_f = self._apply_poly(source_u, source_data, source_coeff)
        target_f = self._apply_poly(target_u, target_data, target_coeff)

        source_logits = self.opal.classify(source_f)
        target_logits = self.opal.classify(target_f)

        source_mask = self._source_mask(source_data)
        cls_loss = F.cross_entropy(source_logits[source_mask], source_data.y[source_mask])

        feat_loss = self._sinkhorn_divergence(source_u, target_u)

        source_z, target_z, source_v, target_v = self._probe_signals(
            source_data,
            target_data,
            probe_coeff=probe_coeff,
            source_coeff=source_coeff,
            target_coeff=target_coeff,
        )

        filt_loss = self._sinkhorn_divergence(source_v, target_v)

        spec_source, low_source, high_source = self._spectrum_regularizer(source_v, source_data)
        spec_target, low_target, high_target = self._spectrum_regularizer(target_v, target_data)
        spec_loss = spec_source + spec_target

        ilip_loss = self._weighted_ilip(source_coeff).pow(2) + self._weighted_ilip(target_coeff).pow(2)

        total_loss = cls_loss
        total_loss = total_loss + self.lambda_feat * feat_loss
        total_loss = total_loss + self.lambda_filt * filt_loss
        total_loss = total_loss + self.lambda_spec * spec_loss
        total_loss = total_loss + self.lambda_ilip * ilip_loss

        logs = {
            "cls_loss": float(cls_loss.detach().cpu().item()),
            "feat_sinkhorn": float(feat_loss.detach().cpu().item()),
            "filt_sinkhorn": float(filt_loss.detach().cpu().item()),
            "spec_loss": float(spec_loss.detach().cpu().item()),
            "spec_low_source": float(low_source.detach().cpu().item()),
            "spec_low_target": float(low_target.detach().cpu().item()),
            "spec_high_source": float(high_source.detach().cpu().item()),
            "spec_high_target": float(high_target.detach().cpu().item()),
            "ilip_loss": float(ilip_loss.detach().cpu().item()),
            "alpha": float(torch.sigmoid(self.spec_alpha).detach().cpu().item()),
        }

        return total_loss, source_logits, target_logits, logs, source_z, target_z

    def _adversarial_probe_objective(self, source_data, target_data):
        source_coeff = self._effective_coeff(self.source_theta).detach()
        target_coeff = self._effective_coeff(self.target_theta).detach()
        probe_coeff = self._effective_coeff(self.probe_theta)

        source_z, target_z, source_v, target_v = self._probe_signals(
            source_data,
            target_data,
            probe_coeff=probe_coeff,
            source_coeff=source_coeff,
            target_coeff=target_coeff,
        )

        filt_loss = self._sinkhorn_divergence(source_v, target_v)
        probe_reg, probe_energy, probe_ilip = self._probe_regularizer(source_z, target_z, probe_coeff)

        objective = self.lambda_filt * filt_loss - self.lambda_probe * probe_reg

        logs = {
            "adv_filt_sinkhorn": float(filt_loss.detach().cpu().item()),
            "adv_probe_reg": float(probe_reg.detach().cpu().item()),
            "adv_probe_energy": float(probe_energy.detach().cpu().item()),
            "adv_probe_ilip": float(probe_ilip.detach().cpu().item()),
            "adv_objective": float(objective.detach().cpu().item()),
        }
        return objective, logs

    def forward_model(self, source_data, target_data):
        probe_coeff = self._effective_coeff(self.probe_theta).detach()
        loss, source_logits, target_logits, logs, _, _ = self._forward_losses(
            source_data,
            target_data,
            probe_coeff=probe_coeff,
        )
        return loss, source_logits, target_logits, logs

    def fit(self, source_data, target_data):
        self.num_source_nodes = int(source_data.x.shape[0])
        self.num_target_nodes = int(target_data.x.shape[0])

        self.source_loader = self.get_loader(source_data, batch_size=self.batch_size)
        self.target_loader = self.get_loader(target_data, batch_size=self.batch_size)

        self.opal = self.init_model()

        main_params = list(self.opal.parameters())
        main_params += [self.source_theta, self.target_theta, self.spec_alpha]

        optimizer = torch.optim.Adam(
            main_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        probe_optimizer = torch.optim.Adam(
            [self.probe_theta],
            lr=self.adv_lr,
            weight_decay=self.adv_weight_decay,
        )

        start_time = time.time()

        from tqdm import tqdm

        for epoch in tqdm(range(self.epoch), desc="Training"):
            epoch_loss = 0.0
            epoch_source_logits = torch.empty((0, self.num_classes), device=self.device)
            epoch_source_labels = torch.empty(0, dtype=torch.long, device=self.device)

            running = {
                "cls_loss": 0.0,
                "feat_sinkhorn": 0.0,
                "filt_sinkhorn": 0.0,
                "spec_loss": 0.0,
                "spec_low_source": 0.0,
                "spec_low_target": 0.0,
                "spec_high_source": 0.0,
                "spec_high_target": 0.0,
                "ilip_loss": 0.0,
                "alpha": 0.0,
                "adv_filt_sinkhorn": 0.0,
                "adv_probe_reg": 0.0,
                "adv_probe_energy": 0.0,
                "adv_probe_ilip": 0.0,
                "adv_objective": 0.0,
            }
            n_batches = 0

            for sampled_source_data, sampled_target_data in zip(self.source_loader, self.target_loader):
                n_batches += 1

                sampled_source_data = sampled_source_data.to(self.device)
                sampled_target_data = sampled_target_data.to(self.device)

                self.opal.train()

                adv_logs = None
                for _ in range(self.adv_steps):
                    probe_optimizer.zero_grad()
                    adv_objective, adv_logs = self._adversarial_probe_objective(
                        sampled_source_data,
                        sampled_target_data,
                    )
                    (-adv_objective).backward()
                    probe_optimizer.step()

                loss, source_logits, _, logs = self.forward_model(sampled_source_data, sampled_target_data)
                epoch_loss += float(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                source_mask = self._source_mask(sampled_source_data)
                epoch_source_logits = torch.cat(
                    (epoch_source_logits, source_logits[source_mask].detach()),
                    dim=0,
                )
                epoch_source_labels = torch.cat(
                    (epoch_source_labels, sampled_source_data.y[source_mask].detach()),
                    dim=0,
                )

                for key in logs:
                    running[key] += float(logs[key])
                if adv_logs is not None:
                    for key in adv_logs:
                        running[key] += float(adv_logs[key])

            train_results = self.metrics(epoch_source_logits, epoch_source_labels)
            target_logits, target_labels = self.predict(target_data, source=False)
            test_results = self.metrics(target_logits, target_labels)
            self.log(epoch, epoch_loss, train_results, test_results)

            if n_batches > 0:
                payload = {f"opal_{k}": v / float(n_batches) for k, v in running.items()}
                payload["epoch"] = epoch + 1
                payload["opal_source_theta_0"] = float(self._effective_coeff(self.source_theta)[0].detach().cpu().item())
                payload["opal_target_theta_0"] = float(self._effective_coeff(self.target_theta)[0].detach().cpu().item())
                payload["opal_probe_theta_0"] = float(self._effective_coeff(self.probe_theta)[0].detach().cpu().item())
                self.wandb.log(payload)

        self.train_time = time.time() - start_time
        self.finish()

    def process_graph(self, data):
        pass

    def predict(self, data, source=False):
        self.opal.eval()

        loader = self.source_loader if source else self.target_loader
        coeff = self._effective_coeff(self.source_theta if source else self.target_theta)

        all_logits = torch.empty((0, self.num_classes), device=self.device)
        all_labels = torch.empty(0, dtype=torch.long, device=self.device)

        for sampled_data in loader:
            sampled_data = sampled_data.to(self.device)
            with torch.no_grad():
                embed = self.opal.encode(sampled_data.x)
                feat = self._apply_poly(embed, sampled_data, coeff)
                logits = self.opal.classify(feat)

            all_logits = torch.cat((all_logits, logits), dim=0)
            all_labels = torch.cat((all_labels, sampled_data.y), dim=0)

        return all_logits, all_labels
