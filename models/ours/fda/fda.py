import time
import torch
import torch.nn.functional as F
from torch import nn

from models.base_model import BaseGDA
from models.ours.fda.fda_base import FDABase
from models.ours.fda.objective import FDAFilterAlignObjective


class FDA(BaseGDA):
    """Filter-aware domain adaptation with feature-MMD + probe/filter-MMD."""

    def __init__(self, config):
        super(FDA, self).__init__(config)
        self.config = config

        self.verbose = config["expt"]["verbose"]
        self.batch_size = config["model"]["batch_size"]

        self.lr = config["model"]["lr"]
        self.weight_decay = config["model"]["weight_decay"]

        self.k = int(config["model"]["K"])
        self.filter_name = str(config["model"].get("filter", "cheb")).lower()
        self.feature_mmd_weight = float(config["model"].get("mmd_weight", 0.1))
        self.probe_mmd_weight = float(
            config["model"].get(
                "probe_mmd_weight",
                config["model"].get("align_weight", 1.0),
            )
        )
        self.nonnegative_filter_params = bool(config["model"].get("nonnegative_filter_params", False))

        self.objective = FDAFilterAlignObjective(
            filter_name=self.filter_name,
            mmd_sampling_num=int(config["model"].get("mmd_sampling_num", 1000)),
            mmd_times=int(config["model"].get("mmd_times", 5)),
            nonnegative_params=self.nonnegative_filter_params,
        )

        init = torch.ones(self.k, device=self.device, dtype=torch.float32) / float(self.k)
        self.source_params = nn.Parameter(init.clone(), requires_grad=True)
        self.target_params = nn.Parameter(init.clone(), requires_grad=True)

        self.fda = None

    def init_model(self, **kwargs):
        return FDABase(self.config).to(self.device)

    def _domain_filter_params(self, domain: str) -> torch.nn.Parameter:
        if domain == "source":
            return self.source_params
        if domain == "target":
            return self.target_params
        raise ValueError(f"Invalid domain: {domain}")

    def _backbone_features(self, data) -> torch.Tensor:
        batch = getattr(data, "batch", None)
        edge_weight = getattr(data, "edge_weight", None)
        return self.fda.feat_bottleneck(
            data.x,
            data.edge_index,
            edge_weight=edge_weight,
            batch=batch,
        )

    def _apply_domain_filter(self, features: torch.Tensor, data, domain: str) -> torch.Tensor:
        edge_weight = getattr(data, "edge_weight", None)
        params = self._domain_filter_params(domain)
        return self.objective.apply_filter(
            features,
            data.edge_index,
            params=params,
            edge_weight=edge_weight,
        )

    def _domain_logits(self, features: torch.Tensor, data) -> torch.Tensor:
        edge_weight = getattr(data, "edge_weight", None)
        logits = self.fda.feat_classifier(features, data.edge_index, edge_weight=edge_weight)
        return F.log_softmax(logits, dim=1)

    def filter_alignment_loss(self, source_features, target_features, source_data, target_data):
        source_edge_weight = getattr(source_data, "edge_weight", None)
        target_edge_weight = getattr(target_data, "edge_weight", None)
        return self.objective.probe_mmd(
            source_embed=source_features,
            target_embed=target_features,
            source_edge_index=source_data.edge_index,
            target_edge_index=target_data.edge_index,
            source_params=self.source_params,
            target_params=self.target_params,
            source_edge_weight=source_edge_weight,
            target_edge_weight=target_edge_weight,
        ).to(self.device)

    def forward_model(self, source_data, target_data):
        source_embed = self._backbone_features(source_data)
        target_embed = self._backbone_features(target_data)

        source_features = self._apply_domain_filter(source_embed, source_data, domain="source")
        target_features = self._apply_domain_filter(target_embed, target_data, domain="target")

        source_logits = self._domain_logits(source_features, source_data)
        target_logits = self._domain_logits(target_features, target_data)

        cls_loss = F.nll_loss(source_logits, source_data.y)
        feat_mmd_loss = self.objective.feature_mmd(source_embed, target_embed).to(self.device)
        probe_mmd_loss = self.filter_alignment_loss(
            source_embed,
            target_embed,
            source_data,
            target_data,
        )

        loss = cls_loss
        loss = loss + self.feature_mmd_weight * feat_mmd_loss
        loss = loss + self.probe_mmd_weight * probe_mmd_loss

        self.wandb.log(
            {
                "cls_loss": cls_loss.item(),
                "feat_mmd_loss": feat_mmd_loss.item(),
                "probe_mmd_loss": probe_mmd_loss.item(),
            }
        )
        return loss, source_logits, target_logits

    def fit(self, source_data, target_data):
        source_loader = self.get_loader(source_data)
        target_loader = self.get_loader(target_data)

        self.fda = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            list(self.fda.parameters()) + [self.source_params, self.target_params],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        start_time = time.time()

        from tqdm import tqdm

        for epoch in tqdm(range(self.epoch), desc="Training"):
            epoch_loss = 0.0
            epoch_source_logits = None
            epoch_source_labels = None

            for sampled_source_data, sampled_target_data in zip(source_loader, target_loader):
                self.fda.train()
                sampled_source_data = sampled_source_data.to(self.device)
                sampled_target_data = sampled_target_data.to(self.device)

                loss, source_logits, _ = self.forward_model(sampled_source_data, sampled_target_data)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred_logits, source_labels = self.predict(sampled_source_data, domain="source")
                if epoch_source_logits is None:
                    epoch_source_logits = pred_logits
                    epoch_source_labels = source_labels
                else:
                    epoch_source_logits = torch.cat((epoch_source_logits, pred_logits))
                    epoch_source_labels = torch.cat((epoch_source_labels, source_labels))

            train_results = self.metrics(epoch_source_logits, epoch_source_labels)
            self.log(epoch, epoch_loss, train_results)

        self.train_time = time.time() - start_time
        self.finish()

    def process_graph(self, data):
        pass

    def predict(self, data, domain="target"):
        self.fda.eval()

        with torch.no_grad():
            embed = self._backbone_features(data)
            features = self._apply_domain_filter(embed, data, domain=domain)
            logits = self._domain_logits(features, data)

        return logits, data.y
