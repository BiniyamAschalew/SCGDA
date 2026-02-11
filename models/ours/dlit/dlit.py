import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base_model import BaseGDA
from models.ours.dlit.dlit_base import DLITBase
from utils.train_utils.mmd import MMD


class DLIT(BaseGDA):
    """DLIT with one shared filter and four losses."""

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config["model"]

        self.mode = model_cfg["mode"]
        self.K = model_cfg.get("K", 8)

        # loss weights
        self.alpha = model_cfg.get("alpha", 0.05)   # IID-filter MMD
        self.beta = model_cfg.get("beta", 0.5)      # feature MMD
        self.gamma = model_cfg.get("gamma", 0.05)   # target entropy
        self.cls_weight = model_cfg.get("cls_weight", 1.0)

        # IID Gaussian source for filter alignment
        self.iid_mean = model_cfg.get("iid_mean", 1.0)
        self.iid_std = model_cfg.get("iid_std", 0.25)

        assert self.num_layers == 2, "unsupported number of layers"
        assert self.mode == "node", "unsupported mode"

        self.dlit = None
        self.source_loader = None
        self.target_loader = None

    def init_model(self):
        return DLITBase(
            features=self.in_dim,
            hidden=self.hid_dim,
            classes=self.num_classes,
            K=self.K,
            dprate=self.dropout,
        ).to(self.device)

    def _sample_iid(self, num_nodes: int, ref_tensor: torch.Tensor):
        shape = (num_nodes, self.hid_dim)
        return ref_tensor.new_empty(shape).normal_(mean=self.iid_mean, std=self.iid_std)

    def _feature_mmd_loss(self, source_data, target_data):
        source_feature = F.relu(self.dlit.lin1(source_data.x))
        target_feature = F.relu(self.dlit.lin1(target_data.x))
        return MMD(source_feature, target_feature)

    def _filter_iid_mmd_loss(self, source_data, target_data):
        iid_source = self._sample_iid(source_data.num_nodes, source_data.x)
        iid_target = self._sample_iid(target_data.num_nodes, target_data.x)
        out_source = self.dlit.prop(iid_source, source_data.edge_index)
        out_target = self.dlit.prop(iid_target, target_data.edge_index)
        return MMD(out_source, out_target)

    @staticmethod
    def entropy_minimization_loss(output):
        probs = F.softmax(output, dim=1)
        log_probs = F.log_softmax(output, dim=1)
        a = torch.sum(probs, dim=0)
        return -torch.sum(probs * log_probs / (a / torch.sum(a)), dim=1).mean()

    def forward_model(self, source_data, target_data):
        feature_mmd = self._feature_mmd_loss(source_data, target_data)
        filter_mmd = self._filter_iid_mmd_loss(source_data, target_data)

        target_logits = self.dlit(target_data, is_source_domain=False)
        entropy_loss = self.entropy_minimization_loss(target_logits)

        source_logits = self.dlit(source_data, is_source_domain=True)
        cls_loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)

        loss = (
            self.beta * feature_mmd
            + self.alpha * filter_mmd
            + self.gamma * entropy_loss
            + self.cls_weight * cls_loss
        )
        return loss, source_logits

    def fit(self, source_data, target_data):
        self.source_loader = self.get_loader(source_data, batch_size=self.batch_size)
        self.target_loader = self.get_loader(target_data, batch_size=self.batch_size)
        self.dlit = self.init_model()

        optimizer = torch.optim.Adam(
            self.dlit.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        for epoch in tqdm(range(self.epoch), desc="Training"):
            epoch_loss = 0.0
            epoch_logits = []
            epoch_labels = []

            for source_batch, target_batch in zip(self.source_loader, self.target_loader):
                self.dlit.train()
                source_batch = source_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                optimizer.zero_grad()
                loss, source_logits = self.forward_model(source_batch, target_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_logits.append(source_logits.detach())
                epoch_labels.append(source_batch.y.detach())

            if not epoch_logits:
                continue

            epoch_source_logits = torch.cat(epoch_logits, dim=0)
            epoch_source_labels = torch.cat(epoch_labels, dim=0)
            train_results = self.metrics(epoch_source_logits, epoch_source_labels)
            self.log(epoch, epoch_loss, train_results)

        if self.verbose >= 1:
            print(
                f"== Best Model from Epoch {self.best_epoch+1:03d} "
                f"with Val Micro-F1: {self.best_val:.4f} =="
            )
        self.finish()

    def predict(self, data, source=False):
        self.dlit.eval()
        loader = self.source_loader if source else self.target_loader
        is_source_domain = bool(source)

        logits_list, labels_list = [], []
        for sampled_data in loader:
            sampled_data = sampled_data.to(self.device)
            with torch.no_grad():
                logits = self.dlit(sampled_data, is_source_domain=is_source_domain)
            logits_list.append(logits)
            labels_list.append(sampled_data.y)

        return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)
