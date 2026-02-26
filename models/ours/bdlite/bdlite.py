import time
import torch
import torch.nn.functional as F

from torch_geometric.loader import NeighborLoader, DataLoader
from tqdm import tqdm

from models.base_model import BaseGDA
from models.ours.bdlite.bdlite_base import BDliteBase
from utils.train_utils.mmd import MMD


class BDlite(BaseGDA):
    """
    BDlite: Bernstein-DLITE style domain adaptation.

    Ported from the original external DLITE implementation while matching
    this repository's BaseGDA training/prediction conventions.
    """

    def __init__(self, config: dict):
        super(BDlite, self).__init__(config)

        self.config = config
        self.mode = config["model"]["mode"]
        self.num_layers = config["model"]["num_layers"]
        self.K = config["model"].get("K", 25)
        self.dprate = config["model"].get("dprate", self.dropout)

        # Loss weights from original DLITE script
        self.alpha = config["model"].get("alpha", 1.0)  # probe filter MMD
        self.beta = config["model"].get("beta", 0.5)    # feature MMD
        self.gamma = config["model"].get("gamma", 0.05) # target entropy
        self.delta = config["model"].get("delta", 0.1)  # probe edge loss
        self.cls_weight = config["model"].get("cls_weight", 1.0)

        # MMD/probe settings
        self.probe_samples = int(config["model"].get("probe_samples", 2048))
        self.feature_mmd_samples = int(config["model"].get("feature_mmd_samples", 256))
        self.feature_mmd_times = int(config["model"].get("feature_mmd_times", 2))
        self.probe_mmd_samples = int(config["model"].get("probe_mmd_samples", 256))
        self.probe_mmd_times = int(config["model"].get("probe_mmd_times", 2))
        self.probe_mmd_dim = int(config["model"].get("probe_mmd_dim", 64))

        self.bdlite = None
        self.probe_proj = None

        assert self.num_layers == 2, "unsupported number of layers"
        assert self.mode == "node", "unsupported mode"

    def init_model(self):
        self.bdlite = BDliteBase(
            features=self.in_dim,
            hidden=self.hid_dim,
            classes=self.num_classes,
            dropout=self.dropout,
            dprate=self.dprate,
            K=self.K,
        ).to(self.device)

        self.probe_proj = None
        if self.probe_mmd_dim > 0 and self.probe_mmd_dim < self.hid_dim:
            proj = torch.randn(self.hid_dim, self.probe_mmd_dim, device=self.device)
            proj = proj / (self.probe_mmd_dim ** 0.5)
            self.probe_proj = proj
        return self.bdlite

    @staticmethod
    def entropy_minimization_loss(output):
        probs = F.softmax(output, dim=1)
        log_probs = F.log_softmax(output, dim=1)
        a = torch.sum(probs, dim=0)
        entropy_loss = -torch.sum(probs * log_probs / (a / torch.sum(a)), dim=1).mean()
        return entropy_loss

    @staticmethod
    def edge_prediction_loss(z, edge_index, num_samples=2048): # let's make this a contrastive loss a bit later
        if edge_index.numel() == 0 or z.size(0) <= 1:
            return z.new_tensor(0.0)

        e = edge_index.size(1)
        k = min(num_samples, e)
        edge_ids = torch.randint(e, (k,), device=edge_index.device)
        pos_u = edge_index[0, edge_ids]
        pos_v = edge_index[1, edge_ids]

        num_nodes = z.size(0)
        neg_u = torch.randint(num_nodes, (k,), device=z.device)
        neg_v = torch.randint(num_nodes, (k,), device=z.device)
        same = neg_u == neg_v
        neg_v = torch.where(same, (neg_v + 1) % num_nodes, neg_v)

        pos_logits = (z[pos_u] * z[pos_v]).sum(dim=1)
        neg_logits = (z[neg_u] * z[neg_v]).sum(dim=1)

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits)
        )
        return 0.5 * (pos_loss + neg_loss)

    def forward_model(self, source_data, target_data):
        source_logits = self.bdlite(source_data, True)
        source_cls_loss = F.cross_entropy(source_logits, source_data.y)

        source_feature = F.relu(self.bdlite.lin1(source_data.x))
        target_feature = F.relu(self.bdlite.lin1(target_data.x))

        feature_mmd_loss = MMD(
            source_feature,
            target_feature,
            sampling_num=self.feature_mmd_samples,
            times=self.feature_mmd_times,
        )

        all_features = torch.cat((source_feature, target_feature), dim=0).detach()
        probe_mean = all_features.mean(dim=0, keepdim=True)
        probe_std = all_features.std(dim=0, keepdim=True) + 1e-6
        probe_features = torch.randn_like(all_features) * probe_std + probe_mean
        source_probe = probe_features[: source_feature.size(0)]
        target_probe = probe_features[source_feature.size(0):]

        source_probe_out = self.bdlite.filter_only(
            source_probe, source_data.edge_index, domain="source", apply_dropout=False
        )
        target_probe_out = self.bdlite.filter_only(
            target_probe, target_data.edge_index, domain="target", apply_dropout=False
        )

        if self.probe_proj is not None:
            source_probe_mmd = source_probe_out @ self.probe_proj
            target_probe_mmd = target_probe_out @ self.probe_proj
        else:
            source_probe_mmd = source_probe_out
            target_probe_mmd = target_probe_out

        probe_filter_mmd_loss = MMD(
            source_probe_mmd,
            target_probe_mmd,
            sampling_num=self.probe_mmd_samples,
            times=self.probe_mmd_times,
        )

        probe_edge_loss = 0.5 * (
            self.edge_prediction_loss(
                source_probe_out, source_data.edge_index, num_samples=self.probe_samples
            )
            + self.edge_prediction_loss(
                target_probe_out, target_data.edge_index, num_samples=self.probe_samples
            )
        )

        target_outputs = self.bdlite(target_data, False)
        target_entropy_loss = self.entropy_minimization_loss(target_outputs)

        total_loss = (
            self.cls_weight * source_cls_loss
            + self.alpha * probe_filter_mmd_loss
            + self.beta * feature_mmd_loss
            + self.gamma * target_entropy_loss
            + self.delta * probe_edge_loss
        )

        loss_dict = {
            "cls": source_cls_loss.item(),
            "feat_mmd": feature_mmd_loss.item(),
            "probe_mmd": probe_filter_mmd_loss.item(),
            "target_entropy": target_entropy_loss.item(),
            "probe_edge": probe_edge_loss.item(),
        }
        return total_loss, source_logits, loss_dict

    def fit(self, source_data, target_data):
        if self.mode == "node":
            if self.batch_size == 0:
                source_batch_size = source_data.x.shape[0]
                target_batch_size = target_data.x.shape[0]
                self.source_loader = NeighborLoader(
                    source_data, self.num_neigh, batch_size=source_batch_size
                )
                self.target_loader = NeighborLoader(
                    target_data, self.num_neigh, batch_size=target_batch_size
                )
            else:
                self.source_loader = NeighborLoader(
                    source_data, self.num_neigh, batch_size=self.batch_size
                )
                self.target_loader = NeighborLoader(
                    target_data, self.num_neigh, batch_size=self.batch_size
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
            raise ValueError(f"Invalid mode: {self.mode}")

        self.init_model()
        optimizer = torch.optim.Adam(
            self.bdlite.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        start_time = time.time()
        for epoch in tqdm(range(self.epoch), desc="Training"):
            epoch_loss = 0.0
            epoch_source_logits = None
            epoch_source_labels = None

            for idx, (sampled_source_data, sampled_target_data) in enumerate(
                zip(self.source_loader, self.target_loader)
            ):
                self.bdlite.train()
                sampled_source_data = sampled_source_data.to(self.device)
                sampled_target_data = sampled_target_data.to(self.device)

                loss, source_logits, loss_dict = self.forward_model(
                    sampled_source_data, sampled_target_data
                )
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.wandb.log(loss_dict)

                if idx == 0:
                    epoch_source_logits = source_logits.detach()
                    epoch_source_labels = sampled_source_data.y
                else:
                    epoch_source_logits = torch.cat((epoch_source_logits, source_logits.detach()))
                    epoch_source_labels = torch.cat((epoch_source_labels, sampled_source_data.y))

            train_results = self.metrics(epoch_source_logits, epoch_source_labels)
            self.log(epoch, epoch_loss, train_results)

        if self.verbose >= 1:
            print(
                f"== Best Model from Epoch {self.best_epoch+1:03d} "
                f"with Val Micro-F1: {self.best_val:.4f} =="
            )

        # Print learned filters for quick inspection
        theta_s = torch.relu(self.bdlite.prop1.temp.detach().cpu())
        theta_t = torch.relu(self.bdlite.prop2.temp.detach().cpu())
        theta_cls = torch.relu(self.bdlite.prop3.temp.detach().cpu())
        print("Theta_source:", [round(float(v), 6) for v in theta_s.tolist()])
        print("Theta_target:", [round(float(v), 6) for v in theta_t.tolist()])
        print("Theta_classifier:", [round(float(v), 6) for v in theta_cls.tolist()])

        self.train_time = time.time() - start_time
        self.finish()

    def predict(self, data, source=False):
        self.bdlite.eval()
        loader = self.source_loader if source else self.target_loader
        domain_flag = True if source else False

        all_logits = []
        all_labels = []
        for sampled_data in loader:
            sampled_data = sampled_data.to(self.device)
            with torch.no_grad():
                logits = self.bdlite(sampled_data, domain_flag)
            all_logits.append(logits)
            all_labels.append(sampled_data.y)

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        return logits, labels

    def process_graph(self, data):
        pass
