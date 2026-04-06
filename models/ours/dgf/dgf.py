import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
import itertools
import time
import numpy as np

from torch_geometric.loader import NeighborLoader, DataLoader

from Learn.Clean_SCGDA.models.base_model import BaseGDA
from Learn.Clean_SCGDA.models.ours.dgf.dgf_base import DGFBase
from Learn.Clean_SCGDA.utils.train_utils.mmd import MMD


class LearnableProbeDistribution(nn.Module):
    """learnable distribution for aligning the filters"""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(1, feature_dim)) # the mean shift
        self.log_scale = nn.Parameter(torch.zeros(1, feature_dim)) # log of the noice scale, to ensure positivity

    def sample(self, base: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.log_scale) + 1e-4
        noise = self.shift + scale * torch.randn_like(base)
        return base + noise

    def regularizer(self) -> torch.Tensor:
        return self.shift.pow(2).mean() + self.log_scale.pow(2).mean()




"""Build on top of DGf with additional adversarial filter alignment"""
class DGF(BaseGDA):

    def __init__(self, config: dict):
        
        super(DGF, self).__init__(config)
        
        self.num_layers=config["model"]["num_layers"]
        self.K=config["model"]["K"]
        self.mode=config["model"]["mode"]
        self.alpha=config["model"]["alpha"]
        self.beta=config["model"]["beta"]
        self.gamma=config["model"]["gamma"]
        self.adv_steps = max(1, int(config["model"].get("adv_steps", 1)))
        self.adv_lr = float(config["model"].get("adv_lr", self.lr))
        self.adv_weight_decay = float(config["model"].get("adv_weight_decay", 0.0))
        self.adv_reg_weight = float(config["model"].get("adv_reg_weight", 1e-3))

        assert self.num_layers==2, 'unsupport number of layers'
        assert self.mode=='node', 'unsupport mode'

        self.probe_dist = LearnableProbeDistribution(feature_dim=self.hid_dim).to(self.device)

    def init_model(self):

        return DGFBase(
            features=self.in_dim,
            hidden=self.hid_dim,
            classes=self.num_classes,
            dprate=self.dropout,
            K=self.K,
            
        ).to(self.device)

    def adv_probe(self, source, target):
        """create probe based on adversarially learned distribution to align the filters"""

        features = torch.cat((source, target), dim=0)
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True).clamp_min(1e-6)
        probe = torch.randn_like(features) * std + mean
        probe = self.probe_dist.sample(probe)

        src_probe, tgt_probe = probe[:source.size(0)], probe[source.size(0):]
        return src_probe, tgt_probe

    def filter_alignment_loss(self, source_feature, target_feature, source_data, target_data, detach_probe=False):
        if detach_probe:
            with torch.no_grad():
                source_probe, target_probe = self.adv_probe(source_feature, target_feature)
        else:
            source_probe, target_probe = self.adv_probe(source_feature, target_feature)

        source_probe = self.dgf.prop1(source_probe, source_data.edge_index)
        target_probe = self.dgf.prop2(target_probe, target_data.edge_index)
        return MMD(source_probe, target_probe)

    def adversarial_probe_step(self, source_feature, target_feature, source_data, target_data):
        filter_loss = self.filter_alignment_loss(
            source_feature.detach(),
            target_feature.detach(),
            source_data,
            target_data,
            detach_probe=False,
        )
        adv_objective = filter_loss - self.adv_reg_weight * self.probe_dist.regularizer()
        return adv_objective


    def forward_model(self, source_data, target_data):

        # source domain cross entropy loss
        source_logits = self.dgf(source_data)
        train_loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
        loss = train_loss

        theta_s = self.dgf.prop1.temp
        theta_t = self.dgf.prop2.temp

        # we don't use theta loss here
        # theta_loss = F.l1_loss(theta_s, theta_t)
        # theta_loss = F.l1_loss(theta_s, theta_t) + torch.sum(torch.abs(theta_s)) + torch.sum(torch.abs(theta_t))
        # loss = loss + theta_loss * self.alpha

        source_feature = F.relu(self.dgf.lin1(source_data.x))
        target_feature = F.relu(self.dgf.lin1(target_data.x))
        mmd_loss = MMD(source_feature, target_feature)
        loss = loss + mmd_loss * self.beta

        # filter alignment loss: minimize against a frozen adversarial probe distribution
        filter_loss = self.filter_alignment_loss(
            source_feature.detach(),
            target_feature.detach(),
            source_data,
            target_data,
            detach_probe=True,
        )
        loss = loss + self.alpha * filter_loss

        target_outputs = self.dgf(target_data, False)
        entropy_loss = self.entropy_minimization_loss(target_outputs)
        loss = loss + entropy_loss * self.gamma 

        return loss, source_logits
    
    def entropy_minimization_loss(self, output):

        probs = F.softmax(output, dim=1)
        log_probs = F.log_softmax(output, dim=1)
        a = torch.sum(probs, dim=0)
        entropy_loss = -torch.sum(probs * log_probs / (a / torch.sum(a)), dim=1).mean()
        
        return entropy_loss

    def fit(self, source_data, target_data):

        if self.mode == 'node':
            self.num_source_nodes, _ = source_data.x.shape
            self.num_target_nodes, _ = target_data.x.shape

            if self.batch_size == 0:
                self.source_batch_size = source_data.x.shape[0]
                self.source_loader = NeighborLoader(
                    source_data,
                    self.num_neigh,
                    batch_size=self.source_batch_size)
                self.target_batch_size = target_data.x.shape[0]
                self.target_loader = NeighborLoader(
                    target_data,
                    self.num_neigh,
                    batch_size=self.target_batch_size)
            else:
                self.source_loader = NeighborLoader(
                    source_data,
                    self.num_neigh,
                    batch_size=self.batch_size)
                self.target_loader = NeighborLoader(
                    target_data,
                    self.num_neigh,
                    batch_size=self.batch_size)
        elif self.mode == 'graph':
            if self.batch_size == 0:
                num_source_graphs = len(source_data)
                num_target_graphs = len(target_data)
                self.source_loader = DataLoader(source_data, batch_size=num_source_graphs, shuffle=True)
                self.target_loader = DataLoader(target_data, batch_size=num_target_graphs, shuffle=True)
            else:
                self.source_loader = DataLoader(source_data, batch_size=self.batch_size, shuffle=True)
                self.target_loader = DataLoader(target_data, batch_size=self.batch_size, shuffle=True)
        else:
            assert self.mode in ('graph', 'node'), 'Invalid train mode'

        self.dgf = self.init_model()

        optimizer = torch.optim.Adam(
            self.dgf.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        adv_optimizer = torch.optim.Adam(
            self.probe_dist.parameters(),
            lr=self.adv_lr,
            weight_decay=self.adv_weight_decay,
        )

        start_time = time.time()

        from tqdm import tqdm
        for epoch in tqdm(range(self.epoch), desc="Training"):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(self.source_loader, self.target_loader)):
                self.dgf.train()
                self.probe_dist.train()

                sampled_source_data = sampled_source_data.to(self.device)
                sampled_target_data = sampled_target_data.to(self.device)

                source_feature = F.relu(self.dgf.lin1(sampled_source_data.x))
                target_feature = F.relu(self.dgf.lin1(sampled_target_data.x))

                for _ in range(self.adv_steps):
                    adv_optimizer.zero_grad()
                    adv_objective = self.adversarial_probe_step(
                        source_feature,
                        target_feature,
                        sampled_source_data,
                        sampled_target_data,
                    )
                    (-adv_objective).backward()
                    adv_optimizer.step()
                    self.dgf.zero_grad()
                
                loss, source_logits = self.forward_model(sampled_source_data, sampled_target_data)
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
            

            epoch_source_preds = epoch_source_logits.argmax(dim=1)
            train_results = self.metrics(epoch_source_logits, epoch_source_labels)

            self.log(epoch, epoch_loss, train_results)


        end_time = time.time()
        training_time = end_time - start_time

        if self.verbose >= 1:
            print(f"== Best Model from Epoch {self.best_epoch+1:03d} with Val Micro-F1: {self.best_val:.4f} ==")

        self.finish()



    def process_graph(self, data):
        pass

    def predict(self, data, source=False):

        self.dgf.eval()

        if source:
            for idx, sampled_data in enumerate(self.source_loader):
                sampled_data = sampled_data.to(self.device)
                with torch.no_grad():
                    logits = self.dgf(sampled_data)

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
                    logits = self.dgf(sampled_data, False)

                    if idx == 0:
                        logits, labels = logits, sampled_data.y
                    else:
                        sampled_logits, sampled_labels = logits, sampled_data.y
                        logits = torch.cat((logits, sampled_logits))
                        labels = torch.cat((labels, sampled_labels))

        return logits, labels
