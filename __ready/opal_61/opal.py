import torch
import torch.nn.functional as F
import itertools
import time

import numpy as np

from torch_geometric.loader import NeighborLoader, DataLoader

from models.base_model import BaseGDA

from models.ours.opal.opal_base import OPALBase
from utils.train_utils.mmd import MMD, Sinkhorn



class OPAL(BaseGDA):

    def __init__(self, config: dict):
        
        super(OPAL, self).__init__(config)
        
        self.num_layers=config["model"]["num_layers"]
        self.K=config["model"]["K"]
        self.mode=config["model"]["mode"]

        self.alpha=config["model"]["alpha"] # filter regularization
        self.beta=config["model"]["beta"] # feature divergence
        self.gamma=config["model"]["gamma"] # entropy minimization
        self.delta=config["model"]["delta"] # filter alignment

        self.cheb_lambda_max = float(config["model"].get("cheb_lambda_max", 2.0))

        if config["model"]["divergence"] == "mmd":
            self.divergence = MMD
        elif config["model"]["divergence"] == "sinkhorn":
            self.divergence = Sinkhorn
        else:
            raise ValueError(f"Unsupported divergence type: {config['model']['divergence']}")

    def init_model(self):

        return OPALBase(
            num_layers=self.num_layers,
            features=self.in_dim,
            hidden=self.hid_dim,
            classes=self.num_classes,
            dropout_rate=self.dropout,
            K=self.K,
            cheb_lambda_max=self.cheb_lambda_max
            
        ).to(self.device)


    def forward_model(self, source_data, target_data):

        # source domain cross entropy loss
        source_logits = self.opal(source_data)
        train_loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
        loss = train_loss

        # feature alignment loss
        source_feature = self.opal.feat_bottleneck(source_data.x, source_data.edge_index, is_source_domain=True)
        target_feature = self.opal.feat_bottleneck(target_data.x, target_data.edge_index, is_source_domain=False)

        div_loss = self.divergence(source_feature, target_feature)
        loss = loss + div_loss * self.beta

        # filter alignment loss
        filter_loss = self.filter_alignment(source_data, target_data) 
        loss = loss + filter_loss * self.delta

        lip_loss = self.integral_lipschitz()
        loss = loss + lip_loss * self.alpha


        # target entropy minimization loss
        target_outputs = self.opal(target_data, False)
        entropy_loss = self.entropy_minimization_loss(target_outputs)
        loss = loss + entropy_loss * self.gamma


        return loss, source_logits

    @staticmethod
    def _cheb_weighted_ilip(coeff: torch.Tensor) -> torch.Tensor:
        """Chebyshev-basis ILIP bound: |T'_k(x)| <= k^2 on x in [-1, 1]."""
        if coeff.numel() <= 1:
            return coeff.new_tensor(0.0)
        orders = torch.arange(1, coeff.numel(), device=coeff.device, dtype=coeff.dtype)
        return torch.sum((orders ** 2) * coeff[1:].abs())

    # def integral_lipschitz(self):
    #     src_coef = self.opal.src_filter.coef
    #     tgt_coef = self.opal.tgt_filter.coef
    #     lip_loss = F.l1_loss(src_coef, tgt_coef)
    #     return lip_loss

    def integral_lipschitz(self):
        src_coef = self.opal.src_filter.coef
        tgt_coef = self.opal.tgt_filter.coef
        src_ilip = self._cheb_weighted_ilip(src_coef)
        tgt_ilip = self._cheb_weighted_ilip(tgt_coef)
        return src_ilip.pow(2) + tgt_ilip.pow(2)

    def filter_alignment(self, source_data, target_data, norm_divergence=True):

        # model actual distribution
        with torch.no_grad():

            source_feature = self.opal.project(source_data.x)
            target_feature = self.opal.project(target_data.x)
        
        source_probe, target_probe = self.generate_probe(source_feature, target_feature)
        source_probe = self.opal.propagate(self.opal.src_filter, source_probe, source_data.edge_index)
        target_probe = self.opal.propagate(self.opal.tgt_filter, target_probe, target_data.edge_index)

        filter_div_loss = self.divergence(source_probe, target_probe)

        # if norm_divergence:

        #     src_norm = source_probe.norm(p=2, dim=1, keepdim=True)
        #     tgt_norm = target_probe.norm(p=2, dim=1, keepdim=True)
        #     norm_loss = self.divergence(src_norm, tgt_norm)

        #     filter_div_loss = filter_div_loss + norm_loss

        return filter_div_loss


    def _sample_probe_indices(self,num_bank: int, out_size: int, device):
        if out_size <= num_bank:
            return torch.randperm(num_bank, device=device)[:out_size]
        return torch.randint(0, num_bank, (out_size,), device=device)


    def generate_probe(self, source_feature, target_feature, use_target=True):
        probe_bank = target_feature.clone()
        if not use_target:
            probe_bank = source_feature.clone()

        source_num = source_feature.size(0)
        target_num = target_feature.size(0)
        probe_num = probe_bank.size(0)

        source_indices = self._sample_probe_indices(probe_num, source_num, probe_bank.device)
        target_indices = self._sample_probe_indices(probe_num, target_num, probe_bank.device)

        source_probe = probe_bank[source_indices]
        target_probe = probe_bank[target_indices]
        return source_probe, target_probe

    
    def entropy_minimization_loss(self, output, eps: float = 1e-8):

        output = torch.nan_to_num(output, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        probs = F.softmax(output, dim=1)
        log_probs = F.log_softmax(output, dim=1)
        class_mass = torch.sum(probs, dim=0)
        class_prior = class_mass / class_mass.sum().clamp_min(float(eps))
        class_prior = class_prior.clamp_min(float(eps))
        entropy_loss = -torch.sum((probs * log_probs) / class_prior, dim=1).mean()
        return torch.nan_to_num(entropy_loss, nan=0.0, posinf=0.0, neginf=0.0)


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

        self.opal = self.init_model()

        optimizer = torch.optim.Adam(
            self.opal.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()

        from tqdm import tqdm
        for epoch in tqdm(range(self.epoch), desc="Training"):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(self.source_loader, self.target_loader)):
                self.opal.train()

                sampled_source_data = sampled_source_data.to(self.device)
                sampled_target_data = sampled_target_data.to(self.device)
                
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
            

            # epoch_source_preds = epoch_source_logits.argmax(dim=1)
            train_results = self.metrics(epoch_source_logits, epoch_source_labels)

            self.log(epoch, epoch_loss, train_results)


        # end_time = time.time()
        # training_time = end_time - start_time

        if self.verbose >= 1:
            print(f"== Best Model from Epoch {self.best_epoch+1:03d} with Val Micro-F1: {self.best_val:.4f} ==")

        self.finish()


    def process_graph(self, data):
        pass


    def predict(self, data, source=False):
        self.opal.eval()

        if source:
            for idx, sampled_data in enumerate(self.source_loader):
                sampled_data = sampled_data.to(self.device)
                with torch.no_grad():
                    logits = self.opal(sampled_data)

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
                    logits = self.opal(sampled_data, False)

                    if idx == 0:
                        logits, labels = logits, sampled_data.y
                    else:
                        sampled_logits, sampled_labels = logits, sampled_data.y
                        logits = torch.cat((logits, sampled_logits))
                        labels = torch.cat((labels, sampled_labels))

        return logits, labels
