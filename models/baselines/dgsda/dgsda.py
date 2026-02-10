import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import numpy as np

from torch_geometric.loader import NeighborLoader, DataLoader

from models.base_model import BaseGDA
from models.baselines.dgsda.dgsda_base import DGSDABase
from utils.train_utils.mmd import MMD


class DGSDA(BaseGDA):

    def __init__(self, config: dict):
        
        super(DGSDA, self).__init__(config)
        
        self.num_layers=config["model"]["num_layers"]
        self.K=config["model"]["K"]
        self.mode=config["model"]["mode"]
        self.alpha=config["model"]["alpha"]
        self.beta=config["model"]["beta"]
        self.gamma=config["model"]["gamma"]
        self.dprate=config["model"].get("dprate", self.dropout)

        assert self.num_layers==2, 'unsupport number of layers'
        assert self.mode=='node', 'unsupport mode'


    def init_model(self):

        return DGSDABase(
            features=self.in_dim,
            hidden=self.hid_dim,
            classes=self.num_classes,
            dropout=self.dropout,
            dprate=self.dprate,
            K=self.K,

        ).to(self.device)


    def forward_model(self, source_data, target_data):

        # source domain cross entropy loss
        source_logits = self.dgsda(source_data)
        train_loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
        loss = train_loss

        theta_s = self.dgsda.prop1.temp
        theta_t = self.dgsda.prop2.temp

        # theta_loss = F.l1_loss(theta_s, theta_t) #43.54
        theta_loss = F.l1_loss(theta_s, theta_t) + torch.sum(torch.abs(theta_s)) + torch.sum(torch.abs(theta_t))
        # 43.54

        loss = loss + theta_loss * self.alpha

        source_feature = F.relu(self.dgsda.lin1(source_data.x))
        target_feature = F.relu(self.dgsda.lin1(target_data.x))
        mmd_loss = MMD(source_feature, target_feature)
        loss = loss + mmd_loss * self.beta

        target_outputs = self.dgsda(target_data, False)
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

        self.dgsda = self.init_model()

        optimizer = torch.optim.Adam(
            self.dgsda.parameters(),
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
                self.dgsda.train()

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
            

            epoch_source_preds = epoch_source_logits.argmax(dim=1)
            train_results = self.metrics(epoch_source_logits, epoch_source_labels)

            self.log(epoch, epoch_loss, train_results)

        end_time = time.time()
        training_time = end_time - start_time

        if self.verbose >= 1:
            print(f"== Best Model from Epoch {self.best_epoch+1:03d} with Val Micro-F1: {self.best_val:.4f} ==")

        self.finish()


    def predict(self, data, source=False):
        self.dgsda.eval()

        if source:
            for idx, sampled_data in enumerate(self.source_loader):
                sampled_data = sampled_data.to(self.device)
                with torch.no_grad():
                    logits = self.dgsda(sampled_data)

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
                    logits = self.dgsda(sampled_data, False)

                    if idx == 0:
                        logits, labels = logits, sampled_data.y
                    else:
                        sampled_logits, sampled_labels = logits, sampled_data.y
                        logits = torch.cat((logits, sampled_logits))
                        labels = torch.cat((labels, sampled_labels))

        return logits, labels
