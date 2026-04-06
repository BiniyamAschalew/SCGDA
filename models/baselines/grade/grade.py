import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import numpy as np

from torch_geometric.loader import NeighborLoader, DataLoader

from Learn.Clean_SCGDA.models.base_model import BaseGDA
from Learn.Clean_SCGDA.models.baselines.grade.grade_base import GRADEBase
from Learn.Clean_SCGDA.models.__layers.reverse_layer import GradReverse
from Learn.Clean_SCGDA.utils.train_utils.mmd import MMD


class GRADE(BaseGDA):


    def __init__(self, config: dict):
        super(GRADE, self).__init__(config)
        
        self.config = config
        self.disc=config["model"]["disc"]
        self.weight=config["model"]["weight"]
        self.mode=config["model"]["mode"]

    def init_model(self):
        model =  GRADEBase(self.config).to(self.device)
        return model


    def forward_model(self, source_data, target_data, alpha):

        # source domain cross entropy loss
        source_logits, source_feats = self.grade(source_data)
        target_logits, target_feats = self.grade(target_data)
        train_loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
        loss = train_loss

        # domain loss
        domain_loss = 0
        if self.disc == 'JS':
            domain_preds = self.grade.discriminator(GradReverse.apply(torch.cat([source_feats, target_feats], dim=0), alpha))
            if self.mode == 'node':
                domain_labels = np.array([0] * source_data.x.size(0) + [1] * target_data.x.size(0))
            else:
                domain_labels = np.array([0] * len(source_data) + [1] * len(target_data))
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=self.device)
            domain_loss = self.grade.criterion(domain_preds, domain_labels)
        elif self.disc == 'MMD':
            if self.mode == 'node':
                mind = min(source_data.x.size(0), target_data.x.size(0))
            else:
                mind = min(len(source_data), len(target_data))
            domain_loss = MMD(source_feats[:mind], target_feats[:mind])
        elif self.disc == 'C':
            ratio = 8
            s_l_f = torch.cat([source_feats, ratio * self.grade.one_hot_embedding(source_data.y)], dim=1)
            t_l_f = torch.cat([target_feats, ratio * F.softmax(target_logits, dim=1)], dim=1)
            domain_preds = self.grade.discriminator(GradReverse.apply(torch.cat([s_l_f, t_l_f], dim=0), alpha))
            if self.mode == 'node':
                domain_labels = np.array([0] * source_data.x.size(0) + [1] * target_data.x.size(0))
            else:
                domain_labels = np.array([0] * len(source_data) + [1] * len(target_data))
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=self.device)
            domain_loss = self.grade.criterion(domain_preds, domain_labels)

        loss = loss + domain_loss * self.weight

        return loss, source_logits, target_logits


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

        self.grade = self.init_model()

        optimizer = torch.optim.Adam(
            self.grade.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()

        from tqdm import tqdm
        for epoch in tqdm(range(self.epoch), desc="Training"):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            alpha = 2 / (1 + np.exp(- 10 * epoch / self.epoch)) - 1

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(self.source_loader, self.target_loader)):
                self.grade.train()

                sampled_source_data = sampled_source_data.to(self.device)
                sampled_target_data = sampled_target_data.to(self.device)
                
                loss, source_logits, target_logits = self.forward_model(sampled_source_data, sampled_target_data, alpha)
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

        #     early_stop = self.early_stop_check(self.grade, result=train_results, epoch=epoch)

        #     if early_stop == "stop":
        #         break
        #     elif early_stop == "save":
        #         torch.save(self.grade.state_dict(), self.best_model_dir)
        # self.grade.load_state_dict(torch.load(self.best_model_dir))

        end_time = time.time()
        training_time = end_time - start_time

        if self.verbose >= 1:
            print(f"== Best Model from Epoch {self.best_epoch+1:03d} with Val Micro-F1: {self.best_val:.4f} ==")

        self.finish()


    def predict(self, data, source=False):

        self.grade.eval()

        if source:
            for idx, sampled_data in enumerate(self.source_loader):
                sampled_data = sampled_data.to(self.device)
                with torch.no_grad():
                    logits, _ = self.grade(sampled_data)

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
                    logits, _ = self.grade(sampled_data)

                    if idx == 0:
                        logits, labels = logits, sampled_data.y
                    else:
                        sampled_logits, sampled_labels = logits, sampled_data.y
                        logits = torch.cat((logits, sampled_logits))
                        labels = torch.cat((labels, sampled_labels))

        return logits, labels
