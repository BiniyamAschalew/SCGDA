import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import numpy as np

from torch_geometric.loader import NeighborLoader, DataLoader

from models.base_model import BaseGDA
from models.baselines.a2gnn.a2gnn_base import A2GNNBase
from utils.train_utils.mmd import MMD


class A2GNN(BaseGDA):
    def __init__(self, config: dict):
        super(A2GNN, self).__init__(config)

        self.config = config
        self.s_pnums=config["model"]["s_pnums"]
        self.t_pnums=config["model"]["t_pnums"]
        self.adv=config["model"]["adv"]
        self.weight=config["model"]["weight"]
        self.mode=config["model"]["mode"]


    def init_model(self):
        model =  A2GNNBase(self.config).to(self.device)
        return model


    def forward_model(self, source_data, target_data, alpha):

        # source domain cross entropy loss
        source_logits = self.a2gnn(source_data, self.s_pnums)
        target_logits = self.a2gnn(target_data, self.t_pnums)

        cls_loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
        loss = cls_loss

        if self.mode == 'node':
            source_batch = None
            target_batch = None
        else:
            source_batch = source_data.batch
            target_batch = target_data.batch

        source_features = self.a2gnn.feat_bottleneck(source_data.x, source_data.edge_index, source_batch, self.s_pnums)
        target_features = self.a2gnn.feat_bottleneck(target_data.x, target_data.edge_index, target_batch, self.t_pnums)

        # Adv loss
        if self.adv:
            source_dlogits = self.a2gnn.domain_classifier(source_features, alpha)
            target_dlogits = self.a2gnn.domain_classifier(target_features, alpha)
            
            domain_label = torch.tensor(
                [0] * source_data.x.shape[0] + [1] * target_data.x.shape[0]
                ).to(self.device)
            
            domain_loss = F.cross_entropy(torch.cat([source_dlogits, target_dlogits], 0), domain_label)
            loss = loss + self.weight * domain_loss
        else:
            # MMD loss
            mmd_loss = MMD(source_features, target_features)
            loss = loss + mmd_loss * self.weight

        return loss, source_logits, target_logits

    def fit(self, source_data, target_data):

        self.num_source_nodes = source_data.x.shape[0]
        self.num_target_nodes = target_data.x.shape[0]

        if self.mode == 'node':
            self.source_loader = self.get_loader(source_data, batch_size=self.batch_size)
            self.target_loader = self.get_loader(target_data, batch_size=self.batch_size)


        elif self.mode == 'graph':

            src_batch_size = self.batch_size
            tgt_batch_size = self.batch_size

            if self.batch_size == 0:
                src_batch_size = len(source_data)
                tgt_batch_size = len(target_data)

            self.source_loader = DataLoader(source_data, batch_size=src_batch_size, shuffle=True)
            self.target_loader = DataLoader(target_data, batch_size=tgt_batch_size, shuffle=True)
       
       
        else:
            assert self.mode in ('graph', 'node'), 'Invalid train mode'

        self.a2gnn = self.init_model()
        optimizer = torch.optim.Adam(
            self.a2gnn.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()

        from tqdm import tqdm
        for epoch in tqdm(range(self.epoch), desc="Training"):
            epoch_loss = 0.0
            epoch_source_logits = torch.empty(0).to(self.device)
            epoch_source_labels = torch.empty(0).to(self.device)

            p = float(epoch) / self.epoch
            alpha = 2. / (1. + np.exp(-10. * p)) - 1

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(self.source_loader, self.target_loader)):
                self.a2gnn.train()

                sampled_source_data = sampled_source_data.to(self.device)
                sampled_target_data = sampled_target_data.to(self.device)
                
                loss, source_logits, target_logits = self.forward_model(sampled_source_data, sampled_target_data, alpha)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                source_labels = sampled_source_data.y
                epoch_source_logits = torch.cat((epoch_source_logits, source_logits))
                epoch_source_labels = torch.cat((epoch_source_labels, source_labels))
            
            epoch_source_preds = epoch_source_logits.argmax(dim=1)
            train_results = self.metrics(epoch_source_logits, epoch_source_labels)

            self.log(epoch, epoch_loss, train_results)

        end_time = time.time()
        training_time = end_time - start_time

        # if self.verbose >= 1:
        #     print(f"== Best Model from Epoch {self.best_epoch+1:03d} with Val Micro-F1: {self.best_val:.4f} ==")

        self.finish()



    def predict(self, data, source=False):

        self.a2gnn.eval()
        loader, pnums = self.target_loader, self.t_pnums

        if source:
            loader, pnums = self.source_loader, self.s_pnums

        all_logits, all_labels = torch.empty(0).to(self.device), torch.empty(0).to(self.device)
        for idx, sampled_data in enumerate(loader):
            sampled_data = sampled_data.to(self.device)
            with torch.no_grad():
                logits = self.a2gnn(sampled_data, pnums)
                labels = sampled_data.y

                all_logits = torch.cat((all_logits, logits), dim=0)
                all_labels = torch.cat((all_labels, labels), dim=0)

                # if idx == 0:
                #     logits, labels = logits, sampled_data.y
                # else:
                #     sampled_logits, sampled_labels = logits, sampled_data.y
                #     logits = torch.cat((logits, sampled_logits))
                #     labels = torch.cat((labels, sampled_labels))
        # else:
        #     for idx, sampled_data in enumerate(self.target_loader):
        #         sampled_data = sampled_data.to(self.device)
        #         with torch.no_grad():
        #             logits = self.a2gnn(sampled_data, self.t_pnums)

        #             if idx == 0:
        #                 logits, labels = logits, sampled_data.y
        #             else:
        #                 sampled_logits, sampled_labels = logits, sampled_data.y
        #                 logits = torch.cat((logits, sampled_logits))
        #                 labels = torch.cat((labels, sampled_labels))

        return logits, labels
