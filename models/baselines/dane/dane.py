import itertools
import time
import copy

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import is_undirected, to_undirected

from models.base_model import BaseGDA
from models.baselines.gnn.gnn_base import GNNBase
from models.__layers.reverse_layer import GradReverse



class DANE(BaseGDA):

    def __init__(
        self, config):
        
        super(DANE, self).__init__(config)
        self.config = config

        self.verbose = config["expt"]["verbose"]
        self.batch_size = config["model"]["batch_size"]
        self.epoch = config["expt"]["epochs"]
        self.mode = config["model"]["mode"]

        self.lr = config["model"]["lr"]
        self.weight_decay = config["model"]["weight_decay"]
        # self.step_size = config["model"]["step_size"]
        # self.gamma = config["model"]["gamma"]

        self.k = config["model"]["k"]
        self.train_mode = config["model"]["train_mode"]
        assert self.train_mode in ["semi", "unsup"], f"train_mode: {self.train_mode} should be semi or unsup"

        self.dane = None

    def init_model(self, **kwargs):

        model =  GNNBase(self.config).to(self.device)
        return model

    
    def forward_model(self, source_data, target_data):

        for dis_epoch in range(5):
            discriminator_loss = self.train_d(source_data, target_data)
        generator_loss = self.train_g(source_data, target_data)

        if self.mode == "graph":
            source_logits = self.dane(source_data.x, source_data.edge_index, batch=source_data.batch)
            target_logits = self.dane(target_data.x, target_data.edge_index, batch=target_data.batch)
        elif self.mode == "node":
            source_logits = self.dane(source_data.x, source_data.edge_index)
            target_logits = self.dane(target_data.x, target_data.edge_index)

        loss = generator_loss + discriminator_loss
        return loss, source_logits, target_logits



    def fit(self, source_data, target_data):

        if self.mode == 'node':
            if not is_undirected(source_data.edge_index):
                source_data.edge_index = to_undirected(source_data.edge_index)
        
            if not is_undirected(target_data.edge_index):
                target_data.edge_index = to_undirected(target_data.edge_index)
        
            self.sample_size = min(source_data.x.shape[0], target_data.x.shape[0])

            if self.batch_size == 0:
                self.source_batch_size = source_data.x.shape[0]
                self.source_loader = NeighborLoader(source_data, self.num_neigh, batch_size=self.source_batch_size)
                self.target_batch_size = target_data.x.shape[0]
                self.target_loader = NeighborLoader(target_data, self.num_neigh, batch_size=self.target_batch_size)
            else:
                self.source_loader = NeighborLoader(source_data, self.num_neigh, batch_size=self.batch_size)
                self.target_loader = NeighborLoader(target_data, self.num_neigh, batch_size=self.batch_size)
        elif self.mode == 'graph':
            self.sample_size = min(len(source_data), len(target_data))

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

        self.dane = self.init_model(**self.kwargs)

        self.domain_discriminator = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, 1)
        ).to(self.device)

        self.g_optimizer = torch.optim.Adam(
            self.dane.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.d_optimizer = torch.optim.Adam(
            self.domain_discriminator.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()

        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(self.source_loader, self.target_loader)):
                self.dane.train()

                sampled_source_data = sampled_source_data.to(self.device)
                sampled_target_data = sampled_target_data.to(self.device)

                loss, source_logits, target_logits = self.forward_model(sampled_source_data, sampled_target_data)
                epoch_loss += loss

                if idx == 0:
                    epoch_source_logits, epoch_source_labels = source_logits, sampled_source_data.y
                else:
                    source_logits, source_labels = source_logits, sampled_source_data.y
                    epoch_source_logits = torch.cat((epoch_source_logits, source_logits))
                    epoch_source_labels = torch.cat((epoch_source_labels, source_labels))
            

            epoch_source_preds = epoch_source_logits.argmax(dim=1)
            train_results = self.metrics(epoch_source_logits, epoch_source_labels)

            self.log(epoch, epoch_loss, train_results)
            early_stop = self.early_stop_check(self.dane, result=train_results, epoch=epoch)

            if early_stop == "stop":
                break
            elif early_stop == "save":
                torch.save(self.dane.state_dict(), self.best_model_dir)

        end_time = time.time()
        training_time = end_time - start_time

        self.dane.load_state_dict(torch.load(self.best_model_dir))
        if self.verbose >= 1:
            print(f"== Best Model from Epoch {self.best_epoch+1:03d} with Val Micro-F1: {self.best_val:.4f} ==")

        self.finish()


    def train_d(self, source_data, target_data):

        self.dane.eval()

        if self.mode == 'node':
            embedding_s = self.dane.feat_bottleneck(source_data.x, source_data.edge_index)
            output_s = self.dane.feat_classifier(embedding_s, source_data.edge_index)

            embedding_t = self.dane.feat_bottleneck(target_data.x, target_data.edge_index)
            output_t = self.dane.feat_classifier(embedding_t, target_data.edge_index)
        else:
            embedding_s = self.dane.feat_bottleneck(source_data.x, source_data.edge_index)
            embedding_s = global_mean_pool(embedding_s, source_data.batch)
            output_s = self.dane.feat_classifier(embedding_s, source_data.edge_index)

            embedding_t = self.dane.feat_bottleneck(target_data.x, target_data.edge_index)
            embedding_t = global_mean_pool(embedding_t, target_data.batch)
            output_t = self.dane.feat_classifier(embedding_t, target_data.edge_index)

        node_s = torch.tensor([1.0 for i in range(0, embedding_s.shape[0])])
        node_t = torch.tensor([1.0 for i in range(0, embedding_t.shape[0])])

        train_idx_s = torch.multinomial(node_s, 8 * self.sample_size, replacement=True)
        train_idx_t = torch.multinomial(node_t, 8 * self.sample_size, replacement=True)

        pre_s = self.domain_discriminator(embedding_s[train_idx_s])
        pre_t = self.domain_discriminator(embedding_t[train_idx_t])

        self.d_optimizer.zero_grad()

        loss = (pre_s ** 2).mean() + ((pre_t - 1) ** 2).mean()
        loss.backward()

        self.d_optimizer.step()

        return loss.item()

    
    def L_GCN(self, embedding, nodes_weight, idx_u, idx_v, k):

        embedding_u = embedding[idx_u]
        embedding_v = embedding[idx_v]

        embedding_neg = [embedding[torch.multinomial(nodes_weight, self.sample_size, replacement=False)] for i in range(0, k)]

        pos = torch.sum(torch.mul(embedding_u, embedding_v), dim=1)
        neg = [torch.sum(torch.mul(embedding_u, embedding_neg[i]) * (-1), dim=1) for i in range(0, k)]
        loss = - torch.sum(F.logsigmoid(pos))
        for i in range(0, k):
            loss = loss - torch.sum(F.logsigmoid(neg[i]))
        return loss


    def L_cluster(self, labelsA, embA, labelsB, embB):

        loss = 0.0
        labelsA = labelsA.detach().cpu().numpy()
        labelsB = labelsB.detach().cpu().numpy()
        for i in range(self.num_classes):
            idxA = np.where(labelsA == i)
            idxB = np.where(labelsB == i)
            if (idxA[0].size > 0 and idxB[0].size > 0):
                loss += torch.sum((torch.mean(embA[idxA], dim=0) - torch.mean(embB[idxB], dim=0)) ** 2)
        answer = loss / self.num_classes
        return answer
    

    def train_g(self, source_data, target_data):
        self.dane.train()

        if self.mode == 'node':
            embedding_s = self.dane.feat_bottleneck(source_data.x, source_data.edge_index)
            output_s = self.dane.feat_classifier(embedding_s, source_data.edge_index)

            embedding_t = self.dane.feat_bottleneck(target_data.x, target_data.edge_index)
            output_t = self.dane.feat_classifier(embedding_t, target_data.edge_index)
        else:
            embedding_s = self.dane.feat_bottleneck(source_data.x, source_data.edge_index)
            embedding_s = global_mean_pool(embedding_s, source_data.batch)
            output_s = self.dane.feat_classifier(embedding_s, source_data.edge_index)

            embedding_t = self.dane.feat_bottleneck(target_data.x, target_data.edge_index)
            embedding_t = global_mean_pool(embedding_t, target_data.batch)
            output_t = self.dane.feat_classifier(embedding_t, target_data.edge_index)

        node_s = torch.tensor([1.0 for i in range(0, embedding_s.shape[0])])
        node_t = torch.tensor([1.0 for i in range(0, embedding_t.shape[0])])

        train_idx_s = torch.multinomial(node_s, 8 * self.sample_size, replacement=True)
        train_idx_t = torch.multinomial(node_t, 8 * self.sample_size, replacement=True)

        pre_s = self.domain_discriminator(embedding_s[train_idx_s])
        pre_t = self.domain_discriminator(embedding_t[train_idx_t])

        l_adv = (pre_t ** 2).mean() + ((pre_s - 1) ** 2).mean()

        if self.mode == 'node':
            sample_edge_s = torch.multinomial(torch.ones(source_data.edge_index.shape[1]), self.sample_size, replacement=False)
            idx_u_s = [source_data.edge_index[0][i] for i in sample_edge_s]
            idx_v_s = [source_data.edge_index[1][i] for i in sample_edge_s]

            sample_edge_t = torch.multinomial(torch.ones(target_data.edge_index.shape[1]), self.sample_size, replacement=False)
            idx_u_t = [target_data.edge_index[0][i] for i in sample_edge_t]
            idx_v_t = [target_data.edge_index[1][i] for i in sample_edge_t]

            _, nodes_weight_s = torch.unique(source_data.edge_index[0], return_counts=True)
            nodes_weight_s = torch.pow(nodes_weight_s, 0.75)

            _, nodes_weight_t = torch.unique(target_data.edge_index[0], return_counts=True)
            nodes_weight_t = torch.pow(nodes_weight_t, 0.75)

            l_gcn1 = self.L_GCN(embedding_s, nodes_weight_s, idx_u_s, idx_v_s, self.k)
            l_gcn2 = self.L_GCN(embedding_t, nodes_weight_t, idx_u_t, idx_v_t, self.k)

            l_gcn = l_gcn1 + l_gcn2
        else:
            l_gcn = 0

        l_ce = F.cross_entropy(output_s, source_data.y)

        if self.train_mode == 'semi':
            label_idx_t = torch.multinomial(node_t, int(self.tgt_rate * len(node_t)), replacement=False)
            l_ce_t = F.cross_entropy(output_t[label_idx_t], target_data.y[label_idx_t])
            l_ce += l_ce_t
            l_c = self.L_cluster(source_data.y, embedding_s, target_data.y[label_idx_t], embedding_t[label_idx_t])
            loss = l_gcn + l_adv * 0.1 + l_ce + l_c
        elif self.train_mode == 'unsup':
            loss = l_gcn + l_ce + l_adv * 0.1

        self.g_optimizer.zero_grad()
        loss.backward()
        self.g_optimizer.step()

        return loss.item()


    def predict(self, data, source=False):
        self.dane.eval()

        if source:
            for idx, sampled_data in enumerate(self.source_loader):
                sampled_data = sampled_data.to(self.device)
                with torch.no_grad():
                    if self.mode == 'graph':
                        logits = self.dane(sampled_data.x, sampled_data.edge_index, batch=sampled_data.batch)
                    else:
                        logits = self.dane(sampled_data.x, sampled_data.edge_index)

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
                    if self.mode == 'graph':
                        logits = self.dane(sampled_data.x, sampled_data.edge_index, batch=sampled_data.batch)
                    else:
                        logits = self.dane(sampled_data.x, sampled_data.edge_index)

                    if idx == 0:
                        logits, labels = logits, sampled_data.y
                    else:
                        sampled_logits, sampled_labels = logits, sampled_data.y
                        logits = torch.cat((logits, sampled_logits))
                        labels = torch.cat((labels, sampled_labels))

        return logits, labels


