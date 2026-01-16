import torch
import warnings
import torch.nn.functional as F
import itertools
import time
import copy

import torch.nn as nn
import scipy.sparse as sp
import numpy as np

from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_dense_adj

# from . import BaseGDA
# from ..nn import ReweightGNN, GradReverse, MixupBase
# from ..utils import logger, MMD
# from ..metrics import eval_macro_f1, eval_micro_f1

from models.base_model import BaseGDA
from models.components.mixup_base import MixupBase
from models.components.reweight_gnn import ReweightGNN
from utils.train_utils.mmd import MMD


class StruRW(BaseGDA):
    def __init__(self, config: dict):
        super(StruRW, self).__init__(config)

        
        self.backbone=config["model"]["gnn"]
        self.lamb=config["model"]["lamb"]
        self.mode=config["model"]["mode"]

        assert self.mode in ['erm', 'mixup', 'mmd', 'adv'], 'unsupport training mode'

        self.bn=config["model"]["bn"]
        self.pooling=config["model"]["pooling"]
        self.cls_dim=config["model"]["cls_dim"]
        self.cls_layers=config["model"]["cls_layers"]
        self.reweight=config["model"]["reweight"]
        self.ew_freq=config["model"]["ew_freq"]
        self.ew_start=config["model"]["ew_start"]
        self.pseudo=config["model"]["pseudo"]

    def init_model(self, **kwargs):


        if self.mode == 'mixup':
            return MixupBase(
                in_dim=self.in_dim,
                hid_dim=self.hid_dim,
                num_classes=self.num_classes,
                num_layers=self.num_layers,
                dropout=self.dropout,
                rw_lmda=self.lamb,
                # **kwargs
                ).to(self.device)
        else:
            return ReweightGNN(
                input_dim=self.in_dim,
                gnn_dim=self.hid_dim,
                output_dim=self.num_classes,
                cls_dim=self.cls_dim,
                gnn_layers=self.num_layers,
                cls_layers=self.cls_layers,
                backbone=self.backbone,
                pooling=self.pooling,
                dropout=self.dropout,
                bn=self.bn,
                rw_lmda=self.lamb,
                # **kwargs
                ).to(self.device)


    def forward_model(self, source_data, target_data, alpha, epoch):

        target_feat, target_logits = self.strurw.forward(target_data, target_data.x)
        target_prob = F.softmax(target_logits, dim=1)
        target_pred = torch.max(target_prob, dim=1)[1]

        if self.reweight and (epoch + 1) >= self.ew_start:
            if self.pseudo:
                if (epoch + 1) % self.ew_freq == 0:
                    self.cal_reweight(source_data, target_data, target_pred)
            else:
                if epoch == self.ew_start - 1:
                    self.cal_reweight(source_data, target_data, target_pred)
        
        source_feat, source_logits = self.strurw.forward(source_data, source_data.x)
        source_prob = F.softmax(source_logits, dim=1)
        source_pred = torch.max(source_prob, dim=1)[1]

        if self.mode == 'erm':
            loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
        elif self.mode == 'adv':
            source_dlogits = self.domain_discriminator(GradReverse.apply(source_feat, alpha))
            target_dlogits = self.domain_discriminator(GradReverse.apply(target_feat, alpha))
            
            domain_label = torch.tensor(
                [0] * source_data.x.shape[0] + [1] * target_data.x.shape[0]
                ).to(self.device)
            
            loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
            domain_loss = F.cross_entropy(torch.cat([source_dlogits, target_dlogits], 0), domain_label)
            loss = loss + domain_loss
        elif self.mode == 'mmd':
            mmd_loss = MMD(source_feat, target_feat)
            loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
            loss = loss + mmd_loss

        return loss, source_logits, target_logits

    def forward_model_mixup(self, source_data, target_data, epoch):

        target_feat = self.strurw.feat_bottleneck(
            target_data.x,
            target_data.edge_index,
            target_data.edge_index,
            1,
            np.arange(target_data.x.shape[0]),
            target_data.edge_weight
        )
        target_logits = self.strurw.feat_classifier(target_feat)
        target_prob = F.softmax(target_logits, dim=1)
        target_pred = torch.max(target_prob, dim=1)[1]

        if self.reweight and (epoch + 1) >= self.ew_start:
            if self.pseudo:
                if (epoch + 1) % self.ew_freq == 0:
                    self.cal_reweight(source_data, target_data, target_pred)
            else:
                if epoch == self.ew_start - 1:
                    self.cal_reweight(source_data, target_data, target_pred)
        
        lam = np.random.beta(4.0, 4.0)
        data_b, id_new_value_old = self.shuffle_data(source_data)
        data_b = data_b.to(self.device)

        source_feat = self.strurw.feat_bottleneck(
            source_data.x,
            source_data.edge_index,
            data_b.edge_index,
            lam,
            id_new_value_old,
            source_data.edge_weight)
        source_logits = self.strurw.feat_classifier(source_feat)

        loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)

        return loss, source_logits, target_logits


    def fit(self, source_data, target_data):

        if source_data.edge_weight is None:
            source_data.edge_weight = torch.ones(source_data.edge_index.shape[1]).to(self.device)
        if target_data.edge_weight is None:
            target_data.edge_weight = torch.ones(target_data.edge_index.shape[1]).to(self.device)

        if self.batch_size == 0:
            self.source_batch_size = source_data.x.shape[0]
            source_loader = NeighborLoader(source_data,
                                self.num_neigh,
                                batch_size=self.source_batch_size)
            self.target_batch_size = target_data.x.shape[0]
            target_loader = NeighborLoader(target_data,
                                self.num_neigh,
                                batch_size=self.target_batch_size)
        else:
            source_loader = NeighborLoader(source_data,
                                self.num_neigh,
                                batch_size=self.batch_size)
            target_loader = NeighborLoader(target_data,
                                self.num_neigh,
                                batch_size=self.batch_size)

        self.strurw = self.init_model(**self.kwargs)

        if self.mode == 'adv':
            self.domain_discriminator = nn.Linear(self.hid_dim, 2).to(self.device)
            models = [self.strurw, self.domain_discriminator]
            params = itertools.chain(*[model.parameters() for model in models])

            optimizer = torch.optim.Adam(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                self.strurw.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )

        start_time = time.time()

        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            p = float(epoch) / self.epoch
            alpha = 2. / (1. + np.exp(-10. * p)) - 1

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(source_loader, target_loader)):
                self.strurw.train()
                
                if self.mode == 'mixup':
                    loss, source_logits, target_logits = self.forward_model_mixup(sampled_source_data, sampled_target_data, epoch)
                else:
                    loss, source_logits, target_logits = self.forward_model(sampled_source_data, sampled_target_data, alpha, epoch)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if idx == 0:
                    epoch_source_logits, epoch_source_labels = self.predict(sampled_source_data)
                else:
                    source_logits, source_labels = self.predict(sampled_source_data)
                    epoch_source_logits = torch.cat((epoch_source_logits, source_logits))
                    epoch_source_labels = torch.cat((epoch_source_labels, source_labels))
            
                # epoch_source_preds = epoch_source_logits.argmax(dim=1)
                # micro_f1_score = eval_micro_f1(epoch_source_labels, epoch_source_preds)

                # logger(epoch=epoch,
                #        loss=epoch_loss,
                #        source_train_acc=micro_f1_score,
                #        time=time.time() - start_time,
                #        verbose=self.verbose,
                #        train=True)
        

            epoch_source_preds = epoch_source_logits.argmax(dim=1)
            train_results = self.metrics(epoch_source_logits, epoch_source_labels)

            self.log(epoch, epoch_loss, train_results)
            early_stop = self.early_stop_check(self.strurw, result=train_results, epoch=epoch)

            if early_stop == "stop":
                break
            elif early_stop == "save":
                torch.save(self.strurw.state_dict(), self.best_model_dir)

        end_time = time.time()
        training_time = end_time - start_time

        self.strurw.load_state_dict(torch.load(self.best_model_dir))
        if self.verbose >= 1:
            print(f"== Best Model from Epoch {self.best_epoch+1:03d} with Val Micro-F1: {self.best_val:.4f} ==")

        self.finish()


    def cal_reweight(self, source_data, target_data, target_pred):

        src_edge_prob, tgt_edge_prob, tgt_true_edge_prob = self.cal_edge_prob_sep(source_data, target_data, target_pred)

        reweight_matrix = torch.div(tgt_edge_prob, src_edge_prob)
        reweight_matrix[torch.isinf(reweight_matrix)] = 1
        reweight_matrix[torch.isnan(reweight_matrix)] = 1

        num_nodes = source_data.x.shape[0] + target_data.x.shape[0]
        label_pred = torch.cat((source_data.y, target_pred))
        graph_label_one_hot = sp.csr_matrix(
            (np.ones(num_nodes), (np.arange(num_nodes), label_pred.cpu().numpy())),
            shape=(num_nodes, self.num_classes))
        src_label_one_hot = sp.csr_matrix(
            (np.ones(source_data.x.shape[0]), (np.arange(source_data.x.shape[0]), source_data.y.cpu().numpy())),
            shape=(source_data.x.shape[0], self.num_classes))
        
        edge_weight = torch.ones(source_data.edge_index.shape[1]).to(self.device)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                idx = np.intersect1d(
                    np.where(np.in1d(source_data.edge_index[0].cpu().numpy(), graph_label_one_hot.getcol(j).nonzero()[0]))[0],
                    np.where(np.in1d(source_data.edge_index[1].cpu().numpy(), src_label_one_hot.getcol(i).nonzero()[0]))[0])
                edge_weight[idx] = reweight_matrix[i][j].item()
        source_data.edge_weight = edge_weight


    def cal_edge_prob_sep(self, src_graph, tgt_graph, tgt_pred):

        src_adj = to_dense_adj(src_graph.edge_index, max_num_nodes=src_graph.x.shape[0])[0].cpu().numpy()
        tgt_adj = to_dense_adj(tgt_graph.edge_index, max_num_nodes=tgt_graph.x.shape[0])[0].cpu().numpy()
    
        num_class = self.num_classes
        num_nodes_src = src_graph.x.shape[0]
        num_nodes_tgt = tgt_graph.x.shape[0]
        src_label = src_graph.y
        tgt_label = tgt_graph.y
    
        src_label_one_hot = sp.csr_matrix(
            (np.ones(num_nodes_src), (np.arange(num_nodes_src), src_label.cpu().numpy())),
            shape=(src_graph.num_nodes, num_class))
        tgt_pred_one_hot = sp.csr_matrix(
            (np.ones(num_nodes_tgt), (np.arange(num_nodes_tgt), tgt_pred.cpu().numpy())),
            shape=(tgt_graph.num_nodes, num_class))
        tgt_label_one_hot = sp.csr_matrix(
            (np.ones(num_nodes_tgt), (np.arange(num_nodes_tgt), tgt_label.cpu().numpy())),
            shape=(tgt_graph.num_nodes, num_class))

        src_node_num = src_label_one_hot.sum(axis=0).T * src_label_one_hot.sum(axis=0)
        tgt_pred_node_sum = tgt_pred_one_hot.sum(axis=0).T * tgt_pred_one_hot.sum(axis=0)
        tgt_node_sum = tgt_label_one_hot.sum(axis=0).T * tgt_label_one_hot.sum(axis=0)

        src_num_edge = (src_label_one_hot.T * src_adj * src_label_one_hot)
        tgt_pred_num_edge = (tgt_pred_one_hot.T * tgt_adj * tgt_pred_one_hot)
        tgt_true_num_edge = (tgt_label_one_hot.T * tgt_adj * tgt_label_one_hot)

        src_edge_prob = src_num_edge / src_node_num
        tgt_edge_prob = tgt_pred_num_edge / (tgt_pred_node_sum + 1e-12)
        tgt_true_edge_prob = tgt_true_num_edge / tgt_node_sum

        src_edge_prob = torch.from_numpy(np.array(src_edge_prob))
        tgt_edge_prob = torch.from_numpy(np.array(tgt_edge_prob))
        tgt_true_edge_prob = torch.from_numpy(np.array(tgt_true_edge_prob))

        return src_edge_prob, tgt_edge_prob, tgt_true_edge_prob
    

    def cal_str_dif_rel(self, pred_mtx, true_mtx):

        cls1_diff = torch.abs(pred_mtx - true_mtx)
        cls0_diff = torch.abs((1 - pred_mtx) - (1 - true_mtx))
        abs_diff = 0.5 * cls0_diff + 0.5 * cls1_diff
        rel_diff_1 = abs_diff / true_mtx
        rel_diff_2 = abs_diff / pred_mtx
        rel_diff = 0.5 * rel_diff_1 + 0.5 * rel_diff_2
        rel_diff[torch.isinf(rel_diff_1)] = rel_diff_2[torch.isinf(rel_diff_1)]
        rel_diff[torch.isinf(rel_diff_2)] = rel_diff_1[torch.isinf(rel_diff_2)]
        rel_diff[torch.isnan(rel_diff)] = 0

        num = true_mtx.size(0) * true_mtx.size(1)

        return torch.sum(abs_diff) / num, torch.sum(rel_diff) / num
    

    def cal_str_diff_ratio(self, pred_mtx, true_mtx):

        intra_prob_pred = torch.diagonal(pred_mtx, 0).repeat_interleave(pred_mtx.size(1)).view(-1, pred_mtx.size(1))
        intra_prob_true = torch.diagonal(true_mtx, 0).repeat_interleave(true_mtx.size(1)).view(-1, true_mtx.size(1))
        pred_ratio = torch.div(pred_mtx, intra_prob_pred)
        true_ratio = torch.div(true_mtx, intra_prob_true)
        pred_ratio[torch.isnan(pred_ratio)] = 1
        true_ratio[torch.isnan(true_ratio)] = 1
        pred_ratio[torch.isinf(pred_ratio)] = pred_mtx[torch.isinf(pred_ratio)]
        true_ratio[torch.isinf(true_ratio)] = true_mtx[torch.isinf(true_ratio)]

        ratio_diff = torch.div(pred_ratio, true_ratio)
        ratio_diff[torch.isnan(ratio_diff)] = 1
        ratio_diff[torch.isinf(ratio_diff)] = 1

        num = true_mtx.size(0) * true_mtx.size(1) - pred_mtx.size(0)
        return (torch.sum(ratio_diff) - torch.sum(torch.diagonal(ratio_diff))) / num


    def calculate_str_diff(self, src_edge_prob, tgt_edge_prob, tgt_true_edge_prob):

        tgt_diff_abs, tgt_diff_rel = self.cal_str_dif_rel(tgt_edge_prob, tgt_true_edge_prob)
        src_tgt_diff_abs, src_tgt_diff_rel = self.cal_str_dif_rel(tgt_edge_prob, src_edge_prob)
        ratio_diff = self.cal_str_diff_ratio(tgt_edge_prob, src_edge_prob)
        log_diff = self.cal_str_dif_log(tgt_edge_prob, src_edge_prob)

        return [src_tgt_diff_abs, src_tgt_diff_rel, ratio_diff], tgt_diff_abs 
    


    def predict(self, data):

        self.strurw.eval()

        with torch.no_grad():
            if self.mode == 'mixup':
                data.edge_weight = torch.ones(data.edge_index.shape[1]).to(self.device)
                logits = self.strurw(data.x, data.edge_index, data.edge_index, 1, np.arange(data.x.shape[0]), data.edge_weight)
            else:
                _, logits = self.strurw(data, data.x)

        return logits, data.y


    def shuffle_data(self, data):

        data = copy.deepcopy(data).to(self.device)
        id_new_value_old = np.arange(data.x.shape[0])
        idx = np.arange(data.x.shape[0])
        id_shuffle = copy.deepcopy(idx)
        np.random.shuffle(id_shuffle)
        id_new_value_old[idx] = id_shuffle
        data = self.id_node(data, id_new_value_old)
        
        return data, id_new_value_old


    def id_node(self, data, id_new_value_old):


        data = copy.deepcopy(data).to(self.device)
        data.x = None
        data.y = data.y[id_new_value_old]

        id_old_value_new = torch.zeros(id_new_value_old.shape[0], dtype=torch.long).to(self.device)
        id_old_value_new[id_new_value_old] = torch.arange(0, id_new_value_old.shape[0], dtype=torch.long).to(self.device)
        row = data.edge_index[0]
        col = data.edge_index[1]
        row = id_old_value_new[row]
        col = id_old_value_new[col]
        data.edge_index = torch.stack([row, col], dim=0)

        return data
