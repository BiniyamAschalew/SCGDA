import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import torch.nn as nn

from torch_geometric.loader import NeighborLoader

from models.baselines.udagcn.udagcn_base import UDAGCNBase
from models.__layers.build_layer import build_layer, build_activation


from models.__layers.reverse_layer import GradReverse
from models.base_model import BaseGDA


class SpecReg(BaseGDA):
    
    def __init__(self, config: dict):
        super(SpecReg, self).__init__(config)

        self.config = config
        self.act = build_activation(config["model"]["activation"])


        self.ppmi=config["model"]["ppmi"]
        self.adv_dim=config["model"]["adv_dim"]
        self.reg_mode=config["model"]["reg_mode"]
        self.gamma_adv=config["model"]["gamma_adv"]

        self.thr_smooth=config["model"]["thr_smooth"]
        self.gamma_smooth=config["model"]["gamma_smooth"]
        self.thr_mfr=config["model"]["thr_mfr"]
        self.gamma_mfr=config["model"]["gamma_mfr"]


    def init_model(self):


        return UDAGCNBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            act=self.act,
            ppmi=self.ppmi,
            adv_dim=self.adv_dim,
        ).to(self.device)


    def forward_model(self, source_data, target_data, alpha, epoch):

        encoded_source = self.udagcn.encode(source_data, "source")
        encoded_target = self.udagcn.encode(target_data, "target")
        source_logits = self.udagcn.cls_model(encoded_source)

        # use source classifier loss:
        cls_loss = self.udagcn.loss_func(source_logits, source_data.y)

        _x_src, _x_tgt = encoded_source.detach(), encoded_target.detach()
        for _ in range(5):
            self.optimizer_critic.zero_grad()
            loss_1 = self.critic(_x_src).mean() - self.critic(_x_tgt).mean()
            loss_2 = self.calculate_gradient_penalty(_x_src, _x_tgt)
            loss_adv = - loss_1 + 10 * loss_2
            loss_adv.backward()
            self.optimizer_critic.step()
            
        loss_grl = self.critic(encoded_source).mean() - self.critic(encoded_target).mean()
        loss = cls_loss + loss_grl * self.gamma_adv

        if self.reg_mode:
            x_src = torch.einsum('nm,md->nd', source_data.eivec, encoded_source)
            x_tgt = torch.einsum('nm,md->nd', target_data.eivec, encoded_target)

            if self.thr_smooth > 0:
                delta_src = (x_src[:-1] - x_src[1:]).abs()
                delta_tgt = (x_tgt[:-1] - x_tgt[1:]).abs()
                loss = loss + (F.relu(delta_src - self.thr_smooth).mean() + F.relu(delta_tgt - self.thr_smooth).mean()) * self.gamma_smooth
            
            if self.thr_mfr > 0:
                loss = loss + (F.relu(x_src.abs() - self.thr_mfr).mean() + F.relu(x_tgt.abs() - self.thr_mfr).mean()) * self.gamma_mfr

        # use target classifier loss:
        target_logits = self.udagcn.cls_model(encoded_target)
        target_probs = F.softmax(target_logits, dim=-1)
        target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)

        loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))

        loss = loss + loss_entropy * (epoch / self.epoch * 0.01)

        return loss, source_logits, target_logits


    def fit(self, source_data, target_data):

        self.num_source_nodes, _ = source_data.x.shape
        self.num_target_nodes, _ = target_data.x.shape

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

        self.udagcn = self.init_model()

        params = itertools.chain(*[model.parameters() for model in self.udagcn.models])

        optimizer = torch.optim.Adam(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # module for SpecReg
        self.critic = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, 1)
        ).to(self.device)

        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), self.lr)

        start_time = time.time()

        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            alpha = min((epoch + 1) / self.epoch, 0.05)

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(source_loader, target_loader)):
                for model in self.udagcn.models:
                    model.train()
                
                loss, source_logits, target_logits = self.forward_model(sampled_source_data, sampled_target_data, alpha, epoch)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if idx == 0:
                    epoch_source_logits, epoch_source_labels = self.predict(sampled_source_data, source=True)
                else:
                    source_logits, source_labels = self.predict(sampled_source_data, source=True)
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
            early_stop = self.early_stop_check(self.udagcn, result=train_results, epoch=epoch)

            if early_stop == "stop":
                break
            elif early_stop == "save":
                torch.save(self.udagcn.state_dict(), self.best_model_dir)

        end_time = time.time()
        training_time = end_time - start_time

        self.udagcn.load_state_dict(torch.load(self.best_model_dir))
        if self.verbose >= 1:
            print(f"== Best Model from Epoch {self.best_epoch+1:03d} with Val Micro-F1: {self.best_val:.4f} ==")

        self.finish()


    def predict(self, data, source=False):

        for model in self.udagcn.models:
            model.eval()

        with torch.no_grad():
            if source:
                encoded_data = self.udagcn.encode(data, 'source')
            else:
                encoded_data = self.udagcn.encode(data, 'target')
            logits = self.udagcn.cls_model(encoded_data)

        return logits, data.y
    

    def calculate_gradient_penalty(self, x_src, x_tgt):

        x = torch.cat([x_src, x_tgt], dim=0).requires_grad_(True)
        x_out = self.critic(x)
        grad_out = torch.ones(x_out.shape, requires_grad=False).to(x_out.device)

        # Get gradient w.r.t. x
        grad = torch.autograd.grad(
            outputs=x_out,
            inputs=x,
            grad_outputs=grad_out,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,)[0]
        grad = grad.view(grad.shape[0], -1)
        grad_penalty = torch.mean((grad.norm(2, dim=1) - 1) ** 2)

        return grad_penalty
