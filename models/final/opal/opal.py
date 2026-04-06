import torch
import torch.nn.functional as F
import itertools
import time

import numpy as np

from torch_geometric.loader import NeighborLoader, DataLoader

from  models.base_model import BaseGDA

from  models.ours.opal.opal_base import OPALBase
from  utils.train_utils.mmd import MMD, Sinkhorn



class OPAL(BaseGDA):

    def __init__(self, config: dict):
        
        super(OPAL, self).__init__(config)
        
        self.num_layers=config["model"]["num_layers"]
        self.K=config["model"]["K"]
        self.mode=config["model"]["mode"]
        self.alpha=config["model"]["alpha"]
        self.beta=config["model"]["beta"]
        self.gamma=config["model"]["gamma"]
        # self.prop_base = config["model"]["prop_base"]
        self.cheb_lambda_max = float(config["model"].get("cheb_lambda_max", 2.0))

        assert self.num_layers==2, 'unsupport number of layers'
        assert self.mode=='node', 'unsupport mode'


    def init_model(self):

        return OPALBase(
            features=self.in_dim,
            hidden=self.hid_dim,
            classes=self.num_classes,
            dropout_rate=self.dropout,
            K=self.K,
            # prop_base=self.prop_base,
            cheb_lambda_max=self.cheb_lambda_max
            
        ).to(self.device)


    def forward_model(self, source_data, target_data):

        # source domain cross entropy loss
        source_logits = self.dgsda(source_data)
        train_loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
        loss = train_loss

        theta_s = self.dgsda.src_filter.coef
        theta_t = self.dgsda.tgt_filter.coef

        # theta_loss = F.l1_loss(theta_s, theta_t)
        theta_loss = F.l1_loss(theta_s, theta_t) + torch.sum(torch.abs(theta_s)) + torch.sum(torch.abs(theta_t))

        loss = loss + theta_loss * self.alpha

        source_feature = F.relu(self.dgsda.lin1(source_data.x))
        target_feature = F.relu(self.dgsda.lin1(target_data.x))
        mmd_loss = MMD(source_feature, target_feature)
        loss = loss + mmd_loss * self.beta

        target_outputs = self.dgsda(target_data, False)
        entropy_loss = self.entropy_minimization_loss(target_outputs)
        loss = loss + entropy_loss * self.gamma 

        return loss, source_logits
    
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

        #     early_stop = self.early_stop_check(self.dgsda, result=train_results, epoch=epoch)

        #     if early_stop == "stop":
        #         break
        #     elif early_stop == "save":
        #         torch.save(self.dgsda.state_dict(), self.best_model_dir)
        # self.dgsda.load_state_dict(torch.load(self.best_model_dir))

        end_time = time.time()
        training_time = end_time - start_time

        if self.verbose >= 1:
            print(f"== Best Model from Epoch {self.best_epoch+1:03d} with Val Micro-F1: {self.best_val:.4f} ==")

        self.finish()



    def process_graph(self, data):
        """
        Process the input graph data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data to be processed.

        Notes
        -----
        This method is currently a placeholder as preprocessing is handled
        through the NeighborLoader and DataLoader classes during training
        and prediction phases.
        """

    def predict(self, data, source=False):
        """
        Make predictions on input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data.
        source : bool, optional
            Whether predicting on source domain. Default: ``False``.

        Returns
        -------
        tuple
            Contains:
            
            - logits : torch.Tensor
                Model predictions.

            - labels : torch.Tensor
                True labels.
        
        Notes
        -----
        The prediction process:
        
        1. **Model Evaluation**: Sets the model to evaluation mode to
           disable dropout and batch normalization updates.
        
        2. **Data Processing**: Uses the appropriate data loader (source
           or target) based on the source parameter.
        
        3. **Inference**: Performs forward pass without gradient computation
           for efficient inference.
        
        4. **Result Aggregation**: Concatenates predictions from multiple
           batches if the data is processed in batches.
        
        The method automatically handles both source and target domain
        prediction modes, with different processing pipelines for each.
        """

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
