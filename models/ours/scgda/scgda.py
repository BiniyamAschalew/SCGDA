"""
SCGDA: Spectral Contrastive Graph Domain Adaptation

Almost identical to DGSDA but with:
- Separate filter depths for source (Ks) and target (Kt)
- Spherical MMD (L2-normalized features before MMD)
"""

import torch
import torch.nn.functional as F
import time

from torch_geometric.loader import NeighborLoader, DataLoader
from tqdm import tqdm

from  models.base_model import BaseGDA
from  models.ours.scgda.scgda_base import SCGDABase
from  utils.train_utils.mmd import MMD


class SCGDA(BaseGDA):
    """
    SCGDA: Spectral Contrastive Graph Domain Adaptation.
    
    Key differences from DGSDA:
    - Separate filter depths: Ks for source, Kt for target
    - Spherical MMD: L2-normalized features before MMD computation
    
    Config parameters (matches DGSDA with additions):
        K (int): Default filter depth (used for K3)
        Ks (int): Source filter depth (default: 1)
        Kt (int): Target filter depth (default: 4)  
        alpha (float): Weight for filter coefficient matching loss
        beta (float): Weight for spherical MMD loss
        gamma (float): Weight for entropy minimization loss
    """

    def __init__(self, config: dict):
        super(SCGDA, self).__init__(config)
        
        self.num_layers = config["model"]["num_layers"]
        self.mode = config["model"]["mode"]
        
        # Filter parameters
        self.K = config["model"].get("K", 8)  # Default/K3 depth
        self.Ks = config["model"].get("Ks", 1)  # Source depth
        self.Kt = config["model"].get("Kt", 4)  # Target depth
        
        # Loss weights (same as DGSDA + new energy loss)
        self.alpha = config["model"].get("alpha", 0.05)
        self.beta = config["model"].get("beta", 0.5)
        self.gamma = config["model"].get("gamma", 0.05)
        self.delta = config["model"].get("delta", 0.1)  # Energy loss weight

        assert self.num_layers == 2, 'unsupported number of layers'
        assert self.mode == 'node', 'unsupported mode'

    def init_model(self):
        return SCGDABase(
            features=self.in_dim,
            hidden=self.hid_dim,
            classes=self.num_classes,
            dprate=self.dropout,
            Ks=self.Ks,
            Kt=self.Kt,
            K3=self.K,
        ).to(self.device)

    def forward_model(self, source_data, target_data):
        # Source domain cross entropy loss
        source_logits = self.scgda(source_data)
        train_loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
        loss = train_loss

        # Filter coefficient alignment loss (L1 between filter temps)
        theta_s = self.scgda.prop1.temp
        theta_t = self.scgda.prop2.temp
        
        # Since Ks != Kt, we compare the mean of coefficients
        theta_loss = F.l1_loss(theta_s.mean(), theta_t.mean())
        loss = loss + theta_loss * self.alpha

        # Spherical MMD loss (L2 normalize features before MMD)
        source_feature = F.relu(self.scgda.lin1(source_data.x))
        target_feature = F.relu(self.scgda.lin1(target_data.x))
        
        # L2 normalize for spherical MMD
        source_feature_norm = F.normalize(source_feature, p=2, dim=1)
        target_feature_norm = F.normalize(target_feature, p=2, dim=1)
        
        mmd_loss = MMD(source_feature_norm, target_feature_norm)
        loss = loss + mmd_loss * self.beta

        # Entropy minimization on target
        target_outputs = self.scgda(target_data, False)
        entropy_loss = self.entropy_minimization_loss(target_outputs)
        loss = loss + entropy_loss * self.gamma

        # Energy loss: KL divergence between filter responses on ones signal
        energy_loss = self.scgda.energy_loss(
            source_data.edge_index,
            target_data.edge_index,
            source_data.num_nodes,
            target_data.num_nodes,
            self.device
        )
        loss = loss + energy_loss * self.delta

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

        self.scgda = self.init_model()

        optimizer = torch.optim.Adam(
            self.scgda.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()

        for epoch in tqdm(range(self.epoch), desc="Training"):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(self.source_loader, self.target_loader)):
                self.scgda.train()

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

    def process_graph(self, data):
        """Placeholder for graph preprocessing."""
        pass

    def predict(self, data, source=False):
        self.scgda.eval()

        if source:
            for idx, sampled_data in enumerate(self.source_loader):
                sampled_data = sampled_data.to(self.device)
                with torch.no_grad():
                    logits = self.scgda(sampled_data)

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
                    logits = self.scgda(sampled_data, False)

                    if idx == 0:
                        logits, labels = logits, sampled_data.y
                    else:
                        sampled_logits, sampled_labels = logits, sampled_data.y
                        logits = torch.cat((logits, sampled_logits))
                        labels = torch.cat((labels, sampled_labels))

        return logits, labels
