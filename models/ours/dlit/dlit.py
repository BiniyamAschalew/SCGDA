"""
DLIT: Disentangled Spectral Graph Domain Adaptation

Core idea:
1. Learn a Chebyshev spectral filter for graph convolution
2. Derive structural roles from filter response on constant signal
3. Align spectral responses between domains (struct_mmd)
4. Align features weighted by structural role similarity (feature_mmd)

The method disentangles structural alignment from feature alignment,
using the learned spectral filter to condition cross-domain transfer.
"""

import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base_model import BaseGDA
from models.ours.dlit.cheb_filter import ChebFilter
from models.ours.dlit.dlit_encoder import DLITEncoder
from utils.train_utils.mmd import MMD, mmd_kernel


class DLIT(BaseGDA):
    """
    DLIT: Disentangled Spectral Graph Domain Adaptation
    
    Learns a spectral GNN with Chebyshev filters, then:
    - Computes K-dimensional structural roles from filter response
    - Aligns structural roles across domains (struct_mmd)
    - Aligns features weighted by role similarity (feature_mmd)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
        
        # DLIT-specific params
        self.cheb_K = config["model"].get("cheb_K", 5)
        self.role_steps = config["model"].get("role_steps", self.cheb_K)
        self.cheb_norm = config["model"].get("cheb_norm", "sym")
        
        # Loss weights
        self.feat_mmd_weight = config["model"].get("feat_mmd_weight", 0.1)
        self.struct_mmd_weight = config["model"].get("struct_mmd_weight", 0.1)
        
        # MMD sampling
        self.mmd_sampling = config["model"].get("mmd_sampling", 1000)
        
        # Models (initialized in init_model)
        self.encoder = None
        self.cheb_filter = None

    def init_model(self):
        """Initialize encoder and Chebyshev filter."""
        self.encoder = DLITEncoder(self.config).to(self.device)
        self.cheb_filter = ChebFilter(
            K=self.cheb_K,
            normalization=self.cheb_norm
        ).to(self.device)

    def compute_roles(self, data):
        """
        Compute structural roles for nodes.
        
        Roles are derived by repeatedly applying the learned Chebyshev filter
        to a constant (all-ones) signal, creating a K-dimensional embedding
        that captures each node's spectral response pattern.
        """
        return self.cheb_filter.compute_roles(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            steps=self.role_steps,
            dtype=data.x.dtype,
        )

    def forward_model(self, source_data, target_data, roles_src_detached, roles_tgt_detached):
        """
        Forward pass with domain adaptation losses.
        
        Args:
            source_data: Source domain graph
            target_data: Target domain graph
            roles_src_detached: Detached source roles for feature weighting
            roles_tgt_detached: Detached target roles for feature weighting
            
        Returns:
            total_loss: Combined classification + alignment losses
            source_logits: Source domain predictions
            loss_dict: Dictionary of individual losses for logging
        """
        # Get features through encoder
        h_src = self.encoder.feat_bottleneck(source_data.x, source_data.edge_index)
        h_tgt = self.encoder.feat_bottleneck(target_data.x, target_data.edge_index)
        
        # Source classification
        logits_src = self.encoder.feat_classifier(h_src, source_data.edge_index)
        loss_cls = F.nll_loss(F.log_softmax(logits_src, dim=1), source_data.y)
        
        # Compute current roles (with gradients for structural alignment)
        roles_src = self.compute_roles(source_data)
        roles_tgt = self.compute_roles(target_data)
        
        # Structural MMD: Align spectral roles between domains
        n_struct = min(self.mmd_sampling, roles_src.size(0), roles_tgt.size(0))
        loss_struct = MMD(roles_src, roles_tgt, sampling_num=n_struct)
        
        # Feature MMD: Weighted by structural role similarity (using detached roles)
        n_feat = min(self.mmd_sampling, h_src.size(0), h_tgt.size(0))
        loss_feat = mmd_kernel(
            h_src, h_tgt,
            roles_src_detached, roles_tgt_detached,
            sampling_num=n_feat,
        )
        
        # Combined loss
        total_loss = (
            loss_cls + 
            self.feat_mmd_weight * loss_feat + 
            self.struct_mmd_weight * loss_struct
        )
        
        loss_dict = {
            "cls": loss_cls.item(),
            "feat_mmd": loss_feat.item(),
            "struct_mmd": loss_struct.item(),
        }
        
        return total_loss, logits_src, loss_dict

    def fit(self, source_data, target_data):
        """
        Train DLIT model.
        
        Training flow:
        1. Initialize models
        2. For each epoch:
           - Compute roles (detached for feature weighting)
           - Forward pass with classification + alignment losses
           - Backprop and update
        """
        self.init_model()
        
        # Setup data loaders
        source_loader = self.get_loader(source_data)
        target_loader = self.get_loader(target_data)
        
        # Optimizer for both encoder and filter
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.cheb_filter.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        start_time = time.time()
        
        for epoch in tqdm(range(self.epoch), desc="DLIT Training", disable=self.verbose < 1):
            epoch_loss = 0
            epoch_logits = []
            epoch_labels = []
            
            for src_batch, tgt_batch in zip(source_loader, target_loader):
                self.encoder.train()
                self.cheb_filter.train()
                
                src_batch = src_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)
                
                # Compute roles (detached for feature MMD weighting)
                with torch.no_grad():
                    roles_src_detached = self.compute_roles(src_batch)
                    roles_tgt_detached = self.compute_roles(tgt_batch)
                
                # Forward pass
                loss, logits, loss_dict = self.forward_model(
                    src_batch, tgt_batch,
                    roles_src_detached, roles_tgt_detached
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_logits.append(logits.detach())
                epoch_labels.append(src_batch.y)
            
            # Compute training metrics
            epoch_logits = torch.cat(epoch_logits, dim=0)
            epoch_labels = torch.cat(epoch_labels, dim=0)
            train_results = self.metrics(epoch_logits, epoch_labels)
            
            # Logging
            self.log(epoch, epoch_loss, train_results)
            
            # Early stopping check
            if self.early_stopping:
                status = self.early_stop_check(self.encoder, train_results, epoch)
                if status == "stop":
                    break
        
        self.train_time = time.time() - start_time
        self.finish()

    def predict(self, data):
        """Make predictions on given data."""
        self.encoder.eval()
        self.cheb_filter.eval()
        
        with torch.no_grad():
            h = self.encoder.feat_bottleneck(data.x, data.edge_index)
            logits = self.encoder.feat_classifier(h, data.edge_index)
        
        return logits, data.y

    def process_graph(self, data):
        """Optional preprocessing (not used for DLIT)."""
        pass
