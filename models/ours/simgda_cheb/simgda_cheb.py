"""
SimGDACheb: SimGDA with Chebyshev polynomial filters.

Same domain adaptation strategy as SimGDA (classification + MMD),
but uses Chebyshev spectral convolutions instead of standard GCN.
"""

import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base_model import BaseGDA
from models.ours.simgda_cheb.simgda_cheb_encoder import SimGDAChebEncoder
from utils.train_utils.mmd import MMD


class SimGDACheb(BaseGDA):
    """
    SimGDA with Chebyshev polynomial graph convolutions.
    
    Uses ChebConv layers for spectral filtering while maintaining
    the simple classification + MMD domain adaptation approach.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
        
        # SimGDA params
        self.mmd_weight = config["model"].get("mmd_weight", 0.1)
        self.mode = config["model"]["mode"]
        
        # Chebyshev params
        self.cheb_K = config["model"].get("cheb_K", 3)
        
        self.encoder = None

    def init_model(self):
        """Initialize the Chebyshev encoder."""
        self.encoder = SimGDAChebEncoder(self.config).to(self.device)
        return self.encoder

    def forward_model(self, source_data, target_data):
        """
        Forward pass with classification and MMD losses.
        
        Args:
            source_data: Source domain graph
            target_data: Target domain graph
            
        Returns:
            loss: Combined classification + MMD loss
            source_logits: Source predictions
            target_logits: Target predictions
        """
        # Extract features
        source_features = self.encoder.feat_bottleneck(
            source_data.x, source_data.edge_index
        )
        target_features = self.encoder.feat_bottleneck(
            target_data.x, target_data.edge_index
        )
        
        # Classification
        source_logits = self.encoder.feat_classifier(
            source_features, source_data.edge_index
        )
        target_logits = self.encoder.feat_classifier(
            target_features, target_data.edge_index
        )
        
        source_logits = F.log_softmax(source_logits, dim=1)
        target_logits = F.log_softmax(target_logits, dim=1)
        
        # Classification loss
        loss_cls = F.nll_loss(source_logits, source_data.y)
        
        # MMD loss
        loss_mmd = MMD(source_features, target_features)
        
        # Combined loss
        loss = loss_cls + self.mmd_weight * loss_mmd
        
        return loss, source_logits, target_logits

    def fit(self, source_data, target_data):
        """Train SimGDACheb model."""
        source_loader = self.get_loader(source_data)
        target_loader = self.get_loader(target_data)
        
        self.encoder = self.init_model()
        
        optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        start_time = time.time()
        
        for epoch in tqdm(range(self.epoch), desc="SimGDACheb Training", disable=self.verbose < 1):
            epoch_loss = 0
            epoch_logits = []
            epoch_labels = []
            
            for src_batch, tgt_batch in zip(source_loader, target_loader):
                self.encoder.train()
                
                src_batch = src_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)
                
                loss, src_logits, tgt_logits = self.forward_model(src_batch, tgt_batch)
                epoch_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Collect predictions for metrics
                logits, labels = self.predict(src_batch)
                epoch_logits.append(logits)
                epoch_labels.append(labels)
            
            # Compute training metrics
            epoch_logits = torch.cat(epoch_logits, dim=0)
            epoch_labels = torch.cat(epoch_labels, dim=0)
            train_results = self.metrics(epoch_logits, epoch_labels)
            
            self.log(epoch, epoch_loss, train_results)
        
        self.train_time = time.time() - start_time
        
        if self.verbose >= 1:
            print(f"== Training completed in {self.train_time:.2f}s ==")
        
        self.finish()

    def predict(self, data):
        """Make predictions on given data."""
        self.encoder.eval()
        
        with torch.no_grad():
            logits = self.encoder(data.x, data.edge_index)
        
        return logits, data.y

    def process_graph(self, data):
        """Optional preprocessing (not used)."""
        pass
