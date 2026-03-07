import time
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

from models.base_model import BaseGDA
from models.baselines.gnn.gnn_base import GNNBase


class GNN(BaseGDA):

    def __init__(
        self, config):
        
        super(GNN, self).__init__(config)
        self.config = config

        self.verbose = config["expt"]["verbose"]
        self.batch_size = config["model"]["batch_size"]
        #self.epoch = config["expt"]["epochs"]


        self.lr = config["model"]["lr"]
        self.use_mask = config["model"]["use_mask"]
        self.weight_decay = config["model"]["weight_decay"]

        self.gnn = None

    def init_model(self, **kwargs):

        model =  GNNBase(self.config).to(self.device)
        return model

    
    def forward_model(self, source_data, target_data):

        source_logits = self.gnn(source_data.x, source_data.edge_index)
        target_logits = self.gnn(target_data.x, target_data.edge_index)

        source_mask = torch.ones_like(source_data.y, dtype=torch.bool, device=self.device)
        if self.use_mask:
            source_mask = source_data.train_mask

        source_logits = source_logits[source_mask]
        source_labels = source_data.y[source_mask]

        loss = F.nll_loss(source_logits, source_labels)
        return loss, source_logits, target_logits


    def fit(self, source_data, target_data):


        source_loader = self.get_loader(source_data)
        target_loader = self.get_loader(target_data)

        self.gnn = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.gnn.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()

        from tqdm import tqdm
        for epoch in tqdm(range(self.epoch), desc="Training"):
            epoch_loss = 0
            epoch_source_logits = torch.empty(0).to(self.device)
            epoch_source_labels = torch.empty(0).to(self.device)

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(source_loader, target_loader)):
                self.gnn.train()
                
                loss, source_logits, target_logits = self.forward_model(sampled_source_data, sampled_target_data)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                source_logits, source_labels = self.predict(sampled_source_data)
                epoch_source_logits = torch.cat((epoch_source_logits, source_logits))
                epoch_source_labels = torch.cat((epoch_source_labels, source_labels))
            
            epoch_source_preds = epoch_source_logits.argmax(dim=1)
            train_results = self.metrics(epoch_source_logits, epoch_source_labels)

            self.log(epoch, epoch_loss, train_results)

        # end_time = time.time()
        # source validation performance

        source_logits, source_labels = self.predict(source_data)
        val_logits = source_logits[source_data.val_mask]
        val_labels = source_labels[source_data.val_mask]
        val_results = self.metrics(val_logits, val_labels)
        self.log(epoch + 3000, epoch_loss, val_results)
        

        self.finish()


    def process_graph(self, data):
        pass

    def predict(self, data):
        self.gnn.eval()

        with torch.no_grad():
            logits = self.gnn(data.x, data.edge_index)

        return logits, data.y