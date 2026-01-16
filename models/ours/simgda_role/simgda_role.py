import time
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import global_mean_pool

from models.base_model import BaseGDA
from models.baselines.gnn.gnn_base import GNNBase

from hypothesis.role.build_role import build_role
from utils.train_utils.mmd import MMD


class SimGDARole(BaseGDA):

    def __init__(
        self, config):
        
        super(SimGDARole, self).__init__(config)
        self.config = config

        self.verbose = config["expt"]["verbose"]
        self.batch_size = config["model"]["batch_size"]
        self.epoch = config["expt"]["epochs"]
    

        self.lr = config["model"]["lr"]
        self.weight_decay = config["model"]["weight_decay"]

        self.mmd_weight = config["model"]["mmd_weight"]
        self.mode = config["model"]["mode"]
        self.simgda = None

        self.role_builder = build_role(config)

    def init_model(self, **kwargs):

        model =  GNNBase(self.config).to(self.device)
        return model

    
    def forward_model(self, source_data, target_data):

        source_features = self.simgda.feat_bottleneck(source_data.x, source_data.edge_index)
        target_features = self.simgda.feat_bottleneck(target_data.x, target_data.edge_index)

        source_logits = self.simgda.feat_classifier(source_features, source_data.edge_index)
        target_logits = self.simgda.feat_classifier(target_features, target_data.edge_index)

        source_logits = F.log_softmax(source_logits, dim=1)
        target_logits = F.log_softmax(target_logits, dim=1)

        loss = F.nll_loss(source_logits, source_data.y)
        mmd_loss = MMD(source_features, target_features).to(self.device)
        loss += self.mmd_weight * mmd_loss


        return loss, source_logits, target_logits


    def fit(self, source_data, target_data):


        source_loader = self.get_loader(source_data)
        target_loader = self.get_loader(target_data)

        source_role, target_role = self.role_builder.build(source_data, target_data)

        self.simgda = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.simgda.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()

        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = torch.empty(0).to(self.device)
            epoch_source_labels = torch.empty(0).to(self.device)

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(source_loader, target_loader)):
                self.simgda.train()
                
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
            early_stop = self.early_stop_check(self.simgda, result=train_results, epoch=epoch)

            if early_stop == "stop":
                break
            elif early_stop == "save":
                torch.save(self.simgda.state_dict(), self.best_model_dir)

        # after training, load the best model
        self.simgda.load_state_dict(torch.load(self.best_model_dir))
        if self.verbose >= 1:
            print(f"== Best Model from Epoch {self.best_epoch+1:03d} with Val Micro-F1: {self.best_val:.4f} ==")

        self.finish()


    def process_graph(self, data):
        pass

    def predict(self, data):
        self.simgda.eval()

        with torch.no_grad():
            logits = self.simgda(data.x, data.edge_index)

        return logits, data.y