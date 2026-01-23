import time
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

from models.base_model import BaseGDA
from models.baselines.mlp.mlp_base import MLPBase


class MLP(BaseGDA):

    def __init__(
        self, config):
        
        super(MLP, self).__init__(config)
        self.config = config

        self.verbose = config["expt"]["verbose"]
        self.batch_size = config["model"]["batch_size"]
        #self.epoch = config["expt"]["epochs"]


        self.lr = config["model"]["lr"]
        self.weight_decay = config["model"]["weight_decay"]

        self.mlp = None

    def init_model(self, **kwargs):

        model =  MLPBase(self.config).to(self.device)
        return model

    
    def forward_model(self, source_data, target_data):

        source_logits = self.mlp(source_data.x, source_data.edge_index)
        target_logits = self.mlp(target_data.x, target_data.edge_index)

        loss = F.nll_loss(source_logits, source_data.y)
        return loss, source_logits, target_logits


    def fit(self, source_data, target_data):


        source_loader = self.get_loader(source_data)
        target_loader = self.get_loader(target_data)

        self.mlp = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.mlp.parameters(),
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
                self.mlp.train()
                
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

        #     early_stop = self.early_stop_check(self.mlp, result=train_results, epoch=epoch)

        #     if early_stop == "stop":
        #         break
        #     elif early_stop == "save":
        #         torch.save(self.mlp.state_dict(), self.best_model_dir)
        # self.mlp.load_state_dict(torch.load(self.best_model_dir))

        # after training, load the best model
        if self.verbose >= 1:
            print(f"== Best Model from Epoch {self.best_epoch+1:03d} with Val Micro-F1: {self.best_val:.4f} ==")

        self.finish()


    def process_graph(self, data):
        pass

    def predict(self, data):
        self.mlp.eval()

        with torch.no_grad():
            logits = self.mlp(data.x, data.edge_index)

        return logits, data.y