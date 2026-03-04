from abc import ABC, abstractmethod
import os
import time
import os.path as osp

import torch
from torch.nn import Linear, Sequential
from torch_geometric.loader import NeighborLoader

from utils.train_utils.metrics import BaseMetric
from utils.expt_utils import WandbHandler
from models.__layers.build_layer import build_activation

class BaseGDA(ABC):

    def __init__(self, config: dict):
        super(BaseGDA, self).__init__()

        # architecture params
        self.in_dim = config["model"]["in_dim"]
        self.hid_dim = config["model"]["hid_dim"]
        self.num_classes = config["model"]["num_classes"]
        self.num_layers = config["model"]["num_layers"]
        self.num_neigh = config["model"]["num_neigh"]

        self.gnn_type = config["model"]["gnn"]
        self.dropout = config["model"]["dropout_ratio"]
        self.weight_decay = config["model"]["weight_decay"]
        self.lr = config["model"]["lr"]
        self.batch_size = config["model"]["batch_size"]

        self.epoch = config["model"].get("epochs", 200)
        self.device = config["expt"]["device"]
        self.verbose = config["expt"]["verbose"]

        self.act = build_activation(config["model"]["activation"])

        # for early stopping
        self.time_stamp = time.strftime("%m_%d_%H_%M", time.localtime())
        src, tgt = config["expt"]["source"], config["expt"]["target"]
        model = config["model"]["name"]

        self.best_model_dir = f"./../../__saved__/models/{model}_{src}_{tgt}_{self.time_stamp}.pt"
        # ensure directory exists
        try:
            os.makedirs(osp.dirname(self.best_model_dir), exist_ok=True)
        except Exception:
            pass

        # start from -inf so the first epoch will be treated as an improvement and saved
        self.best_val = float('-inf')
        self.best_epoch = 0
        self.counter = 0

        self.patience = config["expt"]["patience"]
        self.early_stopping = config["expt"]["early_stopping"]
        self.early_stopping_metric = config["expt"].get("early_stopping_metric", "micro_f1")

        self.kwargs = config["model"]

        # needed in all the other models
        self.metrics = BaseMetric(config)
        self.wandb = WandbHandler(config)

        # self.num_neigh should be a list of length num_layers
        if type(self.num_neigh) is int:
            self.num_neigh = [self.num_neigh] * self.num_layers

        elif type(self.num_neigh) is list and len(self.num_neigh) != self.num_layers:
            raise ValueError(f"""Number of neighbors should have the same length as 
                             hidden layers dimension or the number of layers. 
                             len({self.num_neigh}) != {self.num_layers}""")
        else:
            raise ValueError(f'Number of neighbors {self.num_neigh} must be int or list of int')

    @abstractmethod
    def forward_model(self, data, **kwargs):
        pass

    @abstractmethod
    def fit(self, data, **kwargs):
        pass

    @abstractmethod
    def predict(self, data, **kwargs):
        pass

    @abstractmethod
    def init_model(self, **kwargs):
        pass

    def finish(self):
        self.wandb.finish()
        # delete the saved best model
        if os.path.exists(self.best_model_dir):
            os.remove(self.best_model_dir)


    def get_loader(self, data, batch_size=0):

        # sanitize graph and masks to avoid pyg sampler contiguity issues
        if data.edge_index is not None:
            data.edge_index = data.edge_index.clone().contiguous()

        # if batch_size is 0, we use full batch training
        if batch_size == 0:
            batch_size = data.x.shape[0]

        num_neigh = self.num_neigh
        if isinstance(num_neigh, int):
            num_neigh = [num_neigh] * self.num_layers

        loader = NeighborLoader(data, num_neigh, batch_size=batch_size)
        return loader


    def early_stop_check(self, model, result, epoch):

        cur_metric = result[self.early_stopping_metric]
        improved = cur_metric > self.best_val

        if self.verbose >= 2:
            print(f"Early Stopping Check: Best Val {self.best_val:.4f}, Current Val {cur_metric:.4f}, Improved: {improved}, Counter: {self.counter}/{self.patience}")

        if improved:
            self.best_val = cur_metric
            self.best_epoch = epoch
            self.counter = 0
            # torch.save(model.state_dict(), self.best_model_dir)
            return "save"
        
        else:
            self.counter += 1
            if self.early_stopping and self.counter >= self.patience:
                if self.verbose >= 2:
                    print(f"\n\n Early stopping at epoch {epoch+1:03d}. \n\n")
                return "stop" # stop

        return "continue" # do not stop

    def log(self, epoch, epoch_loss, train_results, test_results=None):

        if self.verbose >= 1:
            print(f"Epoch {epoch+1:03d}, Loss: {epoch_loss:.4f}")
        if self.verbose >= 2:
            print(f"train results: {train_results}")
            if test_results is not None:
                print(f"test results: {test_results}")    

        payload = {
            "epoch": epoch + 1,
            "Loss": epoch_loss,
            "train metrics": train_results,
        }
        if test_results is not None:
            payload["test metrics"] = test_results

        self.wandb.log(payload)
