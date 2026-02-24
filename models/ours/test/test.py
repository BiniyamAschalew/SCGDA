import time
import torch
import torch.nn.functional as F
# from torch_geometric.loader import NeighborLoader
# from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

from models.base_model import BaseGDA
from models.ours.test.test_base import TestBase
from utils.filter_utils import cheb_to_monomial, tensor_to_float_list
from utils.train_utils.mmd import MMD


class Test(BaseGDA):

    def __init__(
        self, config):
        
        super(Test, self).__init__(config)
        self.config = config

        self.verbose = config["expt"]["verbose"]
        self.batch_size = config["model"]["batch_size"]
        #self.epoch = config["expt"]["epochs"]
    

        self.lr = config["model"]["lr"]
        self.weight_decay = config["model"]["weight_decay"]
        # self.step_size = config["model"]["step_size"]
        # self.gamma = config["model"]["gamma"]

        self.mmd_weight = config["model"]["mmd_weight"]
        self.mode = config["model"]["mode"]
        self.test_model = None

    def init_model(self, **kwargs):

        model = TestBase(self.config).to(self.device)
        print(f"\nUsing a base architecture of {model.__class__.__name__}\n")
        return model

    
    def forward_model(self, source_data, target_data):

        source_batch = getattr(source_data, "batch", None)
        target_batch = getattr(target_data, "batch", None)

        source_features = self.test_model.feat_bottleneck(
            source_data.x,
            source_data.edge_index,
            batch=source_batch,
            domain="source",
        )
        target_features = self.test_model.feat_bottleneck(
            target_data.x,
            target_data.edge_index,
            batch=target_batch,
            domain="target",
        )

        source_logits = self.test_model.feat_classifier(
            source_features,
            source_data.edge_index,
            domain="source",
        )
        target_logits = self.test_model.feat_classifier(
            target_features,
            target_data.edge_index,
            domain="target",
        )

        source_logits = F.log_softmax(source_logits, dim=1)
        target_logits = F.log_softmax(target_logits, dim=1)

        loss = F.nll_loss(source_logits, source_data.y)
        mmd_loss = MMD(source_features, target_features).to(self.device)
        loss += self.mmd_weight * mmd_loss


        return loss, source_logits, target_logits


    def fit(self, source_data, target_data):


        source_loader = self.get_loader(source_data)
        target_loader = self.get_loader(target_data)

        self.test_model = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.test_model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()

        for epoch in tqdm(range(self.epoch), desc="Training"):
            epoch_loss = 0
            epoch_source_logits = torch.empty(0).to(self.device)
            epoch_source_labels = torch.empty(0).to(self.device)

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(source_loader, target_loader)):
                self.test_model.train()
                
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

        #     early_stop = self.early_stop_check(self.simgda, result=train_results, epoch=epoch)

        #     if early_stop == "stop":
        #         break
        #     elif early_stop == "save":
        #         torch.save(self.simgda.state_dict(), self.best_model_dir)
        # self.simgda.load_state_dict(torch.load(self.best_model_dir))

        # after training, load the best model
        source_cheb = torch.relu(self.test_model.source_temp.detach())
        target_cheb = torch.relu(self.test_model.target_temp.detach())
        source_mono = cheb_to_monomial(source_cheb)
        target_mono = cheb_to_monomial(target_cheb)

        print("Learned source filter (Cheb):", tensor_to_float_list(source_cheb))
        print("Learned source filter (Monomial):", tensor_to_float_list(source_mono))
        print("Learned target filter (Cheb):", tensor_to_float_list(target_cheb))
        print("Learned target filter (Monomial):", tensor_to_float_list(target_mono))

        if self.verbose >= 1:
            print(f"== Best Model from Epoch {self.best_epoch+1:03d} with Val Micro-F1: {self.best_val:.4f} ==")

        self.finish()


    def process_graph(self, data):
        pass

    def predict(self, data):
        self.test_model.eval()

        with torch.no_grad():
            batch = getattr(data, "batch", None)
            logits = self.test_model(data.x, data.edge_index, batch=batch, domain="source")

        return logits, data.y
