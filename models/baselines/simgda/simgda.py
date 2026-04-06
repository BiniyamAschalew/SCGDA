import time
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import global_mean_pool

from  models.base_model import BaseGDA
from  models.baselines.gnn.gnn_base import GNNBase
from  utils.train_utils.mmd import MMD


def _weighted_nll_loss(
    source_logits,
    source_labels,
    source_mask,
    target_logits,
    target_labels,
    target_mask,
    oracle: bool,
):
    device = source_logits.device
    source_count = int(source_mask.sum().item())
    target_count = int(target_mask.sum().item()) if oracle else 0
    total_count = source_count + target_count

    if total_count <= 0:
        return torch.zeros((), device=device, dtype=source_logits.dtype)

    loss = torch.zeros((), device=device, dtype=source_logits.dtype)
    if source_count > 0:
        loss = loss + F.nll_loss(source_logits[source_mask], source_labels[source_mask]) * source_count
    if oracle and target_count > 0:
        loss = loss + F.nll_loss(target_logits[target_mask], target_labels[target_mask]) * target_count
    return loss / float(total_count)


class SimGDA(BaseGDA):

    def __init__(
        self, config):
        
        super(SimGDA, self).__init__(config)
        self.config = config

        self.verbose = config["expt"]["verbose"]
        self.batch_size = config["model"]["batch_size"]
        #self.epoch = config["expt"]["epochs"]
        self.use_mask = config["model"]["use_mask"]
    

        self.lr = config["model"]["lr"]
        self.weight_decay = config["model"]["weight_decay"]
        # self.step_size = config["model"]["step_size"]
        # self.gamma = config["model"]["gamma"]

        self.mmd_weight = config["model"]["mmd_weight"]

        self.mode = config["model"]["mode"]
        self.simgda = None

    def init_model(self, **kwargs):

        model =  GNNBase(self.config).to(self.device)
        return model

    
    def forward_model(self, source_data, target_data, use_mask=False, oracle=False):

        # source_logits = self.gnn(source_data.x, source_data.edge_index)
        # target_logits = self.gnn(target_data.x, target_data.edge_index)

        source_features = self.simgda.feat_bottleneck(source_data.x, source_data.edge_index)
        target_features = self.simgda.feat_bottleneck(target_data.x, target_data.edge_index)

        source_logits = self.simgda.feat_classifier(source_features, source_data.edge_index)
        target_logits = self.simgda.feat_classifier(target_features, target_data.edge_index)

        source_logits = F.log_softmax(source_logits, dim=1)
        target_logits = F.log_softmax(target_logits, dim=1)

        source_mask = self.get_mask(source_data, use_mask=use_mask, mask_name="train_mask")
        target_mask = self.get_mask(target_data, use_mask=use_mask, mask_name="train_mask")

        loss = _weighted_nll_loss(
            source_logits,
            source_data.y,
            source_mask,
            target_logits,
            target_data.y,
            target_mask,
            oracle=oracle,
        )
        mmd_loss = MMD(source_features[source_mask], target_features[target_mask]).to(self.device)
        loss += self.mmd_weight * mmd_loss


        return loss, source_logits, target_logits


    def fit(self, source_data, target_data, use_mask=False, oracle=False):


        source_loader = self.get_loader(source_data)
        target_loader = self.get_loader(target_data)

        self.simgda = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.simgda.parameters(),
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
                self.simgda.train()
                
                loss, source_logits, target_logits = self.forward_model(
                    sampled_source_data,
                    sampled_target_data,
                    use_mask=use_mask,
                    oracle=oracle,
                )
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_source_logits, train_source_labels, train_source_mask = self.mask_logits_and_labels(
                    source_logits,
                    sampled_source_data,
                    use_mask=use_mask,
                    mask_name="train_mask",
                )
                if int(train_source_mask.sum().item()) > 0:
                    epoch_source_logits = torch.cat((epoch_source_logits, train_source_logits))
                    epoch_source_labels = torch.cat((epoch_source_labels, train_source_labels))
            
            train_results = self.metrics(epoch_source_logits, epoch_source_labels)

            self.log(epoch, epoch_loss, train_results)

        self.finish()


    def process_graph(self, data):
        pass

    def predict(self, data, use_mask=False):
        self.simgda.eval()

        with torch.no_grad():
            logits = self.simgda(data.x, data.edge_index)

        logits, labels, _ = self.mask_logits_and_labels(
            logits,
            data,
            use_mask=use_mask,
            mask_name="val_mask",
        )
        return logits, labels
