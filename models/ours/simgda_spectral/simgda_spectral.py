import time
import torch
import torch.nn.functional as F

from models.base_model import BaseGDA
from models.__components.bernnet import BernNetBase
from models.__components.chebnet import ChebNetBase

from utils.train_utils.mmd import MMD

"""SimGDA with spectral filters (Chebyshev or Bernstein)
and optional warmup for filter alignment using MMD on the filter outputs 
with dummy features. The warmup can be configured to train only the 
filter parameters while keeping the rest of the model frozen, 
and can use different types of dummy feature initializations."""

class SimGDASpectral(BaseGDA):

    def __init__(self, config):
        super(SimGDASpectral, self).__init__(config)
        self.config = config

        self.verbose = config["expt"]["verbose"]
        self.batch_size = config["model"]["batch_size"]
        #self.epoch = config["expt"]["epochs"]

        self.lr = config["model"]["lr"]
        self.weight_decay = config["model"]["weight_decay"]

        self.mmd_weight = config["model"]["mmd_weight"]
        self.mode = config["model"]["mode"]
        self.simgda = None

        self.spectral_type = config["model"].get(
            "spectral_filter", config["model"].get("gnn", "cheb")
        )

        self.warmup_epochs = int(config["model"].get("warmup_epochs", 0))
        self.warmup_init = config["model"].get("warmup_init", "gaussian")
        self.warmup_lr = config["model"].get("warmup_lr", self.lr)
        self.warmup_weight_decay = config["model"].get("warmup_weight_decay", self.weight_decay)
        self.warmup_mmd_weight = config["model"].get("warmup_mmd_weight", 1.0)
        self.warmup_static_value = config["model"].get("warmup_static_value", 1.0)

    def init_model(self, **kwargs):
        spectral_type = self.spectral_type.lower()
        if spectral_type in ("cheb", "chebnet", "chebyshev"):
            model = ChebNetBase(self.config).to(self.device)
        elif spectral_type in ("bern", "bernnet", "bernstein"):
            model = BernNetBase(self.config).to(self.device)
        else:
            raise ValueError(f"Invalid spectral filter: {self.spectral_type}")
        return model

    def _init_dummy_features(self, x_ref, init):
        init = init.lower()
        dtype = x_ref.dtype if x_ref.is_floating_point() else torch.float32
        device = x_ref.device
        if init == "uniform":
            return torch.rand(x_ref.shape, device=device, dtype=dtype)
        if init in ("gaussian", "guassian"):
            return torch.randn(x_ref.shape, device=device, dtype=dtype)
        if init == "static":
            return torch.full(
                x_ref.shape,
                fill_value=float(self.warmup_static_value),
                device=device,
                dtype=dtype,
            )
        raise ValueError(f"Invalid warmup init: {init}")

    def _freeze_for_warmup(self):
        original = {}
        for param in self.simgda.parameters():
            original[id(param)] = param.requires_grad
            param.requires_grad = False

        filter_params = []
        if hasattr(self.simgda, "filter_parameters"):
            filter_params = list(self.simgda.filter_parameters())
            for param in filter_params:
                param.requires_grad = True

        return original, filter_params

    def _restore_after_warmup(self, original_state):
        for param in self.simgda.parameters():
            param.requires_grad = original_state.get(id(param), True)    

    def warmup(self, source_data, target_data, k, init):
        # we train for K epochs to align the output of the filter on the source and target toplogy (for an aligned feature space)
        if k <= 0:
            return

        original_state, filter_params = self._freeze_for_warmup()
        if not filter_params:
            self._restore_after_warmup(original_state)
            return

        optimizer = torch.optim.Adam(
            filter_params,
            lr=self.warmup_lr,
            weight_decay=self.warmup_weight_decay,
        )

        source_loader = self.get_loader(source_data)
        target_loader = self.get_loader(target_data)

        for epoch in range(k):
            epoch_loss = 0.0
            for sampled_source_data, sampled_target_data in zip(source_loader, target_loader):
                self.simgda.train()

                source_x = self._init_dummy_features(sampled_source_data.x, init)
                target_x = self._init_dummy_features(sampled_target_data.x, init)

                source_batch = getattr(sampled_source_data, "batch", None)
                target_batch = getattr(sampled_target_data, "batch", None)

                source_features = self.simgda.feat_bottleneck(
                    source_x, sampled_source_data.edge_index, batch=source_batch
                )
                target_features = self.simgda.feat_bottleneck(
                    target_x, sampled_target_data.edge_index, batch=target_batch
                )

                mmd_loss = MMD(source_features, target_features)
                loss = self.warmup_mmd_weight * mmd_loss
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.verbose >= 2:
                print(f"Warmup Epoch {epoch+1:03d}, Loss: {epoch_loss:.4f}")

        self._restore_after_warmup(original_state)

    def forward_model(self, source_data, target_data):
        source_batch = getattr(source_data, "batch", None)
        target_batch = getattr(target_data, "batch", None)

        source_features = self.simgda.feat_bottleneck(
            source_data.x, source_data.edge_index, batch=source_batch
        )
        target_features = self.simgda.feat_bottleneck(
            target_data.x, target_data.edge_index, batch=target_batch
        )

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

        self.simgda = self.init_model(**self.kwargs)

        if self.warmup_epochs > 0:
            self.warmup(source_data, target_data, self.warmup_epochs, self.warmup_init)

        optimizer = torch.optim.Adam(
            self.simgda.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        start_time = time.time()

        from tqdm import tqdm
        for epoch in tqdm(range(self.epoch), desc="Training"):
            epoch_loss = 0
            epoch_source_logits = torch.empty(0).to(self.device)
            epoch_source_labels = torch.empty(0).to(self.device)

            for sampled_source_data, sampled_target_data in zip(source_loader, target_loader):
                self.simgda.train()

                loss, source_logits, target_logits = self.forward_model(
                    sampled_source_data, sampled_target_data
                )
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                source_logits, source_labels = self.predict(sampled_source_data)
                epoch_source_logits = torch.cat((epoch_source_logits, source_logits))
                epoch_source_labels = torch.cat((epoch_source_labels, source_labels))

            train_results = self.metrics(epoch_source_logits, epoch_source_labels)

            self.log(epoch, epoch_loss, train_results)
            early_stop = self.early_stop_check(self.simgda, result=train_results, epoch=epoch)

            if early_stop == "stop":
                break
            if early_stop == "save":
                torch.save(self.simgda.state_dict(), self.best_model_dir)

        self.simgda.load_state_dict(torch.load(self.best_model_dir))
        if self.verbose >= 1:
            print(
                f"== Best Model from Epoch {self.best_epoch+1:03d} "
                f"with Val Micro-F1: {self.best_val:.4f} =="
            )

        self.finish()

    def process_graph(self, data):
        pass

    def predict(self, data):
        self.simgda.eval()

        with torch.no_grad():
            batch = getattr(data, "batch", None)
            logits = self.simgda(data.x, data.edge_index, batch=batch)

        return logits, data.y







