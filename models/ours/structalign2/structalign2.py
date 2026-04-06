import time
import torch
import torch.nn.functional as F

from  models.base_model import BaseGDA
from  models.__components.bernnet import BernNetBase
from  utils.train_utils.mmd import mmd_kernel


class StructAlign2(BaseGDA):

    def __init__(self, config):
        super(StructAlign2, self).__init__(config)
        self.config = config

        self.verbose = config["expt"]["verbose"]
        self.batch_size = config["model"]["batch_size"]
        #self.epoch = config["expt"]["epochs"]

        self.lr = config["model"]["lr"]
        self.weight_decay = config["model"]["weight_decay"]

        self.mmd_weight = config["model"]["mmd_weight"]
        self.mode = config["model"]["mode"]
        self.simgda = None

        self.struct_align_epochs = int(config["model"].get("struct_align_epochs", 0))
        self.struct_align_init = config["model"].get("struct_align_init", "static")
        self.struct_align_lr = config["model"].get("struct_align_lr", self.lr)
        self.struct_align_weight_decay = config["model"].get("struct_align_weight_decay", self.weight_decay)
        self.struct_align_static_value = config["model"].get("struct_align_static_value", 1.0)
        self.struct_weight = config["model"].get("struct_weight", 1.0)

        self.role_dim = max(1, int(config["model"].get("role_dim", 8)))
        self.role_weight = config["model"].get("role_weight", 1.0)
        self.role_init_value = config["model"].get("role_init_value", 1.0)
        self.struct_steps = max(1, int(config["model"].get("struct_steps", self.role_dim)))

        moment_orders = config["model"].get("moment_orders", [1, 2, 3])
        if isinstance(moment_orders, int):
            moment_orders = list(range(1, moment_orders + 1))
        self.moment_orders = [int(order) for order in moment_orders if int(order) > 0]

        self.kernel_mul = config["model"].get("kernel_mul", 2.0)
        self.kernel_num = config["model"].get("kernel_num", 5)
        self.fix_sigma = config["model"].get("fix_sigma", None)
        self.role_kernel_mul = config["model"].get("role_kernel_mul", self.kernel_mul)
        self.role_kernel_num = config["model"].get("role_kernel_num", self.kernel_num)
        self.role_fix_sigma = config["model"].get("role_fix_sigma", None)
        self.kernel_sampling_num = int(config["model"].get("kernel_sampling_num", 1000))
        self.kernel_times = int(config["model"].get("kernel_times", 5))

    def init_model(self, **kwargs):
        model = BernNetBase(self.config).to(self.device)
        return model

    def _init_signal(self, num_nodes, device, dtype, init, static_value):
        # signals for matching the operator fingerprints
        init = str(init).lower()
        if init == "uniform":
            return torch.rand((num_nodes, 1), device=device, dtype=dtype)
        if init in ("gaussian", "guassian", "normal"):
            return torch.randn((num_nodes, 1), device=device, dtype=dtype)
        if init == "static":
            return torch.full(
                (num_nodes, 1),
                fill_value=float(static_value),
                device=device,
                dtype=dtype,
            )
        raise ValueError(f"Invalid signal init: {init}")

    def _get_role_filters(self):
        if hasattr(self.simgda, "props") and len(self.simgda.props) > 0:
            return list(self.simgda.props)
        raise RuntimeError("StructAlign2 requires bernstein filters.")

    def _spectral_moments(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        dtype = data.x.dtype if data.x.is_floating_point() else torch.float32
        device = data.x.device

        signal = self._init_signal(
            num_nodes,
            device,
            dtype,
            self.struct_align_init,
            self.struct_align_static_value,
        )
        filters = self._get_role_filters()

        moments = []
        for step in range(self.struct_steps):
            prop = filters[step % len(filters)]
            signal = prop(signal, edge_index)
            for order in self.moment_orders:
                moment = signal.pow(order).mean(dim=0)
                moments.append(moment.reshape(-1))

        if not moments:
            return torch.zeros(1, device=device, dtype=dtype)

        return torch.cat(moments, dim=0)

    def _compute_roles(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        dtype = data.x.dtype if data.x.is_floating_point() else torch.float32
        device = data.x.device

        signal = torch.full(
            (num_nodes, 1),
            fill_value=float(self.role_init_value),
            device=device,
            dtype=dtype,
        )
        filters = self._get_role_filters()

        trajectory = []
        for step in range(self.role_dim):
            prop = filters[step % len(filters)]
            signal = prop(signal, edge_index)
            trajectory.append(signal)

        if not trajectory:
            return torch.zeros((num_nodes, 1), device=device, dtype=dtype)

        return torch.cat(trajectory, dim=1)

    def _freeze_for_struct_align(self):
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

    def _restore_after_struct_align(self, original_state):
        for param in self.simgda.parameters():
            param.requires_grad = original_state.get(id(param), True)

    def structural_align(self, source_data, target_data):
        if self.struct_align_epochs <= 0:
            return

        original_state, filter_params = self._freeze_for_struct_align()
        if not filter_params:
            self._restore_after_struct_align(original_state)
            return

        optimizer = torch.optim.Adam(
            filter_params,
            lr=self.struct_align_lr,
            weight_decay=self.struct_align_weight_decay,
        )

        source_loader = self.get_loader(source_data)
        target_loader = self.get_loader(target_data)

        for epoch in range(self.struct_align_epochs):
            epoch_loss = 0.0
            for sampled_source_data, sampled_target_data in zip(source_loader, target_loader):
                self.simgda.train()

                source_moments = self._spectral_moments(sampled_source_data)
                target_moments = self._spectral_moments(sampled_target_data)

                loss = F.mse_loss(source_moments, target_moments)
                loss = self.struct_weight * loss
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.verbose >= 2:
                print(f"StructAlign Epoch {epoch+1:03d}, Loss: {epoch_loss:.4f}")

        self._restore_after_struct_align(original_state)

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

        source_role = self._compute_roles(source_data) * self.role_weight
        target_role = self._compute_roles(target_data) * self.role_weight

        mmd_loss = mmd_kernel(
            source_features,
            target_features,
            source_role,
            target_role,
            sampling_num=self.kernel_sampling_num,
            times=self.kernel_times,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma,
            role_kernel_mul=self.role_kernel_mul,
            role_kernel_num=self.role_kernel_num,
            role_fix_sigma=self.role_fix_sigma,
        )
        loss += self.mmd_weight * mmd_loss

        return loss, source_logits, target_logits

    def fit(self, source_data, target_data):
        source_loader = self.get_loader(source_data)
        target_loader = self.get_loader(target_data)

        self.simgda = self.init_model(**self.kwargs)
        if self.struct_align_epochs > 0:
            self.structural_align(source_data, target_data)

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
