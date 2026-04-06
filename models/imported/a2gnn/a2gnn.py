from pygda.models import A2GNN as A2GNN_imported
from Learn.Clean_SCGDA.models.base_model import BaseGDA


class A2GNN(BaseGDA):
    def __init__(self, config):
        super(A2GNN, self).__init__(config)

        self.config = config

        in_dim = self.config["model"]["in_dim"]
        hid_dim = self.config["model"]["hid_dim"]
        num_classes = self.config["model"]["num_classes"]
        num_layers = self.config["model"]["num_layers"]
        s_pnums = self.config["model"]["s_pnums"]
        t_pnums = self.config["model"]["t_pnums"]

        epochs = self.config["model"].get("epochs", 200)
        lr = self.config["model"]["lr"]
        weight_decay = self.config["model"]["weight_decay"]
        weight = self.config["model"]["weight"]

        adv = self.config["model"].get("adv", False)
        dropout = self.config["model"]["dropout_ratio"]
        mode = self.config["model"].get("mode", "node")

        device = self.config["expt"]["device"]
        batch_size = self.config["model"].get("batch_size", 0)
        num_neigh = self.config["model"].get("num_neigh", -1)
        verbose = self.config["expt"].get("verbose", 2)

        self.model = A2GNN_imported(
            in_dim = in_dim,
            hid_dim = hid_dim,
            num_classes = num_classes,
            mode = mode,
            num_layers = num_layers,
            dropout=dropout,
            act=self.act,
            s_pnums = s_pnums,
            t_pnums = t_pnums,
            adv = adv,
            weight = weight,

            epoch = epochs,
            lr = lr,
            weight_decay = weight_decay,
            device = device,
            batch_size = batch_size,
            num_neigh = num_neigh,
            verbose = verbose,
        )

    def init_model(self):
        return self.model.init_model()

    def forward_model(self, source_data, target_data, alpha):
        return self.model.forward_model(source_data, target_data, alpha)
    
    def fit(self, source_data, target_data):
        return self.model.fit(source_data, target_data)
    
    def predict(self, data, source=False):
        return self.model.predict(data, source)
