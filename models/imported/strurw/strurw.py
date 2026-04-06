from pygda.models import StruRW as StruRWImported
from  models.base_model import BaseGDA

class StruRW(BaseGDA):
    def __init__(self, config):
        super(StruRW, self).__init__(config)

        self.config = config

        in_dim = self.config["model"]["in_dim"]
        hid_dim = self.config["model"]["hid_dim"]
        num_classes = self.config["model"]["num_classes"]
        num_layers = self.config["model"]["num_layers"]

        epochs = self.config["model"]["epochs"]
        lr = self.config["model"]["lr"]
        weight_decay = self.config["model"]["weight_decay"]
        lamb = self.config["model"]["lamb"]

        device = self.config["expt"]["device"]
        batch_size = self.config["model"].get("batch_size", 0)
        num_neigh = self.config["model"].get("num_neigh", -1)
        verbose = self.config["expt"].get("verbose", 2)

        self.model = StruRWImported(
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=self.dropout_ratio,
            act=self.act,
            epoch=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            batch_size=batch_size,
            num_neigh=num_neigh,
            verbose=verbose,
            lamb=lamb,
        )

    def init_model(self):
        return self.model.init_model()

    def forward_model(self, source_data, target_data, alpha):
        return self.model.forward_model(source_data, target_data, alpha)
    
    def fit(self, source_data, target_data):
        return self.model.fit(source_data, target_data)

    def predict(self, data):
        return self.model.predict(data)