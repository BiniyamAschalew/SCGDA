from pygda.models import DGSDA as DGSDAImported

from models.base_model import BaseGDA


class DGSDA(BaseGDA):
    def __init__(self, config):
        super(DGSDA, self).__init__(config)

        self.config = config
        model_cfg = self.config["model"]

        dropout = model_cfg.get("dropout_ratio", model_cfg.get("dp_ratio", model_cfg.get("dprate", 0.0)))

        self.model = DGSDAImported(
            in_dim=model_cfg["in_dim"],
            hid_dim=model_cfg["hid_dim"],
            num_classes=model_cfg["num_classes"],
            mode=model_cfg.get("mode", "node"),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=dropout,
            act=self.act,
            K=model_cfg.get("K", 8),
            alpha=model_cfg.get("alpha", 0.05),
            beta=model_cfg.get("beta", 0.5),
            gamma=model_cfg.get("gamma", 0.05),
            weight_decay=model_cfg["weight_decay"],
            lr=model_cfg["lr"],
            epoch=model_cfg.get("epochs", 200),
            device=self.config["expt"]["device"],
            batch_size=model_cfg.get("batch_size", 0),
            num_neigh=model_cfg.get("num_neigh", -1),
            verbose=self.config["expt"].get("verbose", 2),
        )

    def init_model(self):
        return self.model.init_model()

    def forward_model(self, source_data, target_data):
        return self.model.forward_model(source_data, target_data)

    def fit(self, source_data, target_data):
        return self.model.fit(source_data, target_data)

    def predict(self, data, source=False):
        return self.model.predict(data, source)
