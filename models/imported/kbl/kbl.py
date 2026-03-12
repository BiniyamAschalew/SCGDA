from pygda.models import KBL as KBLImported

from models.base_model import BaseGDA


class KBL(BaseGDA):
    def __init__(self, config):
        super(KBL, self).__init__(config)

        self.config = config
        model_cfg = self.config["model"]

        self.model = KBLImported(
            in_dim=model_cfg["in_dim"],
            hid_dim=model_cfg["hid_dim"],
            num_classes=model_cfg["num_classes"],
            k_cross=model_cfg["k_cross"],
            k_within=model_cfg["k_within"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg.get("dropout_ratio", 0.0),
            act=self.act,
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
        return self.model.predict(data)
