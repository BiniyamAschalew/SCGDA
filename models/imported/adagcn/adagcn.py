from pygda.models import AdaGCN as AdaGCNImported

from  models.base_model import BaseGDA


class AdaGCN(BaseGDA):
    def __init__(self, config):
        super(AdaGCN, self).__init__(config)

        self.config = config
        model_cfg = self.config["model"]

        self.model = AdaGCNImported(
            in_dim=model_cfg["in_dim"],
            hid_dim=model_cfg["hid_dim"],
            num_classes=model_cfg["num_classes"],
            mode=model_cfg.get("mode", "node"),
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg.get("dropout_ratio", 0.0),
            act=self.act,
            gnn_type=model_cfg.get("gnn", "gcn"),
            adv_dim=model_cfg.get("adv_dim", 40),
            gp_weight=model_cfg.get("gp_weight", 5),
            domain_weight=model_cfg.get("domain_weight", 1),
            weight_decay=model_cfg["weight_decay"],
            lr=model_cfg["lr"],
            epoch=model_cfg.get("epochs", 100),
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
