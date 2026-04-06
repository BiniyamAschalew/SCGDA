from pygda.models import PairAlign as PairAlignImported

from Learn.Clean_SCGDA.models.base_model import BaseGDA


class PairAlign(BaseGDA):
    def __init__(self, config):
        super(PairAlign, self).__init__(config)

        self.config = config
        model_cfg = self.config["model"]

        self.model = PairAlignImported(
            in_dim=model_cfg["in_dim"],
            hid_dim=model_cfg["hid_dim"],
            num_classes=model_cfg["num_classes"],
            num_layers=model_cfg.get("num_layers", 2),
            cls_dim=model_cfg.get("cls_dim", 128),
            cls_layers=model_cfg.get("cls_layers", 2),
            dropout=model_cfg.get("dropout_ratio", 0.0),
            backbone=model_cfg.get("backbone", model_cfg.get("gnn", "GS")),
            pooling=model_cfg.get("pooling", "mean"),
            ew_type=model_cfg.get("ew_type", "pseudobeta"),
            rw_lmda=model_cfg.get("rw_lmda", 1.0),
            ls_lambda=model_cfg.get("ls_lambda", 1.0),
            lw_lambda=model_cfg.get("lw_lambda", 0.005),
            label_rw=model_cfg.get("label_rw", False),
            edge_rw=model_cfg.get("edge_rw", False),
            ew_start=model_cfg.get("ew_start", 0),
            ew_freq=model_cfg.get("ew_freq", 10),
            lw_start=model_cfg.get("lw_start", 0),
            lw_freq=model_cfg.get("lw_freq", 10),
            gamma_reg=model_cfg.get("gamma_reg", 1e-4),
            weight_CE_src=model_cfg.get("weight_CE_src", False),
            bn=model_cfg.get("bn", False),
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

    def forward_model(self, source_data, target_data, epoch):
        return self.model.forward_model(source_data, target_data, epoch)

    def fit(self, source_data, target_data):
        return self.model.fit(source_data, target_data)

    def predict(self, data, source=False):
        return self.model.predict(data)
