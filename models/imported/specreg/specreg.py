from pygda.models import SpecReg as SpecRegImported

from  models.base_model import BaseGDA


class SpecReg(BaseGDA):
    def __init__(self, config):
        super(SpecReg, self).__init__(config)

        self.config = config
        model_cfg = self.config["model"]

        self.model = SpecRegImported(
            in_dim=model_cfg["in_dim"],
            hid_dim=model_cfg["hid_dim"],
            num_classes=model_cfg["num_classes"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg.get("dropout_ratio", 0.0),
            act=self.act,
            ppmi=model_cfg.get("ppmi", True),
            adv_dim=model_cfg.get("adv_dim", 40),
            reg_mode=model_cfg.get("reg_mode", True),
            gamma_adv=model_cfg.get("gamma_adv", 0.1),
            thr_smooth=model_cfg.get("thr_smooth", -1),
            gamma_smooth=model_cfg.get("gamma_smooth", 0.01),
            thr_mfr=model_cfg.get("thr_mfr", -1),
            gamma_mfr=model_cfg.get("gamma_mfr", 0.01),
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

    def forward_model(self, source_data, target_data, alpha, epoch):
        return self.model.forward_model(source_data, target_data, alpha, epoch)

    def fit(self, source_data, target_data):
        return self.model.fit(source_data, target_data)

    def predict(self, data, source=False):
        return self.model.predict(data, source)
