import torch

from  models.ours.opal.opal import OPAL
from  models.ours.norm_opal.norm_opal_base import NormOPALBase


class NormOPAL(OPAL):

    def __init__(self, config: dict):
        super(NormOPAL, self).__init__(config)
        self.norm_eps = float(config["model"].get("norm_eps", 1e-5))

    def init_model(self):

        return NormOPALBase(
            num_layers=self.num_layers,
            features=self.in_dim,
            hidden=self.hid_dim,
            classes=self.num_classes,
            dropout_rate=self.dropout,
            K=self.K,
            cheb_lambda_max=self.cheb_lambda_max,
            norm_eps=self.norm_eps,
        ).to(self.device)

    def generate_reference_probe(self, source_data, target_data, layer_idx):
        source_num = source_data.x.size(0)
        target_num = target_data.x.size(0)
        probe_num = max(int(source_num), int(target_num))

        probe_bank = self.opal.sample_reference(
            layer_idx=layer_idx,
            num_nodes=probe_num,
            device=source_data.x.device,
            dtype=source_data.x.dtype,
        )

        source_indices = self._sample_probe_indices(probe_num, int(source_num), probe_bank.device)
        target_indices = self._sample_probe_indices(probe_num, int(target_num), probe_bank.device)

        source_probe = probe_bank[source_indices]
        target_probe = probe_bank[target_indices]
        return source_probe, target_probe

    def filter_alignment(self, source_data, target_data, norm_divergence=True):
        filter_losses = []

        for layer_idx in range(len(self.opal.encoder_norms)):
            source_probe, target_probe = self.generate_reference_probe(
                source_data,
                target_data,
                layer_idx,
            )
            source_probe = self.opal.propagate(self.opal.src_filter, source_probe, source_data.edge_index)
            target_probe = self.opal.propagate(self.opal.tgt_filter, target_probe, target_data.edge_index)
            filter_losses.append(self.divergence(source_probe, target_probe))

        if not filter_losses:
            return source_data.x.new_tensor(0.0)

        return torch.stack(filter_losses).mean()
