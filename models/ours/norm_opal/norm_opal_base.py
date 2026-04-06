import torch
from torch import nn
import torch.nn.functional as F

from Learn.Clean_SCGDA.models.ours.opal.opal_base import OPALBase


class NormOPALBase(OPALBase):

    def __init__(
        self,
        num_layers,
        features,
        hidden,
        classes,
        dropout_rate=0.0,
        K=15,
        cheb_lambda_max=2.0,
        norm_eps=1e-5,
    ):
        super(NormOPALBase, self).__init__(
            num_layers=num_layers,
            features=features,
            hidden=hidden,
            classes=classes,
            dropout_rate=dropout_rate,
            K=K,
            cheb_lambda_max=cheb_lambda_max,
        )

        self.encoder_norms = nn.ModuleList(
            [nn.LayerNorm(hidden, eps=norm_eps) for _ in range(len(self.encoder))]
        )

    def reset_parameters(self):
        super().reset_parameters()
        for norm in self.encoder_norms:
            norm.reset_parameters()

    def feat_bottleneck(self, x, edge_index, is_source_domain=True):

        filter = self.src_filter
        if not is_source_domain:
            filter = self.tgt_filter

        for lin, norm in zip(self.encoder, self.encoder_norms):
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = lin(x)
            x = F.relu(x)
            x = norm(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = self.propagate(filter, x, edge_index)

        return x

    def project(self, x):
        first_lin = self.encoder[0]
        first_norm = self.encoder_norms[0]

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = first_lin(x)
        x = F.relu(x)
        x = first_norm(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x

    def sample_reference(self, layer_idx, num_nodes, device, dtype):
        norm = self.encoder_norms[layer_idx]
        feature_dim = int(norm.weight.numel())

        # LayerNorm outputs a normalized feature vector followed by affine scaling.
        reference = torch.randn(num_nodes, feature_dim, device=device, dtype=dtype)
        weight = norm.weight.view(1, -1).to(device=device, dtype=dtype)
        bias = norm.bias.view(1, -1).to(device=device, dtype=dtype)
        return reference * weight + bias
