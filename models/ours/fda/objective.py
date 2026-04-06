import torch
import torch.nn.functional as F

from  models.__filters.build_filter import build_filter
from  utils.train_utils.mmd import MMD


class FDAFilterAlignObjective:
    """Filter propagation + alignment losses. Learnable params live in caller code."""

    def __init__(
        self,
        filter_name: str,
        mmd_sampling_num: int = 1000,
        mmd_times: int = 5,
        nonnegative_params: bool = False,
        eps: float = 1e-6,
    ):
        self.filter_name = str(filter_name).lower()
        self.filter = build_filter(self.filter_name)()

        self.mmd_sampling_num = int(mmd_sampling_num)
        self.mmd_times = int(mmd_times)
        self.nonnegative_params = bool(nonnegative_params)
        self.eps = float(eps)

    def effective_params(self, params: torch.Tensor) -> torch.Tensor:
        return F.relu(params) if self.nonnegative_params else params

    def apply_filter(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        params: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        params = self.effective_params(params).to(dtype=x.dtype, device=x.device)
        return self.filter(x, edge_index, params, edge_weight=edge_weight)

    @staticmethod
    def make_probe(
        source_embed: torch.Tensor,
        target_embed: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_features = torch.cat((source_embed, target_embed), dim=0).detach()
        probe_mean = all_features.mean(dim=0, keepdim=True)
        probe_std = all_features.std(dim=0, keepdim=True).clamp_min(float(eps))
        source_probe = torch.randn_like(source_embed) * probe_std + probe_mean
        target_probe = torch.randn_like(target_embed) * probe_std + probe_mean
        return source_probe, target_probe

    @staticmethod
    def make_semantic_probe(
        source_embed: torch.Tensor,
        target_embed: torch.Tensor,
        source_labels: torch.Tensor,
        class_conditional: bool = True,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_probe, target_probe = FDAFilterAlignObjective.make_probe(
            source_embed=source_embed,
            target_embed=target_embed,
            eps=eps,
        )
        if not bool(class_conditional):
            return source_probe, target_probe

        semantic_source_probe = torch.empty_like(source_embed)
        labels = source_labels.view(-1).long()
        for cls in torch.unique(labels):
            mask = labels == cls
            if int(mask.sum().item()) == 0:
                continue
            cls_feat = source_embed[mask]
            mu = cls_feat.mean(dim=0, keepdim=True)
            std = cls_feat.std(dim=0, keepdim=True).clamp_min(float(eps))
            semantic_source_probe[mask] = torch.randn_like(cls_feat) * std + mu

        return semantic_source_probe, target_probe

    def feature_mmd(self, source_embed: torch.Tensor, target_embed: torch.Tensor) -> torch.Tensor:
        return MMD(
            source_embed,
            target_embed,
            sampling_num=self.mmd_sampling_num,
            times=self.mmd_times,
        )

    def probe_mmd(
        self,
        source_embed: torch.Tensor,
        target_embed: torch.Tensor,
        source_edge_index: torch.Tensor,
        target_edge_index: torch.Tensor,
        source_params: torch.Tensor,
        target_params: torch.Tensor,
        source_edge_weight: torch.Tensor | None = None,
        target_edge_weight: torch.Tensor | None = None,
        source_probe: torch.Tensor | None = None,
        target_probe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source_probe is None or target_probe is None:
            source_probe, target_probe = self.make_probe(source_embed, target_embed, eps=self.eps)

        source_proj = self.apply_filter(
            source_probe,
            source_edge_index,
            params=source_params,
            edge_weight=source_edge_weight,
        )
        target_proj = self.apply_filter(
            target_probe,
            target_edge_index,
            params=target_params,
            edge_weight=target_edge_weight,
        )
        return self.feature_mmd(source_proj, target_proj)

    def total_alignment_loss(
        self,
        source_embed: torch.Tensor,
        target_embed: torch.Tensor,
        source_edge_index: torch.Tensor,
        target_edge_index: torch.Tensor,
        source_params: torch.Tensor,
        target_params: torch.Tensor,
        feature_mmd_weight: float = 0.1,
        probe_mmd_weight: float = 1.0,
        source_edge_weight: torch.Tensor | None = None,
        target_edge_weight: torch.Tensor | None = None,
        source_probe: torch.Tensor | None = None,
        target_probe: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_mmd_loss = self.feature_mmd(source_embed, target_embed)
        probe_mmd_loss = self.probe_mmd(
            source_embed=source_embed,
            target_embed=target_embed,
            source_edge_index=source_edge_index,
            target_edge_index=target_edge_index,
            source_params=source_params,
            target_params=target_params,
            source_edge_weight=source_edge_weight,
            target_edge_weight=target_edge_weight,
            source_probe=source_probe,
            target_probe=target_probe,
        )
        total = float(feature_mmd_weight) * feat_mmd_loss + float(probe_mmd_weight) * probe_mmd_loss
        return total, feat_mmd_loss, probe_mmd_loss
