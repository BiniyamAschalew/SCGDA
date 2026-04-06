import torch

from Learn.Clean_SCGDA.utils.filter_utils import conditional_mmd, mmd_rbf
from Learn.Clean_SCGDA.utils.ablation_utils.common import as_float


def compute_shift_metrics(
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    source_y: torch.Tensor,
    target_y: torch.Tensor,
    idx_s: torch.Tensor,
    idx_t: torch.Tensor,
    cfg: dict,
) -> tuple[float, float]:
    xs = source_feat[idx_s]
    xt = target_feat[idx_t]
    ys = source_y[idx_s]
    yt = target_y[idx_t]
    val_mmd = mmd_rbf(
        xs,
        xt,
        kernel_mul=cfg["kernel_mul"],
        kernel_num=cfg["kernel_num"],
        fix_sigma=cfg["fix_sigma"],
    )
    val_cmmd = conditional_mmd(
        xs,
        xt,
        ys,
        yt,
        kernel_mul=cfg["kernel_mul"],
        kernel_num=cfg["kernel_num"],
        fix_sigma=cfg["fix_sigma"],
    )
    return as_float(val_mmd), as_float(val_cmmd)
