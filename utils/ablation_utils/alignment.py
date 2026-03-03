import torch
import torch.nn.functional as F
from torch import nn

from models.__components.chebprop import ChebProp
from models.ours.fda.objective import FDAFilterAlignObjective
from utils.filter_utils import make_gaussian_probe
from utils.ablation_utils.common import as_float, project_pair_features, sample_idx
from utils.ablation_utils.metrics import compute_shift_metrics
from utils.ablation_utils.propagation import apply_cheb_once, induced_operator_matrix, operator_to_edges


def filter_smoothness_loss(temp: torch.Tensor) -> torch.Tensor:
    """Legacy coefficient-smoothness helper (kept for compatibility)."""
    if temp.numel() <= 1:
        return torch.zeros((), device=temp.device, dtype=temp.dtype)
    return (temp[1:] - temp[:-1]).pow(2).mean()


def _build_ppmi_positive_edges(data, cfg: dict) -> torch.Tensor:
    """Build PPMI positive edges once for structural BPR regularization."""
    from models.__layers.ppmi_conv import PPMIConv

    conv = PPMIConv(
        in_channels=1,
        out_channels=1,
        use_bias=False,
        path_len=int(cfg.get("ppmi_path_len", 5)),
    )
    ppmi_edge_index, ppmi_edge_weight = conv.norm(
        data.edge_index,
        num_nodes=int(data.x.size(0)),
        edge_weight=None,
        improved=False,
        dtype=data.x.dtype,
    )
    mask = ppmi_edge_index[0] != ppmi_edge_index[1]
    threshold = float(cfg.get("ppmi_pos_threshold", 0.0))
    if threshold > 0:
        mask = mask & (ppmi_edge_weight > threshold)
    return ppmi_edge_index[:, mask]


def build_struct_positive_edges(data, cfg: dict) -> torch.Tensor:
    """Choose positive graph pairs for structural BPR objective."""
    mode = str(cfg.get("struct_loss_mode", "adjacency")).lower()
    if mode == "adjacency":
        return data.edge_index
    if mode == "ppmi":
        return _build_ppmi_positive_edges(data, cfg)
    raise ValueError(f"Unknown struct_loss_mode={mode}. Use 'adjacency' or 'ppmi'.")


def edge_bpr_structure_loss(
    z: torch.Tensor,
    pos_edge_index: torch.Tensor,
    num_samples: int,
    margin: float = 0.0,
) -> torch.Tensor:
    """BPR-style structure loss from positive edges vs random negatives."""
    if pos_edge_index.numel() == 0 or z.size(0) <= 1:
        return z.new_tensor(0.0)

    e = int(pos_edge_index.size(1))
    k = int(min(max(1, num_samples), e))
    pos_ids = torch.randint(e, (k,), device=pos_edge_index.device)
    pos_u = pos_edge_index[0, pos_ids]
    pos_v = pos_edge_index[1, pos_ids]

    num_nodes = int(z.size(0))
    neg_u = pos_u
    neg_v = torch.randint(num_nodes, (k,), device=z.device)
    same = neg_v == neg_u
    neg_v = torch.where(same, (neg_v + 1) % num_nodes, neg_v)

    pos_dist = (z[pos_u] - z[pos_v]).pow(2).sum(dim=1)
    neg_dist = (z[neg_u] - z[neg_v]).pow(2).sum(dim=1)

    # score = -distance ; BPR => -log sigma(score_pos - score_neg)
    # with margin this is softplus(margin + d_pos - d_neg)
    return F.softplus(float(margin) + pos_dist - neg_dist).mean()


def train_probe_aligner(source_data, target_data, cfg: dict) -> dict: # create gaussian probes, project the actual features and the probes, 
    """Train source/target operators using FDA objective + structural regularization."""
    device = source_data.x.device
    k = int(cfg["cheb_k"])

    source_probe, target_probe = make_gaussian_probe(
        source_data.x.detach().float(),
        target_data.x.detach().float(),
    )
    source_align_probe, target_align_probe = project_pair_features(
        source_probe,
        target_probe,
        out_dim=int(cfg["align_feature_dim"]),
        seed=int(cfg["seed"]) + 11,
    )
    source_align_real, target_align_real = project_pair_features(
        source_data.x.detach().float(),
        target_data.x.detach().float(),
        out_dim=int(cfg["align_feature_dim"]),
        seed=int(cfg["seed"]) + 11,
    )

    fda_objective = FDAFilterAlignObjective(
        filter_name=str(cfg.get("fda_filter", "cheb")),
        mmd_sampling_num=int(cfg.get("fda_mmd_sampling_num", cfg.get("mmd_sampling_num", 1000))),
        mmd_times=int(cfg.get("fda_mmd_times", cfg.get("mmd_times", 5))),
        nonnegative_params=bool(cfg.get("fda_nonnegative_params", True)),
    )

    if fda_objective.filter_name != "cheb":
        raise ValueError(
            "shift aligner currently requires fda_filter='cheb' so operator-to-edge conversion "
            "remains valid with Chebyshev basis."
        )

    source_temp = nn.Parameter(torch.full((k,), 1.0 / float(k), device=device))
    target_temp = nn.Parameter(torch.full((k,), 1.0 / float(k), device=device))
    source_edge_logits = nn.Parameter(torch.zeros(source_data.edge_index.size(1), device=device))
    target_edge_logits = nn.Parameter(torch.zeros(target_data.edge_index.size(1), device=device))
    optimizer = torch.optim.Adam(
        [source_temp, target_temp, source_edge_logits, target_edge_logits],
        lr=cfg["align_lr"],
        weight_decay=cfg["align_weight_decay"],
    )

    cheb_prop = ChebProp(k, is_source_domain=False).to(device)
    idx_s = sample_idx(source_data.x.size(0), cfg["metric_sample_size"], device)
    idx_t = sample_idx(target_data.x.size(0), cfg["metric_sample_size"], device)
    source_struct_pos_edges = build_struct_positive_edges(source_data, cfg)
    target_struct_pos_edges = build_struct_positive_edges(target_data, cfg)

    history_rows = []
    edge_eps = float(cfg["edge_eps"])
    struct_bpr_weight = float(cfg.get("struct_bpr_weight", cfg.get("smooth_reg", 0.0)))
    struct_bpr_samples = int(cfg.get("struct_bpr_samples", 2048))
    struct_bpr_margin = float(cfg.get("struct_bpr_margin", 0.0))
    feature_mmd_weight = float(cfg.get("fda_feature_mmd_weight", 0.1))
    probe_mmd_weight = float(cfg.get("fda_probe_mmd_weight", 1.0))

    def _effective_temp(temp: torch.Tensor) -> torch.Tensor:
        return fda_objective.effective_params(temp)

    for epoch in range(1, cfg["align_epochs"] + 1):
        optimizer.zero_grad()

        source_w = F.softplus(source_edge_logits) + edge_eps
        target_w = F.softplus(target_edge_logits) + edge_eps
        src_temp_pos = _effective_temp(source_temp)
        tgt_temp_pos = _effective_temp(target_temp)
        src_probe_out = fda_objective.apply_filter(
            source_align_probe,
            source_data.edge_index,
            params=src_temp_pos,
            edge_weight=source_w,
        )
        tgt_probe_out = fda_objective.apply_filter(
            target_align_probe,
            target_data.edge_index,
            params=tgt_temp_pos,
            edge_weight=target_w,
        )
        align_idx_s = sample_idx(src_probe_out.size(0), int(cfg["align_sample_size"]), device)
        align_idx_t = sample_idx(tgt_probe_out.size(0), int(cfg["align_sample_size"]), device)

        probe_loss = fda_objective.feature_mmd(
            src_probe_out[align_idx_s],
            tgt_probe_out[align_idx_t],
        )
        feat_mmd_loss = fda_objective.feature_mmd(
            source_align_real[align_idx_s],
            target_align_real[align_idx_t],
        )
        align_loss = (
            feature_mmd_weight * feat_mmd_loss
            + probe_mmd_weight * probe_loss
        )

        edge_reg = cfg["edge_reg"] * (source_w.pow(2).mean() + target_w.pow(2).mean())
        temp_reg = cfg["temp_reg"] * (src_temp_pos.pow(2).mean() + tgt_temp_pos.pow(2).mean())

        src_struct = apply_cheb_once(
            source_align_real,
            source_data,
            cheb_prop,
            src_temp_pos,
            source_w,
            cfg["cheb_lambda_max"],
        )
        tgt_struct = apply_cheb_once(
            target_align_real,
            target_data,
            cheb_prop,
            tgt_temp_pos,
            target_w,
            cfg["cheb_lambda_max"],
        )
        src_bpr = edge_bpr_structure_loss(
            src_struct,
            source_struct_pos_edges,
            num_samples=struct_bpr_samples,
            margin=struct_bpr_margin,
        )
        tgt_bpr = edge_bpr_structure_loss(
            tgt_struct,
            target_struct_pos_edges,
            num_samples=struct_bpr_samples,
            margin=struct_bpr_margin,
        )
        struct_reg = struct_bpr_weight * (src_bpr + tgt_bpr)
        loss = align_loss + edge_reg + temp_reg + struct_reg
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % cfg["align_log_interval"] == 0 or epoch == cfg["align_epochs"]:
            with torch.no_grad():
                source_w = F.softplus(source_edge_logits) + edge_eps
                target_w = F.softplus(target_edge_logits) + edge_eps
                src_real = apply_cheb_once(
                    source_align_real,
                    source_data,
                    cheb_prop,
                    _effective_temp(source_temp),
                    source_w,
                    cfg["cheb_lambda_max"],
                )
                tgt_real = apply_cheb_once(
                    target_align_real,
                    target_data,
                    cheb_prop,
                    _effective_temp(target_temp),
                    target_w,
                    cfg["cheb_lambda_max"],
                )
                real_mmd, real_cmmd = compute_shift_metrics(
                    src_real,
                    tgt_real,
                    source_data.y,
                    target_data.y,
                    idx_s,
                    idx_t,
                    cfg,
                )
                history_rows.append(
                    {
                        "epoch": epoch,
                        "feat_mmd": as_float(feat_mmd_loss),
                        "probe_mmd": as_float(probe_loss),
                        "real_mmd": real_mmd,
                        "real_cmmd": real_cmmd,
                        "source_bpr": as_float(src_bpr),
                        "target_bpr": as_float(tgt_bpr),
                        "align_loss": as_float(align_loss),
                        "struct_loss": as_float(struct_reg),
                        "loss": as_float(loss),
                    }
                )

    with torch.no_grad():
        source_w = F.softplus(source_edge_logits) + edge_eps
        target_w = F.softplus(target_edge_logits) + edge_eps
        source_temp_final = _effective_temp(source_temp).detach()
        target_temp_final = _effective_temp(target_temp).detach()

        source_h = induced_operator_matrix(
            source_data,
            cheb_prop,
            source_temp_final,
            source_w,
            cfg["cheb_lambda_max"],
            cfg["operator_chunk_size"],
        )
        target_h = induced_operator_matrix(
            target_data,
            cheb_prop,
            target_temp_final,
            target_w,
            cfg["cheb_lambda_max"],
            cfg["operator_chunk_size"],
        )
        source_aligned_edge_index, source_aligned_edge_weight = operator_to_edges(
            source_h,
            edge_threshold=cfg["operator_edge_threshold"],
            topk=cfg["operator_topk"],
            device=device,
        )
        target_aligned_edge_index, target_aligned_edge_weight = operator_to_edges(
            target_h,
            edge_threshold=cfg["operator_edge_threshold"],
            topk=cfg["operator_topk"],
            device=device,
        )

    return {
        "source_temp": source_temp_final,
        "target_temp": target_temp_final,
        "source_edge_weight": source_w.detach(),
        "target_edge_weight": target_w.detach(),
        "source_aligned_edge_index": source_aligned_edge_index,
        "source_aligned_edge_weight": source_aligned_edge_weight,
        "target_aligned_edge_index": target_aligned_edge_index,
        "target_aligned_edge_weight": target_aligned_edge_weight,
        "history": history_rows,
    }
