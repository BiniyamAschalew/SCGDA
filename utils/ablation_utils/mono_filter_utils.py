"""Reusable utilities for monomial-filter alignment experiments (m02v, mo3v, etc.)."""

import torch
import torch.nn.functional as F
from torch import nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from models.__filters.mono import MonoProp
from utils.ablation_utils.common import sample_idx
from utils.ablation_utils.propagation import Propagation
from utils.expt_utils import set_seed
from utils.filter_utils import mmd_rbf, median_bandwidth
from utils.train_utils.mmd import Sinkhorn


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def cfg_get(cfg, key: str, default):
    """Read a key from an OmegaConf or dict config with a fallback default."""
    if hasattr(cfg, "get"):
        value = cfg.get(key, default)
        return default if value is None else value
    try:
        value = cfg[key]
    except Exception:
        return default
    return default if value is None else value


def resolve_device(device: str) -> str:
    return "cpu" if str(device).startswith("cuda") and not torch.cuda.is_available() else str(device)


# ---------------------------------------------------------------------------
# Label preprocessing
# ---------------------------------------------------------------------------

def remap_labels_contiguous(ys: torch.Tensor, yt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    classes = torch.unique(torch.cat([ys.view(-1), yt.view(-1)], dim=0), sorted=True)
    mapping = {int(c.item()): i for i, c in enumerate(classes)}

    def _remap(y):
        return torch.tensor([mapping[int(v.item())] for v in y.view(-1)], device=y.device, dtype=torch.long)

    return _remap(ys), _remap(yt)


def try_fit_joint_lda(xs, xt, ys, yt, lda_dim, lda_eps):
    n_classes = int(torch.unique(torch.cat([ys, yt], dim=0)).numel())
    out_dim = min(int(lda_dim), int(xs.size(1)), n_classes - 1)
    if out_dim <= 0:
        return None
    x = torch.cat([xs, xt], dim=0)
    y = torch.cat([ys, yt], dim=0)
    lda = LinearDiscriminantAnalysis(n_components=out_dim, tol=float(lda_eps))
    lda.fit(x.detach().cpu().numpy(), y.detach().cpu().numpy())
    w = torch.from_numpy(lda.scalings_).float().to(xs.device)
    return w[:, :out_dim]


def joint_pca_projection(
    source_x: torch.Tensor,
    target_x: torch.Tensor,
    out_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project source and target features via joint PCA (label-unaware)."""
    in_dim = int(source_x.size(1))
    if out_dim <= 0 or out_dim >= in_dim:
        return source_x, target_x
    combined = torch.cat([source_x.detach(), target_x.detach()], dim=0).float()
    mean = combined.mean(dim=0, keepdim=True)
    centered = combined - mean
    _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
    proj = Vt[:out_dim].T.to(device=source_x.device)  # (in_dim, out_dim)
    return (source_x.float() - mean.to(source_x.device)) @ proj, \
           (target_x.float() - mean.to(target_x.device)) @ proj


# ---------------------------------------------------------------------------
# MMD helpers
# ---------------------------------------------------------------------------

def parse_fix_sigma(fix_sigma):
    if isinstance(fix_sigma, str):
        fix = fix_sigma.strip().lower()
        if fix in ("none", "null", ""):
            return None
        return float(fix)
    return fix_sigma


def shared_bandwidth(a: torch.Tensor, b: torch.Tensor, cfg) -> float:
    fix_sigma = parse_fix_sigma(cfg_get(cfg, "fix_sigma", None))
    if fix_sigma is None:
        return median_bandwidth(a, b).item()
    return float(fix_sigma)


def mmd_value(a: torch.Tensor, b: torch.Tensor, cfg, fix_sigma=None) -> float:
    if fix_sigma is None:
        fix_sigma = parse_fix_sigma(cfg_get(cfg, "fix_sigma", None))
    value = mmd_rbf(
        a, b,
        kernel_mul=float(cfg_get(cfg, "kernel_mul", 2.0)),
        kernel_num=int(cfg_get(cfg, "kernel_num", 5)),
        fix_sigma=fix_sigma,
    )
    return float(value.detach().cpu().item())


def mmd_tensor(a: torch.Tensor, b: torch.Tensor, cfg) -> torch.Tensor:
    return mmd_rbf(
        a, b,
        kernel_mul=float(cfg_get(cfg, "kernel_mul", 2.0)),
        kernel_num=int(cfg_get(cfg, "kernel_num", 5)),
        fix_sigma=parse_fix_sigma(cfg_get(cfg, "fix_sigma", None)),
    )


def alignment_tensor(a: torch.Tensor, b: torch.Tensor, cfg) -> torch.Tensor:
    """Probe pushforward alignment loss (MMD or Sinkhorn)."""
    metric = str(cfg_get(cfg, "alignment_metric", "mmd")).lower()
    if metric == "sinkhorn":
        return Sinkhorn(
            a, b,
            sampling_num=max(1, int(cfg_get(cfg, "sinkhorn_sampling_num", cfg_get(cfg, "align_sample_size", 1024)))),
            times=max(1, int(cfg_get(cfg, "sinkhorn_times", 3))),
            blur=float(cfg_get(cfg, "sinkhorn_blur", 0.05)),
            p=int(cfg_get(cfg, "sinkhorn_p", 2)),
            scaling=float(cfg_get(cfg, "sinkhorn_scaling", 0.9)),
            debias=bool(cfg_get(cfg, "sinkhorn_debias", True)),
            backend=str(cfg_get(cfg, "sinkhorn_backend", "auto")),
        )
    return mmd_tensor(a, b, cfg)


def unit_normalize_rows(x: torch.Tensor, eps: float = 1e-8):
    norm = x.norm(p=2, dim=1, keepdim=True).clamp_min(float(eps))
    return x / norm, norm.view(-1)


def compute_mmd_decomposition(hs: torch.Tensor, ht: torch.Tensor, cfg) -> tuple[float, float, float]:
    """Returns (mmd_total, mmd_norm, mmd_angle)."""
    hs_cpu, ht_cpu = hs.detach().cpu(), ht.detach().cpu()
    sigma = shared_bandwidth(hs_cpu, ht_cpu, cfg)
    total = mmd_value(hs_cpu, ht_cpu, cfg, fix_sigma=sigma)
    s_unit, s_norm = unit_normalize_rows(hs_cpu)
    t_unit, t_norm = unit_normalize_rows(ht_cpu)
    norm_sigma = shared_bandwidth(s_norm.unsqueeze(1), t_norm.unsqueeze(1), cfg)
    norm_mmd = mmd_value(s_norm.unsqueeze(1), t_norm.unsqueeze(1), cfg, fix_sigma=norm_sigma)
    angle_sigma = shared_bandwidth(s_unit, t_unit, cfg)
    angle_mmd = mmd_value(s_unit, t_unit, cfg, fix_sigma=angle_sigma)
    return total, norm_mmd, angle_mmd


# ---------------------------------------------------------------------------
# Monomial filter primitives
# ---------------------------------------------------------------------------

def effective_filter(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    t = max(float(temperature), 1e-6)
    return torch.softmax(logits / t, dim=0)


def init_filter_logits(num_terms: int, init_mode: str, device: torch.device) -> torch.Tensor:
    mode = str(init_mode).lower()
    logits = torch.zeros(num_terms, device=device)
    if mode in {"adj", "adjacency", "a1"}:
        logits.fill_(-2.0)
        logits[1 if num_terms > 1 else 0] = 2.0
    elif mode in {"identity", "a0"}:
        logits.fill_(-2.0)
        logits[0] = 2.0
    return logits


def symmetric_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return 0.5 * ((p * (p.log() - q.log())).sum() + (q * (q.log() - p.log())).sum())


def apply_monomial_once(
    mono_prop: MonoProp,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    coeff: torch.Tensor,
    lambda_max: float,
) -> torch.Tensor:
    return mono_prop(x, edge_index, parameters=coeff, edge_weight=None, lambda_max=float(lambda_max))


def propagate_monomial_layers(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    coeff: torch.Tensor,
    max_layers: int,
    lambda_max: float,
    mono_prop: MonoProp,
) -> list[torch.Tensor]:
    layers = [x]
    cur = x
    for _ in range(int(max_layers)):
        cur = apply_monomial_once(mono_prop, cur, edge_index, coeff, lambda_max)
        layers.append(cur)
    return layers


def propagate_adjacency_layers(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    max_layers: int,
    prop: Propagation,
) -> list[torch.Tensor]:
    layers = [x]
    cur = x
    for _ in range(int(max_layers)):
        cur = prop(cur, edge_index)
        layers.append(cur)
    return layers


# ---------------------------------------------------------------------------
# Learnable probe distribution
# ---------------------------------------------------------------------------

class LearnableProbeDistribution(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(1, feature_dim))
        self.log_scale = nn.Parameter(torch.zeros(1, feature_dim))

    def sample(self, base: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.log_scale) + 1e-4
        return base + self.shift + scale * torch.randn_like(base)

    def regularizer(self) -> torch.Tensor:
        return self.shift.pow(2).mean() + self.log_scale.pow(2).mean()


def probe_stats(source_x: torch.Tensor, target_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    all_feat = torch.cat([source_x.detach(), target_x.detach()], dim=0)
    return all_feat.mean(dim=0, keepdim=True), all_feat.std(dim=0, keepdim=True).clamp_min(1e-6)


def sample_gaussian_probe(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return torch.randn_like(x) * std + mean


# ---------------------------------------------------------------------------
# Monomial aligner training
# ---------------------------------------------------------------------------

def train_monomial_aligner(
    *,
    source_data,
    target_data,
    source_x: torch.Tensor,
    target_x: torch.Tensor,
    source_y: torch.Tensor,
    target_y: torch.Tensor,
    source_train_mask: torch.Tensor,
    cfg,
    adversarial: bool,
    seed_offset: int,
) -> dict:
    """Train monomial filter coefficients to align source/target propagation.

    Returns dict with keys: regime, source_coeff, target_coeff, history.
    """
    set_seed(int(cfg_get(cfg, "seed", 0)) + int(seed_offset))
    device = source_x.device
    num_terms = int(cfg_get(cfg, "mono_degree", 3)) + 1
    num_classes = int(torch.max(torch.cat([source_y, target_y], dim=0)).item()) + 1

    mono_prop = MonoProp().to(device)
    source_logits = nn.Parameter(init_filter_logits(num_terms, cfg_get(cfg, "mono_init", "uniform"), device))
    target_logits = nn.Parameter(init_filter_logits(num_terms, cfg_get(cfg, "mono_init", "uniform"), device))
    semantic_head = nn.Linear(source_x.size(1), num_classes).to(device)

    filter_opt = torch.optim.Adam(
        [source_logits, target_logits] + list(semantic_head.parameters()),
        lr=float(cfg_get(cfg, "align_lr", 1e-2)),
        weight_decay=float(cfg_get(cfg, "align_weight_decay", 0.0)),
    )

    source_dist = target_dist = adv_opt = None
    if adversarial:
        source_dist = LearnableProbeDistribution(int(source_x.size(1))).to(device)
        target_dist = LearnableProbeDistribution(int(target_x.size(1))).to(device)
        adv_opt = torch.optim.Adam(
            list(source_dist.parameters()) + list(target_dist.parameters()),
            lr=float(cfg_get(cfg, "adv_lr", 3e-3)),
            weight_decay=float(cfg_get(cfg, "adv_weight_decay", 0.0)),
        )

    p_mean, p_std = probe_stats(source_x, target_x)

    epochs = int(cfg_get(cfg, "align_epochs", 120))
    log_interval = max(1, int(cfg_get(cfg, "align_log_interval", 10)))
    sample_size = int(cfg_get(cfg, "align_sample_size", 1024))
    temp = float(cfg_get(cfg, "mono_filter_temperature", 1.0))
    lmax = float(cfg_get(cfg, "mono_lambda_max", 2.0))
    w_probe = float(cfg_get(cfg, "probe_mmd_weight", 1.0))
    w_sem = float(cfg_get(cfg, "source_semantic_weight", 1.0))
    w_filt = float(cfg_get(cfg, "filter_alignment_weight", 0.1))
    adv_reg_w = float(cfg_get(cfg, "adv_distribution_reg_weight", 1e-3))
    adv_steps = max(1, int(cfg_get(cfg, "adv_steps", 1)))

    history = []
    for epoch in range(1, epochs + 1):
        # adversarial step
        if adversarial and adv_opt is not None:
            for _ in range(adv_steps):
                adv_opt.zero_grad()
                with torch.no_grad():
                    sc = effective_filter(source_logits, temp)
                    tc = effective_filter(target_logits, temp)
                sp = source_dist.sample(sample_gaussian_probe(source_x, p_mean, p_std))
                tp = target_dist.sample(sample_gaussian_probe(target_x, p_mean, p_std))
                sp_push = apply_monomial_once(mono_prop, sp, source_data.edge_index, sc, lmax)
                tp_push = apply_monomial_once(mono_prop, tp, target_data.edge_index, tc, lmax)
                ids = sample_idx(sp_push.size(0), sample_size, device)
                idt = sample_idx(tp_push.size(0), sample_size, device)
                adv_mmd = alignment_tensor(sp_push[ids], tp_push[idt], cfg)
                adv_reg = adv_reg_w * (source_dist.regularizer() + target_dist.regularizer())
                (-adv_mmd + adv_reg).backward()
                adv_opt.step()

        # filter step
        filter_opt.zero_grad()
        sc = effective_filter(source_logits, temp)
        tc = effective_filter(target_logits, temp)

        sp_base = sample_gaussian_probe(source_x, p_mean, p_std)
        tp_base = sample_gaussian_probe(target_x, p_mean, p_std)
        if adversarial and source_dist is not None:
            with torch.no_grad():
                sp_base = source_dist.sample(sp_base)
                tp_base = target_dist.sample(tp_base)

        sp_push = apply_monomial_once(mono_prop, sp_base, source_data.edge_index, sc, lmax)
        tp_push = apply_monomial_once(mono_prop, tp_base, target_data.edge_index, tc, lmax)
        ids = sample_idx(sp_push.size(0), sample_size, device)
        idt = sample_idx(tp_push.size(0), sample_size, device)
        probe_mmd = alignment_tensor(sp_push[ids], tp_push[idt], cfg)

        sr = apply_monomial_once(mono_prop, source_x, source_data.edge_index, sc, lmax)
        tr = apply_monomial_once(mono_prop, target_x, target_data.edge_index, tc, lmax)
        sem_loss = F.cross_entropy(semantic_head(sr)[source_train_mask], source_y[source_train_mask])
        filt_align = symmetric_kl(sc, tc)

        loss = w_probe * probe_mmd + w_sem * sem_loss + w_filt * filt_align
        loss.backward()
        filter_opt.step()

        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            with torch.no_grad():
                sc_now = effective_filter(source_logits, temp)
                tc_now = effective_filter(target_logits, temp)
                sr_now = apply_monomial_once(mono_prop, source_x, source_data.edge_index, sc_now, lmax)
                tr_now = apply_monomial_once(mono_prop, target_x, target_data.edge_index, tc_now, lmax)
                ms = int(cfg_get(cfg, "metric_sample_size", 0))
                mmd_t, mmd_n, mmd_a = compute_mmd_decomposition(
                    sr_now[sample_idx(sr_now.size(0), ms, device)],
                    tr_now[sample_idx(tr_now.size(0), ms, device)],
                    cfg,
                )
                history.append({
                    "epoch": epoch,
                    "probe_mmd": float(probe_mmd.detach().cpu().item()),
                    "semantic_loss": float(sem_loss.detach().cpu().item()),
                    "real_mmd2_total": mmd_t,
                    "real_mmd2_norm": mmd_n,
                    "real_mmd2_angle": mmd_a,
                })

    with torch.no_grad():
        final_sc = effective_filter(source_logits, temp).detach()
        final_tc = effective_filter(target_logits, temp).detach()

    return {
        "regime": "mono_adv_aligned" if adversarial else "mono_aligned",
        "source_coeff": final_sc,
        "target_coeff": final_tc,
        "history": history,
    }
