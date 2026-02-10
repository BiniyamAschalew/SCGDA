import math

import torch
import torch.nn.functional as F

from models.baselines.dgsda.dgsda import DGSDA
from utils.train_utils.mmd import MMD


# ---------------------------------------------------------------------------
# Self-contained helpers (no external dependency on structural_utils)
# ---------------------------------------------------------------------------

def _mmd_multi_kernel(x, y, sigmas):
    """Multi-kernel MMD between two sets of row vectors."""
    xx = torch.cdist(x, x).pow(2)
    yy = torch.cdist(y, y).pow(2)
    xy = torch.cdist(x, y).pow(2)
    total = 0.0
    for sigma in sigmas:
        g = 2.0 * sigma * sigma
        total = total + (torch.exp(-xx / g).mean()
                         + torch.exp(-yy / g).mean()
                         - 2.0 * torch.exp(-xy / g).mean())
    return total / len(sigmas)


def _normalize_embeddings(h, eps=1e-8):
    """Frobenius-normalize: h / (||h||_F / sqrt(n*d))."""
    scale = h.norm(p="fro") / math.sqrt(max(1, h.size(0) * h.size(1)))
    return h / (scale + eps)


def _sample_rows(h, m):
    """Randomly sample m rows from h for size-agnostic comparison."""
    if h.size(0) <= m:
        return h
    idx = torch.randperm(h.size(0), device=h.device)[:m]
    return h[idx]


def _build_edge_pool(edge_index, num_nodes):
    """
    Extract all positive edges (from edge_index) and sample-ready
    negative edges (not in edge_index) using upper-triangle indexing.

    Returns (pos_edges, neg_edges) each of shape (E, 2).
    """
    device = edge_index.device
    adj_set = set()
    ei = edge_index.cpu().numpy()
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        if i < j:
            adj_set.add((i, j))
        elif j < i:
            adj_set.add((j, i))

    pos = torch.tensor(list(adj_set), dtype=torch.long, device=device)

    # Sample negative edges (pairs not in adj_set)
    neg_list = []
    max_neg = pos.size(0) * 2  # collect enough negatives
    rng = torch.randint(0, num_nodes, (max_neg * 2, 2), device=device)
    for k in range(rng.size(0)):
        i, j = int(rng[k, 0]), int(rng[k, 1])
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key not in adj_set:
            neg_list.append(key)
            if len(neg_list) >= max_neg:
                break

    neg = torch.tensor(neg_list, dtype=torch.long, device=device)
    return pos, neg


def _sample_edges(pos_pool, neg_pool, num_samples, device):
    """Sample num_samples positive and negative edges from pools."""
    pos_idx = torch.randint(pos_pool.size(0), (num_samples,), device=device)
    neg_idx = torch.randint(neg_pool.size(0), (num_samples,), device=device)
    return pos_pool[pos_idx], neg_pool[neg_idx]


def _bpr_loss(emb, pos_edges, neg_edges):
    """BPR ranking loss: -logsigmoid(pos_score - neg_score)."""
    pos_scores = (emb[pos_edges[:, 0]] * emb[pos_edges[:, 1]]).sum(dim=1)
    neg_scores = (emb[neg_edges[:, 0]] * emb[neg_edges[:, 1]]).sum(dim=1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


# ---------------------------------------------------------------------------
# DGSDAProbe: DGSDA + probe-output MMD + BPR + energy
# ---------------------------------------------------------------------------

class DGSDAProbe(DGSDA):

    def __init__(self, config: dict):
        super(DGSDAProbe, self).__init__(config)

        self.alpha_probe = config["model"].get("alpha_probe", 0.05)
        self.alpha_bpr = config["model"].get("alpha_bpr", 0.1)
        self.alpha_energy = config["model"].get("alpha_energy", 0.25)
        self.probe_dim = config["model"].get("probe_dim", 32)
        self.mmd_sigmas = config["model"].get(
            "mmd_sigmas", [0.2, 0.5, 1.0, 2.0, 5.0])
        self.edge_samples = config["model"].get("edge_samples", 3000)

        # Edge pools built lazily in fit()
        self._tgt_pos_pool = None
        self._tgt_neg_pool = None

    def fit(self, source_data, target_data):
        # Pre-build edge pool for target graph (done once before training)
        tgt_ei = target_data.edge_index
        n_tgt = target_data.x.size(0)
        self._tgt_pos_pool, self._tgt_neg_pool = _build_edge_pool(tgt_ei, n_tgt)
        super().fit(source_data, target_data)

    def forward_model(self, source_data, target_data):

        # ── shared: source CE ──
        source_logits = self.dgsda(source_data)
        loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)

        # ── theta alignment (L1 only, no sparsity) ──
        # theta_s = self.dgsda.prop1.temp
        # theta_t = self.dgsda.prop2.temp
        # theta_align = F.l1_loss(theta_s, theta_t)
        # loss = loss + self.alpha * theta_align

        # ── probe-output MMD through Bernstein filters ──
        n_s = source_data.num_nodes
        n_t = target_data.num_nodes
        device = source_data.x.device

        X_s = 1.0 + torch.randn(n_s, self.probe_dim, device=device)
        X_t = 1.0 + torch.randn(n_t, self.probe_dim, device=device)

        out_s = self.dgsda.prop1(X_s, source_data.edge_index)
        out_t = self.dgsda.prop2(X_t, target_data.edge_index)

        m = min(n_s, n_t)
        probe_mmd = _mmd_multi_kernel(
            _normalize_embeddings(_sample_rows(out_s, m)),
            _normalize_embeddings(_sample_rows(out_t, m)),
            self.mmd_sigmas,
        )
        loss = loss + self.alpha_probe * probe_mmd

        # ── BPR structural loss on target filter output ──
        if self._tgt_pos_pool is not None and self._tgt_pos_pool.size(0) > 0:
            pos_e, neg_e = _sample_edges(
                self._tgt_pos_pool, self._tgt_neg_pool,
                self.edge_samples, device)
            bpr = _bpr_loss(out_t, pos_e, neg_e)
            loss = loss + self.alpha_bpr * bpr

        # ── energy regularisation ──
        energy = out_t.norm(p="fro") / math.sqrt(
            max(1, out_t.size(0) * out_t.size(1)))
        energy_loss = (energy - 1.0) ** 2
        loss = loss + self.alpha_energy * energy_loss

        # ── shared: feature MMD ──
        source_feature = F.relu(self.dgsda.lin1(source_data.x))
        target_feature = F.relu(self.dgsda.lin1(target_data.x))
        mmd_loss = MMD(source_feature, target_feature)
        loss = loss + self.beta * mmd_loss

        # ── shared: entropy minimisation on target ──
        target_outputs = self.dgsda(target_data, False)
        entropy_loss = self.entropy_minimization_loss(target_outputs)
        loss = loss + self.gamma * entropy_loss

        return loss, source_logits
