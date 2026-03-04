# Agent Memory: Clean_SCGDA

Last updated: 2026-03-04

## Goal Context
- Repository focus: graph domain adaptation under structural shift.
- Main research direction: spectral/filter alignment between source and target to reduce shift and improve cross-domain generalization.
- Current style baseline: PyGDA-like training/evaluation flow with local extensions for filter/operator alignment.

## Reuse-First Rules (Persistent)
- Prefer modular functions over inline experiment-specific blocks.
- Before adding new logic, check `utils/`, `models/__components/`, and `models/__layers/` for reusable pieces.
- Reuse existing utilities for config loading, seeding, MMD/cMMD, data-pair loading, propagation, and reporting.
- If a new helper is needed by multiple files, move it into the closest existing utility module instead of duplicating.
- Keep experiment scripts thin; put reusable logic in utility modules.

## Codebase Map (High-Level)
- `run.py`: single-run pipeline (`build_dataset` -> `build_model` -> `fit` -> `predict` -> metrics).
- `experiment.py`, `benchmark.py`, `hp_tune.py`, `filter_experiment.py`: orchestration wrappers for repeated runs/tuning/evaluation.
- `data/`: dataset loading and transforms (`build_dataset`, domain loaders, SVD transform hooks).
- `models/build_model.py`: registry for baselines, imported models, and local models.
- `models/base_model.py`: shared GDA training scaffolding (metrics, wandb, early-stop/logging helpers).
- `models/ours/`: local methods (SCGDA, FiltADA, SimGDA variants, DLIT, StructAlign2, etc.).
- `models/__components/`: reusable propagation/filter blocks (Cheb/Bern/Mixup/Reweight).
- `models/__layers/`: reusable GNN layers and helpers (`build_layer`, `build_activation`, `FilterGCNConv`, `PPMIConv`, `GradReverse`).
- `utils/ablation_utils/`: end-to-end structure-shift ablation pipeline (alignment, propagation, metrics, transferability, reporting, defaults).
- `experiments/analysis/`: analysis scripts around spectral/filter behavior.
- `experiments/motivation/prob_form/`: synthetic problem-formulation analysis.
- `struct_shift/`: standalone synthetic structural-shift sandbox.

## Most Reused Utility APIs
Import-frequency index across the codebase (AST import scan, 2026-03-04).

### Core `utils` symbols (highest reuse)
- `utils.config_utils.build_config` (17)
- `utils.train_utils.mmd.MMD` (14)
- `utils.expt_utils.set_seed` (12)
- `utils.filter_utils.make_gaussian_probe` (7)
- `utils.filter_utils.mmd_rbf` (6)
- `utils.ablation_utils.common.load_pair` (5)
- `utils.filter_utils.monomial_to_cheb` (5)
- `utils.config_utils.load_config` (5)
- `utils.expt_utils.to_valid_dir` (5)
- `utils.ablation_utils.common.save_table` (4)
- `utils.filter_utils.cheb_to_monomial` (4)
- `utils.train_utils.metrics.BaseMetric` (3)
- `utils.filter_utils.tensor_to_float_list` (3)
- `utils.filter_utils.conditional_mmd` (3)
- `utils.ablation_utils.common.sample_idx` (3)
- `utils.ablation_utils.common.as_float` (3)

### Frequently reused utility modules
- `utils.config_utils`
- `utils.expt_utils`
- `utils.filter_utils`
- `utils.train_utils.mmd`
- `utils.ablation_utils.common`
- `utils.ablation_utils.alignment`
- `utils.ablation_utils.propagation`

## Most Reused Components and Layers
Import-frequency index across the codebase (AST import scan, 2026-03-04).

### Top symbols
- `models.__layers.build_layer.build_activation` (15)
- `models.__layers.reverse_layer.GradReverse` (9)
- `models.__components.chebprop.ChebProp` (9)
- `models.__layers.build_layer.build_layer` (6)
- `models.__layers.filter_gcn_conv.FilterGCNConv` (5)
- `models.__layers.ppmi_conv.PPMIConv` (5)
- `models.__components.bernprop.BernProp` (4)
- `models.__layers.prop_gcn_conv.PropGCNConv` (2)
- `models.__layers.cached_gcn_conv.CachedGCNConv` (2)
- `models.__layers.attention.Attention` (2)
- `models.__components.chebnet.ChebNetBase` (2)
- `models.__components.bernnet.BernNetBase` (2)

### Reusable module hotspots
- `models.__layers.build_layer`
- `models.__layers.reverse_layer`
- `models.__components.chebprop`
- `models.__layers.filter_gcn_conv`
- `models.__layers.ppmi_conv`

## Implementation Checklist (Before Writing New Code)
- Search for equivalent utility first: `rg "def <name>|class <name>" utils models/__components models/__layers`.
- Reuse existing pair loading/sampling/metrics paths before adding new local helpers.
- For new spectral-alignment experiments, prefer these existing building blocks:
- `make_gaussian_probe`, `mmd_rbf`, `conditional_mmd`
- `load_pair`, `sample_idx`, `save_table`
- `ChebProp`, `BernProp`, `FilterGCNConv`, `PPMIConv`
- `build_activation`, `build_layer`, `GradReverse`

## Maintenance Rule for This File
- When adding a new reusable utility/component used in 3+ files, add it to the relevant list above.
- Refresh counts periodically (or before major refactors) using the command below.

```bash
python - <<'PY'
import ast, os
from collections import Counter

util_sym, comp_sym = Counter(), Counter()
for dp, _, files in os.walk('.'):
    if '/.git' in dp:
        continue
    for fn in files:
        if not fn.endswith('.py'):
            continue
        p = os.path.join(dp, fn)
        try:
            t = ast.parse(open(p, encoding='utf-8').read(), filename=p)
        except Exception:
            continue
        for n in ast.walk(t):
            if isinstance(n, ast.ImportFrom) and n.module:
                for a in n.names:
                    if a.name == '*':
                        continue
                    key = f"{n.module}:{a.name}"
                    if n.module.startswith('utils.'):
                        util_sym[key] += 1
                    if n.module.startswith('models.__components.') or n.module.startswith('models.__layers.'):
                        comp_sym[key] += 1
print("Top utils:", util_sym.most_common(20))
print("Top components/layers:", comp_sym.most_common(20))
PY
```
