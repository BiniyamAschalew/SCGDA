# Triplet Bound Experiment

Run with the project environment:

```bash
conda run -n pygda python -m experiments.analysis.airport_triplet_bound.run --dataset airport
```

Benchmark-style launcher with editable constants at the top of
[`experiment.py`](/home/bini/projects/GDA/Clean_SCGDA/experiments/analysis/airport_triplet_bound/experiment.py):

```bash
conda run -n pygda python -m experiments.analysis.airport_triplet_bound.experiment
```

Switching to another supported 3-domain dataset is one argument:

```bash
conda run -n pygda python -m experiments.analysis.airport_triplet_bound.run --dataset citation
```

Useful options:

```bash
conda run -n pygda python -m experiments.analysis.airport_triplet_bound.run \
  --dataset citation \
  --seeds 0 1 2 \
  --epochs 200 \
  --device cuda:0
```

Fast reduced run:

```bash
conda run -n pygda python -m experiments.analysis.airport_triplet_bound.run \
  --dataset citation \
  --seeds 0 1 2 \
  --epochs 20 \
  --grad-nodes-cap 8 \
  --device cpu \
  --run-name citation_quick
```

The experiment expects a dataset config with exactly 3 domains. For each ordered triplet it trains:

- `source_only`: source train-mask labels only
- `source_mmd_target`: source train-mask labels plus source-target MMD alignment
- `oracle`: source and target train-mask labels

Final metrics are evaluated on `val_mask`. The outputs include:

- source/target/reference micro/macro-F1
- pairwise validation MMDs
- source-train vs source-val and source-train vs target-val shift MMDs
- gradient-based sensitivity proxies
- oracle target performance and bound-style proxy terms
- raw per-seed rows, aggregated summaries, component correlations, and plots

By default the runner resamples the train/val masks per seed and automatically loads pair-specific tuned `SimGDA` hyperparameters from `__hps__/prev_tuned/simgda/<dataset>/<source>_<target>/best.yaml` when available. Use `--reuse-saved-masks` if you want to keep the processed split unchanged.
