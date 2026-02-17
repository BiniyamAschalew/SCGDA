# Synthetic MMD DA Analysis

This folder provides a minimal, reproducible analysis of how MMD-based domain adaptation aligns source and target feature distributions on synthetic data.

## Run

```bash
python -m prob_form.run_analysis --epochs 200 --lambda-mmd 1.0 --seed 7 --projection-seed 17
```

Optional:

```bash
python -m prob_form.run_analysis --output-dir prob_form/results/demo --snapshot-stride 5 --latent-dim 8
```

## Outputs

Each run saves:

- `synthetic_data.npz`: generated source/target features and labels
- `metrics.csv`: per-epoch metrics for `baseline` and `mmd`
- `snapshots/<regime>/epoch_XXXX.npz`: saved latent embeddings over time
- `raw_features_fixed_projection.png`: source/target raw-space view via fixed projection
- `embedding_evolution_fixed_projection.png`: latent evolution with fixed projection seed
- `training_curves.png`: loss/MMD/source-acc/target-acc curves
- `config.json`: run configuration

## Notes

- Target labels are only used for evaluation.
- `metrics.csv` includes both global MMD and class-conditional MMD (`conditional_raw_mmd`, `conditional_latent_mmd`).
- The projection for visualization is fixed once by seed and reused at every epoch to avoid changing geometry from random projection differences.
- Lower (conditional) MMD does not guarantee better target accuracy: alignment can reduce class separation and cause class mixing.
