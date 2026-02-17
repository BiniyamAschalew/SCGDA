# Problem Formulation: Synthetic MMD Domain Adaptation

We consider unsupervised domain adaptation with labeled source samples and unlabeled target samples.

- Source domain: \(\mathcal{D}_s=\{(x_i^s,y_i^s)\}_{i=1}^{n_s}\)
- Target domain: \(\mathcal{D}_t=\{x_j^t\}_{j=1}^{n_t}\)

The model is:

- Encoder \(E_\theta: \mathbb{R}^{d} \rightarrow \mathbb{R}^{p}\)
- Classifier \(C_\phi: \mathbb{R}^{p} \rightarrow \mathbb{R}^{K}\)

with latent features:

\[
z_i^s = E_\theta(x_i^s), \quad z_j^t = E_\theta(x_j^t)
\]

and source logits:

\[
\hat{y}_i^s = C_\phi(z_i^s)
\]

We optimize:

\[
\mathcal{L}(\theta,\phi)=\underbrace{\mathcal{L}_{cls}\big(C_\phi(E_\theta(x^s)),y^s\big)}_{\text{source cross-entropy}}
+\lambda\underbrace{\mathrm{MMD}^2\big(E_\theta(x^s),E_\theta(x^t)\big)}_{\text{distribution alignment}}
\]

where \(\lambda \ge 0\) balances discriminative source training and domain alignment.

## Synthetic setup

We generate class-conditional Gaussian source features and create target features with controlled shift:

- rotation-like linear transform
- global translation
- per-class offset
- covariance/scale change

This gives a known source-target mismatch while preserving class semantics for evaluation.

## What we analyze

1. Baseline (\(\lambda=0\)): source-only supervision.
2. MMD adaptation (\(\lambda>0\)): source CE + latent MMD.
3. Embedding evolution over epochs under a **fixed projection seed**, so the projection operator is constant across snapshots and movement reflects representation changes rather than projection randomness.
4. Conditional MMD diagnostics: class-wise latent alignment averaged across classes (using labels for analysis).
