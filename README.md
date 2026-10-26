# JEPA & Self-Supervised Representation Learning Portfolio

## Why this exists

Second entry in a data science/AI portfolio, after the
[Cookie Cats A/B test project](https://github.com/KayaD707/cookiecats-ab-test-analysis).
Where that project demonstrated statistical rigor (causal inference, experimental design),
this one demonstrates the other half of an AI Researcher / Data Scientist skill set:
actually building and evaluating deep learning models — specifically in the area of
self-supervised representation learning and world models, directly connected to ongoing
study of Yann LeCun's 2022 position paper "A Path Towards Autonomous Machine Intelligence."

**The through-line across all three projects:** comparing things in *representation
space*, built from raw sensory input (images, sequences), rather than comparing in
language/token space the way LLMs do. This is a deliberate choice of direction, not the
default "build a chatbot" portfolio piece.

## The three projects, and how they build on each other

### Project A — I-JEPA on a small image dataset (starting point)

Implements a simplified Joint-Embedding Predictive Architecture (Assran et al., CVPR
2023) on CIFAR-10: mask part of an image, encode the visible context, predict the
*representation* (never the raw pixels) of the masked region. Evaluated via linear
probing against a baseline. Chosen as the starting point because it has an established,
standard evaluation protocol — the safest way to land a first real, defensible result
before attempting anything more exploratory. **Model, training loop, and both baselines
(random-init, autoencoder) built; pretraining run in progress — see
[`project-a-ijepa-cifar10/PLAN.md`](project-a-ijepa-cifar10/PLAN.md).**

### Project B — Contrastive vs. non-contrastive self-supervised learning, compared

Trains a contrastive method (SimCLR-style) and a non-contrastive, JEPA/BYOL-style method
on the same dataset, and compares the representations each learns. Connects to Jean
Ponce's (ENS Paris / PRAIRIE Institute) 2026 ICLR paper on non-contrastive SSL theory.
Builds directly on Project A's encoder (imported directly) and Project D's VICReg loss
(used as the non-contrastive method, rather than reimplementing BYOL) — genuine reuse,
not starting from zero. **New NT-Xent contrastive loss implemented, both training
scripts and shared evaluation written; training run queued — see
[`project-b-contrastive-vs-noncontrastive/PLAN.md`](project-b-contrastive-vs-noncontrastive/PLAN.md).**

### Project C — Predictive coding on toy sensory sequences (most exploratory)

A small recurrent JEPA-style model predicting future *representations* from past frames
of a simple synthetic scene — directly implementing the idea from Mohammadi, Halvagal &
Zenke's "Understanding cortical computation through the lens of joint-embedding
predictive architectures" (bioRxiv, 2025). The closest project on this list to the
specific "comparison at the sensory level" research direction. No standard benchmark
exists for this, so the evaluation protocol was designed from scratch: a bouncing-ball
physics simulator (free, exact ground-truth position) and two honest baselines
(last-position-repeated, constant-velocity extrapolation), with a specific eye on
whether the model beats constant-velocity *at bounces*, where it's known to fail.
**Simulator, recurrent model, training, and evaluation all implemented and
smoke-tested; full run queued — see
[`project-c-predictive-coding-sensory/PLAN.md`](project-c-predictive-coding-sensory/PLAN.md).**

### Project D — VICReg and the collapse problem (mechanism deep-dive)

A smaller, more focused project born from a specific question about LeCun's citation
on world-model training: *how do the two encoder functions gx/gy actually get learned
without just collapsing to a constant?* Implements VICReg (Bardes, Ponce & LeCun,
2022) — the cleanest available answer — as three separate, switchable loss terms, and
runs a 3-way ablation (invariance-only / variance-only / full VICReg) on a toy 2D
dataset where the resulting representations can be plotted directly. Confirms, visibly
and numerically, both known failure modes: full collapse (invariance-only) and the
subtler redundant-dimensions collapse (variance-only, no covariance term). **Toy demo
complete; a second stage on real CIFAR-10 images (two augmented crops as the two views,
same three-condition ablation, evaluated with Project A's linear-probing protocol) is
running — see
[`project-d-vicreg-collapse/PLAN.md`](project-d-vicreg-collapse/PLAN.md) and
[`docs/PROGRESS_LOG.md`](project-d-vicreg-collapse/docs/PROGRESS_LOG.md).**

## Shared practices across all three

Same discipline as the Cookie Cats project, adapted from statistics to ML experiments:

- A `PLAN.md` in each project folder, written before serious code, stating the
  objective, the baseline, the evaluation metric, and what would count as success or
  failure — pre-registration discipline, applied to ML instead of a hypothesis test.
- Every training run tracked in **Weights & Biases** — no "it worked" claims without a
  logged comparison against a stated baseline.
- Code is heavily commented for learning, not just correctness — every non-obvious line
  explains *why*, not just what.
- A `docs/` folder per project holding the reasoning log: design decisions, dead ends,
  and critical self-assessment — not just a polished final result.
- One phase/session at a time, pushed to GitHub as it goes — same gradual pace as the
  Cookie Cats project.
- PyTorch, deliberately — the field-standard framework for this type of work, and a
  confirmed skill gap being closed on purpose through this portfolio.

## Status

| Project | Status |
|---|---|
| A — I-JEPA on CIFAR-10 | Code complete; pretraining run in progress |
| B — Contrastive vs. non-contrastive | Code complete; training run queued |
| C — Predictive coding on sensory sequences | Code complete, smoke-tested; full run queued |
| D — VICReg and the collapse problem | Toy demo complete; CIFAR-10 stage running |
