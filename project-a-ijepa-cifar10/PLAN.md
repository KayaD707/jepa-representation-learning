# Project A — Plan: I-JEPA on CIFAR-10

Written before serious code, same discipline as the Cookie Cats pre-registration: commit
to the objective, the baseline, and the evaluation metric now, so success later can't be
quietly redefined to match whatever happened to come out of training.

## Objective

Implement a simplified Joint-Embedding Predictive Architecture (JEPA), following Assran
et al.'s I-JEPA (CVPR 2023), and test whether it learns image representations that are
genuinely useful for a downstream task — not just whether the training loss goes down.
"Useful" is defined concretely below, before training starts.

## Why CIFAR-10, why this architecture, why now

- CIFAR-10 (60,000 32×32 color images, 10 classes) is small enough to train on a laptop
  CPU in reasonable time, and has a long history as a benchmark, so there are known
  reference points for what a reasonable linear-probe accuracy looks like.
- I-JEPA specifically, over a full generative/contrastive approach, because it's the
  direct architectural idea behind the paper driving this whole second portfolio (LeCun's
  2022 position paper) — predicting in representation space, not pixel space, is the
  entire point of doing this project rather than a generic image classifier.
- This is Project A specifically because it has an established evaluation protocol
  (linear probing) to lean on — the goal this session is a defensible first result, not
  the most ambitious possible one.

## The method, in plain terms

1. Take an image, split it into patches.
2. Mask out a block of patches (the "target"). Keep the rest visible (the "context").
3. A **context encoder** turns the visible patches into representations.
4. A **target encoder** (a slowly-updated copy of the context encoder, not trained
   directly by gradient descent — this is a deliberate design choice in I-JEPA to avoid
   representational collapse) turns the masked patches into target representations.
5. A **predictor** takes the context representations and tries to predict the target
   representations — never the raw pixel values of the masked patches.
6. The loss is the distance between predicted and actual target representations.

## Baseline(s) — what "useful" is measured against

A representation is only interesting if it beats doing nothing clever:

1. **Random-init baseline:** run the same encoder architecture with random, untrained
   weights, extract "representations," and linear-probe those. This is the floor — if
   trained I-JEPA doesn't clearly beat this, something is broken.
2. **Plain autoencoder baseline:** an encoder-decoder trained to reconstruct raw pixels
   (conceptually close to the VAE work from Mindtrace, so this baseline is cheap to build
   on existing intuition), linear-probed the same way. This is the more interesting bar —
   it tests whether predicting in *representation* space specifically beats predicting in
   *pixel* space, which is the actual scientific claim behind JEPA, not just "beats
   nothing."

## Evaluation protocol: linear probing

- Freeze the trained encoder's weights entirely.
- Train a single linear classifier on top of the frozen representations, using CIFAR-10's
  labels, on a train split.
- Report accuracy on a held-out test split.
- Compare across: random-init baseline, autoencoder baseline, I-JEPA.
- This protocol is decided now, before any model is trained, specifically so the
  evaluation can't quietly shift to whichever metric flatters the result.

## What counts as success — stated before training

- **Minimum bar:** I-JEPA's linear-probe accuracy clearly exceeds the random-init
  baseline. (If it doesn't, the implementation has a real bug — this should be an easy
  bar to clear.)
- **The actual claim being tested:** I-JEPA's linear-probe accuracy exceeds the plain
  autoencoder baseline — i.e., predicting in representation space produces more useful
  features than predicting in pixel space, on this dataset, at this scale.
- **Being honest in advance:** at CIFAR-10 scale with a laptop CPU budget, it's genuinely
  possible the autoencoder baseline wins, or the two tie within noise. JEPA's advantages
  are best documented at larger scale/data than this project can afford. A negative or
  null result here is a real, reportable finding about where the architecture's
  advantages do and don't show up at small scale — not a failed project.

## Explicitly out of scope for this first pass

- Vision Transformer patch embeddings (the original I-JEPA architecture) — a small CNN
  encoder is used instead, to keep training tractable on CPU. This is a real
  simplification from the paper, stated here rather than glossed over.
- Any claim about ImageNet-scale performance — this is a small-scale, CPU-budget
  replication of the *idea*, not an attempt to reproduce the paper's actual numbers.
- GPU training — not available on this machine (confirmed: AMD/Intel integrated
  graphics, no NVIDIA/CUDA).

## Tooling

- **PyTorch** — deliberate choice, see the repo-level README for why.
- **Weights & Biases** — every run logged: architecture variant, hyperparameters, training
  loss curve, final linear-probe accuracy. No result claimed without a logged run backing
  it.

## Known risks, named upfront (being critical before being surprised)

- **Collapse risk:** JEPA-style architectures are specifically prone to representational
  collapse (the model learns to output the same constant representation regardless of
  input, trivially minimizing the loss) if the target-encoder update rule isn't right.
  This is a known failure mode in the literature, not a hypothetical — watching for it
  explicitly (e.g., checking representation variance, not just the loss curve) is part of
  the plan, not an afterthought if things go wrong.
- **CPU training time:** without a GPU, training budget is a real constraint on how large
  a model/how many epochs are feasible — this may force a smaller architecture than ideal,
  which is a legitimate limitation to state in the final write-up, not hide.
- **Small-scale null result:** as stated above, it's genuinely possible the autoencoder
  baseline wins at this scale. Deciding now that this would still be a valid, reportable
  outcome — not retroactively lowering the bar for "success" if it happens.

## Status

Plan written. Environment fully set up and verified (torch 2.2.2 + torchvision 0.17.2 in
an isolated `.venv` — see [`docs/SETUP_NOTES.md`](docs/SETUP_NOTES.md)). Data loading and
context/target masking (`data.py`) complete and visually verified — see
[`docs/PROGRESS_LOG.md`](docs/PROGRESS_LOG.md). Linear probe will be implemented directly
in PyTorch rather than with scikit-learn. Not yet built: the I-JEPA model itself (context
encoder, target encoder, predictor) and the training loop.
