# Project D — Plan: VICReg and the Collapse Problem

Written before serious code, same discipline as Projects A/B/C: state the objective, the
baseline, and what would count as success or failure, before training anything.

## Where this project comes from

Directly from this passage in LeCun's 2022 position paper, about training a predictive
world model for planning:

> given two inputs x and y, learn two functions that compute representations
> sx = gx(x) and sy = gy(y) such that (1) sx and sy are maximally informative about x and
> y and (2) sy can easily be predicted from sx.

Project A already builds *one* way of implementing something like this (I-JEPA: a
predictor network, plus an EMA-updated target encoder, specifically to avoid collapse).
But the question that prompted this project was more basic than "does I-JEPA work" — it
was **how do gx and gy actually get learned, mechanically, so that they don't just both
collapse to outputting a constant?** That question deserves its own small, transparent
project rather than being answered only implicitly inside a bigger architecture.

## Why VICReg specifically

**VICReg** (Bardes, Ponce & LeCun, 2022, "Variance-Invariance-Covariance
Regularization") is the cleanest available implementation of exactly the two-part
principle quoted above, with no extra machinery (no EMA target network, no predictor
network, no stop-gradient tricks) — which makes it the right tool to isolate and *see*
the mechanism, rather than the right tool to get the best representations. Each term in
its loss maps onto one clause of the citation:

| Citation clause | VICReg loss term | What it mechanically does |
|---|---|---|
| "(1) sx and sy are maximally informative about x and y" | **Variance term** | Pushes the standard deviation of each representation dimension, across a batch, above a threshold — if every dimension stays "spread out," the representation can't be collapsing to a single constant point. |
| "(1) ... informative ..." (the other half) | **Covariance term** | Pushes the off-diagonal covariances between different representation dimensions toward zero — if dimensions aren't just copies/rescalings of each other, the representation can't be collapsing onto a lower-dimensional redundant subspace either. |
| "(2) sy can easily be predicted from sx" | **Invariance term** | Simple mean-squared error between sx and sy — the only term that actually pulls the two representations *together*. |

The reason collapse is even a risk: term (2) alone, with nothing else, is trivially
minimized by making gx and gy both output the same constant for every input — sx and sy
would then be "perfectly" predictable from each other (distance zero), for free, without
learning anything about x or y at all. Terms (1a) and (1b) exist specifically to make
that trivial solution *not* minimize the loss. This is the whole mechanism, and this
project's job is to make it visible, not just assert it.

## The method, in plain terms

1. Take an input and produce two different "views" of it, x and y (for the toy demo:
   two independently-noised observations of the same hidden 2D factor; later, for the
   image version: two differently-augmented crops of the same CIFAR-10 image).
2. Two encoder networks, gx and gy, map x and y to representations sx and sy. (Simplest
   version: gx and gy share weights — a "Siamese" network — which is the common
   practical simplification and what VICReg's own paper does; kept as one network called
   `encoder` in code, applied twice, rather than two separately-trained networks, and
   this simplification is stated here rather than silently assumed.)
3. Compute three loss terms — invariance, variance, covariance — as described above.
4. Backpropagate the combined loss into gx/gy directly. (No EMA copy, no stop-gradient,
   no separate predictor network — this is what makes VICReg simpler than I-JEPA and
   therefore clearer for this specific question.)

## Experiment design — the actual test

The real content of this project is an **ablation**, not a single training run. Train
the same encoder on the same data under three loss conditions, and look at what happens
to the learned representations in each case:

1. **Invariance-only** (no variance or covariance term): the predicted failure mode —
   representations should collapse toward a constant (or near-constant) point.
2. **Variance-only, no covariance** (prevents constant collapse, but not redundancy):
   representations should stay spread out, but may collapse onto a smaller effective
   subspace than the data actually needs (dimensions becoming near-copies of each
   other).
3. **Full VICReg** (all three terms): representations should stay spread out *and*
   non-redundant *and* predictive of each other — i.e. actually informative.

## Baseline(s)

This project's "baseline" is condition 1 above (invariance-only) — it is not a strawman,
it is the literal trivial solution the other two terms are specifically designed to rule
out. The interesting comparison is baseline (1) vs. full VICReg (3), with condition (2)
as a diagnostic middle step showing that variance alone isn't sufficient either.

## Evaluation protocol

Two levels, cheap first, then more realistic:

- **Toy synthetic data (first):** 2D representations, so they can be plotted directly as
  a scatter plot. Collapse is visible by eye: condition (1) should visibly bunch into a
  tiny cluster; condition (3) should visibly spread out and separate the underlying
  hidden factor. Also report representation variance and the off-diagonal covariance
  magnitude numerically, not just visually, for all three conditions.
- **CIFAR-10 (second, if the toy result is clean):** same three conditions, higher-
  dimensional representations, evaluated via the same linear-probing protocol as Project
  A, reusing its `data.py` loading utilities. This connects the toy mechanism back to a
  real result comparable to Project A's.

## What counts as success — stated before running anything

- **Minimum bar:** condition (1) visibly/numerically collapses (representation variance
  drops sharply toward zero) and condition (3) does not. If invariance-only doesn't
  collapse on this toy setup, something about the toy data or encoder is accidentally
  preventing it (e.g. the encoder architecture itself can't represent a constant
  function), and that needs to be understood before trusting any other result.
- **The actual claim being tested:** full VICReg (3) produces representations that are
  simultaneously high-variance, low-redundancy, AND mutually predictive — i.e., all
  three loss terms are doing real, necessary work, not just the invariance term.
- **Being honest in advance:** a toy 2D synthetic demo is a deliberately easy setting.
  It is meant to make the *mechanism* visible, not to be a rigorous benchmark result —
  that's what the CIFAR-10 follow-up step is for, if this project reaches it.

## Explicitly out of scope for this project

- Re-implementing I-JEPA's EMA/predictor mechanism for comparison — that's Project A,
  already in progress, and uses a different (also valid) anti-collapse strategy.
- Any claim that VICReg is "better than" I-JEPA — they solve the same problem
  differently; this project exists to understand one mechanism clearly, not to rank them.
- Training a model intended to be used downstream for anything — purely a mechanism
  demonstration.

## Tooling

- **PyTorch**, reusing the already-working `torch==2.2.2` / `torchvision==0.17.2`
  environment set up for Project A (`project-a-ijepa-cifar10/.venv`) rather than
  re-creating a new isolated venv from scratch — the dependency set is the same
  (torch, torchvision, numpy<2, matplotlib), and that environment is already confirmed
  working on this machine (see Project A's `SETUP_NOTES.md`). No new environment-setup
  risk taken on here.
- **Weights & Biases** deferred until the CIFAR-10 stage — for the toy demo, a single
  saved scatter plot plus printed numbers is more direct and easier to read than a W&B
  dashboard for 2D data.

## Known risks, named upfront

- **The toy demo might not show clean collapse** if the toy data/encoder is too simple
  (e.g. a linear encoder on Gaussian data might not have an easy "collapse" direction to
  fall into) — if so, this is itself an interesting, reportable finding about when
  collapse is and isn't an easy local minimum, not a failed demo.
- **CIFAR-10 stage may not be reached** in the first session — explicitly fine; the toy
  demo alone already answers the "how do these functions get learned" question this
  project was started to address.

## Status

Plan written. Nothing else started yet.
