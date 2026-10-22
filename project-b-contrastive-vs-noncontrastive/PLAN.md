# Project B — Contrastive vs. Non-Contrastive Self-Supervised Learning

**Status: outlined, not started.** This is a placeholder plan so the idea isn't lost
between sessions — not a committed pre-registration yet. That gets written properly,
pre-registration-style, in the session this project actually starts.

## The question

Do contrastive and non-contrastive self-supervised methods learn meaningfully different
representations on the same data, or do they converge to similarly useful features by a
different route?

## Rough shape of the plan

- **Contrastive baseline:** a SimCLR-style method — pulls representations of two
  augmented views of the same image together, pushes representations of different images
  apart. Needs explicit negative examples.
- **Non-contrastive comparison:** a JEPA/BYOL-style method — predicts one view's
  representation from another, with no explicit negative examples, relying on
  architectural tricks (stop-gradient, a predictor network, or similar) to avoid
  representational collapse.
- **Shared infrastructure:** reuse Project A's encoder architecture and linear-probing
  evaluation harness rather than rebuilding from scratch — the point of doing B second is
  to not start from zero.
- **What "interesting" would look like:** either the two methods converge to similar
  linear-probe accuracy via different routes (interesting on its own), or one clearly
  outperforms the other on this dataset (worth understanding why, not just reporting
  which won).

## Key reading already identified

- Ponce et al., "Dual Perspectives on Non-Contrastive Self-Supervised Learning," ICLR
  2026 — the theoretical question of why non-contrastive methods avoid collapse without
  negative examples.
- The original SimCLR and BYOL papers, for the contrastive/non-contrastive baselines
  respectively.

## Open questions to resolve when this project actually starts

- Same dataset as Project A (CIFAR-10) for a clean comparison, or something slightly
  richer to give the comparison more room to show a difference?
- What counts as "collapse" operationally, and how to detect it quantitatively rather
  than just visually inspecting embeddings?
