# Project B — Plan: Contrastive vs. Non-Contrastive Self-Supervised Learning

Promoted from placeholder to a real pre-registration now that Project A (I-JEPA) and
Project D (VICReg) exist and give this project real infrastructure to reuse, resolving
the open questions the placeholder left unresolved.

## The question

Do a contrastive method and a non-contrastive method, trained on the same data with the
same encoder architecture and the same training budget, learn representations that are
similarly useful downstream — or does one clearly win, and if so, why?

## Resolving the placeholder's open questions

- **Dataset:** CIFAR-10, same as Projects A and D — keeps every comparison across the
  whole portfolio apples-to-apples (same images, same linear-probing protocol, same
  encoder family), rather than introducing a new confound.
- **Collapse, operationalized quantitatively:** the same two numbers used in Projects A
  and D — per-dimension representation standard deviation (collapse to a point) and
  mean squared off-diagonal covariance (collapse to a redundant subspace) — computed on
  a held-out batch after training. Not just visual inspection.

## The two methods being compared

- **Contrastive: SimCLR-style (NT-Xent loss).** Two augmented views of the same image
  are pulled together; views of *every other image in the batch* are explicitly pushed
  apart as negatives. This is new code for this project (`contrastive_loss.py`) — no
  existing part of this portfolio implements a contrastive loss yet, since both A
  (I-JEPA) and D (VICReg) are non-contrastive methods.
- **Non-contrastive: VICReg, reused directly from Project D (`vicreg_loss.py`)**, not a
  new BYOL-style implementation. This is a deliberate scope decision, stated plainly: the
  placeholder originally considered "JEPA/BYOL-style... stop-gradient, a predictor
  network" (i.e., Project A's own anti-collapse mechanism) as the non-contrastive
  comparison. Project A already demonstrates that EMA/stop-gradient route. Reusing
  VICReg instead gives Project B a *different* non-contrastive mechanism (explicit
  variance/covariance regularization, no asymmetric network needed) to contrast against
  SimCLR — a more informative three-way picture across the portfolio (EMA/predictor vs.
  explicit regularization vs. contrastive) than reimplementing BYOL would have been,
  without writing a near-duplicate of Project A.

## Shared infrastructure, and what's genuinely new

| Piece | Source |
|---|---|
| Encoder architecture | Imported directly from `project-a-ijepa-cifar10/model.py` (its `Encoder` class) — same architecture Project A and its baselines use, so EMBED_DIM and capacity are held constant across every method in the portfolio. |
| Non-contrastive loss | Imported directly from `project-d-vicreg-collapse/vicreg_loss.py` |
| Two-augmented-views data pipeline | Re-implemented locally (`data_two_view.py`), matching Project D's `cifar_demo.py` construction — kept as a local copy rather than a cross-import, since Project D's own file is a script, not a shared module, and duplicating ~20 lines of transform code is simpler than restructuring D to export it. |
| Contrastive loss (NT-Xent) | **New** — `contrastive_loss.py` |
| Linear-probe evaluation | Re-implemented locally (`eval_linear_probe.py`), same pattern as Projects A and D |

## What "interesting" looks like (stated before training)

- If both methods land within a few points of each other on linear-probe accuracy:
  genuinely interesting — two very different training signals (explicit negatives vs.
  explicit regularization, no negatives) converging on similarly useful features.
- If one clearly wins: the collapse diagnostics (std, covariance) and the training
  curves should be examined to understand *why*, not just reported as a leaderboard
  result. A plausible candidate explanation named in advance: SimCLR's negatives give it
  a directly batch-size-dependent amount of signal per step (more negatives = harder,
  more informative task), while VICReg's signal per step doesn't scale with batch size
  the same way — if batch size here is small (see CPU constraint below), that could
  disadvantage the contrastive method specifically at this scale, not in general.

## Constraints and known risks

- Same CPU-only training budget constraint as Projects A and D — batch size will likely
  be smaller than papers in this area typically use, which specifically matters more for
  SimCLR (contrastive methods are known to be batch-size-sensitive, since batch size
  controls the number of negatives) than for VICReg. This is named here, in advance, as a
  real limitation of this comparison's conclusions, not discovered after the fact.
- Training subset size and epoch budget will be matched exactly between the two methods
  (same images, same number of optimizer steps) so any accuracy difference reflects the
  method, not an uneven training budget.

## Status

Plan finalized. Code not yet written.
