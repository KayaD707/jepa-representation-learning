# Project B — Progress Log

## Session 1 (2026-10-26/27) — built, trained, and a clean, interpretable result

Implemented `contrastive_loss.py` (NT-Xent, new), reused Project A's `Encoder` and
Project D's `vicreg_loss.py` (non-contrastive comparison), and trained both methods on
the same 5,000-image CIFAR-10 subset, same encoder architecture, same 500 optimizer
steps, same batch size (256) — see PLAN.md for why matching the training budget exactly
matters for a fair comparison.

### Results (chance level = 10%)

| Method | Linear-probe accuracy | Final repr std |
|---|---|---|
| random-init | 29.11% | — |
| contrastive (NT-Xent) | 50.07% | 0.4558 |
| **non-contrastive (VICReg)** | **53.85%** | 0.3651 |

Both self-supervised methods clearly clear the random-init floor (+21pp, +25pp) — the
minimum bar from PLAN.md's shared infrastructure table. VICReg wins by 3.8 percentage
points.

### Reading the result honestly

- **The 3.8pp gap is real but modest** — not large enough to call one method
  conclusively better in general. PLAN.md named the likely explanation in advance:
  SimCLR-style contrastive learning is known to be batch-size-sensitive (batch size
  controls the number of negatives available per step), and a batch size of 256 here
  is small by the standards of SimCLR's own paper (which uses batch sizes in the
  thousands). This result is consistent with that named risk, not a surprise finding —
  the gap could plausibly narrow or close with a larger batch, which wasn't tested here
  (CPU budget).
- **One finding worth flagging, consistent with Project D's lesson:** the contrastive
  method's final representation actually has a *higher* raw standard deviation
  (0.4558) than VICReg's (0.3651), yet scores *lower* accuracy. Raw spread alone still
  doesn't predict usefulness — same lesson Project D demonstrated directly (the
  variance-only ablation had healthy std but high redundancy and worse accuracy). Not
  re-measured here with a covariance diagnostic for the contrastive encoder specifically
  (NT-Xent doesn't log one the way vicreg_loss.py does) — a natural follow-up if this
  project is revisited.
- **random-init's 29.11% is higher than it might look at first** — a fresh, untrained
  CNN with this architecture still extracts some useful low-level structure (edges,
  color blobs) purely from its convolutional inductive bias, a well-known phenomenon
  in vision (sometimes called the "random features" baseline). It's a real floor, not
  an unusually weak one — both trained methods still clear it by a wide margin.

### What this means for PLAN.md's actual question

"Do a contrastive method and a non-contrastive method... learn representations that are
similarly useful downstream, or does one clearly win, and if so, why?" — the honest
answer at this scale: non-contrastive (VICReg) wins by a modest, plausibly
batch-size-explained margin, not an overwhelming one. Both converge to clearly useful
representations via very different mechanisms (explicit negatives vs. explicit
variance/covariance regularization, no negatives) — the more interesting part of
PLAN.md's framing ("either outcome is informative") holds either way.

## Status

Complete. No further work planned unless batch-size sensitivity is worth testing
directly later (would need a larger batch size, likely requiring a smaller training
subset or fewer steps to stay within CPU budget).
