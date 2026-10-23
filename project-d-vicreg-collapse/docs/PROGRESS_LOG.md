# Project D — Progress Log

## Session 1 (2026-10-23) — toy demo, collapse made visible

Built `vicreg_loss.py` (the three VICReg terms as separate, individually-switchable
functions — see PLAN.md for why each term maps onto one clause of the citation) and
`toy_demo.py` (4 Gaussian-blob clusters in 2D, two independently-noised views x/y of
each cluster center, a small shared 2-layer encoder, trained under 3 ablation
conditions).

Reused Project A's existing `.venv` (`torch==2.2.2`) rather than setting up a new
environment — same dependency set, already confirmed working on this machine.

**Result — ran clean on the first attempt, matched the prediction in PLAN.md:**

| Condition | Final per-dim std | Off-diagonal covariance | Visual result |
|---|---|---|---|
| invariance-only | [0.032, 0.043] | 0.0000 | Full collapse — all 4 true clusters crushed into a region 0.04 wide |
| variance-only | [1.242, 1.387] | 2.5590 | No point-collapse, but high cross-dimension correlation — a subtler, partial collapse (not predicted as clearly as this in PLAN.md, but anticipated as a possibility) |
| full VICReg | [1.354, 1.312] | 0.0010 | Healthy spread, near-zero redundancy, all 4 clusters cleanly separated with no label supervision |

See `collapse_demo.png` for the actual scatter plots — looked at directly, not just
the printed numbers (same discipline as Project A's masking sanity check).

**One honest note:** the "variance-only" middle condition separated the clusters
visually *better* than expected for something meant to demonstrate a failure mode —
the real tell was the covariance number (2.56 vs 0.001), not the scatter plot by eye.
Worth remembering: representation collapse isn't always visually obvious even when
it's numerically real — this is itself a useful finding about why VICReg's paper
reports the covariance term's value explicitly rather than relying on plots.

## Next up

If continuing: scale this same three-condition ablation up to CIFAR-10 (two augmented
crops per image as x/y, reusing Project A's `data.py` loading code), and evaluate with
the same linear-probing protocol as Project A, per PLAN.md's second evaluation stage.
Not started yet — the toy demo was the part that directly answered the original
question ("how do these functions get learned without collapsing"), so stopping here
for this session is a deliberate checkpoint, not an unfinished dangling task.
