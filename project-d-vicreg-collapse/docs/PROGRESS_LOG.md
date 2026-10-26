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

## Session 2 (2026-10-26) — CIFAR-10 stage: the mechanism, now with a real accuracy number

Built `cifar_demo.py`: same 3-condition ablation (invariance-only / variance-only /
full VICReg), now on real CIFAR-10 images with two randomly-augmented crops per image
as the two views, trained on a 5,000-image subset (CPU budget), evaluated with the same
linear-probing protocol as Project A (frozen encoder, linear classifier, held-out test
accuracy).

**Results (chance level = 10%):**

| Condition | Mean std | Off-diagonal cov | Linear-probe accuracy |
|---|---|---|---|
| invariance-only | 0.0057 | 0.0000 | 24.45% |
| variance-only | 1.0632 | 80.4580 | 28.98% |
| full VICReg | 0.3291 | 0.0458 | **45.09%** |

The ordering matches PLAN.md's prediction exactly: full VICReg clearly beats both
ablations, and variance-only (healthy spread, but massively redundant dimensions —
off-diagonal covariance 80.46 vs. full VICReg's 0.046, roughly 1750x higher) beats
invariance-only but falls well short of full VICReg. This is the same story the toy
demo told, now with a real downstream number attached: redundant dimensions aren't
just a theoretical concern, they measurably cost accuracy.

**One finding not predicted by PLAN.md, worth stating honestly:** invariance-only —
the condition whose representation std (0.0057) shows it's about as collapsed as
Project D's toy demo ever got — still reached 24.45% accuracy, well above the 10%
chance level. A representation that's "collapsed" by the std diagnostic apparently
isn't collapsed to *zero* information; some residual, non-collapsed signal (plausibly
coming from the convolutional layers before the final collapsed projection, or from
crude class-correlated statistics like average brightness/color that survive even a
near-constant embedding) still leaks through enough for a linear probe to do
noticeably better than guessing. Worth remembering: "collapse," as measured by std, is
a matter of degree, not a clean binary — a diagnostic showing severe collapse doesn't
mean the representation carries *zero* usable information, just dramatically less than
a healthy one.

## Next up

Project D's two planned stages (toy demo, CIFAR-10 stage) are both complete. No further
work planned here unless a new question comes up — this project was scoped tightly
around one specific mechanism question, not meant to be extended indefinitely like
Projects A/B/C.
