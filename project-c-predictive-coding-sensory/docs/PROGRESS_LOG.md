# Project C — Progress Log

## Session 1 (2026-10-26) — built, trained, and a genuine negative result (with a methodology bug caught along the way)

Built `sim.py` (bouncing-ball physics, visually confirmed via `sim_sanity_check.png`),
`recurrent_model.py` (frame encoder, GRU, predictor, reusing Project A's EMA/stop-gradient
utilities directly), `train.py`, and `eval.py`.

### The training loss rose, not fell — and that was the first honest flag

Training loss went 0.0124 → 0.0059 (epochs 1–3) then climbed steadily to 0.0150 by
epoch 20. A rising loss is unusual enough that it was checked immediately rather than
assumed to be fine: a diagnostic comparing the final encoder's representation to the
predicted representation found no collapse (healthy std ≈ 0.84, comparable norms,
MSE between predicted and actual next-frame representation ≈ 0.032 against a
per-dimension variance of ≈ 0.7 — i.e., the predictor is capturing the large majority
of representation-level variance). The rising loss is most likely explained by the
representation's scale drifting upward slightly over training (nothing in this
architecture constrains embedding norm, unlike VICReg's explicit variance cap) — not
investigated further this session, flagged here for anyone picking this back up.

### The real finding: the first evaluation run was comparing apples to oranges

The first full run of `eval.py` reported the model's predicted-position error as
**0.1005** against baselines of 0.000625 (last-position-repeated) and 0.000075
(constant-velocity) — a roughly 130x gap, which would normally read as "the model is
badly broken." Before writing that up as the result, it was checked directly: a
diagnostic found that even decoding the **true, actual** next-frame representation
(not the model's prediction — the best possible case) through the trained linear
readout only achieves an MSE of ≈0.032 and a position correlation of ≈0.65, nowhere
near perfect.

**The bug: the original baselines operated on exact, noiseless ground-truth position
numbers straight from the simulator, while the model has to recover position through a
genuinely lossy pipeline (render to a 32×32 image → CNN encoder → linear readout).**
Comparing the two was never a fair test of "does the learned dynamics model predict
motion well" — it was mostly measuring the unavoidable cost of going through pixels at
all, which has nothing to do with whether the recurrent predictor is any good.

**Fix:** `eval.py` now reports both sets of baselines, clearly labeled —
`oracle_position_baselines()` (exact state, reference only, not a fair comparison) and
the new `estimated_position_baselines()` (computed on positions decoded from the exact
same encoder+readout pipeline the model uses, so every method being compared goes
through the same lossy image pipeline and only the *prediction mechanism* differs). A
readout sanity-check number (MSE decoding true representations → position, 0.0312) is
now also reported as the honest accuracy ceiling any image-based method here could
reach.

### Corrected results (fair comparison — chance level not applicable; lower is better)

| Method | Overall MSE | Bounce MSE | Non-bounce MSE |
|---|---|---|---|
| [oracle] last-position-repeated | 0.000625 | 0.000640 | 0.000625 |
| [oracle] constant-velocity | 0.000075 | 0.001102 | 0.000039 |
| last-position-repeated (image-estimated) | 0.062689 | 0.105146 | 0.061173 |
| constant-velocity (image-estimated) | 0.079639 | 0.121015 | 0.078162 |
| **model (predicted representation)** | **0.095138** | **0.156765** | **0.092937** |

Readout sanity ceiling (decoding the *true* representation): MSE = 0.031183.

### Honest interpretation

- **The model does not beat either baseline, even under the corrected, fair
  comparison.** This is a genuine negative result for PLAN.md's "actual claim being
  tested," not a success to spin.
- **One interesting, unplanned side-finding:** under the fair (image-estimated)
  comparison, constant-velocity is actually *worse* than last-position-repeated
  (0.0796 vs 0.0627) — the reverse of the oracle ordering. This makes sense once
  stated: constant-velocity differences two already-noisy position *estimates*,
  which amplifies estimation noise, whereas last-position-repeated only uses one
  noisy estimate. A real, reportable finding about naive-baseline fragility under
  noisy inputs, not a bug.
- **Why the model likely underperforms even the fair baselines:** the readout
  sanity ceiling (0.031 MSE, ~0.65 correlation) shows the *encoder itself* struggles to
  make position cleanly linearly decodable in the first place — a representation-level
  bottleneck that exists before the recurrent predictor ever gets involved. The
  predictor then has to predict a *moving target* (the next frame's already-imperfect
  representation) through a GRU trained only on MSE-in-representation-space, with no
  explicit incentive to preserve exactly the handful of directions in that space that
  happen to encode position. PLAN.md named a different risk in advance (memorizing
  bounce geometry without generalizing) — this is a different, more basic limitation:
  the encoder's representation isn't precise enough at the position level for this
  particular linear-readout evaluation to show a win, independent of whether the
  recurrent dynamics model is doing anything useful at a representation level (its own
  ~0.032 prediction MSE in representation space suggests it may be).
- **Not fixed this session, named as a next step:** training the encoder with an
  explicit auxiliary signal that encourages position-relevant information specifically
  (not attempted here, since it would compromise the "fully self-supervised, label-free
  pretraining" framing that's the whole point of this project), or a deeper/longer-trained
  encoder that might reach a better readout ceiling on its own.

## Next up

If continuing: investigate the rising-training-loss embedding-drift question directly
(log embedding norm per epoch), and/or try a longer training run or larger encoder to
see whether the readout ceiling (0.031 MSE) improves with more capacity/data before
concluding the representation-precision limitation is fundamental rather than just
under-trained.
