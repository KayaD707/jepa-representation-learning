# Project C — Plan: Predictive Coding on a Toy Bouncing-Ball Sequence

Promoted from placeholder to a real pre-registration, now that Projects A and B exist
to build judgment and reusable utilities from (as the placeholder itself anticipated).

## The question

Can a small recurrent model learn to predict a FUTURE frame's *representation* (never
its raw pixels) well enough that the prediction, decoded back to the ball's true (x, y)
position, beats naive physics baselines — including specifically at the one moment
those baselines are known to fail: a bounce.

## Resolving the placeholder's open questions

- **Simplest non-trivial synthetic environment:** a single ball bouncing elastically
  inside a fixed square box, rendered as a 32×32 grayscale frame (a filled circle at the
  ball's position). Chosen over pure constant-velocity motion specifically *because* of
  the bounces — a model that just learns "repeat the last displacement" will do fine
  between bounces but should fail exactly at the bounce frame, which is what makes this
  task non-trivial and gives the evaluation something real to detect.
- **The honest baseline:** two of them, not one, both computed directly on the known
  ground-truth position (free from the simulator, not from the image):
  1. **Last-position-repeated** (assumes zero velocity) — the floor.
  2. **Constant-velocity linear extrapolation** (`pos[t] + (pos[t] - pos[t-1])`) — the
     real bar. This is a strong baseline between bounces and a specifically *bad* one at
     a bounce, which is exactly the gap the learned model needs to close to be
     interesting.

## Architecture

- **Frame encoder:** a small CNN (new, 1-channel input, not reused from Projects A/B
  since those are 3-channel CIFAR-10 encoders at a different resolution/domain) mapping
  one 32×32 grayscale frame to an embedding vector.
- **Target encoder:** an EMA-updated, stop-gradient copy of the frame encoder — reusing
  `make_target_encoder` / `ema_update` directly from Project A's `model.py`. These two
  functions are architecture-agnostic (they operate on any two matching PyTorch
  modules), so importing them here is genuine reuse, not a strained analogy — same
  anti-collapse mechanism as Project A, applied over time instead of over a masked
  region.
- **Recurrent module:** a GRU cell carrying a hidden state forward across frames,
  updated by each new frame's encoded representation.
- **Predictor:** a small MLP mapping the current hidden state to a predicted
  *next-frame* representation — compared against the target encoder's actual encoding
  of the next frame (stop-gradient, EMA-updated, same role as Project A's target
  encoder).

## Evaluation protocol

1. **Position readout (sanity check, the "minimum bar"):** train a linear regressor
   from the (frozen, trained) frame encoder's representation to the ball's true (x, y)
   position. If this fails, the representation isn't encoding position at all, and
   nothing downstream can be trusted.
2. **Predicted-position accuracy (the actual claim being tested):** feed the *predicted*
   next-frame representation (the recurrent model's output, never the real next frame)
   through that same trained readout, and compare the resulting position estimate
   against ground truth — measured in mean squared pixel-position error, directly
   comparable to the two baselines above.
3. **Per-bounce breakdown:** report the same error metric split into "bounce frames"
   and "non-bounce frames" separately. The headline comparison is not just "did the
   model beat the baselines on average" but specifically "did it beat constant-velocity
   extrapolation at bounces, where constant-velocity is known to be wrong."

## What counts as success — stated before training

- **Minimum bar:** the position readout works at all (step 1 above), and the model's
  predicted-position error clearly beats the last-position-repeated baseline.
- **The actual claim being tested:** the model's predicted-position error at bounce
  frames specifically beats constant-velocity extrapolation — i.e., the learned
  representation has captured something about the box's walls/bounce dynamics that a
  naive physics baseline, by construction, cannot.
- **Being honest in advance:** a GRU with enough capacity might simply memorize typical
  bounce geometry for this fixed box size without learning anything that would
  generalize to a differently-shaped box — this project does not test generalization to
  a new environment, only prediction quality within the one simulated box, and that
  limit is stated here rather than glossed over.

## Explicitly out of scope

- Any claim about real sensory data (video, robotics) — this is a toy synthetic
  environment, chosen specifically because ground-truth position is free and exact,
  which a real video dataset would not give.
- Testing whether the representation generalizes to a different box size, ball speed, or
  number of balls — a single fixed environment only.

## Tooling

PyTorch, reusing Project A's already-working environment (no new venv). No W&B for this
project's first pass — the position-space MSE comparison table is the direct result,
same reasoning as Project D's toy-stage choice to skip it initially.

## Status

Plan finalized. Code not yet written.
