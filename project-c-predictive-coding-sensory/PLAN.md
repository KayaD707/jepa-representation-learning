# Project C — Predictive Coding on Toy Sensory Sequences

**Status: outlined, not started.** Placeholder plan so the idea isn't lost between
sessions. The real, committed plan gets written pre-registration-style once this project
actually starts — likely after Projects A and B, since this is the most exploratory of
the three and benefits most from the judgment built up doing the other two first.

## The question

Can a small recurrent model learn to predict *future representations* (not future raw
pixels) from past frames of a simple synthetic visual scene — directly testing the
"compare at the sensory level, in representation space" idea from the Mohammadi/Halvagal/
Zenke paper, rather than relying on a pre-built benchmark the way Projects A and B do.

## Rough shape of the plan

- **A synthetic environment, deliberately simple:** something like a moving shape on a
  plain background, or a toy physics simulation (a ball bouncing, a pendulum) — simple
  enough to generate unlimited labeled "ground truth" data (the true next position/state)
  for evaluation, which a real video dataset wouldn't give for free.
- **Architecture:** an encoder for each frame, a recurrent module carrying state forward,
  and a predictor trained to predict the *next frame's representation* from the current
  state — the recurrent JEPA shape described in the Zenke lab paper, not a generative
  model that decodes back to pixels.
- **The hard part, honestly:** there's no standard benchmark for this, unlike Projects A
  and B. The evaluation has to be designed, not borrowed — e.g., does the learned
  representation support predicting the true future position of the object better than a
  naive baseline (last-position-repeated, or a simple linear extrapolation)?

## Key reading already identified

- Mohammadi, Halvagal & Zenke, "Understanding cortical computation through the lens of
  joint-embedding predictive architectures," bioRxiv, 2025 — the direct template for this
  project's architecture and motivation.
- Rao & Ballard, 1999, and Friston, 2010 — the neuroscience predictive-coding background
  motivating why this comparison should happen in representation space at all.

## Open questions to resolve when this project actually starts

- What's the simplest synthetic environment that's still non-trivial to predict (pure
  constant-velocity motion might be too easy to need a learned representation at all)?
- What's the honest baseline this needs to beat? A naive physics extrapolation is a much
  higher bar than "no baseline at all," and picking the right one matters as much as the
  model itself.
