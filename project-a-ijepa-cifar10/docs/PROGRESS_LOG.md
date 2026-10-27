# Progress Log — Project A

One entry per session, so the reasoning stays visible rather than arriving as a finished
result. Environment setup (session 1) is logged separately in
[`SETUP_NOTES.md`](SETUP_NOTES.md) since it was its own, fairly long story.

## Session 2 — data loading and masking

**Goal for today:** write and verify the CIFAR-10 loading and context/target masking
logic — the one piece of this project that's genuinely specific to JEPA, as opposed to
a generic image pipeline.

**What was built:** [`data.py`](../data.py) — `load_cifar10()` (standard loading +
normalization), `sample_target_block()` (randomly picks a rectangular patch-grid region
as the target, with random scale/aspect, following the spirit of the I-JEPA paper's own
sampling), `make_context_target_masks()` (converts that patch-grid region into pixel-space
boolean masks), and `JEPAMaskedCIFAR10` (the dataset wrapper that actually produces
(context_image, target_image) training pairs).

**Verification, not just "it ran":** rather than trust that the shapes lined up, the
script generates an actual visualization (`masking_sanity_check.png`) showing the
original image, the context view, and the target view side by side for several examples.
Looking at it directly confirmed: the context view has a clean rectangular hole, the
target view shows *only* that same region, the two are exact complements with no overlap,
and the target block's size/shape/position vary across examples (confirming the random
sampling logic isn't secretly always producing the same block).

**A mistake caught by looking at the actual image, not just the code:** the first
version of the sanity-check script's own print statements described the masked region as
"blacked out." Looking at the generated image, it's clearly a muted grey/beige, not
black. Tracing why: masking sets pixels to 0 in *normalized* space (where the per-channel
mean is close to 0, not where raw pixel value 0 means black), and the display function
un-normalizes everything for viewing — so a normalized 0 maps back to roughly CIFAR-10's
*average* color, not black. This doesn't affect what the model actually trains on (it
only ever sees the normalized tensor, where the masked region genuinely is a clean,
consistent 0) — it was purely a wrong claim in a comment, corrected after actually
looking at the output rather than assuming the comment was right.

**Why this is worth stating plainly:** this is the second time in this project alone
(after the environment saga) that looking at the actual result — not trusting that code
which runs without an error is therefore correct — caught something. That's the habit
worth keeping, not a one-off.

**Status:** data pipeline complete and visually verified. Not yet done: the actual I-JEPA
model (context encoder, target encoder, predictor) and the training loop — next session.

## Session 3 (2026-10-26/27) — model, training, baselines, and a genuine negative result

Built `model.py` (Encoder, Predictor, Decoder, `make_target_encoder`/`ema_update`),
`train_ijepa.py`, `train_autoencoder_baseline.py`, and `eval_linear_probe.py`, per
PLAN.md's evaluation protocol: random-init, autoencoder, and I-JEPA, all compared with
the same frozen-encoder linear probe.

**Training itself showed no collapse.** I-JEPA's representation std rose from 0.24 to a
peak of 0.53 around epoch 9–10, then settled in a healthy 0.48–0.53 band through epoch
25 — never approaching zero. The training loss *rising* slightly over most of training
(0.018 → ~0.022) looked alarming at first glance, but is consistent with a spreading
(not collapsing) representation: a higher-variance target naturally produces a higher
raw MSE even for an equally-good predictor, so this isn't itself evidence of a problem.

### The real result (final linear-probe comparison, chance level = 10%)

| Method | Accuracy |
|---|---|
| autoencoder | **40.44%** |
| random-init | 29.11% |
| **i-jepa** | **26.09%** |

**I-JEPA fails PLAN.md's own stated minimum bar** — it does not clearly exceed
random-init; it's actually slightly *below* it, and well below the autoencoder
baseline. This is a genuine negative result, not a near-miss to round up.

### Ruling out a hypothesis before accepting the result

Before writing this up, one real possibility was checked directly, the same discipline
used in Project C: the context encoder was trained *exclusively* on masked images (a
block zeroed out), but `eval_linear_probe.py` evaluates it on full, unmasked images —
a plausible train/eval distribution mismatch. A quick diagnostic confirmed the
distribution shift is real and large (embedding norm 0.80 on full images vs. 5.33 on
masked images — the BatchNorm statistics learned during training clearly don't transfer
cleanly to unmasked inputs). **But evaluating on masked images instead did not fix the
result** — accuracy on a 10k/2k subset was 22.05% (full images) vs. 20.60% (masked
images), both still below that subset's random-init level. The distribution mismatch is
real but is not the explanation for the underperformance.

### The honest explanation

PLAN.md named this risk explicitly in advance, in its own "out of scope" section: using
a small CNN instead of a Vision Transformer, and predicting ONE global pooled vector
instead of per-patch representations, is a real simplification of I-JEPA, not a neutral
implementation detail. With a single global vector produced by 4 stride-2 convs down to
a 2×2 feature map, the "predict the hidden region's representation from the visible
region's representation" task likely reduces to something close to "predict the hidden
region's coarse color/brightness statistics from the visible region's" — a task
solvable without learning the kind of fine-grained, edge/shape-sensitive features that
a *random*, untrained CNN already happens to produce (a well-documented phenomenon:
random CNN features are a surprisingly strong baseline, precisely because convolution's
inductive bias captures generic edge/texture detectors for free). Training specifically
toward the coarse-statistics task plausibly pushed the encoder's weights *away* from
those generically useful random features, without replacing them with anything more
useful for 10-class classification — explaining why trained I-JEPA here ends up
*below* its own untrained starting point.

### Reading this against the project's own stated goal

PLAN.md said, before any training happened: *"it's genuinely possible the autoencoder
baseline wins... A negative or null result here is a real, reportable finding about
where the architecture's advantages do and don't show up at small scale — not a failed
project."* That is exactly what happened, and more starkly than PLAN.md's own
hedge anticipated: not just a tie or a narrow loss, but pixel-space prediction
(autoencoder, 40.44%) clearly beating representation-space prediction (I-JEPA, 26.09%)
at this scale, with this architecture. The result argues specifically against the
global-vector CNN simplification, not against JEPA's underlying idea — Project D
(VICReg, also non-contrastive, also predicting in representation space, but operating
on standard augmented views rather than spatial block-masking) reached 45.09% on a
similar evaluation, which suggests the *masking-based* version of this idea is the
specific piece that struggled here, not "representation-space prediction" in general.

**Status:** Complete. All three methods trained and compared; a genuine, well-understood
negative result for I-JEPA's minimum bar, with one competing explanation (train/eval
distribution mismatch) tested and ruled out before accepting the architectural
explanation above.
