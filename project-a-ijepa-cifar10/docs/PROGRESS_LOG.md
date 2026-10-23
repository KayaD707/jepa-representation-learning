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
