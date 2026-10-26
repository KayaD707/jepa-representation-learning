"""
model.py -- the actual I-JEPA pieces: context encoder, target encoder, predictor, plus
a decoder used only by the autoencoder baseline (not by I-JEPA itself).

ARCHITECTURE CHOICE, stated plainly (see PLAN.md "out of scope"): the original I-JEPA
uses a Vision Transformer and predicts representations of individual PATCHES. This
project uses a small CNN instead, and -- following the simplification already baked
into data.py's pixel-space masking -- predicts ONE global representation of the hidden
region from ONE global representation of the visible region, rather than per-patch
representations. This is simpler than the paper, and it is the right simplification to
state honestly rather than silently narrow.

THE ENCODER is the one piece of architecture shared by all three things being compared
in this project (random-init baseline, autoencoder baseline, I-JEPA) -- using the exact
same Encoder class for all three is what makes the eventual linear-probe comparison
fair: any accuracy difference has to come from HOW the weights were trained, not from
one model simply being bigger.
"""

import copy

import torch
import torch.nn as nn

EMBED_DIM = 128  # size of the final representation vector -- same for all 3 methods compared


class Encoder(nn.Module):
    """Small CNN: 32x32x3 image -> EMBED_DIM-dimensional vector.

    Four stride-2 conv blocks halve the spatial size each time (32 -> 16 -> 8 -> 4 -> 2),
    doubling channel count each time to compensate for the lost spatial resolution with
    more "depth" of features. AdaptiveAvgPool2d(1) then collapses whatever spatial size
    is left into a single value per channel -- this is what turns a spatial feature map
    into the single global vector this project's simplified (non-ViT, non-patch-level)
    method needs. BatchNorm after each conv stabilizes training -- without it, a CNN
    this deep trained from scratch on CPU is noticeably more prone to stalling early in
    training.
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # 32x32 -> 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 4x4 -> 2x2
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 2x2 -> 1x1, any remaining spatial detail averaged away
        )
        self.project = nn.Linear(256, embed_dim)

    def forward(self, x):
        features = self.conv(x)            # (batch, 256, 1, 1)
        features = features.flatten(1)     # (batch, 256)
        return self.project(features)      # (batch, embed_dim)


class Predictor(nn.Module):
    """Small MLP: context representation -> predicted target representation.

    This is the piece that does the actual "predict the future/hidden thing in
    representation space" work. It is deliberately simple (one hidden layer) -- in
    I-JEPA, the heavy lifting is supposed to happen in the encoders learning good
    representations, not in the predictor finding a clever way to cheat the loss.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """EMBED_DIM vector -> reconstructed 32x32x3 image. Used ONLY by the autoencoder
    baseline (train_autoencoder_baseline.py) -- I-JEPA itself never reconstructs
    pixels, which is the entire point of the comparison (predicting in representation
    space vs. predicting in pixel space, per PLAN.md's "actual claim being tested").

    Mirrors the encoder's downsampling in reverse: a linear layer expands back to a
    (256, 2, 2) feature map, then four stride-2 transposed convs double the spatial
    size each time (2 -> 4 -> 8 -> 16 -> 32) while halving channels, ending at 3
    channels (RGB) with a Tanh to keep output values in a bounded range.
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.expand = nn.Linear(embed_dim, 256 * 2 * 2)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 2x2 -> 4x4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 4x4 -> 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 8x8 -> 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     # 16x16 -> 32x32
            nn.Tanh(),
        )

    def forward(self, embedding):
        x = self.expand(embedding)
        x = x.view(-1, 256, 2, 2)
        return self.deconv(x)


def make_target_encoder(context_encoder: Encoder) -> Encoder:
    """Creates the target encoder as an exact weight-for-weight copy of the context
    encoder at initialization, then freezes it (requires_grad=False on every
    parameter).

    Freezing matters: the target encoder is NEVER updated by backpropagation. Its only
    update mechanism is the EMA step below. This is a deliberate anti-collapse design
    in I-JEPA (named in PLAN.md) -- if the target encoder could also freely move to
    minimize the loss, the easiest joint solution for both encoders would again be the
    same constant-output collapse discussed in Project D. By making the target slow
    and non-gradient-updated, the context encoder is forced to do the work of matching
    a comparatively stable, slowly-moving target, rather than the two encoders
    "meeting in the middle" at a trivial point.
    """
    target_encoder = copy.deepcopy(context_encoder)
    for param in target_encoder.parameters():
        param.requires_grad = False
    return target_encoder


@torch.no_grad()
def ema_update(target_encoder: Encoder, context_encoder: Encoder, momentum: float = 0.996):
    """Nudges target_encoder's weights slightly toward context_encoder's current
    weights: target = momentum * target + (1 - momentum) * context.

    momentum=0.996 means the target encoder moves only 0.4% of the way toward the
    context encoder's weights after each training step -- it changes slowly and
    smoothly, in contrast to the context encoder's weights, which jump around however
    gradient descent happens to move them step to step. This smoothness is what makes
    the target a stable, slowly-evolving thing to predict, rather than a jittery one.
    momentum is kept constant here (a stated simplification -- some I-JEPA/BYOL
    variants ramp momentum up over training; not done here, to keep the training loop
    easy to read and reason about).
    """
    for target_param, context_param in zip(target_encoder.parameters(), context_encoder.parameters()):
        target_param.data.mul_(momentum).add_(context_param.data, alpha=1 - momentum)
