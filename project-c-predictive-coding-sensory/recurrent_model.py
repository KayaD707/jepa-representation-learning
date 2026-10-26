"""
recurrent_model.py -- the recurrent JEPA pieces for Project C: a per-frame encoder, a GRU
carrying state across time, and a predictor mapping the current hidden state to a
predicted NEXT-FRAME representation. See PLAN.md for how this connects to Project A's
anti-collapse mechanism (EMA target encoder, reused directly, not reimplemented).
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project-a-ijepa-cifar10"))
from model import make_target_encoder, ema_update  # noqa: E402  -- architecture-agnostic, genuine reuse

EMBED_DIM = 32  # smaller than Projects A/B/D's 64-128 -- this task (one ball's position)
                # carries far less information than a natural image, so a smaller
                # representation is a deliberate match to task complexity, not a shortcut.
HIDDEN_DIM = 64


class FrameEncoder(nn.Module):
    """32x32x1 grayscale frame -> EMBED_DIM vector. New architecture (1 input channel,
    not reused from Projects A/B which are 3-channel CIFAR-10 encoders) -- see PLAN.md."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # 32x32 -> 16x16
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.project = nn.Linear(64, embed_dim)

    def forward(self, x):
        return self.project(self.conv(x).flatten(1))


class RecurrentPredictor(nn.Module):
    """A GRUCell carrying a hidden state forward across frames, plus an MLP head that
    turns the current hidden state into a predicted NEXT-FRAME representation.

    Using a GRUCell (processed one timestep at a time in train.py's loop) rather than
    a full nn.GRU run over the whole sequence at once is deliberate here -- the
    training loop needs the hidden state at every intermediate timestep anyway (to
    predict every next frame, not just the last one), so stepping manually is more
    direct than unrolling with nn.GRU and then slicing out intermediate states.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.gru_cell = nn.GRUCell(input_size=embed_dim, hidden_size=hidden_dim)
        self.predict_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def step(self, frame_repr: torch.Tensor, hidden: torch.Tensor):
        """One timestep: update the hidden state with the current frame's
        representation, and predict the NEXT frame's representation from the
        (just-updated) hidden state. Returns (new_hidden, predicted_next_repr)."""
        new_hidden = self.gru_cell(frame_repr, hidden)
        predicted_next = self.predict_head(new_hidden)
        return new_hidden, predicted_next

    def init_hidden(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.gru_cell.hidden_size, device=device)


class PositionReadout(nn.Module):
    """A simple linear regressor: EMBED_DIM representation -> (x, y) position.
    Trained SEPARATELY, after the encoder/predictor are frozen -- this is the
    regression equivalent of the linear-probe classifier used in Projects A and B
    (PLAN.md's "position readout" evaluation step)."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 2)

    def forward(self, x):
        return self.linear(x)


__all__ = ["FrameEncoder", "RecurrentPredictor", "PositionReadout", "make_target_encoder", "ema_update", "EMBED_DIM"]
