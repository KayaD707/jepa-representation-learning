"""
train.py -- trains the recurrent JEPA model on bouncing-ball sequences.

For each sequence, step through time: encode the current frame, update the GRU hidden
state, predict the NEXT frame's representation from that hidden state, and compare it
against the target encoder's (stop-gradient, EMA-updated) encoding of the actual next
frame. Loss is averaged over every timestep in the sequence, so the model is trained to
predict well at every point in the sequence, not just at the end.

Run with:
    ..\\project-a-ijepa-cifar10\\.venv\\Scripts\\python.exe train.py
"""

import os
import time

import torch
import torch.nn as nn

from sim import BouncingBallDataset
from recurrent_model import FrameEncoder, RecurrentPredictor, make_target_encoder, ema_update

N_TRAIN_SEQUENCES = 1000
SEQ_LEN = 20
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
EMA_MOMENTUM = 0.99  # faster-moving target than Project A's 0.996 -- sequences here are
                     # short (20 steps) and the task is much simpler, so the target can
                     # afford to track the context encoder a bit more closely per step
CHECKPOINT_DIR = "checkpoints"


def run_epoch(frames_batch, context_encoder, target_encoder, recurrent, optimizer, loss_fn, train: bool):
    """frames_batch: (batch, seq_len, 1, H, W). Returns the average per-timestep loss
    for this batch. If train=True, also backpropagates and steps the optimizer/EMA."""
    batch_size, seq_len = frames_batch.shape[0], frames_batch.shape[1]
    hidden = recurrent.init_hidden(batch_size)
    losses = []

    for t in range(seq_len - 1):  # no "next frame" to predict after the last timestep
        current_frame = frames_batch[:, t]
        next_frame = frames_batch[:, t + 1]

        frame_repr = context_encoder(current_frame)
        hidden, predicted_next_repr = recurrent.step(frame_repr, hidden)

        with torch.no_grad():
            target_next_repr = target_encoder(next_frame)

        step_loss = loss_fn(predicted_next_repr, target_next_repr)
        losses.append(step_loss)

    total_loss = torch.stack(losses).mean()

    if train:
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        ema_update(target_encoder, context_encoder, momentum=EMA_MOMENTUM)
        # hidden carries gradient history across the whole unrolled sequence; detaching
        # here isn't needed since hidden is reinitialized fresh for every new batch
        # (no truncated-BPTT carry-over between batches in this project).

    return total_loss.item()


def main():
    torch.manual_seed(0)

    print(f"Generating {N_TRAIN_SEQUENCES} training sequences ({SEQ_LEN} frames each)...")
    train_dataset = BouncingBallDataset(n_sequences=N_TRAIN_SEQUENCES, seq_len=SEQ_LEN, seed=0)

    context_encoder = FrameEncoder()
    target_encoder = make_target_encoder(context_encoder)
    recurrent = RecurrentPredictor()

    optimizer = torch.optim.Adam(
        list(context_encoder.parameters()) + list(recurrent.parameters()), lr=LR
    )
    loss_fn = nn.MSELoss()

    n_batches = N_TRAIN_SEQUENCES // BATCH_SIZE
    start_time = time.time()
    print(f"Starting training: {EPOCHS} epochs, {n_batches} batches/epoch, batch size {BATCH_SIZE}")

    for epoch in range(EPOCHS):
        permutation = torch.randperm(N_TRAIN_SEQUENCES)
        epoch_loss_sum = 0.0

        for b in range(n_batches):
            idx = permutation[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            frames_batch = train_dataset.frames[idx]
            loss = run_epoch(frames_batch, context_encoder, target_encoder, recurrent,
                              optimizer, loss_fn, train=True)
            epoch_loss_sum += loss

        avg_loss = epoch_loss_sum / n_batches
        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1:2d}/{EPOCHS}  loss={avg_loss:.5f}  elapsed={elapsed:.0f}s")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(context_encoder.state_dict(), os.path.join(CHECKPOINT_DIR, "context_encoder.pt"))
    torch.save(recurrent.state_dict(), os.path.join(CHECKPOINT_DIR, "recurrent.pt"))
    print(f"\nSaved checkpoints to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
