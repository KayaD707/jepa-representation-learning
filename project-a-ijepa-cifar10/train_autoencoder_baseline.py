"""
train_autoencoder_baseline.py -- the second baseline named in PLAN.md: an
encoder-decoder trained to reconstruct raw pixels.

This is the more interesting of the two baselines (the other being an untrained,
random-init encoder, which needs no training script at all -- see
eval_linear_probe.py). The scientific question this baseline exists to answer:
does predicting in REPRESENTATION space (I-JEPA) produce more useful features than
predicting in PIXEL space (this autoencoder), at this dataset size and model scale?
That comparison is only fair if both methods share the exact same Encoder
architecture and the exact same EMBED_DIM bottleneck -- which is why both this script
and train_ijepa.py import Encoder from the same model.py rather than each defining
their own.

Unlike I-JEPA, this trains on the ORIGINAL, unmasked image -- an autoencoder's job is
to compress-then-reconstruct the whole image through the EMBED_DIM bottleneck, not to
fill in a hidden region. (Using the masked context image instead would turn this into
an inpainting task, which is a different, not-quite-comparable problem -- stated here
so the choice is visible, not accidental.)

Run with (from project-a-ijepa-cifar10/):
    .venv\\Scripts\\python.exe train_autoencoder_baseline.py
"""

import csv
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import load_cifar10
from model import Encoder, Decoder

BATCH_SIZE = 256
EPOCHS = 25  # matched to train_ijepa.py's epoch count, so training BUDGET is equal -- a
             # fair comparison needs both methods to see the same amount of data/updates.
LR = 1e-3
CHECKPOINT_PATH = "autoencoder_encoder.pt"
LOG_PATH = "docs/autoencoder_training_log.csv"


def main():
    device = torch.device("cpu")
    torch.manual_seed(0)

    print("Loading CIFAR-10 (unmasked, for autoencoder reconstruction)...")
    train_dataset = load_cifar10(train=True, download=True)
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)
    loss_fn = nn.MSELoss()

    os.makedirs("docs", exist_ok=True)
    log_rows = []

    print(f"Starting autoencoder pretraining: {EPOCHS} epochs, batch size {BATCH_SIZE}, "
          f"{len(train_dataset)} training images, device={device}")
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_loss_sum = 0.0
        epoch_std_sum = 0.0
        n_batches = 0

        for images, _labels in loader:  # labels unused -- still self-supervised, reconstruction is the signal
            images = images.to(device)

            embedding = encoder(images)
            reconstruction = decoder(embedding)
            loss = loss_fn(reconstruction, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            # Same collapse diagnostic as train_ijepa.py -- an autoencoder is actually
            # much LESS prone to collapse than I-JEPA/VICReg-style methods, since a
            # constant embedding can't reconstruct varying images with low error. Tracked
            # anyway, for an apples-to-apples comparison in docs/RESULTS.md.
            epoch_std_sum += embedding.detach().std(dim=0).mean().item()
            n_batches += 1

        avg_loss = epoch_loss_sum / n_batches
        avg_std = epoch_std_sum / n_batches
        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1:2d}/{EPOCHS}  recon_loss={avg_loss:.5f}  "
              f"repr_std={avg_std:.5f}  elapsed={elapsed:.0f}s")
        log_rows.append({"epoch": epoch + 1, "loss": avg_loss, "repr_std": avg_std, "elapsed_s": elapsed})

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "repr_std", "elapsed_s"])
        writer.writeheader()
        writer.writerows(log_rows)

    torch.save(encoder.state_dict(), CHECKPOINT_PATH)
    print(f"\nSaved trained autoencoder encoder to {CHECKPOINT_PATH}")
    print(f"Saved training log to {LOG_PATH}")


if __name__ == "__main__":
    main()
