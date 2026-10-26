"""
train_ijepa.py -- the actual I-JEPA pretraining loop.

For each batch: encode the visible context, encode the hidden target (with the
slowly-updated target encoder, no gradient), predict the target representation from
the context representation, and minimize the distance between prediction and actual
target representation.

COLLAPSE MONITORING is included deliberately, not as an afterthought: Project D
demonstrated concretely that a representation can trivially "solve" a predict-the-
other-view loss by collapsing to a near-constant output. I-JEPA avoids this a
DIFFERENT way than VICReg (via the asymmetric slow-moving target + stop-gradient,
rather than explicit variance/covariance loss terms -- see PLAN.md and the repo
README for how A and D relate). Tracking representation std during training is how we
check that this asymmetric-target mechanism is actually working here too, rather than
assuming it from the architecture alone.

Run with (from project-a-ijepa-cifar10/):
    .venv\\Scripts\\python.exe train_ijepa.py
"""

import csv
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import JEPAMaskedCIFAR10, load_cifar10
from model import Encoder, Predictor, make_target_encoder, ema_update, EMBED_DIM

BATCH_SIZE = 256
EPOCHS = 25
LR = 1e-3
EMA_MOMENTUM = 0.996
CHECKPOINT_PATH = "ijepa_encoder.pt"
LOG_PATH = "docs/ijepa_training_log.csv"


def main():
    device = torch.device("cpu")  # confirmed no CUDA on this machine, see SETUP_NOTES.md
    torch.manual_seed(0)

    print("Loading CIFAR-10 and building the JEPA-masked dataset...")
    raw_train = load_cifar10(train=True, download=True)
    masked_train = JEPAMaskedCIFAR10(raw_train)
    loader = DataLoader(masked_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # num_workers=0 deliberately: Windows uses "spawn" for multiprocessing workers, which
    # has caused friction before on this machine (see SETUP_NOTES.md) -- not worth the
    # risk for a CPU run that's already I/O-light (CIFAR-10 fits in memory easily).

    context_encoder = Encoder().to(device)
    target_encoder = make_target_encoder(context_encoder).to(device)
    predictor = Predictor().to(device)

    optimizer = torch.optim.Adam(
        list(context_encoder.parameters()) + list(predictor.parameters()), lr=LR
    )
    loss_fn = nn.MSELoss()

    os.makedirs("docs", exist_ok=True)
    log_rows = []

    print(f"Starting I-JEPA pretraining: {EPOCHS} epochs, batch size {BATCH_SIZE}, "
          f"{len(masked_train)} training images, device={device}")
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_loss_sum = 0.0
        epoch_std_sum = 0.0
        n_batches = 0

        for context_img, target_img in loader:
            context_img = context_img.to(device)
            target_img = target_img.to(device)

            sx = context_encoder(context_img)
            with torch.no_grad():
                sy = target_encoder(target_img)
            predicted_sy = predictor(sx)

            loss = loss_fn(predicted_sy, sy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update happens AFTER the optimizer step, so the target encoder moves
            # toward the context encoder's just-updated weights, not its pre-update ones.
            ema_update(target_encoder, context_encoder, momentum=EMA_MOMENTUM)

            epoch_loss_sum += loss.item()
            # Representation std, averaged across embedding dims, for THIS batch's context
            # representations -- the same collapse diagnostic used in Project D. Detached
            # from the graph since this is pure monitoring, not part of the loss.
            epoch_std_sum += sx.detach().std(dim=0).mean().item()
            n_batches += 1

        avg_loss = epoch_loss_sum / n_batches
        avg_std = epoch_std_sum / n_batches
        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1:2d}/{EPOCHS}  loss={avg_loss:.5f}  "
              f"repr_std={avg_std:.5f}  elapsed={elapsed:.0f}s")
        log_rows.append({"epoch": epoch + 1, "loss": avg_loss, "repr_std": avg_std, "elapsed_s": elapsed})

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "repr_std", "elapsed_s"])
        writer.writeheader()
        writer.writerows(log_rows)

    torch.save(context_encoder.state_dict(), CHECKPOINT_PATH)
    print(f"\nSaved trained context encoder to {CHECKPOINT_PATH}")
    print(f"Saved training log to {LOG_PATH}")
    print(
        "\nIf repr_std collapsed toward 0 over training, that's the same failure mode "
        "demonstrated in Project D, now showing up (or not) in this different "
        "architecture -- worth checking the log, not just assuming the asymmetric "
        "target prevented it just because the architecture is designed to."
    )


if __name__ == "__main__":
    main()
