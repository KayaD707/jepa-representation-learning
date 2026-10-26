"""
train_contrastive.py -- trains Project A's Encoder architecture using the NT-Xent
(SimCLR-style) contrastive loss from contrastive_loss.py.

Run with (from project-b-contrastive-vs-noncontrastive/, using Project A's venv --
no new environment needed, same torch/torchvision install):
    ..\\project-a-ijepa-cifar10\\.venv\\Scripts\\python.exe train_contrastive.py
"""

import csv
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project-a-ijepa-cifar10"))
from model import Encoder, EMBED_DIM  # noqa: E402  (import after sys.path edit, intentional)

from contrastive_loss import nt_xent_loss
from data_two_view import TwoViewCIFAR10, load_cifar10_raw

TRAIN_SUBSET_SIZE = 5000  # matched exactly with train_noncontrastive.py -- see PLAN.md
BATCH_SIZE = 256          # larger batch matters more here than for VICReg: batch size
                          # directly sets how many negatives each image gets (see PLAN.md)
N_STEPS = 500             # matched exactly with train_noncontrastive.py
LR = 1e-3
TEMPERATURE = 0.5
CHECKPOINT_PATH = "contrastive_encoder.pt"
LOG_PATH = "docs/contrastive_training_log.csv"


def main():
    torch.manual_seed(0)

    print("Loading CIFAR-10 two-view dataset (subset, for CPU training speed)...")
    raw_train = load_cifar10_raw(train=True)
    subset_indices = torch.randperm(len(raw_train))[:TRAIN_SUBSET_SIZE].tolist()
    dataset = TwoViewCIFAR10(Subset(raw_train, subset_indices))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    # drop_last=True: NT-Xent's negative-index bookkeeping assumes every batch has the
    # same size; a smaller final batch would need special-casing for no real benefit.

    encoder = Encoder()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)

    os.makedirs("docs", exist_ok=True)
    log_rows = []
    start_time = time.time()
    step = 0

    print(f"Starting contrastive (NT-Xent) pretraining: {N_STEPS} steps, batch size {BATCH_SIZE}")
    while step < N_STEPS:
        for view1, view2 in loader:
            z1, z2 = encoder(view1), encoder(view2)
            loss = nt_xent_loss(z1, z2, temperature=TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            repr_std = torch.cat([z1, z2], dim=0).detach().std(dim=0).mean().item()

            if step % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  step {step:4d}/{N_STEPS}  loss={loss.item():.4f}  "
                      f"repr_std={repr_std:.4f}  elapsed={elapsed:.0f}s")
            log_rows.append({"step": step, "loss": loss.item(), "repr_std": repr_std})

            step += 1
            if step >= N_STEPS:
                break

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "repr_std"])
        writer.writeheader()
        writer.writerows(log_rows)

    torch.save(encoder.state_dict(), CHECKPOINT_PATH)
    print(f"\nSaved trained contrastive encoder to {CHECKPOINT_PATH}")
    print(f"Saved training log to {LOG_PATH}")


if __name__ == "__main__":
    main()
