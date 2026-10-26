"""
train_noncontrastive.py -- trains Project A's Encoder architecture using Project D's
VICReg loss, as the non-contrastive comparison point for train_contrastive.py. See
PLAN.md for why VICReg was chosen here over re-implementing BYOL.

Run with (from project-b-contrastive-vs-noncontrastive/):
    ..\\project-a-ijepa-cifar10\\.venv\\Scripts\\python.exe train_noncontrastive.py
"""

import csv
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project-a-ijepa-cifar10"))
from model import Encoder  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project-d-vicreg-collapse"))
from vicreg_loss import combined_vicreg_loss  # noqa: E402

from data_two_view import TwoViewCIFAR10, load_cifar10_raw

TRAIN_SUBSET_SIZE = 5000  # matched exactly with train_contrastive.py -- see PLAN.md
BATCH_SIZE = 256          # matched exactly with train_contrastive.py, even though VICReg
                          # itself isn't batch-size-sensitive the way NT-Xent is -- an
                          # equal training budget is what makes the comparison fair
N_STEPS = 500             # matched exactly with train_contrastive.py
LR = 1e-3
CHECKPOINT_PATH = "noncontrastive_encoder.pt"
LOG_PATH = "docs/noncontrastive_training_log.csv"


def main():
    torch.manual_seed(0)

    print("Loading CIFAR-10 two-view dataset (same subset construction as the contrastive run)...")
    raw_train = load_cifar10_raw(train=True)
    subset_indices = torch.randperm(len(raw_train))[:TRAIN_SUBSET_SIZE].tolist()
    dataset = TwoViewCIFAR10(Subset(raw_train, subset_indices))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    encoder = Encoder()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)

    os.makedirs("docs", exist_ok=True)
    log_rows = []
    start_time = time.time()
    step = 0

    print(f"Starting non-contrastive (VICReg) pretraining: {N_STEPS} steps, batch size {BATCH_SIZE}")
    while step < N_STEPS:
        for view1, view2 in loader:
            sx, sy = encoder(view1), encoder(view2)
            losses = combined_vicreg_loss(sx, sy, inv_weight=1.0, var_weight=1.0, cov_weight=1.0)

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            repr_std = torch.cat([sx, sy], dim=0).detach().std(dim=0).mean().item()

            if step % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  step {step:4d}/{N_STEPS}  total={losses['total'].item():.4f}  "
                      f"inv={losses['invariance'].item():.4f}  var={losses['variance'].item():.4f}  "
                      f"cov={losses['covariance'].item():.4f}  repr_std={repr_std:.4f}  "
                      f"elapsed={elapsed:.0f}s")
            log_rows.append({
                "step": step, "total": losses["total"].item(),
                "invariance": losses["invariance"].item(),
                "variance": losses["variance"].item(),
                "covariance": losses["covariance"].item(),
                "repr_std": repr_std,
            })

            step += 1
            if step >= N_STEPS:
                break

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "total", "invariance", "variance", "covariance", "repr_std"])
        writer.writeheader()
        writer.writerows(log_rows)

    torch.save(encoder.state_dict(), CHECKPOINT_PATH)
    print(f"\nSaved trained non-contrastive encoder to {CHECKPOINT_PATH}")
    print(f"Saved training log to {LOG_PATH}")


if __name__ == "__main__":
    main()
