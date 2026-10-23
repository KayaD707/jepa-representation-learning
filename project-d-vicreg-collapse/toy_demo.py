"""
toy_demo.py -- make representation collapse (and its prevention) visible on data
simple enough to literally plot.

THE SETUP:

We invent a hidden "true" structure: 4 cluster centers arranged in a square. Each
input is one of these 4 clusters, observed through noise. x and y are two
INDEPENDENT noisy observations of the same cluster center -- this is the toy
stand-in for "two views of the same thing" (e.g. two augmentations of the same image).
Critically, the loss functions in vicreg_loss.py NEVER see which cluster an input
belongs to -- this is unsupervised, exactly like real VICReg/JEPA training. The
cluster id is only used afterward, to COLOR the plot, so a human can see whether the
learned representation accidentally recovered the hidden structure or not.

A single small encoder (shared weights for both x and y -- see PLAN.md's "Siamese"
note) is trained three times from three different random initializations, one per
loss condition:

    1. invariance-only   -- expected to collapse
    2. variance-only     -- expected to avoid collapse, but allow redundant dimensions
    3. full VICReg       -- expected to stay spread out AND separate the 4 clusters

Run this with the Project A venv's Python (same torch install, already confirmed
working on this machine):

    project-a-ijepa-cifar10\\.venv\\Scripts\\python.exe project-d-vicreg-collapse\\toy_demo.py
"""

import torch
import torch.nn as nn

from vicreg_loss import combined_vicreg_loss, variance_loss, covariance_loss

torch.manual_seed(0)

N_PER_CLUSTER = 200
REPR_DIM = 2  # kept at 2 so the learned representation can be plotted directly, no PCA needed
NOISE_STD = 0.25
N_STEPS = 500
LR = 0.01

# Four cluster centers at the corners of a square -- an arbitrary but clearly-separated
# hidden structure for the model to (hopefully) recover.
CLUSTER_CENTERS = torch.tensor([
    [2.0, 2.0],
    [2.0, -2.0],
    [-2.0, 2.0],
    [-2.0, -2.0],
])


def make_toy_batch():
    """Returns (x, y, cluster_ids). x and y are two independently-noised observations
    of the same cluster center -- the toy "two views of the same thing." cluster_ids
    is returned ONLY for plotting later; it is never passed into training."""
    cluster_ids = torch.randint(0, 4, (N_PER_CLUSTER * 4,))
    centers = CLUSTER_CENTERS[cluster_ids]
    x = centers + torch.randn_like(centers) * NOISE_STD
    y = centers + torch.randn_like(centers) * NOISE_STD
    return x, y, cluster_ids


class ToyEncoder(nn.Module):
    """A small nonlinear encoder, shared (same weights) for both x and y -- the
    "Siamese network" simplification of gx/gy noted in PLAN.md. Nonlinear (one
    hidden layer + ReLU) so the encoder genuinely has to learn something, rather than
    the task being solvable by a trivial linear map."""

    def __init__(self, in_dim=2, hidden_dim=32, repr_dim=REPR_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_one_condition(name: str, var_weight: float, cov_weight: float):
    """Trains a fresh encoder under one loss condition and returns the final
    representations (for plotting) plus the final variance/covariance numbers (for
    the printed summary) -- both computed on one held-out large batch."""
    encoder = ToyEncoder()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)

    for step in range(N_STEPS):
        x, y, _ = make_toy_batch()
        sx, sy = encoder(x), encoder(y)
        losses = combined_vicreg_loss(sx, sy, inv_weight=1.0, var_weight=var_weight, cov_weight=cov_weight)
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        if step % 100 == 0 or step == N_STEPS - 1:
            print(
                f"  [{name}] step {step:4d}  "
                f"inv={losses['invariance'].item():.4f}  "
                f"var={losses['variance'].item():.4f}  "
                f"cov={losses['covariance'].item():.4f}"
            )

    # Final evaluation batch -- larger, and representations kept for plotting.
    with torch.no_grad():
        x, y, cluster_ids = make_toy_batch()
        sx = encoder(x)
        final_std = sx.std(dim=0)  # per-dimension spread -- near 0 in both dims means collapse
        final_cov = covariance_loss(sx).item()

    return sx.numpy(), cluster_ids.numpy(), final_std.numpy(), final_cov


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = [
        ("invariance-only", 0.0, 0.0),
        ("variance-only", 1.0, 0.0),
        ("full VICReg", 1.0, 1.0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    summary_lines = []

    for ax, (name, var_w, cov_w) in zip(axes, conditions):
        print(f"\nTraining condition: {name}")
        reprs, cluster_ids, final_std, final_cov = train_one_condition(name, var_w, cov_w)

        scatter = ax.scatter(reprs[:, 0], reprs[:, 1], c=cluster_ids, cmap="tab10", s=8, alpha=0.6)
        ax.set_title(f"{name}\nstd={final_std.round(3)}  off-diag cov={final_cov:.4f}")
        ax.set_xlabel("representation dim 0")
        ax.set_ylabel("representation dim 1")

        summary_lines.append(
            f"{name:18s} final per-dim std = {final_std}  off-diagonal covariance = {final_cov:.4f}"
        )

    plt.tight_layout()
    plt.savefig("collapse_demo.png", dpi=120)

    print("\n" + "=" * 70)
    print("SUMMARY (also see collapse_demo.png):")
    for line in summary_lines:
        print("  " + line)
    print("=" * 70)
    print(
        "\nExpected pattern: 'invariance-only' should show near-zero std (all points "
        "bunched at one spot, regardless of true cluster color) -- that IS collapse, "
        "caught directly, not just asserted. 'full VICReg' should show four separated, "
        "color-coded clusters -- the representation has recovered the hidden structure "
        "without ever being told the cluster labels."
    )
