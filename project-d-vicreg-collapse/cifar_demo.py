"""
cifar_demo.py -- Project D's second stage (PLAN.md): the same invariance-only /
variance-only / full-VICReg ablation as toy_demo.py, but on real CIFAR-10 images
instead of synthetic 2D points, evaluated with the same linear-probing protocol used
in Project A (so the mechanism shown in the toy demo can be connected to a real,
comparable number).

WHAT COUNTS AS "TWO VIEWS" HERE: toy_demo.py used two independently-noised
observations of a hidden cluster center. For real images, the standard
SimCLR/VICReg-style substitute is two independently, randomly AUGMENTED crops of the
SAME photo (random crop + flip + color jitter) -- different pixels, same underlying
content, exactly matching the citation's "two inputs x and y" that should share
information worth predicting.

This file deliberately defines its OWN small encoder (ImageEncoder below) rather than
importing Project A's model.Encoder, even though the architecture is very similar --
keeping Project D self-contained (per the repo README, each project stands on its own
except where a project's own PLAN.md explicitly says otherwise) and avoiding a
dependency that would break if Project A's model.py changes later for I-JEPA-specific
reasons. The CIFAR LOADING code (not the encoder) still reuses Project A's existing
downloaded dataset folder, to avoid a redundant ~170MB re-download.

Run with:
    ..\\project-a-ijepa-cifar10\\.venv\\Scripts\\python.exe cifar_demo.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from vicreg_loss import combined_vicreg_loss, covariance_loss

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
EMBED_DIM = 64
TRAIN_SUBSET_SIZE = 5000  # kept small deliberately -- CPU training budget, same constraint named in Project A's PLAN.md
BATCH_SIZE = 128
N_STEPS = 400
LR = 1e-3

# Two independent random augmentations of the same photo stand in for "two views x, y
# of the same thing" -- the same role toy_demo.py's two independent noise draws played.
AUGMENT_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.4, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# No augmentation -- used only when extracting features for the linear probe, where we
# want a stable, repeatable representation of each real image, not a random crop of it.
EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


class TwoViewCIFAR10(Dataset):
    """Wraps a plain (untransformed) CIFAR-10 dataset and returns two independently
    augmented views of the same image. The label is read and thrown away here on
    purpose -- same self-supervised discipline as Project A's JEPAMaskedCIFAR10."""

    def __init__(self, base_dataset_no_transform):
        self.base = base_dataset_no_transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, _label = self.base[idx]
        return AUGMENT_TRANSFORM(image), AUGMENT_TRANSFORM(image)


class ImageEncoder(nn.Module):
    """A small CNN mapping a 32x32x3 CIFAR-10 image to an EMBED_DIM vector. Smaller
    (3 conv blocks, not 4) than Project A's Encoder, since this project trains THREE
    conditions back to back and is meant to stay fast, not to chase maximum accuracy."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 32x32 -> 16x16
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.project = nn.Linear(128, embed_dim)

    def forward(self, x):
        return self.project(self.conv(x).flatten(1))


def train_one_condition(name: str, var_weight: float, cov_weight: float, loader) -> ImageEncoder:
    encoder = ImageEncoder()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)

    step = 0
    while step < N_STEPS:
        for view1, view2 in loader:
            sx, sy = encoder(view1), encoder(view2)
            losses = combined_vicreg_loss(sx, sy, inv_weight=1.0, var_weight=var_weight, cov_weight=cov_weight)
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"  [{name}] step {step:4d}  inv={losses['invariance'].item():.4f}  "
                      f"var={losses['variance'].item():.4f}  cov={losses['covariance'].item():.4f}")
            step += 1
            if step >= N_STEPS:
                break

    return encoder


@torch.no_grad()
def extract_eval_embeddings(encoder: ImageEncoder, dataset) -> tuple:
    encoder.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    embeddings, labels = [], []
    for images, lbls in loader:
        embeddings.append(encoder(images))
        labels.append(lbls)
    return torch.cat(embeddings), torch.cat(labels)


def linear_probe_accuracy(train_emb, train_lbl, test_emb, test_lbl, embed_dim: int) -> float:
    probe = nn.Linear(embed_dim, 10)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    n = train_emb.shape[0]
    for _epoch in range(30):
        perm = torch.randperm(n)
        for start in range(0, n, 512):
            idx = perm[start:start + 512]
            loss = loss_fn(probe(train_emb[idx]), train_lbl[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        preds = probe(test_emb).argmax(dim=1)
        return (preds == test_lbl).float().mean().item()


if __name__ == "__main__":
    torch.manual_seed(0)

    print("Loading CIFAR-10 (reusing Project A's already-downloaded copy)...")
    raw_train = datasets.CIFAR10(root="../project-a-ijepa-cifar10/cifar_data", train=True, download=True)
    raw_test = datasets.CIFAR10(root="../project-a-ijepa-cifar10/cifar_data", train=False, download=True,
                                 transform=EVAL_TRANSFORM)

    # A fixed random subset for self-supervised TRAINING, kept small for CPU speed
    # (see TRAIN_SUBSET_SIZE comment above). The linear probe afterward is trained and
    # evaluated on the FULL train/test sets' embeddings -- that part is cheap (no CNN
    # forward needed per probe-epoch) so there's no reason to subset it too.
    subset_indices = torch.randperm(len(raw_train))[:TRAIN_SUBSET_SIZE].tolist()
    two_view_dataset = TwoViewCIFAR10(Subset(raw_train, subset_indices))
    loader = DataLoader(two_view_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Separate, eval-transformed (non-augmented) copies for linear-probe feature extraction.
    probe_train_dataset = datasets.CIFAR10(root="../project-a-ijepa-cifar10/cifar_data", train=True, download=True,
                                            transform=EVAL_TRANSFORM)

    conditions = [
        ("invariance-only", 0.0, 0.0),
        ("variance-only", 1.0, 0.0),
        ("full VICReg", 1.0, 1.0),
    ]

    results = {}
    for name, var_w, cov_w in conditions:
        print(f"\nTraining condition: {name}")
        encoder = train_one_condition(name, var_w, cov_w, loader)

        with torch.no_grad():
            # Representation health check on one batch, same diagnostic as toy_demo.py.
            sample_imgs, _ = next(iter(DataLoader(probe_train_dataset, batch_size=512, shuffle=True)))
            sample_repr = encoder(sample_imgs)
            final_std = sample_repr.std(dim=0).mean().item()
            final_cov = covariance_loss(sample_repr).item()

        print(f"  Extracting embeddings and running linear probe for '{name}'...")
        train_emb, train_lbl = extract_eval_embeddings(encoder, probe_train_dataset)
        test_emb, test_lbl = extract_eval_embeddings(encoder, raw_test)
        accuracy = linear_probe_accuracy(train_emb, train_lbl, test_emb, test_lbl, EMBED_DIM)

        results[name] = {"std": final_std, "cov": final_cov, "accuracy": accuracy}
        print(f"  -> {name}: mean std={final_std:.4f}  off-diag cov={final_cov:.4f}  "
              f"linear-probe accuracy={accuracy * 100:.2f}%")

    print("\n" + "=" * 70)
    print("CIFAR-10 ABLATION RESULTS (chance level = 10.00%)")
    print("=" * 70)
    for name, r in results.items():
        print(f"  {name:18s}  std={r['std']:.4f}  cov={r['cov']:.4f}  accuracy={r['accuracy'] * 100:6.2f}%")
    print("=" * 70)

    with open("cifar_ablation_results.txt", "w") as f:
        f.write("CIFAR-10 VICReg ablation (chance level = 10.00%)\n")
        for name, r in results.items():
            f.write(f"{name}: std={r['std']:.4f}  cov={r['cov']:.4f}  accuracy={r['accuracy'] * 100:.2f}%\n")
    print("\nSaved cifar_ablation_results.txt")
