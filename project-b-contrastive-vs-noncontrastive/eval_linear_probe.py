"""
eval_linear_probe.py -- same protocol as Project A's eval_linear_probe.py (frozen
encoder, linear classifier on top, test accuracy), applied here to compare the
contrastive and non-contrastive checkpoints produced by train_contrastive.py and
train_noncontrastive.py. Also includes a random-init encoder as the same floor
baseline used throughout the portfolio.

Run with (from project-b-contrastive-vs-noncontrastive/, after both training scripts
have produced their checkpoints):
    ..\\project-a-ijepa-cifar10\\.venv\\Scripts\\python.exe eval_linear_probe.py
"""

import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project-a-ijepa-cifar10"))
from model import Encoder  # noqa: E402

from data_two_view import load_cifar10_eval

PROBE_EPOCHS = 30
PROBE_LR = 1e-2
BATCH_SIZE = 512


@torch.no_grad()
def extract_embeddings(encoder: Encoder, dataset):
    encoder.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    embeddings, labels = [], []
    for images, lbls in loader:
        embeddings.append(encoder(images))
        labels.append(lbls)
    return torch.cat(embeddings), torch.cat(labels)


def train_and_evaluate_probe(train_emb, train_lbl, test_emb, test_lbl, embed_dim: int) -> float:
    probe = nn.Linear(embed_dim, 10)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    loss_fn = nn.CrossEntropyLoss()
    n = train_emb.shape[0]
    for _epoch in range(PROBE_EPOCHS):
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            loss = loss_fn(probe(train_emb[idx]), train_lbl[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        preds = probe(test_emb).argmax(dim=1)
        return (preds == test_lbl).float().mean().item()


def main():
    torch.manual_seed(0)

    print("Loading CIFAR-10 train/test splits...")
    train_dataset = load_cifar10_eval(train=True)
    test_dataset = load_cifar10_eval(train=False)

    methods = {"random-init": Encoder()}

    contrastive_encoder = Encoder()
    contrastive_encoder.load_state_dict(torch.load("contrastive_encoder.pt", map_location="cpu"))
    methods["contrastive (NT-Xent)"] = contrastive_encoder

    noncontrastive_encoder = Encoder()
    noncontrastive_encoder.load_state_dict(torch.load("noncontrastive_encoder.pt", map_location="cpu"))
    methods["non-contrastive (VICReg)"] = noncontrastive_encoder

    results = {}
    for name, encoder in methods.items():
        print(f"\nExtracting embeddings for '{name}'...")
        train_emb, train_lbl = extract_embeddings(encoder, train_dataset)
        test_emb, test_lbl = extract_embeddings(encoder, test_dataset)

        print(f"Training linear probe for '{name}'...")
        accuracy = train_and_evaluate_probe(train_emb, train_lbl, test_emb, test_lbl, embed_dim=train_emb.shape[1])
        results[name] = accuracy
        print(f"  -> '{name}' linear-probe test accuracy: {accuracy * 100:.2f}%")

    print("\n" + "=" * 70)
    print("PROJECT B RESULTS (chance level = 10.00%)")
    print("=" * 70)
    for name, accuracy in results.items():
        print(f"  {name:28s}  {accuracy * 100:6.2f}%")
    print("=" * 70)

    os.makedirs("docs", exist_ok=True)
    with open("docs/linear_probe_results.txt", "w") as f:
        f.write("Project B linear probe test accuracy (chance level = 10.00%)\n")
        for name, accuracy in results.items():
            f.write(f"{name}: {accuracy * 100:.2f}%\n")
    print("\nSaved docs/linear_probe_results.txt")


if __name__ == "__main__":
    main()
