"""
eval_linear_probe.py -- the evaluation protocol decided in PLAN.md, BEFORE any model
was trained: freeze an encoder, train ONLY a linear classifier on top of its frozen
representations, and report test accuracy. Run once per method (random-init,
autoencoder, I-JEPA) and compare.

WHY A LINEAR PROBE, SPECIFICALLY: a linear classifier has very little capacity of its
own to solve the task -- it can't learn new features, only draw a hyperplane through
whatever features the frozen encoder already produced. So if linear-probe accuracy is
high, the INFORMATION needed to tell classes apart must already have been present,
linearly accessible, in the encoder's representation before this script ever saw a
label. That is what "useful representation" is being used to mean throughout this
project (see PLAN.md).

This script is written to evaluate ALL THREE methods in one run, since they need to be
read off the same table to mean anything (PLAN.md's whole point is the comparison, not
any one number alone).

Run with (from project-a-ijepa-cifar10/, after both training scripts have produced
their checkpoints):
    .venv\\Scripts\\python.exe eval_linear_probe.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import load_cifar10
from model import Encoder

PROBE_EPOCHS = 30
PROBE_LR = 1e-2
BATCH_SIZE = 512


@torch.no_grad()
def extract_embeddings(encoder: Encoder, dataset) -> tuple:
    """Runs the (frozen) encoder over an entire dataset once, and returns all
    embeddings and labels as two plain tensors.

    Doing this ONCE per encoder, up front, rather than re-running the CNN forward pass
    on every linear-probe training epoch, is what makes linear-probe TRAINING itself
    extremely cheap afterward -- the probe only ever sees pre-computed vectors, never
    images, from this point on.
    """
    encoder.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_embeddings, all_labels = [], []
    for images, labels in loader:
        all_embeddings.append(encoder(images))
        all_labels.append(labels)
    return torch.cat(all_embeddings), torch.cat(all_labels)


def train_and_evaluate_probe(train_embeddings, train_labels, test_embeddings, test_labels, embed_dim: int) -> float:
    """Trains a single nn.Linear(embed_dim, 10) classifier by ordinary cross-entropy
    gradient descent on the frozen embeddings, then returns test-set accuracy.

    This is intentionally the simplest possible classifier -- the point of a linear
    probe is specifically to AVOID giving the evaluation extra modeling power that
    could compensate for a weak encoder.
    """
    probe = nn.Linear(embed_dim, 10)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    loss_fn = nn.CrossEntropyLoss()

    n_train = train_embeddings.shape[0]
    for epoch in range(PROBE_EPOCHS):
        permutation = torch.randperm(n_train)
        for start in range(0, n_train, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            logits = probe(train_embeddings[idx])
            loss = loss_fn(logits, train_labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        test_logits = probe(test_embeddings)
        predictions = test_logits.argmax(dim=1)
        accuracy = (predictions == test_labels).float().mean().item()
    return accuracy


def main():
    torch.manual_seed(0)

    print("Loading CIFAR-10 train/test splits (unmasked -- the probe needs to see the "
          "full image and its real label, same as any normal classification setup)...")
    train_dataset = load_cifar10(train=True, download=True)
    test_dataset = load_cifar10(train=False, download=True)

    methods = {}

    print("\nBuilding random-init encoder (baseline 1: never trained at all)...")
    random_encoder = Encoder()
    methods["random-init"] = random_encoder

    print("Loading autoencoder-pretrained encoder (baseline 2)...")
    autoencoder_encoder = Encoder()
    autoencoder_encoder.load_state_dict(torch.load("autoencoder_encoder.pt", map_location="cpu"))
    methods["autoencoder"] = autoencoder_encoder

    print("Loading I-JEPA-pretrained encoder (the method actually being tested)...")
    ijepa_encoder = Encoder()
    ijepa_encoder.load_state_dict(torch.load("ijepa_encoder.pt", map_location="cpu"))
    methods["i-jepa"] = ijepa_encoder

    results = {}
    for name, encoder in methods.items():
        print(f"\nExtracting embeddings for '{name}'...")
        train_emb, train_lbl = extract_embeddings(encoder, train_dataset)
        test_emb, test_lbl = extract_embeddings(encoder, test_dataset)

        print(f"Training linear probe for '{name}' ({PROBE_EPOCHS} epochs)...")
        accuracy = train_and_evaluate_probe(train_emb, train_lbl, test_emb, test_lbl, embed_dim=train_emb.shape[1])
        results[name] = accuracy
        print(f"  -> '{name}' linear-probe test accuracy: {accuracy * 100:.2f}%")

    print("\n" + "=" * 60)
    print("FINAL RESULTS (chance level for 10 balanced classes = 10.00%)")
    print("=" * 60)
    for name, accuracy in results.items():
        print(f"  {name:15s}  {accuracy * 100:6.2f}%")
    print("=" * 60)
    print(
        "\nMinimum bar (PLAN.md): i-jepa should clearly exceed random-init.\n"
        "Actual claim being tested (PLAN.md): i-jepa should exceed autoencoder.\n"
        "Stated in advance as a real possibility: autoencoder could win or tie at this "
        "small scale -- that would be a valid, reportable null result, not a failure."
    )

    with open("docs/linear_probe_results.txt", "w") as f:
        f.write("Linear probe test accuracy (chance level = 10.00%)\n")
        for name, accuracy in results.items():
            f.write(f"{name}: {accuracy * 100:.2f}%\n")


if __name__ == "__main__":
    main()
