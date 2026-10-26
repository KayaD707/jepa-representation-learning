"""
contrastive_loss.py -- NT-Xent (Normalized Temperature-scaled Cross Entropy), the loss
behind SimCLR. This is the genuinely new piece of code in this project (see PLAN.md):
neither Project A (I-JEPA) nor Project D (VICReg) use negative examples at all, so
nothing in this portfolio yet implements the contrastive approach to avoiding
collapse.

THE CORE IDEA, contrasted with VICReg (Project D): VICReg prevents collapse by
explicitly regularizing representation variance and covariance -- no notion of
"different images" is used at all. NT-Xent takes a completely different route: collapse
is prevented implicitly, by making the loss explicitly reward representations of
DIFFERENT images being far apart, not just representations of the SAME image's two
views being close together. If every image collapsed to the same representation, every
"different image" pair would score as maximally similar too, which this loss directly
and heavily penalizes. This is why a contrastive method needs explicit negative
examples (other images in the batch) and a non-contrastive one (VICReg, I-JEPA) does
not -- it's two different answers to the same collapse problem named in Project D's
PLAN.md.
"""

import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """z1, z2: (batch_size, embed_dim) -- embeddings of two augmented views of the same
    batch of images, z1[i] and z2[i] being the two views of image i.

    Step by step:
    1. Stack both views into one (2*batch_size, embed_dim) tensor and L2-normalize each
       row -- after normalizing, a dot product between two rows IS their cosine
       similarity, which is what "temperature-scaled similarity" means here.
    2. Compute the full (2N, 2N) pairwise similarity matrix in one matrix multiply.
    3. For each row i, its POSITIVE is the other view of the same image (if i is in the
       first N rows, its positive is i+N, and vice versa) -- every other row is
       automatically a negative, with no explicit negative-sampling step needed, since
       "every other image in this batch" already serves as negatives for free.
    4. Treat this as an ordinary classification problem: row i's "correct class" is its
       positive index, and cross-entropy over the similarity row (divided by
       temperature) is exactly NT-Xent. This reframing is why one call to
       F.cross_entropy implements the whole loss -- no custom loop needed.
    """
    batch_size = z1.shape[0]
    device = z1.device

    z = torch.cat([z1, z2], dim=0)               # (2N, D)
    z = F.normalize(z, dim=1)                    # unit vectors -> dot product = cosine similarity
    similarity = (z @ z.T) / temperature         # (2N, 2N)

    # A row's similarity with ITSELF is always 1.0 (cosine sim of a vector with itself)
    # -- it must be excluded, or every row's trivially-easiest "positive" would be
    # itself, which teaches the model nothing.
    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    similarity = similarity.masked_fill(self_mask, float("-inf"))

    # Row i < N's positive partner is row i + N (the other view of the same image);
    # row i >= N's positive partner is row i - N. Concatenating these two ranges gives
    # the "correct class" index for every row, in order.
    positive_indices = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=device),
        torch.arange(0, batch_size, device=device),
    ])

    return F.cross_entropy(similarity, positive_indices)
