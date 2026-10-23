"""
vicreg_loss.py -- the three loss terms that answer "how do gx and gy get learned
without collapsing?"

See PLAN.md for the full citation and reasoning. In short: given two views x and y of
the same underlying thing, two encoders sx = gx(x), sy = gy(y) need to be trained so
that sx and sy end up (1) predictive of each other AND (2) actually informative about
the input, rather than both just outputting a constant. This file implements that as
three separate, addable loss terms, each doing one specific job:

    invariance_loss  -- pulls sx and sy together           (does clause (2) alone)
    variance_loss    -- stops a representation collapsing  (part of clause (1))
                         to a single point
    covariance_loss  -- stops dimensions of a representation (the other part of (1))
                         from being redundant copies of each other

Keeping them as three separate functions (rather than one opaque "vicreg_loss"
function) is deliberate -- the whole point of this project is to see what each term
does on its own, including what breaks when one is left out. combined_vicreg_loss()
at the bottom just adds them with weights, and lets any weight be set to 0 for the
ablation in toy_demo.py.
"""

import torch


def invariance_loss(sx: torch.Tensor, sy: torch.Tensor) -> torch.Tensor:
    """Mean squared distance between the two representations.

    This is the ONLY term that pulls sx and sy toward each other. On its own, it is
    trivially minimized by both encoders learning to output the same constant vector
    for every input -- zero distance, zero loss, zero information learned. That trivial
    solution is exactly the "collapse" this whole project is about; the other two
    functions below exist specifically to make it not be the minimum anymore.
    """
    return torch.nn.functional.mse_loss(sx, sy)


def variance_loss(s: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Penalizes any representation DIMENSION whose spread (std, across the batch) is
    below target_std.

    s has shape (batch_size, repr_dim). For each of the repr_dim columns, compute the
    standard deviation of that column's values across the batch. If a dimension's
    values barely vary across different inputs, its std will be near 0 -- a direct,
    measurable sign that this dimension is on its way to carrying no information (in
    the extreme, total collapse, every dimension's std is exactly 0, since every input
    maps to the exact same vector).

    relu(target_std - std) is a one-sided "hinge": once std already reaches
    target_std, the penalty is exactly 0, so this term stops pushing once a dimension
    is "spread out enough" -- it's a floor, not a reward for infinite variance.
    """
    std = torch.sqrt(s.var(dim=0, unbiased=True) + eps)  # one std value per dimension
    return torch.relu(target_std - std).mean()


def covariance_loss(s: torch.Tensor) -> torch.Tensor:
    """Penalizes CORRELATION between different representation dimensions.

    variance_loss alone stops a representation from collapsing to a single point, but
    it says nothing about dimensions collapsing onto EACH OTHER -- e.g. dimension 2
    could just always equal 3x dimension 1. Every dimension would individually have
    healthy variance, yet the representation would still only really be using 1 degree
    of freedom, not repr_dim of them. That's a subtler, partial collapse.

    This term computes the (repr_dim x repr_dim) covariance matrix of s across the
    batch, then penalizes the squared OFF-diagonal entries (the diagonal is each
    dimension's own variance, already handled by variance_loss, so it's excluded here).
    Driving the off-diagonal entries toward 0 pushes different dimensions to carry
    genuinely different information rather than redundant copies of each other.
    """
    batch_size, repr_dim = s.shape
    s_centered = s - s.mean(dim=0, keepdim=True)
    cov = (s_centered.T @ s_centered) / (batch_size - 1)  # (repr_dim, repr_dim)

    off_diagonal_mask = ~torch.eye(repr_dim, dtype=torch.bool, device=s.device)
    off_diagonal_sq_sum = cov[off_diagonal_mask].pow(2).sum()

    # Normalize by repr_dim so this term's scale doesn't automatically grow just because
    # a bigger representation has more off-diagonal entries to sum over.
    return off_diagonal_sq_sum / repr_dim


def combined_vicreg_loss(
    sx: torch.Tensor,
    sy: torch.Tensor,
    inv_weight: float = 1.0,
    var_weight: float = 1.0,
    cov_weight: float = 1.0,
) -> dict:
    """Combines the three terms with weights, and reports each one separately.

    Setting var_weight=0 and cov_weight=0 reproduces the "invariance-only" collapse
    condition from PLAN.md's ablation; setting cov_weight=0 alone reproduces the
    "variance-only" condition. Returning a dict of the individual terms (not just the
    total) is deliberate -- the ablation experiment needs to see each term's value on
    its own, not just whichever combination happens to be switched on.
    """
    inv = invariance_loss(sx, sy)
    # Variance and covariance are each computed on BOTH views and averaged -- both
    # encoders need the anti-collapse pressure, not just one of them.
    var = 0.5 * (variance_loss(sx) + variance_loss(sy))
    cov = 0.5 * (covariance_loss(sx) + covariance_loss(sy))

    total = inv_weight * inv + var_weight * var + cov_weight * cov

    return {"total": total, "invariance": inv, "variance": var, "covariance": cov}
