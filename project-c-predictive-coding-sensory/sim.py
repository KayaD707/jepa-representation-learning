"""
sim.py -- the toy physics environment: one ball, bouncing elastically inside a square
box, rendered as a 32x32 grayscale frame per timestep. See PLAN.md for why this
specific environment (simple, but non-trivial exactly at bounces).

Positions and velocities are tracked in a continuous [0, 1] x [0, 1] coordinate space
(not pixels) -- this is the "ground truth" the whole evaluation protocol leans on,
free and exact because it comes from the simulator itself, not from a vision model
trying to recover it from pixels.
"""

import torch

IMAGE_SIZE = 32
BALL_RADIUS = 0.08   # in [0,1] coordinate units
DT = 0.05            # time step size -- small enough that motion between frames is smooth


def _render_frame(position: torch.Tensor) -> torch.Tensor:
    """Renders one (1, IMAGE_SIZE, IMAGE_SIZE) grayscale frame: 1.0 inside the ball,
    0.0 outside. Built with a coordinate grid and a distance check rather than drawing
    APIs, since this needs to run thousands of times per training run and stay fast
    and dependency-free (no PIL/cv2 needed)."""
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, IMAGE_SIZE), torch.linspace(0, 1, IMAGE_SIZE), indexing="ij"
    )
    dist_sq = (xs - position[0]) ** 2 + (ys - position[1]) ** 2
    frame = (dist_sq <= BALL_RADIUS ** 2).float()
    return frame.unsqueeze(0)  # (1, H, W)


def simulate_sequence(seq_len: int, generator: torch.Generator) -> tuple:
    """Simulates one sequence of `seq_len` frames with elastic wall bounces.

    Elastic bounce rule: whenever the ball's center would cross a wall (accounting for
    its radius), the corresponding velocity component is simply negated -- the
    standard, simplest "perfectly elastic, no energy loss" bounce model.

    Returns (frames, positions): frames is (seq_len, 1, H, W), positions is
    (seq_len, 2) -- the ground-truth (x, y) position at every timestep, in [0,1]
    coordinates.
    """
    # Random start position, kept away from the walls by at least one radius so the
    # ball starts fully inside the box.
    position = BALL_RADIUS + torch.rand(2, generator=generator) * (1 - 2 * BALL_RADIUS)
    # Random initial velocity, direction and speed both randomized.
    velocity = (torch.rand(2, generator=generator) * 2 - 1) * 0.6  # units per second, in [-0.6, 0.6]

    frames, positions = [], []
    for _ in range(seq_len):
        frames.append(_render_frame(position))
        positions.append(position.clone())

        position = position + velocity * DT
        # Bounce off each wall independently -- reflect position back in and flip that
        # axis's velocity sign, accounting for the ball's radius so it visually bounces
        # off the wall, not off the box's exact center-point boundary.
        for axis in (0, 1):
            if position[axis] < BALL_RADIUS:
                position[axis] = BALL_RADIUS + (BALL_RADIUS - position[axis])
                velocity[axis] = -velocity[axis]
            elif position[axis] > 1 - BALL_RADIUS:
                position[axis] = (1 - BALL_RADIUS) - (position[axis] - (1 - BALL_RADIUS))
                velocity[axis] = -velocity[axis]

    return torch.stack(frames), torch.stack(positions)


def find_bounce_timesteps(positions: torch.Tensor) -> torch.Tensor:
    """Returns a boolean tensor, shape (seq_len,), True at timestep t if a bounce
    happened BETWEEN t-1 and t (detected by the velocity direction flipping, inferred
    from a sign change in consecutive displacement vectors). Timestep 0 is always
    False (no previous step to compare against). Used only for the evaluation's
    per-bounce breakdown (PLAN.md) -- never seen by the model during training."""
    displacement = positions[1:] - positions[:-1]            # (seq_len-1, 2)
    sign_flipped = (displacement[1:] * displacement[:-1] < 0).any(dim=1)  # (seq_len-2,)
    is_bounce = torch.zeros(positions.shape[0], dtype=torch.bool)
    is_bounce[2:] = sign_flipped  # offset by 2: need two prior displacements to detect a flip
    return is_bounce


class BouncingBallDataset:
    """Pre-generates a fixed set of sequences at construction time (not lazily per
    __getitem__) so train/test sequences are reproducible across runs given the same
    seed, and so repeated epochs reuse the exact same data rather than regenerating
    (slightly) different random sequences every pass."""

    def __init__(self, n_sequences: int, seq_len: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.frames, self.positions = [], []
        for _ in range(n_sequences):
            frames, positions = simulate_sequence(seq_len, generator)
            self.frames.append(frames)
            self.positions.append(positions)
        self.frames = torch.stack(self.frames)        # (n_sequences, seq_len, 1, H, W)
        self.positions = torch.stack(self.positions)  # (n_sequences, seq_len, 2)

    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, idx):
        return self.frames[idx], self.positions[idx]


if __name__ == "__main__":
    # Visual sanity check -- same discipline as Project A's data.py: look at the actual
    # output before trusting the physics is right.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    generator = torch.Generator().manual_seed(0)
    frames, positions = simulate_sequence(seq_len=40, generator=generator)
    bounces = find_bounce_timesteps(positions)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    shown_steps = torch.linspace(0, 39, 16).long()
    for ax, t in zip(axes.flat, shown_steps):
        ax.imshow(frames[t, 0], cmap="gray", vmin=0, vmax=1)
        title = f"t={t.item()}"
        if bounces[t]:
            title += " BOUNCE"
        ax.set_title(title, fontsize=8, color="red" if bounces[t] else "black")
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("sim_sanity_check.png", dpi=120)
    print(f"Saved sim_sanity_check.png -- {bounces.sum().item()} bounce(s) detected out of 40 steps")
    print("Confirm visually: the ball should move smoothly and reverse direction at "
          "frames marked BOUNCE, staying fully inside the box at every frame.")
