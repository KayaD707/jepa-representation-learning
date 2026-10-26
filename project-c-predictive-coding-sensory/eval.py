"""
eval.py -- Project C's evaluation protocol (PLAN.md): a position readout (sanity
check), then the real test -- does the model's PREDICTED representation, decoded back
to a position, beat the two honest baselines (last-position-repeated, constant-velocity
extrapolation), especially at bounce frames.

Run with (after train.py has produced checkpoints/):
    ..\\project-a-ijepa-cifar10\\.venv\\Scripts\\python.exe eval.py
"""

import os

import torch
import torch.nn as nn

from sim import BouncingBallDataset, find_bounce_timesteps
from recurrent_model import FrameEncoder, RecurrentPredictor, PositionReadout

N_TEST_SEQUENCES = 200
SEQ_LEN = 20
READOUT_EPOCHS = 100
READOUT_LR = 1e-2


@torch.no_grad()
def extract_frame_representations(encoder: FrameEncoder, frames: torch.Tensor) -> torch.Tensor:
    """frames: (n_sequences, seq_len, 1, H, W) -> (n_sequences, seq_len, embed_dim).
    Flattens sequence/time together for one batched forward pass, then reshapes back
    -- faster than looping per-sequence for something this cheap (pure CNN forward,
    no recurrence needed here)."""
    n, t, c, h, w = frames.shape
    flat = frames.reshape(n * t, c, h, w)
    reprs = encoder(flat)
    return reprs.reshape(n, t, -1)


def train_position_readout(encoder: FrameEncoder, dataset: BouncingBallDataset) -> PositionReadout:
    """Trains a linear regressor from frozen frame representations to true (x, y)
    position -- PLAN.md's step 1, the sanity check that the encoder captures position
    information at all."""
    encoder.eval()
    reprs = extract_frame_representations(encoder, dataset.frames)   # (n, t, embed_dim)
    positions = dataset.positions                                     # (n, t, 2)

    flat_reprs = reprs.reshape(-1, reprs.shape[-1])
    flat_positions = positions.reshape(-1, 2)

    readout = PositionReadout(embed_dim=reprs.shape[-1])
    optimizer = torch.optim.Adam(readout.parameters(), lr=READOUT_LR)
    loss_fn = nn.MSELoss()

    n = flat_reprs.shape[0]
    for _epoch in range(READOUT_EPOCHS):
        permutation = torch.randperm(n)
        for start in range(0, n, 512):
            idx = permutation[start:start + 512]
            loss = loss_fn(readout(flat_reprs[idx]), flat_positions[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return readout


@torch.no_grad()
def run_recurrent_predictions(context_encoder: FrameEncoder, recurrent: RecurrentPredictor,
                               frames: torch.Tensor) -> torch.Tensor:
    """Rolls the trained recurrent model forward over test sequences (no gradient) and
    returns the predicted NEXT-FRAME representation at every timestep.
    Shape: (n_sequences, seq_len - 1, embed_dim) -- one prediction per (frame, next
    frame) pair, same indexing convention as train.py's run_epoch."""
    context_encoder.eval()
    recurrent.eval()
    n, seq_len = frames.shape[0], frames.shape[1]
    hidden = recurrent.init_hidden(n)
    predictions = []
    for t in range(seq_len - 1):
        frame_repr = context_encoder(frames[:, t])
        hidden, predicted_next = recurrent.step(frame_repr, hidden)
        predictions.append(predicted_next)
    return torch.stack(predictions, dim=1)  # (n, seq_len-1, embed_dim)


def _last_repeated_and_constant_velocity(positions: torch.Tensor) -> dict:
    """Shared math for both baseline families below -- takes ANY (n, seq_len, 2)
    position-like tensor (exact ground truth, or readout-estimated) and returns the
    same two baseline predictions computed from it. Kept as one function so the two
    baseline families below can't accidentally compute the math differently."""
    last_repeated = positions[:, :-1]  # predicts t+1 using position at t, for t=0..seq_len-2

    constant_velocity = last_repeated.clone()
    displacement = positions[:, 1:-1] - positions[:, :-2]  # position[t] - position[t-1], for t=1..seq_len-2
    constant_velocity[:, 1:] = positions[:, 1:-1] + displacement  # undefined at t=0 -> falls back to last_repeated

    return {"last-position-repeated": last_repeated, "constant-velocity": constant_velocity}


def oracle_position_baselines(true_positions: torch.Tensor) -> dict:
    """Baselines computed on the EXACT ground-truth position -- an oracle comparison,
    included for reference only. NOT a fair comparison to the model: the model only
    ever sees rendered pixels and has to recover position through a lossy
    encoder + linear readout, while these baselines get the exact numbers directly.
    See PLAN.md's post-mortem for why this distinction matters and was missed at
    first."""
    return {f"[oracle] {name}": pred for name, pred in _last_repeated_and_constant_velocity(true_positions).items()}


def estimated_position_baselines(estimated_positions: torch.Tensor) -> dict:
    """The FAIR comparison: the exact same two baseline formulas, but computed on
    positions ESTIMATED from pixels via the same frozen encoder + linear readout the
    model itself uses -- so every method being compared here goes through the same
    lossy image -> representation -> position pipeline, and only the *prediction
    mechanism* (naive physics heuristic vs. the learned recurrent model) differs."""
    return _last_repeated_and_constant_velocity(estimated_positions)


def main():
    print(f"Generating {N_TEST_SEQUENCES} test sequences (different seed than training)...")
    test_dataset = BouncingBallDataset(n_sequences=N_TEST_SEQUENCES, seq_len=SEQ_LEN, seed=999)

    print("Loading trained checkpoints...")
    context_encoder = FrameEncoder()
    context_encoder.load_state_dict(torch.load("checkpoints/context_encoder.pt", map_location="cpu"))
    recurrent = RecurrentPredictor()
    recurrent.load_state_dict(torch.load("checkpoints/recurrent.pt", map_location="cpu"))

    print(f"Training position readout ({READOUT_EPOCHS} epochs) on the test set's own "
          f"frame representations...")
    # NOTE: trained on the SAME test set's frames purely as a decoding tool (mapping
    # representation -> position), not on labels the recurrent model ever saw during
    # its own training -- this is standard for a linear-probe-style readout. A held-out
    # third split would be more rigorous for a paper; stated here as a simplification
    # for this portfolio-scale project, not hidden.
    readout = train_position_readout(context_encoder, test_dataset)

    print("Running recurrent model forward over test sequences...")
    predicted_reprs = run_recurrent_predictions(context_encoder, recurrent, test_dataset.frames)

    with torch.no_grad():
        model_predicted_positions = readout(predicted_reprs)  # (n, seq_len-1, 2)

        # Estimated position at EVERY timestep, via the same encoder+readout pipeline
        # the model uses -- this is what makes estimated_position_baselines() a fair
        # comparison (see its docstring, and PLAN.md's post-mortem).
        all_frame_reprs = extract_frame_representations(context_encoder, test_dataset.frames)
        estimated_positions = readout(all_frame_reprs)  # (n, seq_len, 2)

        # Sanity check, reported alongside the results: how good is the readout itself,
        # decoding the TRUE current frame's representation? This number is the ceiling
        # any image-based method here could possibly reach -- if it's far from 0, that
        # itself explains a lot of the gap against the oracle baselines.
        readout_sanity_mse = torch.nn.functional.mse_loss(estimated_positions, test_dataset.positions).item()

    true_next_positions = test_dataset.positions[:, 1:]  # (n, seq_len-1, 2) -- ground truth for t+1

    baselines = oracle_position_baselines(test_dataset.positions)
    baselines.update(estimated_position_baselines(estimated_positions))
    baselines["model (predicted representation)"] = model_predicted_positions

    print(f"\nReadout sanity check -- MSE decoding TRUE frame representations to position: "
          f"{readout_sanity_mse:.6f}")
    print("(this is the accuracy ceiling for ANY image-based method here -- compare the")
    print(" oracle baselines below against THIS number, not against 0, to judge the gap fairly)")

    # Bounce flag per (sequence, predicted-timestep): did a bounce happen approaching
    # the frame being predicted (t+1)?
    bounce_flags = torch.stack([find_bounce_timesteps(test_dataset.positions[i]) for i in range(N_TEST_SEQUENCES)])
    bounce_flags_next = bounce_flags[:, 1:]  # align with true_next_positions' t+1 indexing

    print("\n" + "=" * 70)
    print("PROJECT C RESULTS -- mean squared position error (lower is better)")
    print("=" * 70)
    results_lines = []
    for name, predicted in baselines.items():
        error = (predicted - true_next_positions).pow(2).sum(dim=-1)  # (n, seq_len-1)
        overall_mse = error.mean().item()
        bounce_mse = error[bounce_flags_next].mean().item() if bounce_flags_next.any() else float("nan")
        non_bounce_mse = error[~bounce_flags_next].mean().item()
        line = (f"{name:32s}  overall={overall_mse:.6f}  "
                f"bounce={bounce_mse:.6f}  non-bounce={non_bounce_mse:.6f}")
        print(" " + line)
        results_lines.append(line)
    print("=" * 70)
    print(f"\n({bounce_flags_next.sum().item()} bounce timesteps out of {bounce_flags_next.numel()} total "
          f"in the test set)")

    os.makedirs("docs", exist_ok=True)
    with open("docs/results.txt", "w") as f:
        f.write("Project C: mean squared position error (lower is better)\n")
        f.write(f"Readout sanity check (MSE decoding TRUE representations to position): {readout_sanity_mse:.6f}\n")
        f.write("-- this is the accuracy ceiling for any image-based method here\n")
        f.write(f"({bounce_flags_next.sum().item()} bounce timesteps out of {bounce_flags_next.numel()} total)\n\n")
        for line in results_lines:
            f.write(line + "\n")
    print("\nSaved docs/results.txt")


if __name__ == "__main__":
    main()
