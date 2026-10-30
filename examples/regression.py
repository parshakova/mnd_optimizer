"""Fit a linear model with MND."""

import argparse

import torch

from mndopt import MND


def run_regression(steps=300, device="cpu", seed=42):
    """Return initial and per-step losses; preserve CPU RNG state."""
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        inputs = torch.randn(64, 4).to(device)
        model = torch.nn.Linear(4, 1).to(device)
    targets = inputs @ torch.tensor([[0.3], [-0.7], [0.2], [0.5]], device=device) + 0.1
    optimizer = MND(model.parameters())

    def mse():
        return torch.nn.functional.mse_loss(model(inputs), targets)

    losses = [mse().item()]
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        mse().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            losses.append(mse().item())
    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    losses = run_regression(args.steps, args.device)
    print(f"MND: loss {losses[0]:.6g} -> {losses[-1]:.6g} after {args.steps} steps")
