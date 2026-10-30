"""MND and AdamW numerical checks."""

import math
import random

import pytest
import torch

from mndopt import MND


SHAPES = [(2, 3), (3, 2), (3, 3), (1, 4), (4, 1)]


def scalar_step(gradient, mean, row, col, beta, eps):
    """Scalar reference for one MND step."""
    rows, cols = len(row), len(col)
    residual = [
        [gradient[i][j] - mean[i][j] for j in range(cols)]
        for i in range(rows)
    ]
    next_mean = [
        [beta * mean[i][j] + (1 - beta) * gradient[i][j] for j in range(cols)]
        for i in range(rows)
    ]
    gain = beta * (1 - beta)
    next_row = [
        beta * row[i] + gain / cols * sum(
            residual[i][j] ** 2 / max(col[j], eps) for j in range(cols)
        ) for i in range(rows)
    ]
    next_col = [
        beta / rows * col[j] * sum(row[i] / max(next_row[i], eps) for i in range(rows))
        + gain / rows * sum(residual[i][j] ** 2 / max(next_row[i], eps) for i in range(rows))
        for j in range(cols)
    ]
    scale = math.sqrt(max(sum(next_col), eps) / max(sum(next_row), eps))
    next_row = [value * scale for value in next_row]
    next_col = [value / scale for value in next_col]
    direction = [
        [next_mean[i][j] / math.sqrt(max(next_row[i], eps) * max(next_col[j], eps))
         for j in range(cols)] for i in range(rows)
    ]
    return direction, next_mean, next_row, next_col


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("eps", [1e-9, 0.5])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_mnd_matches_independent_equations(shape, eps, dtype):
    rng = random.Random(15)
    rows, cols = shape
    parameter = torch.nn.Parameter(torch.tensor(
        [[rng.gauss(0, 1) for _ in range(cols)] for _ in range(rows)], dtype=dtype,
    ))
    weights = parameter.detach().double().tolist()
    mean = [[0.0] * cols for _ in range(rows)]
    row, col = [1 / rows] * rows, [1 / cols] * cols
    optimizer = MND([parameter], lr=0.007, beta=0.8, eps=eps, weight_decay=0.03)
    tolerance = 2e-12 if dtype == torch.float64 else 2e-6
    for iteration in range(8):
        gradient = torch.tensor([
            [rng.gauss(0, 1) if iteration != 2 else 0 for _ in range(cols)]
            for _ in range(rows)
        ], dtype=dtype)
        direction, mean, row, col = scalar_step(
            gradient.double().tolist(), mean, row, col, 0.8, eps,
        )
        weights = [[(1 - 0.007 * 0.03) * weights[i][j] - 0.007 * direction[i][j]
                    for j in range(cols)] for i in range(rows)]
        parameter.grad = gradient
        optimizer.step()
        state = optimizer.state[parameter]
        for actual, expected in [(parameter, weights), (state["momentum"], mean),
                                 (state["row_factor"], row), (state["col_factor"], col)]:
            torch.testing.assert_close(actual, torch.tensor(expected, dtype=dtype),
                                       rtol=tolerance, atol=tolerance)
        assert state["step"] == iteration + 1


def test_column_prior_and_initialization():
    parameter = torch.nn.Parameter(torch.zeros(1, 2, dtype=torch.float64))
    parameter.grad = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    optimizer = MND([parameter], beta=0.9)
    optimizer.step()
    state = optimizer.state[parameter]
    covariance = torch.outer(state["row_factor"], state["col_factor"])
    torch.testing.assert_close(covariance, torch.tensor([[0.54, 0.81]], dtype=torch.float64))


@pytest.mark.parametrize("shape", [(), (4,), (2, 3), (2, 2, 2)])
@pytest.mark.parametrize("betas", [(0.95, 0.95), (0.9, 0.999), (0.0, 0.0)])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_adamw_routing_matches_pytorch(shape, betas, dtype):
    generator = torch.Generator().manual_seed(9)
    parameter = torch.nn.Parameter(torch.randn(shape, generator=generator, dtype=dtype))
    other = torch.nn.Parameter(parameter.detach().clone())
    optimizer = MND([parameter], matrix=False, lr=0.002, adam_betas=betas,
                    adam_eps=1e-6, weight_decay=0.02)
    reference = torch.optim.AdamW([other], lr=0.002, betas=betas, eps=1e-6,
                                 weight_decay=0.02, foreach=False)
    for _ in range(8):
        gradient = torch.randn(shape, generator=generator, dtype=dtype)
        parameter.grad, other.grad = gradient.clone(), gradient.clone()
        optimizer.step()
        reference.step()
        tolerance = 2e-13 if dtype == torch.float64 else 2e-6
        torch.testing.assert_close(parameter, other, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_noncontiguous_parameters_and_gradients(dtype):
    initial = torch.arange(12, dtype=dtype).reshape(3, 4).T / 10
    parameter = torch.nn.Parameter(initial.clone())
    other = torch.nn.Parameter(initial.contiguous())
    assert not parameter.is_contiguous()
    optimizer, reference = MND([parameter]), MND([other])
    for index in range(4):
        gradient = (torch.arange(12, dtype=dtype).reshape(3, 4) + index).T / 10
        parameter.grad, other.grad = gradient, gradient.contiguous()
        optimizer.step()
        reference.step()
    tolerance = 1e-14 if dtype == torch.float64 else 3e-7
    torch.testing.assert_close(parameter, other, rtol=tolerance, atol=tolerance)


def test_trace_balance_and_covariance_finiteness():
    parameter = torch.nn.Parameter(torch.zeros(3, 5, dtype=torch.float64))
    optimizer = MND([parameter])
    generator = torch.Generator().manual_seed(3)
    for _ in range(30):
        parameter.grad = torch.randn(3, 5, generator=generator, dtype=torch.float64)
        optimizer.step()
        state = optimizer.state[parameter]
        row, col = state["row_factor"], state["col_factor"]
        assert torch.isfinite(row).all() and (row > 0).all()
        assert torch.isfinite(col).all() and (col > 0).all()
        torch.testing.assert_close(row.sum(), col.sum(), rtol=1e-14, atol=1e-14)
