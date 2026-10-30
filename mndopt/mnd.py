"""Matrix Normal Diagonal (MND) optimizer for PyTorch."""

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

__all__ = ["MND"]


def _validate_group(group: dict[str, Any]) -> None:
    for name in ("lr", "weight_decay", "eps", "adam_eps"):
        value = group[name]
        allow_zero = name in ("lr", "weight_decay")
        if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
            bound = "nonnegative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {bound}")
    if not 0 < group["beta"] < 1:
        raise ValueError("beta must be strictly between 0 and 1")
    if group["matrix"] is not None and type(group["matrix"]) is not bool:
        raise ValueError("matrix must be None, True, or False")
    betas = group["adam_betas"]
    if len(betas) != 2 or any(not 0 <= beta < 1 for beta in betas):
        raise ValueError("adam_betas must contain two values in [0, 1)")


class MND(Optimizer):
    """Matrix Normal Diagonal with decoupled weight decay.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 0.003,
        beta: float = 0.925,
        eps: float = 1e-9,
        weight_decay: float = 0.0,
        *,
        matrix: bool | None = None,
        adam_betas: tuple[float, float] = (0.95, 0.95),
        adam_eps: float = 1e-8,
    ) -> None:
        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            weight_decay=weight_decay,
            matrix=matrix,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
        )
        super().__init__(params, defaults)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        group = dict(self.defaults, **param_group)
        _validate_group(group)
        parameters = group["params"]
        if isinstance(parameters, Tensor):
            parameters = [parameters]
        elif isinstance(parameters, set):
            raise TypeError("Optimizer parameters must have a deterministic order")
        else:
            parameters = list(parameters)
        if group["matrix"] is True and any(p.ndim != 2 for p in parameters):
            raise ValueError("matrix=True requires two-dimensional parameters")
        super().add_param_group(dict(param_group, params=parameters))

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        """Take one step; return the optional closure loss."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if (
                    parameter.grad.layout != torch.strided
                    or parameter.layout != torch.strided
                ):
                    raise RuntimeError("MND requires dense parameters and gradients")
                if parameter.dtype not in (torch.float32, torch.float64):
                    raise RuntimeError("MND requires real float32 or float64 parameters")
                if not parameter.numel():
                    raise ValueError("MND requires nonempty parameters")
                matrix = group["matrix"]
                if matrix is None:
                    matrix = parameter.ndim == 2
                if matrix and parameter.ndim != 2:
                    raise ValueError("matrix=True requires two-dimensional parameters")
                gradient = parameter.grad
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["matrix"] = matrix
                    if matrix:
                        rows, cols = gradient.shape
                        state["momentum"] = torch.zeros_like(gradient)
                        state["row_factor"] = gradient.new_full((rows,), 1 / rows)
                        state["col_factor"] = gradient.new_full((cols,), 1 / cols)
                    else:
                        state["exp_avg"] = torch.zeros_like(gradient)
                        state["exp_avg_sq"] = torch.zeros_like(gradient)
                if state["matrix"] != matrix:
                    raise ValueError("Cannot change parameter routing after initialization")
                state["step"] += 1
                with torch.autocast(device_type=parameter.device.type, enabled=False):
                    if matrix:
                        direction = _mnd_update(
                            gradient,
                            state["momentum"],
                            state["row_factor"],
                            state["col_factor"],
                            group["beta"],
                            group["eps"],
                        )
                    else:
                        direction = _adamw_update(
                            gradient,
                            state["exp_avg"],
                            state["exp_avg_sq"],
                            group["adam_betas"],
                            group["adam_eps"],
                            state["step"],
                        )
                parameter.mul_(1 - lr * weight_decay)
                parameter.add_(direction, alpha=-lr)
        return loss


def _mnd_update(
    gradient: Tensor,
    momentum: Tensor,
    row: Tensor,
    col: Tensor,
    beta: float,
    eps: float,
) -> Tensor:
    """Update state; return the MND direction."""
    rows, cols = gradient.shape
    gain = beta * (1 - beta)
    residual = gradient - momentum
    momentum.mul_(beta).add_(gradient, alpha=1 - beta)
    squared = residual.square()
    new_row = beta * row + gain / cols * (squared @ col.clamp_min(eps).reciprocal())
    inv_row = new_row.clamp_min(eps).reciprocal()
    # Column prior: old row factor, new inverse.
    new_col = beta / rows * torch.dot(inv_row, row) * col
    new_col = new_col + gain / rows * (squared.T @ inv_row)
    # Balance traces; preserve covariance.
    scale = (new_col.sum().clamp_min(eps) / new_row.sum().clamp_min(eps)).sqrt()
    row.copy_(scale * new_row)
    col.copy_(new_col / scale)
    return (
        momentum
        * row.clamp_min(eps).rsqrt()[:, None]
        * col.clamp_min(eps).rsqrt()[None, :]
    )


def _adamw_update(
    gradient: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    betas: tuple[float, float],
    eps: float,
    step: int,
) -> Tensor:
    """Update moments; return the AdamW direction."""
    mean_decay, variance_decay = betas
    exp_avg.mul_(mean_decay).add_(gradient, alpha=1 - mean_decay)
    exp_avg_sq.mul_(variance_decay).addcmul_(gradient, gradient, value=1 - variance_decay)
    mean = exp_avg / (1 - mean_decay**step)
    variance = exp_avg_sq / (1 - variance_decay**step)
    return mean / (variance.clamp_min(0).sqrt() + eps)
