from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    x: torch.Tensor,
    steps: int = 100,
    t0: float = 0.0,
    t1: float = 1.0,
) -> torch.Tensor:
    """Solve dx/dt = v_phi(x,t) from t0 to t1 using Euler steps."""
    dt = (t1 - t0) / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), t0 + i * dt, device=x.device, dtype=x.dtype)
        x = x + dt * model(x, t)
    return x


@torch.no_grad()
def midpoint_sample(
    model: nn.Module,
    x: torch.Tensor,
    steps: int = 50,
    t0: float = 0.0,
    t1: float = 1.0,
) -> torch.Tensor:
    """Second-order midpoint ODE solver for the learned flow."""
    dt = (t1 - t0) / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), t0 + i * dt, device=x.device, dtype=x.dtype)
        v = model(x, t)
        x_mid = x + 0.5 * dt * v
        t_mid = torch.full((x.shape[0],), t0 + (i + 0.5) * dt, device=x.device, dtype=x.dtype)
        x = x + dt * model(x_mid, t_mid)
    return x
