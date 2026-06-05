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
    t_scale: float = 1.0,
) -> torch.Tensor:
    """Solve dx/dt = v_phi(x, t/t_scale) from t0 to t1 using Euler steps.

    `t_scale` rescales the time the MODEL sees without changing the integration
    variable: the trajectory replication integrates in GLOBAL time t in [0, T-1] but
    feeds the network normalized time t/(T-1) in [0,1] (matching how it was trained;
    see `fm.flow_matching.trajectory_cfm_loss`). The default t_scale=1.0 is the plain
    [0,1] noise->data case and leaves behaviour unchanged.
    """
    dt = (t1 - t0) / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), (t0 + i * dt) / t_scale, device=x.device, dtype=x.dtype)
        x = x + dt * model(x, t)
    return x


@torch.no_grad()
def midpoint_sample(
    model: nn.Module,
    x: torch.Tensor,
    steps: int = 50,
    t0: float = 0.0,
    t1: float = 1.0,
    t_scale: float = 1.0,
) -> torch.Tensor:
    """Second-order midpoint ODE solver for the learned flow.

    `t_scale` rescales the model's time input only (see `euler_sample`); default 1.0
    is the standard [0,1] case.
    """
    dt = (t1 - t0) / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), (t0 + i * dt) / t_scale, device=x.device, dtype=x.dtype)
        v = model(x, t)
        x_mid = x + 0.5 * dt * v
        t_mid = torch.full(
            (x.shape[0],), (t0 + (i + 0.5) * dt) / t_scale, device=x.device, dtype=x.dtype
        )
        x = x + dt * model(x_mid, t_mid)
    return x
