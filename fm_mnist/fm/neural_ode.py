from __future__ import annotations

import torch
from torch import nn


class NeuralODEVectorField(nn.Module):
    """Small vector field for Sec. 5.1 NODE intuition."""

    def __init__(self, dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([x, t], dim=-1))


@torch.no_grad()
def integrate_ode(field: nn.Module, x: torch.Tensor, steps: int = 100) -> torch.Tensor:
    """Forward Euler integration of dx/dt = f_phi(x,t)."""
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
        x = x + dt * field(x, t)
    return x


def exact_trace_jacobian(field: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Pedagogical exact divergence Tr(d f / d x), expensive in high dimensions."""
    x = x.requires_grad_(True)
    y = field(x, t)
    trace = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for d in range(x.shape[1]):
        grad = torch.autograd.grad(y[:, d].sum(), x, create_graph=True)[0][:, d]
        trace = trace + grad
    return trace
