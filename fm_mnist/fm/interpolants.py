from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class Interpolant:
    """x_t = a(t) x0 + b(t) x1, u_t = d/dt x_t."""

    name: str

    def coefficients(self, t: torch.Tensor, ndim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a, b, da/dt, db/dt with shape broadcastable to x."""
        # what's the shape of x? [B, 1, 28, 28]
        if t.ndim == 1:
            t = t.view(t.shape[0], *([1] * (ndim - 1))) 
            # torch.view: return a new tensor with new shape
        if self.name in {"linear", "rectified", "canonical_affine"}:
            # Chapter 5 canonical affine / RF: a_t = 1 - t, b_t = t
            a = 1.0 - t
            b = t
            da = -torch.ones_like(t)
            db = torch.ones_like(t)
        elif self.name == "trig":
            # Albergo-style trigonometric interpolant in the textbook table.
            a = torch.cos(0.5 * math.pi * t)
            b = torch.sin(0.5 * math.pi * t)
            da = -0.5 * math.pi * torch.sin(0.5 * math.pi * t)
            db = 0.5 * math.pi * torch.cos(0.5 * math.pi * t)
        elif self.name == "vp_like":
            # Simple VP-shaped schedule written in FM convention.
            b = torch.sin(0.5 * math.pi * t)
            a = torch.sqrt(torch.clamp(1.0 - b**2, min=1e-8))
            db = 0.5 * math.pi * torch.cos(0.5 * math.pi * t)
            da = -(b * db) / a.clamp_min(1e-8)
        else:
            raise ValueError(f"unknown interpolant: {self.name}")
        return a, b, da, db

    def sample_xt_and_velocity(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a, b, da, db = self.coefficients(t, ndim=x0.ndim)
        xt = a * x0 + b * x1
        ut = da * x0 + db * x1
        return xt, ut
