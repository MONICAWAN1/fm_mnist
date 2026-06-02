from __future__ import annotations

import torch
from torch import nn


class AffineCoupling(nn.Module):
    """RealNVP-style affine coupling layer for Sec. 5.1 normalizing flows.

    Forward direction maps data x -> latent z and returns log|det dz/dx|.
    Inverse direction maps latent z -> data x and returns log|det dx/dz|.
    """

    def __init__(self, dim: int = 2, hidden: int = 128, flip: bool = False):
        super().__init__()
        if dim != 2:
            raise ValueError("This minimal example assumes dim=2.")
        self.flip = flip
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def _st(self, x_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s, t = self.net(x_a).chunk(2, dim=-1)
        s = 2.0 * torch.tanh(s)  # stabilize log-scale
        return s, t

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.flip:
            x = torch.flip(x, dims=[1])
        x_a, x_b = x[:, :1], x[:, 1:]
        s, t = self._st(x_a)
        z_b = (x_b - t) * torch.exp(-s)
        z = torch.cat([x_a, z_b], dim=-1)
        logdet = -s.squeeze(-1)
        if self.flip:
            z = torch.flip(z, dims=[1])
        return z, logdet

    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.flip:
            z = torch.flip(z, dims=[1])
        z_a, z_b = z[:, :1], z[:, 1:]
        s, t = self._st(z_a)
        x_b = z_b * torch.exp(s) + t
        x = torch.cat([z_a, x_b], dim=-1)
        logdet = s.squeeze(-1)
        if self.flip:
            x = torch.flip(x, dims=[1])
        return x, logdet


class RealNVP2D(nn.Module):
    """Tiny normalizing flow with exact log likelihood."""

    def __init__(self, n_layers: int = 6, hidden: int = 128):
        super().__init__()
        self.layers = nn.ModuleList([
            AffineCoupling(dim=2, hidden=hidden, flip=bool(i % 2)) for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        z = x
        for layer in self.layers:
            z, logdet = layer(z)
            total = total + logdet
        return z, total

    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        x = z
        for layer in reversed(self.layers):
            x, logdet = layer.inverse(x)
            total = total + logdet
        return x, total

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, logdet = self.forward(x)
        base = -0.5 * (z.pow(2).sum(dim=-1) + x.shape[1] * torch.log(torch.tensor(2 * torch.pi, device=x.device)))
        return base + logdet

    @torch.no_grad()
    def sample(self, n: int, device: str | torch.device = "cpu") -> torch.Tensor:
        z = torch.randn(n, 2, device=device)
        x, _ = self.inverse(z)
        return x
