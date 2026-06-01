from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Maps scalar t in [0,1] to sinusoidal features."""

    def __init__(self, dim: int = 64):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(0, math.log(1000.0), half, device=t.device, dtype=t.dtype)
        )
        args = 2 * math.pi * t * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class VelocityMLP(nn.Module):
    """Time-conditioned velocity field v_phi(x_t, t): R^D x [0,1] -> R^D."""

    def __init__(self, dim: int = 2, hidden: int = 128, time_dim: int = 64):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(dim + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time_embed(t)
        return self.net(torch.cat([x, emb], dim=-1))


class _ResBlock(nn.Module):
    """Conv-GroupNorm-SiLU residual block with additive time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class VelocityUNet(nn.Module):
    """Time-conditioned velocity field v_phi(x_t, t) for 28x28 grayscale images.

    Same call signature as VelocityMLP: forward(x, t) -> velocity of identical shape.
    Architecture: 2 down-blocks (28->14->7), 1 mid-block, 2 up-blocks with skip
    connections. Time injected via additive per-channel bias in every ResBlock.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32, time_dim: int = 64):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        self.stem = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.down1 = _ResBlock(c1, c1, time_dim)
        self.down2 = _ResBlock(c1, c2, time_dim)
        self.mid = _ResBlock(c2, c3, time_dim)
        self.up2 = _ResBlock(c3 + c2, c2, time_dim)
        self.up1 = _ResBlock(c2 + c1, c1, time_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.Conv2d(c1, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)

        h0 = self.stem(x)
        s1 = self.down1(h0, t_emb)
        s2 = self.down2(F.avg_pool2d(s1, 2), t_emb)
        h = self.mid(F.avg_pool2d(s2, 2), t_emb)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up2(torch.cat([h, s2], dim=1), t_emb)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up1(torch.cat([h, s1], dim=1), t_emb)
        return self.out(h)
