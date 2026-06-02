from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import torch

from fm.data import sample_gaussian_mixture, sample_prior, sample_two_moons
from fm.interpolants import Interpolant


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize CFM interpolated samples x_t.")
    p.add_argument("--batch", type=int, default=2000)
    p.add_argument("--data", choices=["moons", "gmm"], default="moons")
    p.add_argument("--interpolant", choices=["linear", "trig", "vp_like"], default="linear")
    p.add_argument("--times", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--out", type=str, default="outputs/interpolants/xt_grid.png")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def sample_data(kind: str, n: int, device: torch.device) -> torch.Tensor:
    if kind == "moons":
        return sample_two_moons(n, device=device)
    return sample_gaussian_mixture(n, device=device)


def plot_xt_grid(
    path: Path,
    x0: torch.Tensor,
    x1: torch.Tensor,
    interpolant: Interpolant,
    times: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    cols = len(times)
    plt.figure(figsize=(2.6 * cols, 2.6))
    for i, t_value in enumerate(times):
        t = torch.full((x0.shape[0],), t_value, device=x0.device, dtype=x0.dtype)
        xt, ut = interpolant.sample_xt_and_velocity(x0, x1, t)
        pts = xt.detach().cpu()

        plt.subplot(1, cols, i + 1)
        plt.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.55)
        plt.title(f"t={t_value:g}")
        plt.axis("equal")
        plt.axis("off")

        if i == cols // 2:
            few = pts[:40]
            vel = ut[:40].detach().cpu()
            plt.quiver(
                few[:, 0],
                few[:, 1],
                vel[:, 0],
                vel[:, 1],
                angles="xy",
                scale_units="xy",
                scale=18,
                width=0.004,
                alpha=0.45,
            )

    plt.suptitle(f"{interpolant.name} interpolant: x_t from prior to target", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    x0 = sample_prior(args.batch, device=device)
    x1 = sample_data(args.data, args.batch, device=device)
    interpolant = Interpolant(args.interpolant)

    out = Path(args.out)
    plot_xt_grid(out, x0, x1, interpolant, args.times)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
