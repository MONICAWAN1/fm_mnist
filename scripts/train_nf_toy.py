from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import torch
from tqdm import trange

from fm.data import sample_two_moons, sample_gaussian_mixture
from fm.normalizing_flow import RealNVP2D


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--data", choices=["moons", "gmm"], default="moons")
    p.add_argument("--out", type=str, default="outputs/nf")
    return p.parse_args()


def sample_data(kind: str, n: int, device: torch.device) -> torch.Tensor:
    return sample_two_moons(n, device=device) if kind == "moons" else sample_gaussian_mixture(n, device=device)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = RealNVP2D().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    bar = trange(args.steps)
    for step in bar:
        x = sample_data(args.data, args.batch, device=device)
        loss = -model.log_prob(x).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            bar.set_description(f"nll={loss.item():.4f}")

    target = sample_data(args.data, 2000, device=device).detach().cpu()
    samples = model.sample(2000, device=device).detach().cpu()
    plt.figure(figsize=(6, 3))
    for i, (title, pts) in enumerate([("target data", target), ("NF samples", samples)]):
        plt.subplot(1, 2, i + 1)
        plt.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.6)
        plt.title(title)
        plt.axis("equal")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out / "nf_samples.png", dpi=180)
    torch.save(model.state_dict(), out / "realnvp2d.pt")
    print(f"saved {out / 'nf_samples.png'}")


if __name__ == "__main__":
    main()
