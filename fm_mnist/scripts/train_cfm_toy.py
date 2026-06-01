from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

from fm.data import sample_prior, sample_two_moons, sample_gaussian_mixture
from fm.flow_matching import conditional_flow_matching_loss, ot_conditional_flow_matching_loss
from fm.interpolants import Interpolant
from fm.networks import VelocityMLP
from fm.sampling import midpoint_sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CFM / OT-CFM on 2D toy data.")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--data", choices=["moons", "gmm"], default="moons")
    p.add_argument("--interpolant", choices=["linear", "trig", "vp_like"], default="linear")
    p.add_argument("--coupling", choices=["independent", "ot"], default="independent")
    p.add_argument("--ot_method", choices=["exact", "hungarian"], default="exact")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="outputs/cfm_toy")
    p.add_argument("--sample_n", type=int, default=2000)
    p.add_argument("--sample_steps", type=int, default=10)
    return p.parse_args()


def sample_data(kind: str, n: int, device: torch.device) -> torch.Tensor:
    if kind == "moons":
        return sample_two_moons(n, device=device)
    return sample_gaussian_mixture(n, device=device)


def save_plot(path: Path, prior: torch.Tensor, target: torch.Tensor, generated: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 3))
    for i, (title, pts) in enumerate([("prior", prior), ("target data", target), ("generated", generated)]):
        plt.subplot(1, 3, i + 1)
        pts = pts.detach().cpu()
        plt.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.6)
        plt.title(title)
        plt.axis("equal")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = VelocityMLP(dim=2).to(device)
    interpolant = Interpolant(args.interpolant)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    bar = trange(args.steps)
    for step in bar:
        x1 = sample_data(args.data, args.batch, device)
        x0 = sample_prior(args.batch, shape=(2,), device=device)
        if args.coupling == "ot":
            loss = ot_conditional_flow_matching_loss(model, x0, x1, interpolant, ot_method=args.ot_method)
        else:
            loss = conditional_flow_matching_loss(model, x0, x1, interpolant)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            bar.set_description(f"loss={loss.item():.4f}")

    model.eval()
    prior = sample_prior(args.sample_n, shape=(2,), device=device)
    target = sample_data(args.data, args.sample_n, device)
    generated = midpoint_sample(model, prior.clone(), steps=args.sample_steps)
    save_plot(out / "cfm_samples.png", prior, target, generated)
    torch.save(model.state_dict(), out / "velocity_mlp.pt")
    print(f"saved {out / 'cfm_samples.png'}")


if __name__ == "__main__":
    main()
