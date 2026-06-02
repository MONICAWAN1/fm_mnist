from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from tqdm import trange

from fm.data import sample_prior, get_mnist_loader
from fm.flow_matching import conditional_flow_matching_loss, ot_conditional_flow_matching_loss
from fm.interpolants import Interpolant
from fm.networks import VelocityUNet
from fm.sampling import midpoint_sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--interpolant", choices=["linear", "trig", "vp_like"], default="linear")
    p.add_argument("--coupling", choices=["independent", "ot"], default="independent")
    p.add_argument("--ot_method", choices=["exact", "hungarian"], default="exact")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="outputs/cfm_mnist")
    p.add_argument("--sample_n", type=int, default=16)
    p.add_argument("--sample_steps", type=int, default=10)
    return p.parse_args()


def save_image_grid(path: Path, generated: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = generated.detach().cpu()
    images = images.clamp(-1, 1) * 0.5 + 0.5
    grid = make_grid(images, nrow=generated.shape[0])
    save_image(grid, path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # OTPlanSampler uses np.random.choice for plan sampling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is not installed. Run `python -m pip install wandb` in your conda env."
        ) from exc
    wandb_run = wandb.init(
        project="fm-mnist",
        config=vars(args) | {"device": str(device)},
    )

    model = VelocityUNet(in_channels=1, base_channels=32).to(device)
    interpolant = Interpolant(args.interpolant)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    loader = get_mnist_loader(args.batch, root=args.root, train=True)
    loader_iter = iter(loader)

    def next_batch() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal loader_iter
        try:
            return next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            return next(loader_iter)

    bar = trange(args.steps)
    for step in bar:
        x1, _ = next_batch()
        x1 = x1.to(device)
        x0 = sample_prior(x1.shape[0], shape=(1, 28, 28), device=device)
        if args.coupling == "ot":
            loss = ot_conditional_flow_matching_loss(model, x0, x1, interpolant, ot_method=args.ot_method)
        else:
            loss = conditional_flow_matching_loss(model, x0, x1, interpolant)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_value = loss.item()
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss": loss_value,
                    "train/lr": opt.param_groups[0]["lr"],
                },
                step=step,
            )
        if step % 100 == 0:
            bar.set_description(f"loss={loss_value:.4f}")

    model.eval()
    prior = sample_prior(args.sample_n, shape=(1, 28, 28), device=device)
    # target, _ = next_batch()
    # target = target[: args.sample_n].to(device)
    generated = midpoint_sample(model, prior.clone(), steps=args.sample_steps)
    save_image_grid(out / "cfm_samples.png", generated=generated)
    torch.save(model.state_dict(), out / "velocity_mlp.pt")
    if wandb_run is not None:
        wandb_run.finish()
    print(f"saved {out / 'cfm_samples.png'}")


if __name__ == "__main__":
    main()
