from __future__ import annotations

import os

# macOS OpenMP guard (see train_cfm_spatial.py): pin OMP to one thread BEFORE
# numpy/torch import so OpenBLAS + torch's libomp don't collide and segfault.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import trange

from fm.data import load_eb_velocity_trajectory, sample_eb_batch
from fm.flow_matching import trajectory_cfm_loss
from fm.interpolants import Interpolant
from fm.networks import VelocityMLP

# Config that must stay fixed across a resume (weights + data pipeline consistency).
_RESUMABLE_CONFIG = (
    "data", "dim", "hidden", "batch", "lr", "interpolant",
    "coupling", "ot_method", "n_pcs", "leaveout", "whiten", "seed",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replicate Tong et al.'s EB trajectory OT-CFM (leave-one-out interpolation)."
    )
    p.add_argument("--data", type=str, default="data/eb_velocity_v5.npz")
    p.add_argument("--dim", type=int, default=0, help="model dim; set automatically to --n_pcs")
    p.add_argument("--n_pcs", type=int, default=5,
                   help="PCA components (their trajectory config uses 5)")
    p.add_argument("--whiten", action="store_true", default=True,
                   help="per-PC StandardScaler whitening (paper default True)")
    p.add_argument("--no_whiten", dest="whiten", action="store_false")
    p.add_argument("--hidden", type=int, default=64,
                   help="MLP width (their VelocityNet uses 64)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--interpolant", choices=["linear", "trig", "vp_like"], default="linear")
    p.add_argument("--coupling", choices=["independent", "ot"], default="ot")
    p.add_argument("--ot_method", choices=["exact", "hungarian"], default="exact")
    p.add_argument("--leaveout", type=int, default=-1,
                   help="intermediate timepoint to hold out for interpolation (e.g. 1, 2, 3); -1 = none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="outputs/cfm_eb")
    p.add_argument("--save_every", type=int, default=5000, help="checkpoint every N steps (0 -> only final)")
    p.add_argument("--resume", type=str, default=None, help="path to a checkpoint to resume from")
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def save_checkpoint(path: Path, model, opt, step: int, config: dict, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # drop bulky per-timepoint plotting arrays from the saved stats
    stats_slim = {k: v for k, v in stats.items() if k != "phate_by_t"}
    torch.save(
        {"model": model.state_dict(), "optimizer": opt.state_dict(),
         "step": step, "config": config, "stats": stats_slim},
        path,
    )


def main() -> None:
    args = parse_args()

    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        ckpt_cfg = resume_state["config"]
        for k in _RESUMABLE_CONFIG:
            if k in ckpt_cfg:
                setattr(args, k, ckpt_cfg[k])
        print(f"resuming from {args.resume} @ step {resume_state['step']}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # sample_eb_batch + OTPlanSampler both use np.random
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("wandb not installed; `pip install wandb` or pass --no_wandb.") from exc
        wandb_run = wandb.init(project="fm-eb-trajectory", config=vars(args) | {"device": str(device)})

    timepoint_data, stats = load_eb_velocity_trajectory(
        args.data, n_pcs=args.n_pcs, whiten=args.whiten,
    )
    n_times = stats["n_times"]
    pca_dim = stats["dim"]
    if resume_state is not None and args.dim != pca_dim:
        raise ValueError(f"checkpoint dim={args.dim} != PCA dim={pca_dim}; data/config mismatch")
    args.dim = pca_dim
    if not (0 < args.leaveout < n_times - 1) and args.leaveout != -1:
        # only interior timepoints can be interpolated (need a before and an after)
        raise ValueError(f"--leaveout must be an interior timepoint in 1..{n_times-2}, or -1")
    print(f"timepoints={n_times} sizes={[len(d) for d in timepoint_data]} dim={args.dim} "
          f"leaveout={args.leaveout}")

    model = VelocityMLP(dim=args.dim, hidden=args.hidden).to(device)
    interpolant = Interpolant(args.interpolant)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        opt.load_state_dict(resume_state["optimizer"])
        start_step = resume_state["step"]

    config = {k: v for k, v in vars(args).items() if k not in {"resume", "no_wandb"}}

    if start_step >= args.steps:
        print(f"checkpoint already at step {start_step} >= --steps {args.steps}; nothing to train")
    bar = trange(start_step, args.steps, initial=start_step, total=args.steps)
    for step in bar:
        batches = sample_eb_batch(timepoint_data, args.batch, device=device)
        loss = trajectory_cfm_loss(
            model, batches, interpolant,
            leaveout=args.leaveout, coupling=args.coupling, ot_method=args.ot_method,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_value = loss.item()
        if wandb_run is not None:
            wandb_run.log({"train/loss": loss_value, "train/lr": opt.param_groups[0]["lr"]}, step=step)
        if step % 100 == 0:
            bar.set_description(f"loss={loss_value:.4f}")
        done = step + 1
        if args.save_every and done % args.save_every == 0:
            save_checkpoint(out / f"ckpt_step{done:06d}.pt", model, opt, done, config, stats)
            save_checkpoint(out / "ckpt_last.pt", model, opt, done, config, stats)

    final_step = max(args.steps, start_step)
    save_checkpoint(out / "ckpt_final.pt", model, opt, final_step, config, stats)
    save_checkpoint(out / "ckpt_last.pt", model, opt, final_step, config, stats)
    print(f"saved checkpoints to {out} (final: ckpt_final.pt @ step {final_step})")
    print(f"evaluate with: python scripts/eval_cfm_eb.py --checkpoint {out / 'ckpt_final.pt'}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
