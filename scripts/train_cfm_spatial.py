from __future__ import annotations

import os

# On macOS, PyTorch ships its own libomp while numpy's OpenBLAS pulls in a
# second OpenMP runtime. With multiple OMP threads the two collide and segfault
# (exit 139) at larger batch sizes. Pin OMP to one thread BEFORE numpy/torch are
# imported so OpenBLAS picks it up. Override by exporting OMP_NUM_THREADS first.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import trange

from fm.data import sample_prior, load_spatial_pca, load_eb_velocity
from fm.flow_matching import conditional_flow_matching_loss, ot_conditional_flow_matching_loss
from fm.interpolants import Interpolant
from fm.networks import VelocityMLP

# Config keys that describe the model/data/training and must stay fixed across a
# resume (so the checkpoint's weights and the data pipeline remain consistent).
# Everything else (steps, save_every, out, resume, no_wandb) is a per-invocation
# control the CLI is free to change.
_RESUMABLE_CONFIG = (
    "data", "dim", "hidden", "batch", "lr", "interpolant",
    "coupling", "ot_method", "test_frac", "target_sum", "n_pcs", "seed",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train OT-CFM over spatial expression (with checkpointing).")
    p.add_argument("--data", type=str, default="data/adata_Zhuang_Zhuang-ABCA-1.001.h5ad")
    p.add_argument("--dim", type=int, default=0, help="model dim; set automatically to --n_pcs")
    p.add_argument("--n_pcs", type=int, default=100, help="PCA components (torchcfm paper uses 100)")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--interpolant", choices=["linear", "trig", "vp_like"], default="linear")
    p.add_argument("--coupling", choices=["independent", "ot"], default="ot")
    p.add_argument("--ot_method", choices=["exact", "hungarian"], default="exact")
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--target_sum", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="outputs/cfm_spatial")
    p.add_argument("--save_every", type=int, default=2000, help="checkpoint every N steps (0 -> only final)")
    p.add_argument("--resume", type=str, default=None, help="path to a checkpoint to resume from")
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def save_checkpoint(path: Path, model, opt, step: int, config: dict, stats: dict) -> None:
    """Persist everything needed to resume training or run evaluation later.

    Stores model + optimizer state, the step reached, the resolved config, and the
    preprocessing `stats` (so eval can invert the transform without recomputing).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "config": config,
            "stats": stats,
        },
        path,
    )


def main() -> None:
    args = parse_args()

    # resume: restore model/data/training config from the checkpoint 
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        ckpt_cfg = resume_state["config"]
        for k in _RESUMABLE_CONFIG:
            if k in ckpt_cfg:
                setattr(args, k, ckpt_cfg[k])
        print(f"resuming from {args.resume} @ step {resume_state['step']}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # OTPlanSampler uses np.random.choice for plan sampling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Run `python -m pip install wandb`, or pass --no_wandb."
            ) from exc
        wandb_run = wandb.init(project="fm-spatial", config=vars(args) | {"device": str(device)})

    # `.npz` -> Tong et al.'s EB data, which already ships a PCA embedding (no
    # counts to normalize/PCA); `.h5ad` -> raw-count slide, normalize+log1p+PCA.
    # Both return the same (train_loader, x_test, stats) so the loop is identical.
    if args.data.endswith(".npz"):
        train_loader, x_test, stats = load_eb_velocity(
            args.data,
            batch_size=args.batch,
            n_pcs=args.n_pcs,
            test_frac=args.test_frac,
            seed=args.seed,
        )
    else:
        train_loader, x_test, stats = load_spatial_pca(
            args.data,
            batch_size=args.batch,
            n_pcs=args.n_pcs,
            target_sum=args.target_sum,
            test_frac=args.test_frac,
            seed=args.seed,
        )
    # model dim = whitened-PCA dimensionality (n_pcs, capped by data)
    pca_dim = x_test.shape[1]
    if resume_state is not None and args.dim != pca_dim:
        raise ValueError(f"checkpoint dim={args.dim} != PCA dim={pca_dim}; data/config mismatch")
    args.dim = pca_dim
    print(f"train batches/epoch={len(train_loader)}  test cells={x_test.shape[0]}  PCA dim={args.dim}")

    model = VelocityMLP(dim=args.dim, hidden=args.hidden).to(device)
    interpolant = Interpolant(args.interpolant)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        opt.load_state_dict(resume_state["optimizer"])
        start_step = resume_state["step"]

    # config snapshot saved into every checkpoint (drop per-invocation controls)
    config = {k: v for k, v in vars(args).items() if k not in {"resume", "no_wandb"}}

    loader_iter = iter(train_loader)

    def next_batch() -> torch.Tensor:
        nonlocal loader_iter
        try:
            (x1,) = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            (x1,) = next(loader_iter)
        return x1

    if start_step >= args.steps:
        print(f"checkpoint already at step {start_step} >= --steps {args.steps}; nothing to train")
    bar = trange(start_step, args.steps, initial=start_step, total=args.steps)
    for step in bar:
        x1 = next_batch().to(device)
        x0 = sample_prior(x1.shape[0], shape=(args.dim,), device=device)
        if args.coupling == "ot":
            loss = ot_conditional_flow_matching_loss(model, x0, x1, interpolant, ot_method=args.ot_method)
        else:
            loss = conditional_flow_matching_loss(model, x0, x1, interpolant)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_value = loss.item()
        if wandb_run is not None:
            wandb_run.log({"train/loss": loss_value, "train/lr": opt.param_groups[0]["lr"]}, step=step)
        if step % 100 == 0:
            bar.set_description(f"loss={loss_value:.4f}")
        # periodic checkpoint (step + 1 = number of completed steps)
        done = step + 1
        if args.save_every and done % args.save_every == 0:
            save_checkpoint(out / f"ckpt_step{done:06d}.pt", model, opt, done, config, stats)
            save_checkpoint(out / "ckpt_last.pt", model, opt, done, config, stats)

    # ---- final checkpoint ----
    final_step = max(args.steps, start_step)
    save_checkpoint(out / "ckpt_final.pt", model, opt, final_step, config, stats)
    save_checkpoint(out / "ckpt_last.pt", model, opt, final_step, config, stats)
    print(f"saved checkpoints to {out} (final: ckpt_final.pt @ step {final_step})")
    print(f"evaluate with: python scripts/eval_cfm_spatial.py --checkpoint {out / 'ckpt_final.pt'}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
