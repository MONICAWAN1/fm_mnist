from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # macOS OpenMP guard (see train script)

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # headless: render figures to files
import numpy as np
import torch

from fm.data import load_eb_velocity_trajectory
from fm.eval import compute_trajectory_distances, trajectory_overlay
from fm.networks import VelocityMLP
from fm.sampling import euler_sample, midpoint_sample

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate an EB trajectory OT-CFM checkpoint (leave-one-out EMD)."
    )
    p.add_argument("--checkpoint", type=str, required=True, help="ckpt_*.pt from train_cfm_eb.py")
    p.add_argument("--out", type=str, default="outputs/cfm_eb_eval")
    p.add_argument("--sample_steps", type=int, default=100, help="Euler steps per unit of global time")
    p.add_argument("--solver", choices=["euler", "midpoint"], default="euler",
                   help="ODE solver (paper uses euler)")
    p.add_argument("--seed", type=int, default=0, help="seed for EMD subsampling reproducibility")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def _resolve_data_path(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    if (REPO / path).exists():
        return str(REPO / path)
    raise FileNotFoundError(f"data file not found: {path} (also tried {REPO / path})")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    leaveout = cfg.get("leaveout", -1)
    print(f"loaded {args.checkpoint} (step {ckpt['step']}, dim={cfg['dim']}, leaveout={leaveout})")

    model = VelocityMLP(dim=cfg["dim"], hidden=cfg["hidden"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # rebuild the SAME whitened-PCA per-timepoint data the model trained on
    timepoint_data, stats = load_eb_velocity_trajectory(
        _resolve_data_path(cfg["data"]), n_pcs=cfg["n_pcs"], whiten=cfg["whiten"],
    )
    n_times = stats["n_times"]
    t_scale = max(n_times - 1, 1)
    integrate = euler_sample if args.solver == "euler" else midpoint_sample

    # For each transition i -> i+1, predict the timepoint-(i+1) marginal by integrating
    # one unit of GLOBAL time from the TRUE timepoint-i cells (re-seeding from ground
    # truth each step, exactly as torchcfm's trajectory eval does). The held-out
    # transition (i = leaveout-1) is the unbiased interpolation EMD.
    preds, trues = [], []
    pred_heldout = None
    for i in range(n_times - 1):
        x0 = timepoint_data[i].to(device)
        with torch.no_grad():
            pred = integrate(model, x0.clone(), steps=args.sample_steps,
                             t0=float(i), t1=float(i + 1), t_scale=t_scale)
        pred_np = pred.cpu().numpy()
        preds.append(pred_np)
        trues.append(timepoint_data[i + 1].numpy())
        if leaveout > 0 and (i + 1) == leaveout:
            pred_heldout = pred_np

    metrics = compute_trajectory_distances(preds, trues, leaveout=leaveout, seed=args.seed)
    print("eval:", {k: round(v, 4) for k, v in metrics.items()})
    if leaveout > 0:
        print(f"\n>>> interpolation EMD (held-out t{leaveout}): "
              f"1-Wasserstein={metrics['eval/t_out/ot_w1']:.4f}  "
              f"2-Wasserstein={metrics['eval/t_out/ot_w2']:.4f}")
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # figure: real cells by timepoint (whitened PCA) + predicted held-out overlay
    trajectory_overlay(
        out, [d.numpy() for d in timepoint_data], pred_heldout=pred_heldout, leaveout=leaveout,
    )

    if args.wandb:
        import wandb

        run = wandb.init(project="fm-eb-trajectory", job_type="eval",
                         config=cfg | {"checkpoint": args.checkpoint})
        run.log(metrics)
        fpath = out / "trajectory_overlay.png"
        if fpath.exists():
            run.log({"eval/trajectory_overlay": wandb.Image(str(fpath))})
        run.finish()

    print(f"saved figures + metrics.json to {out}")


if __name__ == "__main__":
    main()
