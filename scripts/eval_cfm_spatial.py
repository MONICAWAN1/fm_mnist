from __future__ import annotations

import os

# Same macOS OpenMP guard as the training script (see its comment). Must precede
# numpy/torch import.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # headless CLI: render figures to files, no display
import numpy as np
import torch

from fm.data import sample_prior, load_spatial_pca, load_eb_velocity, invert_pca_expression
from fm.eval import run_evaluation
from fm.networks import VelocityMLP
from fm.sampling import midpoint_sample

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate + evaluate a trained OT-CFM checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True, help="path to a ckpt_*.pt from training")
    p.add_argument("--out", type=str, default="outputs/cfm_spatial_eval")
    p.add_argument("--sample_n", type=int, default=0, help="0 -> match #real test cells")
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--label_key", type=str, default="class", help=".obs column used to colour the UMAP")
    p.add_argument("--seed", type=int, default=0, help="seed for sampling/subsampling (reproducibility)")
    p.add_argument("--save_adata", action="store_true", help="also write the generated slide as .h5ad")
    p.add_argument("--wandb", action="store_true", help="log metrics + figures to Weights & Biases")
    return p.parse_args()


def _resolve_data_path(path: str) -> str:
    """Checkpoints store the data path as given at train time (often repo-relative)."""
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
    cfg, stats = ckpt["config"], ckpt["stats"]
    print(f"loaded {args.checkpoint} (trained to step {ckpt['step']}, dim={cfg['dim']})")

    # rebuild the trained model
    model = VelocityMLP(dim=cfg["dim"], hidden=cfg["hidden"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # recover the SAME held-out test split + labels the model never saw, using the
    # checkpoint's data config (deterministic given the seed). This refits the same
    # whitened-PCA the model trained on (n_pcs from the checkpoint).
    is_eb = cfg["data"].endswith(".npz")
    if is_eb:
        # EB ships a PCA embedding only; its label column is the collection
        # timepoint `sample_labels` (ignore --label_key, which is a gene-space .obs).
        eb_label_key = "sample_labels"
        _, x_test, stats_reload = load_eb_velocity(
            _resolve_data_path(cfg["data"]),
            batch_size=cfg["batch"],
            n_pcs=cfg["n_pcs"],
            test_frac=cfg["test_frac"],
            seed=cfg["seed"],
            label_keys=[eb_label_key],
        )
        labels = stats_reload["test_labels"][eb_label_key]
        print(f"test cells={x_test.shape[0]}  timepoint levels ({eb_label_key})={labels.nunique()}")
    else:
        _, x_test, stats_reload = load_spatial_pca(
            _resolve_data_path(cfg["data"]),
            batch_size=cfg["batch"],
            n_pcs=cfg["n_pcs"],
            target_sum=cfg["target_sum"],
            test_frac=cfg["test_frac"],
            seed=cfg["seed"],
            label_keys=[args.label_key],
        )
        labels = stats_reload["test_labels"][args.label_key]
        print(f"test cells={x_test.shape[0]}  cell-type levels ({args.label_key})={labels.nunique()}")

    # generation step: transport the prior to the model's whitened-PCA space
    n_gen = args.sample_n if args.sample_n > 0 else x_test.shape[0]
    torch.manual_seed(args.seed)
    prior = sample_prior(n_gen, shape=(cfg["dim"],), device=device)
    gen = midpoint_sample(model, prior.clone(), steps=args.sample_steps)

    # model-space coords (whitened PCA). The EB embedding has no recoverable gene
    # counts, so we score EMD/MMD purely in PCA space (counts=None skips the
    # gene-level diagnostics); the spatial slide additionally inverts to counts.
    gen_coords = gen.cpu().numpy()
    real_coords = x_test.numpy()
    if is_eb:
        gen_counts = real_counts = None
    else:
        gen_counts = invert_pca_expression(gen_coords, stats)
        real_counts = invert_pca_expression(real_coords, stats)

    # EMD/MMD scored in PCA space (their way); gene metrics from the inverted counts
    metrics = run_evaluation(out, real_coords, gen_coords, real_counts, gen_counts,
                             labels=labels, seed=args.seed)
    print("eval:", {k: round(v, 4) for k, v in metrics.items()})
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if args.save_adata and is_eb:
        print("warning: --save_adata is not supported for the EB PCA embedding (no gene counts); skipping")
    elif args.save_adata:
        try:
            import anndata as ad
            import pandas as pd

            adata_gen = ad.AnnData(
                X=gen_counts.astype(np.float32),
                var=pd.DataFrame(index=stats["var_names"]),
            )
            adata_gen.write_h5ad(out / "adata_generated.h5ad")
            print(f"wrote {out / 'adata_generated.h5ad'}")
        except Exception as exc:  
            print(f"warning: could not write adata_generated.h5ad ({exc})")

    if args.wandb:
        import wandb

        run = wandb.init(project="fm-spatial", job_type="eval", config=cfg | {"checkpoint": args.checkpoint})
        run.log(metrics)
        for fname in ("pca_overlay.png", "per_gene_moments.png", "marginal_hists.png", "umap_overlay.png"):
            fpath = out / fname
            if fpath.exists():
                run.log({f"eval/{fname[:-4]}": wandb.Image(str(fpath))})
        run.finish()

    print(f"saved figures + metrics.json to {out}")


if __name__ == "__main__":
    main()
