from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# NOTE: we deliberately do NOT call matplotlib.use("Agg") here. The functions only
# savefig + close, which works under any backend, so importing this module from a
# notebook leaves inline plotting intact. Headless CLIs should set Agg themselves.


def mmd2_rbf(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth_mults: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
    max_n: int = 2000,
    seed: int = 0,
) -> float:
    """Unbiased squared MMD between two sample sets under a mixture of RBF kernels.

    MMD is a kernel two-sample distance: ~0 iff the two empirical distributions
    match (in the RKHS), with NO per-row pairing — it compares the clouds, which
    is exactly the generated-vs-real-test question. The mixture kernel
    k = sum_s exp(-||a-b||^2 / (2 * sigma^2 * mult_s)) with sigma^2 set by the
    median heuristic avoids picking a single bandwidth. The unbiased U-statistic
    drops the kernel-matrix diagonal, so it can be slightly negative when the two
    distributions are identical (sampling noise) — that is expected, do not clip.

    Both sets are subsampled to `max_n` rows to keep the O(n^2) kernel matrices
    tractable; `seed` makes that subsample reproducible.
    """
    rng = np.random.default_rng(seed)
    if x.shape[0] > max_n:
        x = x[rng.choice(x.shape[0], max_n, replace=False)]
    if y.shape[0] > max_n:
        y = y[rng.choice(y.shape[0], max_n, replace=False)]

    def sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d2 = (a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2.0 * (a @ b.T)
        return np.maximum(d2, 0.0)  # clamp tiny negatives from float error

    dxx, dyy, dxy = sqdist(x, x), sqdist(y, y), sqdist(x, y)
    # median heuristic on the pooled off-diagonal squared distances
    med = float(np.median(np.concatenate([
        dxx[np.triu_indices_from(dxx, k=1)],
        dyy[np.triu_indices_from(dyy, k=1)],
        dxy.ravel(),
    ])))
    med = med if med > 0 else 1.0
    n, m = x.shape[0], y.shape[0]

    total = 0.0
    for mult in bandwidth_mults:
        denom = 2.0 * med * mult
        kxx, kyy, kxy = np.exp(-dxx / denom), np.exp(-dyy / denom), np.exp(-dxy / denom)
        np.fill_diagonal(kxx, 0.0)  # unbiased: exclude self-similarity
        np.fill_diagonal(kyy, 0.0)
        total += kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2.0 * kxy.mean()
    return float(total)


def ot_distance(
    x: np.ndarray,
    y: np.ndarray,
    max_n: int = 4000,
    seed: int = 0,
) -> tuple[float, float]:
    """Exact optimal-transport (Wasserstein-2) distance between two sample sets.

    Uniform marginals, squared-Euclidean ground cost, solved with POT's exact EMD.
    Returns (W2_squared, W2). Like MMD this is a pairing-free distributional
    distance, but it is a true metric on distributions (the OT cost the model's
    OT-CFM coupling approximates at training time). Sets are subsampled to `max_n`
    rows so the EMD LP stays tractable; `seed` makes that reproducible.
    """
    import ot as pot

    rng = np.random.default_rng(seed)
    if x.shape[0] > max_n:
        x = x[rng.choice(x.shape[0], max_n, replace=False)]
    if y.shape[0] > max_n:
        y = y[rng.choice(y.shape[0], max_n, replace=False)]
    a = np.ones(x.shape[0]) / x.shape[0]
    b = np.ones(y.shape[0]) / y.shape[0]
    M = pot.dist(x, y, metric="sqeuclidean")  # squared-Euclidean ground cost
    w2_sq = float(pot.emd2(a, b, M))  # exact EMD -> W2^2
    return w2_sq, float(np.sqrt(max(w2_sq, 0.0)))


def umap_overlay(
    out: Path,
    real_emb: np.ndarray,
    gen_emb: np.ndarray,
    labels,
    seed: int = 0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> None:
    """UMAP of real test cells coloured by cell type, with generated cells overlaid.

    `real_emb`/`gen_emb` are a shared low-dim space (e.g. PCA-50). UMAP is fit on
    the real cells; generated cells are projected onto the *same* embedding with
    `reducer.transform`, so their position is comparable. Real cells are coloured
    by `labels` (one category per cell type); generated cells are drawn as black x.
    """
    import umap  # umap-learn; lazy import keeps module load light

    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed
    )
    emb_real = reducer.fit_transform(real_emb)
    emb_gen = reducer.transform(gen_emb)

    import pandas as pd

    cats = pd.Categorical(np.asarray(labels).astype(str))
    uniq = list(cats.categories)
    cmap = plt.get_cmap("tab20", max(len(uniq), 1))

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, name in enumerate(uniq):
        m = cats.codes == i
        ax.scatter(emb_real[m, 0], emb_real[m, 1], s=6, alpha=0.7,
                   color=cmap(i), label=name, linewidths=0)
    ax.scatter(emb_gen[:, 0], emb_gen[:, 1], s=12, alpha=0.6,
               color="black", marker="x", label="generated", linewidths=0.6)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title("UMAP: real test cells (by class) + generated")
    ax.legend(markerscale=2, fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "umap_overlay.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_eval_figures(out: Path, real: np.ndarray, gen: np.ndarray) -> dict:
    """real/gen are count-like (N, G) arrays in the SAME space. Returns scalar metrics."""
    import scanpy as sc

    out.mkdir(parents=True, exist_ok=True)
    # work in log1p space for visualization (matches how the model was trained).
    # sc.pp.log1p on an ndarray == np.log1p (natural log, base=None); copy=True so
    # we don't mutate the caller's count arrays in place.
    real_l, gen_l = sc.pp.log1p(real, copy=True), sc.pp.log1p(gen, copy=True)

    # 1) PCA overlay: fit on real, project both
    pca = PCA(n_components=2).fit(real_l)
    r2d, g2d = pca.transform(real_l), pca.transform(gen_l)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(r2d[:, 0], r2d[:, 1], s=6, alpha=0.5, label="real (test)")
    ax.scatter(g2d[:, 0], g2d[:, 1], s=6, alpha=0.5, label="generated")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(); ax.set_title("PCA overlay")
    fig.tight_layout(); fig.savefig(out / "pca_overlay.png", dpi=120); plt.close(fig)

    # 2) per-gene mean & variance (real vs gen) should lie on y = x
    rm, gm = real_l.mean(0), gen_l.mean(0)
    rv, gv = real_l.var(0), gen_l.var(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for axx, r, g, name in ((axes[0], rm, gm, "mean"), (axes[1], rv, gv, "variance")):
        lo = float(min(r.min(), g.min())); hi = float(max(r.max(), g.max()))
        axx.scatter(r, g, s=5, alpha=0.4)
        axx.plot([lo, hi], [lo, hi], "k--", lw=1)
        axx.set_xlabel(f"real per-gene {name}"); axx.set_ylabel(f"gen per-gene {name}")
        axx.set_title(f"per-gene {name} (log1p)")
    fig.tight_layout(); fig.savefig(out / "per_gene_moments.png", dpi=120); plt.close(fig)

    # 3) marginal histograms for the highest-variance genes
    top = np.argsort(rv)[::-1][:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, gidx in zip(axes.ravel(), top):
        ax.hist(real_l[:, gidx], bins=40, alpha=0.5, density=True, label="real")
        ax.hist(gen_l[:, gidx], bins=40, alpha=0.5, density=True, label="gen")
        ax.set_title(f"gene #{gidx}"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "marginal_hists.png", dpi=120); plt.close(fig)

    # scalar metrics
    def r2(a, b):
        ss_res = float(((a - b) ** 2).sum())
        ss_tot = float(((a - a.mean()) ** 2).sum()) + 1e-12
        return 1.0 - ss_res / ss_tot

    cov_r, cov_g = np.cov(real_l, rowvar=False), np.cov(gen_l, rowvar=False)
    metrics = {
        "eval/mean_r2": r2(rm, gm),
        "eval/var_r2": r2(rv, gv),
        "eval/cov_frobenius": float(np.linalg.norm(cov_r - cov_g)),
        "eval/cov_frobenius_rel": float(np.linalg.norm(cov_r - cov_g) / (np.linalg.norm(cov_r) + 1e-12)),
        # kernel two-sample distance over the full expression vectors (~0 == match)
        "eval/mmd2_rbf": mmd2_rbf(real_l, gen_l),
    }
    return metrics


def run_evaluation(
    out: Path,
    real_counts: np.ndarray,
    gen_counts: np.ndarray,
    labels=None,
    n_pcs: int = 50,
    seed: int = 0,
) -> dict:
    """Full evaluation: moment/MMD metrics + figures, plus OT distance and a UMAP.

    `real_counts`/`gen_counts` are count-like (N, G). `labels` is a per-real-cell
    cell-type series (e.g. the 'class' column) used to colour the UMAP; pass None
    to skip the UMAP. OT distance and UMAP both operate in a PCA-`n_pcs` space fit
    on the real test cells (denoises before the high-dim comparison).
    """
    import scanpy as sc

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # existing moment-based metrics + figures (does its own log1p internally)
    metrics = save_eval_figures(out, real_counts, gen_counts)

    # shared PCA space for OT + UMAP, fit on real test cells
    real_l = sc.pp.log1p(real_counts, copy=True)
    gen_l = sc.pp.log1p(gen_counts, copy=True)
    k = min(n_pcs, real_l.shape[1], max(real_l.shape[0] - 1, 1))
    pca = PCA(n_components=k, random_state=seed).fit(real_l)
    real_p, gen_p = pca.transform(real_l), pca.transform(gen_l)

    w2_sq, w2 = ot_distance(real_p, gen_p, seed=seed)
    metrics["eval/ot_w2sq_pca"] = w2_sq
    metrics["eval/ot_w2_pca"] = w2

    if labels is not None:
        umap_overlay(out, real_p, gen_p, labels, seed=seed)

    return metrics
