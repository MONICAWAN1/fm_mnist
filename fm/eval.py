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
    sigma_list: tuple[float, ...] | None = None,
    bandwidth_mults: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
    biased: bool = False,
    max_n: int = 2000,
    seed: int = 0,
) -> float:
    """Mixture-of-RBF squared MMD between two sample sets.

    Mirrors torchcfm's `mix_rbf_mmd2` / `_mix_rbf_kernel` / `_mmd2`
    (references/conditional-flow-matching/runner/src/models/components/mmd.py):
    pool X and Y, build the full Gram matrix, sum an RBF kernel over several
    bandwidths, and form MMD^2 from the XX / YY / XY blocks. RBF is the
    *characteristic*-kernel variant; unlike their `linear_mmd2` (difference in
    means only) or `poly_mmd2` (low-order moments via a crude consecutive-pair
    estimator), it detects ANY distributional difference between the two clouds —
    no per-row pairing — which is exactly the generated-vs-real-test question.

    Cherry-picked for gene expression on this data:
      * bandwidth: median heuristic (sigma^2 = median pairwise sq-distance * mult,
        summed over `bandwidth_mults`) instead of their FIXED sigma_list=[0.01..100],
        which was tuned to their data scale and would mostly saturate/vanish on our
        log1p / PCA distances. Pass an explicit `sigma_list` to override it.
      * estimator: `biased=False` (unbiased U-statistic, diagonal dropped) by
        default — at our modest sample size (~589 cells) the biased V-statistic's
        diagonal term inflates MMD^2 by ~d/m. The unbiased value can dip slightly
        negative when the distributions match; that is expected, do not clip. Set
        `biased=True` to mirror their default (non-negative, lower variance — nicer
        for a per-step training curve).

    Generalises their equal-size assumption to allow n != m. Both sets are
    subsampled to `max_n` rows (seeded) for tractable O(n^2) Gram matrices.
    """
    import torch

    rng = np.random.default_rng(seed)
    if x.shape[0] > max_n:
        x = x[rng.choice(x.shape[0], max_n, replace=False)]
    if y.shape[0] > max_n:
        y = y[rng.choice(y.shape[0], max_n, replace=False)]

    X = torch.as_tensor(x, dtype=torch.float32)
    Y = torch.as_tensor(y, dtype=torch.float32)
    n, m = X.shape[0], Y.shape[0]

    # pooled Gram -> squared distances (their _mix_rbf_kernel trick)
    Z = torch.cat((X, Y), dim=0)
    ZZT = Z @ Z.t()
    diag = torch.diag(ZZT).unsqueeze(1)
    d2 = (diag - 2.0 * ZZT + diag.t()).clamp_min(0.0)  # ||z_i - z_j||^2

    if sigma_list is None:
        off = d2[~torch.eye(n + m, dtype=torch.bool)]  # off-diagonal pairwise sq-dists
        med = float(off.median())
        med = med if med > 0 else 1.0
        sigma2_list = [med * mult for mult in bandwidth_mults]  # sigma^2 per kernel
    else:
        sigma2_list = [float(s) ** 2 for s in sigma_list]

    K = torch.zeros_like(d2)
    for s2 in sigma2_list:
        K = K + torch.exp(-d2 / (2.0 * s2))  # gamma = 1/(2 sigma^2), kernels summed

    Kxx, Kyy, Kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    if biased:  # V-statistic: keep the diagonal
        mmd2 = Kxx.sum() / (n * n) + Kyy.sum() / (m * m) - 2.0 * Kxy.sum() / (n * m)
    else:  # U-statistic: drop self-similarity on the diagonal
        sum_xx = Kxx.sum() - torch.diagonal(Kxx).sum()
        sum_yy = Kyy.sum() - torch.diagonal(Kyy).sum()
        mmd2 = sum_xx / (n * (n - 1)) + sum_yy / (m * (m - 1)) - 2.0 * Kxy.sum() / (n * m)
    return float(mmd2)


def ot_distance(
    x: np.ndarray,
    y: np.ndarray,
    method: str | None = "exact",
    reg: float = 0.05,
    power: int = 2,
    max_n: int = 4000,
    seed: int = 0,
) -> float:
    """Wasserstein-`power` distance between two sample sets (Euclidean ground cost).

    Mirrors torchcfm's `wasserstein` helper
    (references/conditional-flow-matching/torchcfm/optimal_transport.py): uniform
    marginals, cost M = ||x_i - y_j|| (squared when power==2), solved with POT's
    exact EMD (`emd2`) or entropic Sinkhorn (`partial(sinkhorn2, reg=reg)`), with a
    high `numItermax` so the LP/iteration doesn't bail early. For power==2 the sqrt
    is taken, so the return is the true W2 distance (not W2^2).

    Pairing-free and a true metric on distributions — the OT cost OT-CFM's coupling
    approximates at training time. Sets are subsampled to `max_n` rows (seeded) so
    the solve stays tractable in high dimension; the spatial test set (~589 cells)
    is well under this, so no subsampling happens there.
    """
    import ot as pot
    import torch

    assert power == 1 or power == 2
    rng = np.random.default_rng(seed)
    if x.shape[0] > max_n:
        x = x[rng.choice(x.shape[0], max_n, replace=False)]
    if y.shape[0] > max_n:
        y = y[rng.choice(y.shape[0], max_n, replace=False)]

    # ot_fn takes (a, b, M): marginals a, b and cost matrix M
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        from functools import partial

        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x.shape[0]), pot.unif(y.shape[0])
    M = torch.cdist(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.cpu().numpy(), numItermax=int(1e7))
    if power == 2:
        ret = np.sqrt(ret)
    return float(ret)


def compute_trajectory_distances(
    pred_list: list[np.ndarray],
    true_list: list[np.ndarray],
    leaveout: int = -1,
    seed: int = 0,
) -> dict:
    """Per-transition distribution distances for the EB trajectory task.

    `pred_list[i]` is the model's predicted cloud at timepoint i+1 (obtained by
    integrating the learned field one unit of global time from the TRUE timepoint-i
    cells); `true_list[i]` is the real timepoint-(i+1) cloud. We score each
    transition exactly as torchcfm's `compute_distribution_distances` does — W1 and
    W2 via the same exact-EMD `wasserstein` (mirrored in `ot_distance`), plus a
    mixture-RBF MMD — then average over transitions.

    The headline interpolation metric is `t_out/*`: when `leaveout > 0`, the
    transition i = leaveout-1 predicts the held-out timepoint that training never
    paired, so its W1 (`t_out/ot_w1`) is the unbiased EMD the paper reports.
    """
    metrics = {}
    w1s, w2s, mmds = [], [], []
    for i, (a, b) in enumerate(zip(pred_list, true_list)):
        w1 = ot_distance(a, b, power=1, seed=seed)
        w2 = ot_distance(a, b, power=2, seed=seed)
        mmd = mmd2_rbf(a, b, seed=seed)
        tp = i + 1  # this transition predicts timepoint i+1
        metrics[f"eval/t{tp}/ot_w1"] = w1
        metrics[f"eval/t{tp}/ot_w2"] = w2
        metrics[f"eval/t{tp}/mmd2"] = mmd
        w1s.append(w1); w2s.append(w2); mmds.append(mmd)
        if leaveout > 0 and tp == leaveout:
            metrics["eval/t_out/ot_w1"] = w1  # the interpolation EMD (paper headline)
            metrics["eval/t_out/ot_w2"] = w2
            metrics["eval/t_out/mmd2"] = mmd
    metrics["eval/mean/ot_w1"] = float(np.mean(w1s))
    metrics["eval/mean/ot_w2"] = float(np.mean(w2s))
    metrics["eval/mean/mmd2"] = float(np.mean(mmds))
    return metrics


def trajectory_overlay(
    out: Path,
    timepoint_data: list[np.ndarray],
    pred_heldout: np.ndarray | None = None,
    leaveout: int = -1,
) -> None:
    """PC1/PC2 scatter of real cells coloured by timepoint, predicted held-out overlaid.

    Everything is the model-space (whitened PCA) data the EMD is computed in, so the
    picture is faithful to what the metric sees. `pred_heldout` (model's prediction at
    the held-out timepoint) is drawn as black x to eyeball the interpolation.
    """
    cmap = plt.get_cmap("viridis", max(len(timepoint_data), 1))
    fig, ax = plt.subplots(figsize=(7, 6))
    for t, X in enumerate(timepoint_data):
        tag = f"t{t}" + (" (held out)" if t == leaveout else "")
        ax.scatter(X[:, 0], X[:, 1], s=6, alpha=0.5, color=cmap(t), label=tag, linewidths=0)
    if pred_heldout is not None:
        ax.scatter(pred_heldout[:, 0], pred_heldout[:, 1], s=14, alpha=0.7,
                   color="black", marker="x", linewidths=0.6,
                   label=f"predicted t{leaveout}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("EB trajectory (whitened PCA): real cells by timepoint + prediction")
    ax.legend(markerscale=2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out / "trajectory_overlay.png", dpi=130)
    plt.close(fig)


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
    # R2 measures the difference between generated cell gene expression mean with 
    # aggregated real cells gene expressoin mean
    def r2(a, b):
        ss_res = float(((a - b) ** 2).sum())
        ss_tot = float(((a - a.mean()) ** 2).sum()) + 1e-12
        return 1.0 - ss_res / ss_tot

    cov_r, cov_g = np.cov(real_l, rowvar=False), np.cov(gen_l, rowvar=False)
    # gene-level diagnostics only; the distributional distances (MMD, Wasserstein)
    # are computed by run_evaluation in the model's whitened-PCA space, the same
    # space/way torchcfm scores them.
    metrics = {
        "eval/mean_r2": r2(rm, gm),
        "eval/var_r2": r2(rv, gv),
        "eval/cov_frobenius": float(np.linalg.norm(cov_r - cov_g)),
        "eval/cov_frobenius_rel": float(np.linalg.norm(cov_r - cov_g) / (np.linalg.norm(cov_r) + 1e-12)),
    }
    return metrics


def run_evaluation(
    out: Path,
    real_coords: np.ndarray,
    gen_coords: np.ndarray,
    real_counts: np.ndarray | None,
    gen_counts: np.ndarray | None,
    labels=None,
    seed: int = 0,
) -> dict:
    """Full evaluation for the whitened-PCA pipeline.

    `real_coords`/`gen_coords` are the model-space (whitened-PCA) arrays — the
    distributional distances are computed directly here, in the SAME space and the
    SAME way as torchcfm (`wasserstein` -> exact EMD with Euclidean cost; W1 and W2;
    plus a mixture-RBF MMD). `real_counts`/`gen_counts` are those coords inverted to
    gene counts, used for the gene-level diagnostics AND the UMAP (which is built in
    gene space). Pass them as None for a PCA-only dataset (e.g. the EB `.npz`, which
    ships only a PCA embedding with no recoverable counts): the gene-level
    diagnostics are skipped and the UMAP is built directly on the PCA coords.
    `labels` (per-real-cell label, e.g. cell type or collection timepoint) colours
    the UMAP; pass None to skip it.
    """
    import scanpy as sc

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pca_only = real_counts is None or gen_counts is None

    # gene-level diagnostics (per-gene moments, covariance) + figures — gene space only
    metrics = {} if pca_only else save_eval_figures(out, real_counts, gen_counts)

    # distributional distances in the model space, exactly as torchcfm scores them
    metrics["eval/ot_w1"] = ot_distance(real_coords, gen_coords, power=1, seed=seed)
    metrics["eval/ot_w2"] = ot_distance(real_coords, gen_coords, power=2, seed=seed)
    metrics["eval/mmd2"] = mmd2_rbf(real_coords, gen_coords, seed=seed)

    if pca_only:
        # PCA overlay directly in model space (PC1 vs PC2) for a quick visual check
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(real_coords[:, 0], real_coords[:, 1], s=6, alpha=0.5, label="real (test)")
        ax.scatter(gen_coords[:, 0], gen_coords[:, 1], s=6, alpha=0.5, label="generated")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
        ax.set_title("whitened-PCA overlay")
        fig.tight_layout(); fig.savefig(out / "pca_overlay.png", dpi=120); plt.close(fig)

    if labels is not None:
        if pca_only:
            # already PCA: UMAP directly on the whitened-PCA coords (no gene space).
            umap_overlay(out, real_coords, gen_coords, labels, seed=seed)
        else:
            # UMAP in GENE space, not the model's whitened-PCA space: per-PC whitening
            # flattens the cell-type structure. Embed from log1p gene counts via PCA-50
            # (the standard scanpy UMAP input) so the class clusters match prior versions.
            real_l = sc.pp.log1p(real_counts, copy=True)
            gen_l = sc.pp.log1p(gen_counts, copy=True)
            k = min(50, real_l.shape[1], max(real_l.shape[0] - 1, 1))
            pca = PCA(n_components=k, random_state=seed).fit(real_l)
            umap_overlay(out, pca.transform(real_l), pca.transform(gen_l), labels, seed=seed)

    return metrics
