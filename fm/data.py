from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import make_moons
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset


def sample_prior(batch_size: int, shape=(2,), device: str | torch.device = "cpu") -> torch.Tensor:
    """p_prior = N(0, I). In Chapter 5 notation, this is often x0 ~ p_src."""
    return torch.randn(batch_size, *shape, device=device)

def get_mnist_loader(batch_size : int, root="./data", train=True, shuffle=True) -> torch.Tensor:
    "Load mnist data and sample data from mnist"
    # transform raw PIL data to torch tensor with values 0...1 and rescales pixels from [0, 1] to [-1, 1]
    tf = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Lambda(lambda x:x*2-1)
    ])
    dataset_train = datasets.MNIST(root=root, train=train, download=True, transform=tf,)
    return DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)

def sample_two_moons(batch_size: int, noise: float = 0.06, device: str | torch.device = "cpu") -> torch.Tensor:
    """Toy p_data = p_tgt. Returns centered/scaled two moons."""
    x, _ = make_moons(n_samples=batch_size, noise=noise)
    x = torch.tensor(x, dtype=torch.float32, device=device)
    x = x - x.mean(dim=0, keepdim=True)
    x = x / x.std(dim=0, keepdim=True)
    return x


def _split_indices(n_obs: int, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic train/test row split. Returns (train_idx, test_idx).

    Shared by `load_spatial_expression` so any caller passing the same
    (n_obs, test_frac, seed) recovers the identical test cells.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_obs)
    n_test = int(round(test_frac * n_obs))
    return perm[n_test:], perm[:n_test]


def load_spatial_expression(
    path: str,
    batch_size: int,
    target_sum: float | None = None,
    standardize: bool = True,
    test_frac: float = 0.1,
    seed: int = 0,
    shuffle: bool = True,
    label_keys: list[str] | None = None,
) -> tuple[DataLoader, torch.Tensor, dict]:
    """Load an `.h5ad` of raw counts for unconditional OT-CFM over expression.

    Pipeline (scanpy): `normalize_total` -> `log1p` -> per-gene `scale` (z-score).
    We transport N(0, I) -> expression, so the z-score (fit on the TRAIN split
    only) puts the data on roughly the same scale as the Gaussian prior. The data
    point is one cell's 1122-D expression vector; spatial coords and metadata are
    intentionally ignored.

    Returns (train_loader, x_test, stats) where `stats` holds everything needed to
    invert the transform at generation time (see `invert_expression`). When
    `label_keys` is given, `stats["test_labels"]` is a DataFrame of those `.obs`
    columns for the test cells (row-aligned with `x_test`) — used by evaluation to
    colour the UMAP by cell type.
    """
    import scanpy as sc

    adata = sc.read_h5ad(path)
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()
    adata.X = adata.X.astype(np.float32)

    # effective normalization target (scanpy uses median cell depth when None)
    eff_target = (
        float(np.median(np.asarray(adata.X).sum(axis=1)))
        if target_sum is None
        else float(target_sum)
    )

    # 1) library-size normalize + 2) log1p. Both are per-cell / elementwise
    #    target_sum=None tells scanpy to normalize to the median cell depth.
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    # split original dataset so that gene stats are fit on TRAIN cells only
    train_idx, test_idx = _split_indices(adata.n_obs, test_frac, seed)
    adata_train, adata_test = adata[train_idx].copy(), adata[test_idx].copy()

    # 3) per-gene z-score. sc.pp.scale fits + applies on train and records the
    #    per-gene mean/std in .var; we reuse those stats for test and inversion.
    n_genes = adata.n_vars
    if standardize:
        sc.pp.scale(adata_train, zero_center=True)
        mean = adata_train.var["mean"].to_numpy(dtype=np.float32)
        std = adata_train.var["std"].to_numpy(dtype=np.float32)
        std = np.where(std == 0.0, 1.0, std)  # scanpy zeroes constant genes; avoid /0
        adata_test.X = (adata_test.X - mean) / std
    else:
        mean = np.zeros(n_genes, np.float32)
        std = np.ones(n_genes, np.float32)

    X_train = np.asarray(adata_train.X, dtype=np.float32)
    X_test = np.asarray(adata_test.X, dtype=np.float32)

    stats = {
        "gene_mean": mean.astype(np.float32),
        "gene_std": std.astype(np.float32),
        "target_sum": eff_target,
        "standardize": standardize,
        "var_names": list(map(str, adata.var_names)),
    }
    if label_keys is not None:
        # .obs labels for the test cells, row-aligned with X_test (reset_index so
        # positional indexing matches the tensor rows).
        stats["test_labels"] = adata_test.obs[list(label_keys)].reset_index(drop=True)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=0,
    )
    return train_loader, torch.from_numpy(X_test), stats


def invert_expression(x: np.ndarray, stats: dict) -> np.ndarray:
    """Map a continuous FM output back to count-like expression.

    Inverse of `load_spatial_expression`'s transform: undo z-score, undo log1p,
    clamp to non-negative (counts cannot be negative).
    """
    # denormalized and undo log1p
    x = np.asarray(x, dtype=np.float32)
    x = x * stats["gene_std"] + stats["gene_mean"]
    x = np.expm1(x)
    return np.clip(x, 0.0, None)


def sample_gaussian_mixture(batch_size: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """A simple multimodal p_data for visualizing transport."""
    centers = torch.tensor(
        [[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]],
        dtype=torch.float32,
        device=device,
    )
    idx = torch.randint(0, len(centers), (batch_size,), device=device)
    return centers[idx] + 0.25 * torch.randn(batch_size, 2, device=device)
