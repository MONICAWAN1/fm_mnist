from __future__ import annotations

import torch
from sklearn.datasets import make_moons
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


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


def sample_gaussian_mixture(batch_size: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """A simple multimodal p_data for visualizing transport."""
    centers = torch.tensor(
        [[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]],
        dtype=torch.float32,
        device=device,
    )
    idx = torch.randint(0, len(centers), (batch_size,), device=device)
    return centers[idx] + 0.25 * torch.randn(batch_size, 2, device=device)
