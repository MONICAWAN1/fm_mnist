from __future__ import annotations

from functools import lru_cache

import torch
from torchcfm.optimal_transport import OTPlanSampler


@lru_cache(maxsize=4)
def _get_sampler(method: str) -> OTPlanSampler:
    return OTPlanSampler(method=method if method != "hungarian" else "exact")


def ot_reorder(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: str = "exact",
    replace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder (x0, x1) along the batch dim by a minibatch OT plan.

    Cost is squared Euclidean on flat features; image shapes (B,C,H,W) are
    flattened internally by OTPlanSampler.get_map.
    method="exact" uses POT's EMD (Hungarian-style) with replacement-sampled
    pairs. method="hungarian" uses scipy.optimize.linear_sum_assignment via
    sample_plan_with_scipy, which preserves all x1 rows and is deterministic.
    """
    sampler = _get_sampler(method)
    if method == "hungarian":
        return sampler.sample_plan_with_scipy(x0, x1)
    return sampler.sample_plan(x0, x1, replace=replace)
