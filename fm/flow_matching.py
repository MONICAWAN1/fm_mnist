from __future__ import annotations

import torch
from torch import nn

from .interpolants import Interpolant
from .ot import ot_reorder


def conditional_flow_matching_loss(
    model: nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    interpolant: Interpolant,
) -> torch.Tensor:
    """LCFM-style simulation-free regression.

    Chapter 5 notation:
      x0 ~ p_src (often p_prior)
      x1 ~ p_tgt (often p_data)
      x_t = a_t x0 + b_t x1
      u_t = d/dt x_t = a'_t x0 + b'_t x1
      min_phi E || v_phi(x_t,t) - u_t ||^2

    The model sees only (x_t, t), so at optimum it learns the marginal velocity
    v*(x,t) = E[u_t | X_t = x].
    """
    batch = x0.shape[0]
    # print("x0", x0.shape)
    # print("x1", x1.shape)
    t = torch.rand(batch, device=x0.device, dtype=x0.dtype)
    # print("t", t.shape)
    xt, ut = interpolant.sample_xt_and_velocity(x0, x1, t)
    # print("xt", xt.shape)
    # print("ut", ut.shape)
    pred = model(xt, t)
    # print("pred", pred.shape)
    return torch.mean((pred - ut) ** 2)


def ot_conditional_flow_matching_loss(
    model: nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    interpolant: Interpolant,
    ot_method: str = "exact",
    replace: bool = True,
) -> torch.Tensor:
    """OT-CFM (Tong et al., 2024).

    Pair (x0, x1) along the batch dim by a minibatch OT plan (squared-L2 cost),
    then compute the standard CFM regression. Identical to
    conditional_flow_matching_loss except for the reorder line; diff the two
    to see the entire OT-CFM patch.
    """
    x0, x1 = ot_reorder(x0, x1, method=ot_method, replace=replace)
    t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
    xt, ut = interpolant.sample_xt_and_velocity(x0, x1, t)
    pred = model(xt, t)
    return torch.mean((pred - ut) ** 2)


@torch.no_grad()
def estimate_marginal_velocity_by_binning(
    x0: torch.Tensor,
    x1: torch.Tensor,
    interpolant: Interpolant,
    t_value: float,
    query: torch.Tensor,
    bandwidth: float = 0.25,
) -> torch.Tensor:
    """Numerically illustrates v(x,t)=E[u_t | X_t=x] using kernel weights.

    This is not used for training. It is a pedagogical helper for seeing why the
    trained velocity is a conditional expectation rather than always x1-x0.
    """
    t = torch.full((x0.shape[0],), t_value, device=x0.device, dtype=x0.dtype)
    xt, ut = interpolant.sample_xt_and_velocity(x0, x1, t)
    d2 = torch.cdist(query, xt) ** 2
    weights = torch.softmax(-d2 / (2 * bandwidth**2), dim=1)
    return weights @ ut
