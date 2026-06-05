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


def trajectory_cfm_loss(
    model: nn.Module,
    timepoint_batches: list[torch.Tensor],
    interpolant: Interpolant,
    *,
    leaveout: int = -1,
    coupling: str = "ot",
    ot_method: str = "exact",
) -> torch.Tensor:
    """OT-CFM over an ordered sequence of timepoints (Tong et al., 2024, EB data).

    This is the paper's TRAJECTORY task, distinct from the noise->data losses above:
    `timepoint_batches[i]` is a same-size minibatch of cells at timepoint i (see
    `fm.data.sample_eb_batch`), and for each consecutive pair (i, i+1) we fit a
    conditional flow transporting the timepoint-i marginal to the timepoint-(i+1)
    marginal. A SINGLE field covers the whole trajectory, conditioned on a global
    time normalized to [0, 1] before the model sees it (`t_global / (T-1)`): the
    sinusoidal embedding is built for [0,1] and is periodic with period <=1, so
    feeding raw integer-offset times would alias timepoint i onto i+1.

    Leave-one-out (`leaveout > 0`) reproduces `cfm_module.py`'s interpolation
    benchmark: the held-out timepoint forms no pairs. Instead the pair ending at it
    (i = leaveout-1) is BRIDGED across it to leaveout+1 over a 2-unit interval, so
    the per-unit velocity is halved (`ut/2`) and the global time spans two units
    (`2t + t_start`). The separate leaveout -> leaveout+1 pair is skipped (the bridge
    already covers that interval). Integrating the trained field from the true
    leaveout-1 cells then predicts the unseen leaveout marginal, which the EMD scores.
    `leaveout <= 0` means "no held-out timepoint" (matches their `> 0` guard).

    `coupling="ot"` resamples each pair by a minibatch squared-L2 OT plan (this is
    what makes it OT-CFM rather than vanilla CFM); "independent" keeps the random
    pairing from `sample_eb_batch`.
    """
    n_times = len(timepoint_batches)
    t_scale = max(n_times - 1, 1)
    losses = []
    for t_start in range(n_times - 1):
        # the leaveout -> leaveout+1 interval is covered by the bridge from leaveout-1
        if leaveout > 0 and t_start == leaveout:
            continue
        bridged = leaveout > 0 and (t_start + 1 == leaveout)
        x0 = timepoint_batches[t_start]
        x1 = timepoint_batches[t_start + 2] if bridged else timepoint_batches[t_start + 1]

        if coupling == "ot":
            x0, x1 = ot_reorder(x0, x1, method=ot_method)
        elif coupling != "independent":
            raise ValueError(f"unknown coupling: {coupling}")

        t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
        xt, ut = interpolant.sample_xt_and_velocity(x0, x1, t)
        if bridged:
            # displacement spans 2 units of global time -> half the per-unit velocity,
            # and local [0,1] maps onto global [t_start, t_start+2].
            ut = ut / 2.0
            t_global = 2.0 * t + t_start
        else:
            t_global = t + t_start
        pred = model(xt, t_global / t_scale)
        losses.append(torch.mean((pred - ut) ** 2))
    return torch.stack(losses).mean()


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
