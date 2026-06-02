# Chapter 5 Flow Matching Implementations

This mini-project is meant to be read alongside Chapter 5 of *The Principles of Diffusion Models*.
It is deliberately small: 2D toy data, PyTorch MLPs, and explicit equations.

## Files and textbook mapping

| File | Corresponds to | What to look at |
|---|---|---|
| `flow_matching_ch5/data.py` | empirical `pdata`, prior `pprior` | two-moons and Gaussian mixture toy distributions |
| `flow_matching_ch5/networks.py` | learnable velocity field `v_phi(x_t, t)` | time-conditioned MLP |
| `flow_matching_ch5/normalizing_flow.py` | Sec. 5.1 Normalizing flows | exact change-of-variables with affine coupling layers |
| `flow_matching_ch5/neural_ode.py` | Sec. 5.1 Neural ODEs | ODE integration and trace/Jacobian intuition |
| `flow_matching_ch5/interpolants.py` | Sec. 5.3 interpolants | `x_t = a_t x_0 + b_t x_1` and target `d/dt x_t` |
| `flow_matching_ch5/flow_matching.py` | Sec. 5.2 FM / CFM | marginal FM idea and conditional regression loss |
| `flow_matching_ch5/sampling.py` | sampling from trained velocity | Euler / midpoint ODE integration from prior to data |
| `scripts/train_cfm_toy.py` | runnable CFM demo | trains canonical affine flow matching on 2D toys |
| `scripts/train_nf_toy.py` | runnable NF demo | trains a simple RealNVP-style normalizing flow |

## Install

```bash
pip install torch matplotlib scikit-learn tqdm
```

## Run

```bash
cd ch5_flow_matching_impl
python scripts/train_cfm_toy.py --steps 2000 --out outputs/cfm
python scripts/train_nf_toy.py --steps 2000 --out outputs/nf
```

The CFM script saves samples during training. The core loss is:

```python
x_t = a(t) * x0 + b(t) * x1
u_t = da_dt(t) * x0 + db_dt(t) * x1
loss = mean(||v_phi(x_t, t) - u_t||^2)
```

For the canonical affine / rectified-flow schedule, `a(t)=1-t`, `b(t)=t`, so:

```python
x_t = (1 - t) * x0 + t * x1
u_t = x1 - x0
```

This is the “conditional target” used for simulation-free training. The trained field approximates the marginal velocity `v(x,t)=E[u_t | x_t=x]`.
