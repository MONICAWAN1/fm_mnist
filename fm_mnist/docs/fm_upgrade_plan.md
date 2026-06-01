# Next-step FM adaptations for `fm_mnist`

## Context

Your `fm_mnist` baseline is complete and working: continuous CFM with the linear interpolant, a 28×28 `VelocityUNet` (~530k params), MNIST data loader, and a wandb-logged training script. From here the natural next moves for image generation are **three orthogonal upgrades** that each layer cleanly on top of the baseline without rewriting it:

1. **OT-CFM** — minibatch-OT-coupled training pairs (Tong et al. 2023, [arXiv:2302.00482](https://arxiv.org/abs/2302.00482)). Pure data-side change; no model change.
2. **Class-conditional generation + Classifier-Free Guidance (CFG)** — feed the digit label into the model, drop it sometimes, blend conditional/unconditional at sample time. Modest model + loop change.
3. **Higher-order ODE sampler** — replace your hand-rolled Euler/midpoint with the guide's `ODESolver` so you can experiment with `midpoint`, `heun3`, `dopri5`. Inference-only change.

Each phase is independent, has its own exit criterion, and is roughly 30 / 80 / 15 lines of new code respectively. Constraints unchanged: laptop CPU, MNIST 28×28, code-first math depth. The fourth obvious adaptation — Discrete FM — is **out of scope** by your choice; it's mentioned at the end only as a forward pointer.

The recommended order is **A → B → C**: smallest-blast-radius first, then structural model change, then a pure inference swap. Verify each before moving on; debugging compounded changes is unnecessarily painful.

---

## Adaptation A — OT-CFM training pairs (Tong et al. 2023)

### Why

Vanilla CFM draws `x0 ~ N(0,I)` and `x1 ~ p_data` **independently** every step. That gives a high-variance training target `u_t = x1 − x0` because each `x1` is paired with an arbitrary `x0`. OT-CFM keeps the same loss but **re-pairs each minibatch** via an exact OT plan so that close-by `x0`s map to close-by `x1`s. Result: lower-variance target, straighter trajectories (the §5.4 "Reflow" intuition done in one shot), and faster convergence at the same sample quality.

### Files to modify

| File | Change | Lines |
|---|---|---|
| [fm_mnist/requirements.txt](flow_matching_code/fm_mnist/requirements.txt) | Add `torchcfm` (which uses `POT` under the hood for exact OT) | +1 |
| [fm_mnist/scripts/train_cfm_mnist.py](flow_matching_code/fm_mnist/scripts/train_cfm_mnist.py) | Instantiate `OTPlanSampler` once; in the training loop, flatten `(x0, x1)` to `(B, -1)`, call `sample_plan`, reshape back, then pass to the existing `conditional_flow_matching_loss` | ~20 |
| [fm_mnist/flow_matching_ch5/data.py](flow_matching_code/fm_mnist/flow_matching_ch5/data.py) | No change — the existing `get_mnist_loader()` and `sample_prior()` already produce the right shapes | 0 |

### Implementation sketch

```python
# top of train_cfm_mnist.py
from torchcfm import OTPlanSampler
ot_sampler = OTPlanSampler(method="exact")  # Hungarian algorithm via POT

# in the training loop, after drawing x0 and x1:
B = x0.shape[0]
x0_flat = x0.view(B, -1)
x1_flat = x1.view(B, -1)
x0_flat, x1_flat = ot_sampler.sample_plan(x0_flat, x1_flat)
x0 = x0_flat.view_as(x0)
x1 = x1_flat.view_as(x1)
loss = conditional_flow_matching_loss(model, x0, x1, interpolant)
```

Add a `--ot` flag to gate this so you can A/B against the baseline with the same model and seed.

### Verification

1. **Loss-curve A/B.** Train two runs from the same seed: baseline vs `--ot`. The OT run should show a noticeably lower loss plateau and less stepwise variance. Log both to wandb on one chart.
2. **Path-length proxy.** After training, sample 64 images using `euler_sample` with `steps=10` vs `steps=50`. OT-CFM should produce more consistent samples at lower step counts (straighter paths integrate better with coarser steps).
3. **Exit criterion:** OT run trains in ≤ same wall-clock and produces visibly cleaner digits at 10–25 Euler steps than baseline does at the same step count.

### Hand-rolled alternative (optional, pedagogical)

If you want to see the math without a new dependency, replace `OTPlanSampler` with two lines using `scipy.optimize.linear_sum_assignment`:

```python
import scipy.optimize as so
cost = torch.cdist(x0.view(B,-1), x1.view(B,-1)).cpu().numpy() ** 2
row, col = so.linear_sum_assignment(cost)
x0 = x0[row]; x1 = x1[col]
```

This is exact 1:1 matching (Hungarian); identical result to `OTPlanSampler(method="exact")` for equal-size minibatches.

---

## Adaptation B — Class-conditional generation + CFG

### Why

You currently sample arbitrary digits from random noise. Class conditioning lets you *request* a specific digit. CFG (Ho & Salimans 2022) then lets you trade off diversity for sample quality at inference. This is the single biggest qualitative jump for not much code, and the pattern transfers verbatim to any future conditional generation task.

### Files to modify

| File | Change | Lines |
|---|---|---|
| [fm_mnist/flow_matching_ch5/networks.py](flow_matching_code/fm_mnist/flow_matching_ch5/networks.py) | Add `num_classes` arg to `VelocityUNet.__init__`. Add `nn.Embedding(num_classes + 1, time_dim, padding_idx=num_classes)`. In `forward`, accept optional `y: Tensor \| None`; build `cls_emb` (use the null index `num_classes` when `y is None`); add `cls_emb` to the sinusoidal `t_emb` before passing to ResBlocks. | ~25 |
| [fm_mnist/flow_matching_ch5/data.py](flow_matching_code/fm_mnist/flow_matching_ch5/data.py) | Modify `get_mnist_loader()` to yield `(image, label)` pairs instead of dropping labels. | ~3 |
| [fm_mnist/scripts/train_cfm_mnist.py](flow_matching_code/fm_mnist/scripts/train_cfm_mnist.py) | Pull `y` from the loader. With probability `p_drop = 0.1`, replace `y` with the null index `num_classes`. Pass `y` to the model. Loss is unchanged. | ~10 |
| [fm_mnist/flow_matching_ch5/flow_matching.py](flow_matching_code/fm_mnist/flow_matching_ch5/flow_matching.py) | Extend `conditional_flow_matching_loss` signature with `y=None` and forward it to `model(xt, t, y)`. | ~3 |
| [fm_mnist/flow_matching_ch5/sampling.py](flow_matching_code/fm_mnist/flow_matching_ch5/sampling.py) | Add `cfg_sample(model, x, y, steps, scale)` that runs Euler/midpoint with `v = (1 + scale) * model(x, t, y) − scale * model(x, t, null_y)`. | ~20 |
| [fm_mnist/scripts/train_cfm_mnist.py](flow_matching_code/fm_mnist/scripts/train_cfm_mnist.py) (sampling section) | At sample time, build a 10×10 grid: 10 classes × 10 samples per class. Save with `make_grid(nrow=10)`. | ~10 |

### Reference patterns to copy

- **Embedding setup**: [flow_matching_guide/examples/image/models/unet.py:L505-L507](flow_matching_code/flow_matching_guide/examples/image/models/unet.py#L505-L507) — `self.label_emb = nn.Embedding(num_classes + 1, time_embed_dim, padding_idx=num_classes)`. The `padding_idx=num_classes` is the unconditional/null token.
- **Random drop**: [flow_matching_guide/examples/image/training/train_loop.py:L68-L71](flow_matching_code/flow_matching_guide/examples/image/training/train_loop.py#L68-L71) — `if torch.rand(1) < args.class_drop_prob: conditioning = {} else: conditioning = {"label": labels}`. Adapt the pattern; you can do it per-sample (cleaner) instead of per-batch.
- **CFG inference**: [flow_matching_guide/examples/image/training/eval_loop.py:L36-L71](flow_matching_code/flow_matching_guide/examples/image/training/eval_loop.py#L36-L71) — `result = (1 + cfg_scale) * cond − cfg_scale * uncond`.

### Suggested hyperparameters for MNIST

- `num_classes = 10`
- `class_drop_prob = 0.1` (matches DDPM CFG paper defaults)
- `cfg_scale` ∈ {0.0, 1.0, 2.0, 3.0} — sweep at sample time
- Don't retrain to tune `cfg_scale`; it's a pure inference knob.

### Verification

1. **Per-class samples**: a 10×10 grid where each row is class `i` should show clearly that class.
2. **Guidance sweep**: a 4×10 grid at `cfg_scale ∈ {0, 1, 2, 3}` — higher scale = sharper digits, less diversity.
3. **Unconditional fallback**: passing `y=None` (or `y=num_classes`) at `cfg_scale=0` should be indistinguishable from the original unconditional run — confirms you didn't break the baseline path.
4. **Exit criterion:** all rows of the per-class grid are recognizably the requested digit; `cfg_scale=2` looks sharper than `cfg_scale=0`.

---

## Adaptation C — Higher-order ODE sampler via the guide's `ODESolver`

### Why

Your `euler_sample` and `midpoint_sample` work but offer no way to use adaptive solvers or higher-order fixed-step schemes. The guide ships `ODESolver` as a thin wrapper over `torchdiffeq` with support for `euler`, `midpoint`, `heun3`, `dopri5`, etc. Swapping to it costs ~15 LOC, lets you compare solver quality on the same trained model, and teaches the library's `ModelWrapper` interface — which you'll need anyway if you ever use other parts of the guide.

### Files to modify

| File | Change | Lines |
|---|---|---|
| [fm_mnist/flow_matching_ch5/sampling.py](flow_matching_code/fm_mnist/flow_matching_ch5/sampling.py) | Add a `GuideModelWrapper(ModelWrapper)` subclass with `forward(self, x, t, **extra)` that returns `self.model(x, t, extra.get("y"))`. Add a `guide_sample(model, x_init, method="midpoint", step_size=0.02, **kwargs)` helper that builds an `ODESolver` and calls `.sample(...)`. | ~25 |
| [fm_mnist/scripts/train_cfm_mnist.py](flow_matching_code/fm_mnist/scripts/train_cfm_mnist.py) | Replace the existing `midpoint_sample` call at sample time with `guide_sample(model, x_init, method=args.solver)`. Add `--solver` arg with choices `["euler", "midpoint", "heun3", "dopri5"]`. | ~5 |
| [fm_mnist/requirements.txt](flow_matching_code/fm_mnist/requirements.txt) | Add `torchdiffeq` if not already present. The guide pulls it in via its own install. | +1 |

### Reference patterns

- **Wrapper interface**: [flow_matching_guide/flow_matching/utils/model_wrapper.py](flow_matching_code/flow_matching_guide/flow_matching/utils/model_wrapper.py) — abstract class with `forward(x, t, **extra) -> Tensor`.
- **Solver invocation**: [flow_matching_guide/flow_matching/solver/ode_solver.py:L59-L65](flow_matching_code/flow_matching_guide/flow_matching/solver/ode_solver.py#L59-L65) — `solver.sample(x_init=..., step_size=0.001, time_grid=torch.tensor([0., 1.]))`.
- **Production example with CFG**: [flow_matching_guide/examples/image/training/eval_loop.py:L104-L105](flow_matching_code/flow_matching_guide/examples/image/training/eval_loop.py#L104-L105) — uses a CFG-scaled wrapper as the velocity model.

### Verification

Tabulate sample quality vs NFE (number of function evaluations) on the *same* trained model:

| Method | NFE = 10 | NFE = 25 | NFE = 50 | NFE = 100 |
|---|---|---|---|---|
| `euler` | blurry | OK | OK | OK |
| `midpoint` | OK | sharp | sharp | sharp |
| `heun3` | sharp | sharp | sharp | sharp |
| `dopri5` (adaptive) | — | — | — | — |

Save each as a 4×4 image grid. Confirm that for the same trained model, `heun3` at NFE=10 looks like `euler` at NFE=50, etc.

**Exit criterion**: a side-by-side grid showing higher-order solvers reach acceptable quality at lower NFE than Euler.

---

## Cross-cutting verification

Run **A**, **B**, **C** as separate wandb runs and save the artifacts side by side:

```
fm_mnist/output/
  baseline/                 # current state
  ot_cfm/                   # Phase A
  cfg/                      # Phase B (class-conditional)
  ot_cfm_cfg/               # A + B combined
  ot_cfm_cfg_heun3/         # A + B + C combined
```

For each, save: training loss curve, a 4×4 unconditional sample grid (or 10×10 class-conditional grid for B/C), and the trained checkpoint.

**Single end-to-end success criterion**: the `ot_cfm_cfg_heun3` run produces sharper, class-controllable digits at NFE=25 than the baseline produces unconditionally at NFE=50.

---

## Files-touched index

**Phase A (OT-CFM):**
- [fm_mnist/scripts/train_cfm_mnist.py](flow_matching_code/fm_mnist/scripts/train_cfm_mnist.py) — add OT pairing in loop
- [fm_mnist/requirements.txt](flow_matching_code/fm_mnist/requirements.txt) — add `torchcfm`

**Phase B (CFG):**
- [fm_mnist/flow_matching_ch5/networks.py](flow_matching_code/fm_mnist/flow_matching_ch5/networks.py) — add `num_classes`, `label_emb`, modify `forward`
- [fm_mnist/flow_matching_ch5/data.py](flow_matching_code/fm_mnist/flow_matching_ch5/data.py) — yield `(x, label)`
- [fm_mnist/flow_matching_ch5/flow_matching.py](flow_matching_code/fm_mnist/flow_matching_ch5/flow_matching.py) — thread `y` through loss
- [fm_mnist/flow_matching_ch5/sampling.py](flow_matching_code/fm_mnist/flow_matching_ch5/sampling.py) — add `cfg_sample`
- [fm_mnist/scripts/train_cfm_mnist.py](flow_matching_code/fm_mnist/scripts/train_cfm_mnist.py) — random class drop, conditional sample grid

**Phase C (higher-order solver):**
- [fm_mnist/flow_matching_ch5/sampling.py](flow_matching_code/fm_mnist/flow_matching_ch5/sampling.py) — add `GuideModelWrapper`, `guide_sample`
- [fm_mnist/scripts/train_cfm_mnist.py](flow_matching_code/fm_mnist/scripts/train_cfm_mnist.py) — `--solver` arg
- [fm_mnist/requirements.txt](flow_matching_code/fm_mnist/requirements.txt) — add `torchdiffeq`

**Reference reading (no edits):**
- [flow_matching_guide/examples/image/models/unet.py:L505-L507](flow_matching_code/flow_matching_guide/examples/image/models/unet.py#L505-L507) — CFG label embedding
- [flow_matching_guide/examples/image/training/train_loop.py:L68-L71](flow_matching_code/flow_matching_guide/examples/image/training/train_loop.py#L68-L71) — class drop
- [flow_matching_guide/examples/image/training/eval_loop.py:L36-L71](flow_matching_code/flow_matching_guide/examples/image/training/eval_loop.py#L36-L71) — CFG inference wrapper
- [flow_matching_guide/flow_matching/solver/ode_solver.py:L30-L99](flow_matching_code/flow_matching_guide/flow_matching/solver/ode_solver.py#L30-L99) — solver API
- [flow_matching_guide/flow_matching/utils/model_wrapper.py](flow_matching_code/flow_matching_guide/flow_matching/utils/model_wrapper.py) — wrapper base class

---

## Optional Phase D — Discrete FM (out of scope)

If you later want a fundamentally different paradigm rather than another upgrade, the natural step is **discrete FM**: tokenize each MNIST pixel to one of 256 (+1 mask) values and treat generation as evolving a categorical distribution over the token grid. This is a much bigger surgery: new `MixtureDiscreteProbPath`, new generalized-KL loss, new CTMC solver, and a UNet head that outputs `(B, C, H, W, 257)` logits rather than `(B, C, H, W)` velocities. The guide has the full pipeline ready in [examples/image/models/discrete_unet.py](flow_matching_code/flow_matching_guide/examples/image/models/discrete_unet.py) and the path/loss/solver files cited in the main plan. Mentioned here only so you know where to look when the time comes.
