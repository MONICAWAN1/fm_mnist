# Modify `fm_example` for MNIST image generation

## Context

The high-level study plan in [`inside-the-flow-matching-code-folder-distributed-moore.md`](inside-the-flow-matching-code-folder-distributed-moore.md) covers learning Flow Matching from 2D toys to an MNIST implementation. This plan is the **execution-level sibling** for Phase 3 of that document: it answers *"what are the exact places to change in the CH5 codebase to turn it into a simple MNIST image generator?"*

The premise is that the CH5 code is **dimension-agnostic almost everywhere** — the FM loss, sampling loop, and time embedding all just thread tensors through without caring about shape. So we don't need a rewrite; we need three surgical changes plus one new architecture file. The hardware target is laptop/CPU; the math depth is light (no schedule comparison work — stick to the linear interpolant).

Approach: create a new sibling folder `flow_matching_code/fm_example/mnist/` and a sibling script under `scripts/` so the original 2D code stays intact as a reference. Reuse `flow_matching.py`, `interpolants.py` (with one tiny fix), and `sampling.py` by import — do **not** copy them.

---

## Change-point table

| # | File | Action | What changes |
|---|---|---|---|
| 1 | [flow_matching_ch5/data.py](flow_matching_code/fm_example/flow_matching_ch5/data.py) | **Extend** | Add `sample_mnist_batch()` returning `(B, 1, 28, 28)` in `[-1, 1]`; generalize `sample_prior` to accept a shape tuple. |
| 2 | [flow_matching_ch5/interpolants.py:L14-L17](flow_matching_code/fm_example/flow_matching_ch5/interpolants.py#L14-L17) | **One-line fix** | Generalize the `t = t[:, None]` reshape to match image rank. Current `(B, 1)` doesn't broadcast against `(B, 1, 28, 28)`. |
| 3 | [flow_matching_ch5/networks.py](flow_matching_code/fm_example/flow_matching_ch5/networks.py) | **Extend** | Keep `SinusoidalTimeEmbedding` ([L8-L26](flow_matching_code/fm_example/flow_matching_ch5/networks.py#L8-L26)) — reuse verbatim. Add a new `VelocityUNet` class. Leave `VelocityMLP` alone. |
| 4 | [flow_matching_ch5/flow_matching.py:L28-L36](flow_matching_code/fm_example/flow_matching_ch5/flow_matching.py#L28-L36) | **Cleanup** | Remove the six `print(...)` debug statements — they'll spam an MNIST training run. The MSE loss itself is rank-agnostic and needs no change. |
| 5 | [flow_matching_ch5/sampling.py](flow_matching_code/fm_example/flow_matching_ch5/sampling.py) | **Reuse as-is** | `x = x + dt * model(x, t)` works for any tensor shape. |
| 6 | [scripts/train_cfm_toy.py](flow_matching_code/fm_example/scripts/train_cfm_toy.py) | **Copy → new file** | Create `scripts/train_cfm_mnist.py`. Swap data sampler, swap model class, replace the matplotlib scatter plot with `torchvision.utils.save_image(make_grid(...))`. |

The three files that **need real new code** are #1, #3, and #6. The interpolant change (#2) is one line. Everything else is delete-or-reuse.

---

## Detail per file

### 1. `data.py` — add MNIST sampler

Append a function modeled on `sample_two_moons` for symmetry:

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loader(batch_size, root="./data", device="cpu"):
    tf = transforms.Compose([
        transforms.ToTensor(),               # → (1, 28, 28) in [0, 1]
        transforms.Lambda(lambda x: x * 2 - 1),  # → [-1, 1]
    ])
    ds = datasets.MNIST(root, train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
```

Generalize `sample_prior`:

```python
def sample_prior(batch_size, shape=(2,), device="cpu"):
    return torch.randn(batch_size, *shape, device=device)
```

The default `shape=(2,)` keeps the original 2D scripts working.

### 2. `interpolants.py` — fix `t` broadcasting

Currently [L16-L17](flow_matching_code/fm_example/flow_matching_ch5/interpolants.py#L16-L17):

```python
if t.ndim == 1:
    t = t[:, None]                 # always lands at (B, 1)
```

Replace with:

```python
if t.ndim == 1:
    t = t.view(t.shape[0], *([1] * (ndim - 1)))
```

…and add an `ndim` argument to `coefficients(t, ndim=2)`, then have `sample_xt_and_velocity` pass `ndim=x0.ndim`:

```python
def sample_xt_and_velocity(self, x0, x1, t):
    a, b, da, db = self.coefficients(t, ndim=x0.ndim)
    ...
```

This keeps backward compatibility for 2D toys (`ndim=2`) while supporting 4D image tensors (`ndim=4`).

### 3. `networks.py` — add `VelocityUNet`

Reuse [`SinusoidalTimeEmbedding`](flow_matching_code/fm_example/flow_matching_ch5/networks.py#L8-L26) as-is. Add a minimal UNet (~120 lines) with this shape pipeline for 28×28 grayscale:

```
input  (B, 1,  28, 28)
  → conv  → (B, 32, 28, 28)  + time bias
  → down  → (B, 64, 14, 14)  + time bias
  → down  → (B,128,  7,  7)  + time bias        ← bottleneck
  → up    → (B, 64, 14, 14)  + skip + time bias
  → up    → (B, 32, 28, 28)  + skip + time bias
  → conv  → (B,  1, 28, 28)
```

- **Blocks**: two `Conv → GroupNorm → SiLU → Conv → GroupNorm` residual blocks per resolution. No attention (MNIST is small enough to skip it).
- **Time conditioning**: project the 64-D sinusoidal embedding through `nn.Linear(64, channels)` per block, then add as a per-channel bias to the post-conv feature map (additive FiLM). Simpler than full FiLM, sufficient at this scale.
- **Critical interface**: keep `forward(self, x, t)` identical to `VelocityMLP.forward`. That way `flow_matching.py` and `sampling.py` work unchanged.
- **Reference, do not copy**: [flow_matching_guide/examples/image/models/unet.py](flow_matching_code/flow_matching_guide/examples/image/models/unet.py) is the production version. Use it for design ideas only; it's much heavier than you need.

### 4. `flow_matching.py` — strip prints

Delete the six `print(...)` lines at [L28-L36](flow_matching_code/fm_example/flow_matching_ch5/flow_matching.py#L28-L36). Leave the rest. The loss is `torch.mean((pred - ut) ** 2)`, which averages across every dimension regardless of rank.

### 5. `sampling.py` — no changes

Both `euler_sample` and `midpoint_sample` only call `x + dt * model(x, t)` and `torch.full((x.shape[0],), ...)`. Both work identically for `(B, 2)` and `(B, 1, 28, 28)` inputs.

### 6. `scripts/train_cfm_mnist.py` — new script

Skeleton copied from [train_cfm_toy.py:L49-L74](flow_matching_code/fm_example/scripts/train_cfm_toy.py#L49-L74), with these swaps:

- Replace `sample_data(...)` with iteration over the MNIST DataLoader from #1.
- Replace `sample_prior(B, device=...)` with `sample_prior(B, shape=(1, 28, 28), device=...)`.
- Replace `VelocityMLP(dim=2)` with `VelocityUNet(in_channels=1, base_channels=32)`.
- Replace `save_plot` (scatter) with:
  ```python
  from torchvision.utils import save_image, make_grid
  save_image(make_grid(samples.clamp(-1, 1) * 0.5 + 0.5, nrow=4), out / f"step_{step}.png")
  ```
- Every 500 steps, sample 16 images via `midpoint_sample(model, sample_prior(16, shape=(1,28,28)), steps=50)` and save the grid.

**CPU-friendly hyperparameters**: batch 64, lr 2e-4, 5–10k steps, AdamW, linear interpolant. Expect ~30 min to first recognizable digits.

---

## Debugging strategy (do not skip)

1. **Overfit a single batch first.** Train on 64 fixed images for 2000 steps with shuffle off. If the model can't memorize 64 images, your wiring is wrong and you'll waste hours on the full set.
2. **Verify shapes once.** On the first training step, print `xt.shape` and `ut.shape` — both should be `(B, 1, 28, 28)`. If you see broadcasting errors here, the interpolant fix (#2) is incomplete.
3. **Watch the `t=1` boundary.** If trained samples look like Gaussian blobs, the solver may be integrating past the well-conditioned interval — start integration at `t=ε` for `ε ≈ 1e-3` rather than `t=0` (already true for `midpoint_sample`'s default).
4. **CPU iteration budget**: if 30 minutes doesn't yield recognizable digits, halve `base_channels` (32 → 16) before dropping resolution (`Resize(14)`) — the channel reduction usually buys back enough speed without losing too much capacity.

---

## Verification

| Test | How | Expected |
|---|---|---|
| Sanity overfit | `python scripts/train_cfm_mnist.py --overfit-batch 64 --steps 2000` (add an `--overfit-batch` flag that re-uses the same batch) | Loss drops near zero; generated samples look like the 64 training images. |
| End-to-end smoke | `python scripts/train_cfm_mnist.py --steps 500 --out outputs/mnist_smoke` | No errors; `outputs/mnist_smoke/step_500.png` exists, shapes look right, even if quality is poor. |
| Full training | `python scripts/train_cfm_mnist.py --steps 5000 --out outputs/mnist` on CPU (~30–60 min) | `outputs/mnist/step_5000.png` shows a 4×4 grid of recognizable handwritten digits; loss curve drops monotonically. |
| Reuse integrity | Run the original `python scripts/train_cfm_toy.py --steps 500 --out outputs/cfm_smoke` afterward | Still produces the 3-panel moons PNG — proves the interpolant change is backward-compatible. |

**Definition of done**: `outputs/mnist/step_5000.png` shows recognizable digits **and** `train_cfm_toy.py` still works unchanged.

---

## Optional follow-up (not part of this plan)

Once the hand-written MNIST FM works, swap in the official library's `AffineProbPath` + `ODESolver` (from [flow_matching_code/flow_matching_guide/](flow_matching_code/flow_matching_guide/)) as a separate experiment. The only required code changes are wrapping `VelocityUNet` to satisfy [`ModelWrapper`](flow_matching_code/flow_matching_guide/flow_matching/utils/model_wrapper.py) — the loss and sampling come from the library. This is the cleanest way to internalize what the official abstractions are buying you.
