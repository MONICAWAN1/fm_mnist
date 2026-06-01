# Flow Matching Study Plan — from 2D toys to your own MNIST FM model

## Context

You want a working understanding of Flow Matching (FM): the theory, how to run experiments, how to modify code, and ultimately how to write your own FM model for a simple image generation task. You have two codebases sitting side-by-side:

- [flow_matching_code/fm_example/](flow_matching_code/fm_example/) — a tiny, pedagogical 2D implementation that maps 1:1 to Chapter 5 of *Principles of Diffusion Models* ([Literature/principle_of_diffusion.pdf](Literature/principle_of_diffusion.pdf)).
- [flow_matching_code/flow_matching_guide/](flow_matching_code/flow_matching_guide/) — the official Meta/FAIR library accompanying the Lipman et al. *Flow Matching Guide and Code* paper. Production-scale: U-Nets, distributed training, ODE/SDE solvers, discrete + Riemannian variants.

Your chosen constraints (informing every decision below):
- **Dataset target:** MNIST (28×28 grayscale).
- **Hardware:** Laptop / CPU only — so no CIFAR-10, no ImageNet, no distributed training. The `examples/image/` folder of the official repo is for reference reading only.
- **Math depth:** Code-first; derive only the CFM loss and interpolant equations on paper.

The plan is layered: each phase builds intuition that the next phase relies on, and ends with a runnable artifact. Total time estimate: **~10–15 focused hours** spread over a few sessions.

---

## Phase 0 — Setup & smoke test (30 min)

Get both repos running before you read anything in depth. If something is broken, you want to know now.

1. Create a fresh venv/conda env. Install the CH5 deps:
   ```
   pip install -r flow_matching_code/fm_example/requirements.txt
   ```
2. Install the official library in editable mode (for later phases):
   ```
   pip install -e flow_matching_code/flow_matching
   ```
3. Smoke-test CH5 (CPU is fine — this takes ~30 seconds):
   ```
   cd flow_matching_code/fm_example
   python scripts/train_cfm_toy.py --steps 500 --out outputs/cfm_smoke
   ```
   Open `outputs/cfm_smoke/cfm_samples.png` — you should see three panels: Gaussian prior, two-moons target, FM-generated samples that resemble the moons.

**Exit criterion:** the 3-panel PNG looks roughly right.

---

## Phase 1 — Master the CH5 code on 2D toys (3–4 hours)

The CH5 folder is the right place to *start* because every concept in Chapter 5 is concretely realized in ~50–100 lines of plain PyTorch. Read in this order; skim the textbook section listed beside each file when you want the math.

| Order | File | CH5 section | Why |
|---|---|---|---|
| 1 | [data.py](flow_matching_code/fm_example/flow_matching_ch5/data.py) | Pre-5.1 (p_data, p_prior) | The simplest possible source/target distributions. |
| 2 | [interpolants.py](flow_matching_code/fm_example/flow_matching_ch5/interpolants.py) | Sec. 5.3 | `x_t = a(t)·x0 + b(t)·x1`, `u_t = a'(t)·x0 + b'(t)·x1`. The 3 schedules (linear/trig/VP) are all you need to understand path design. |
| 3 | [networks.py](flow_matching_code/fm_example/flow_matching_ch5/networks.py) | velocity field `v_φ(x_t,t)` | Sinusoidal time embedding + 2-layer MLP. Notice how `t` is concatenated — same trick will apply to your MNIST UNet. |
| 4 | [flow_matching.py](flow_matching_code/fm_example/flow_matching_ch5/flow_matching.py) | Sec. 5.2 (CFM) | The 1-equation loss. `estimate_marginal_velocity_by_binning` is optional but visualizes *why* the trained velocity is a conditional expectation, not `x1−x0`. |
| 5 | [sampling.py](flow_matching_code/fm_example/flow_matching_ch5/sampling.py) | Sampling | Euler vs midpoint ODE integration from `t=0→1`. |
| 6 | [train_cfm_toy.py](flow_matching_code/fm_example/scripts/train_cfm_toy.py) | End-to-end | The full loop tying it all together. |

**Active learning — run these experiments:**
1. **Visualize paths.** Run [scripts/sample_t.py](flow_matching_code/fm_example/scripts/sample_t.py) with `--interpolant linear`, then `trig`, then `vp`. Compare the resulting grids. Confirm you can predict the velocity arrows by hand for the linear case.
2. **Swap the interpolant.** Re-run `train_cfm_toy.py` with each of the 3 schedules and the two data types (`moons`, `gmm`). What changes in convergence speed and sample quality?
3. **Break the loss on purpose.** In [flow_matching.py:L30](flow_matching_code/fm_example/flow_matching_ch5/flow_matching.py#L30), replace `torch.rand` with `torch.zeros`. Retrain. The samples will collapse — understand why (the model only ever sees `t=0`, never learns the path).
4. **Cleanup chore (optional but recommended):** the print statements at [flow_matching.py:L28-L36](flow_matching_code/fm_example/flow_matching_ch5/flow_matching.py#L28-L36) are debug noise. Remove or guard them — you'll regret them when training MNIST.

**Optional side files** (read only if curious about the broader context Chapter 5 sets up):
- [neural_ode.py](flow_matching_code/fm_example/flow_matching_ch5/neural_ode.py) — Sec. 5.1, the NODE precursor to FM.
- [normalizing_flow.py](flow_matching_code/fm_example/flow_matching_ch5/normalizing_flow.py) — Sec. 5.1, RealNVP as a likelihood-based baseline; you can run [train_nf_toy.py](flow_matching_code/fm_example/scripts/train_nf_toy.py) and visually compare it to FM samples.

**Exit criterion:** you can sketch the CFM training loop from memory on paper, and you understand why simulation-free training works.

---

## Phase 2 — Bridge to the official library on 2D (2 hours)

Now translate the same ideas into the official library's vocabulary. This phase is short because there is no new theory — only new abstractions.

1. **Read the API skeleton first.** These are the four files that contain ~80% of what you need:
   - [flow_matching/path/path.py](flow_matching_code/flow_matching_guide/flow_matching/path/path.py) — `ProbPath` abstract base + `PathSample` dataclass.
   - [flow_matching/path/affine.py](flow_matching_code/flow_matching_guide/flow_matching/path/affine.py) — `AffineProbPath` and `CondOTProbPath`. Notice this is the library equivalent of CH5's `interpolants.py`.
   - [flow_matching/path/scheduler/scheduler.py](flow_matching_code/flow_matching_guide/flow_matching/path/scheduler/scheduler.py) — schedulers produce α_t, σ_t and their derivatives. Map each scheduler to one of the 3 CH5 interpolant schedules.
   - [flow_matching/solver/ode_solver.py](flow_matching_code/flow_matching_guide/flow_matching/solver/ode_solver.py) — `ODESolver.sample()` is the library version of CH5's Euler/midpoint loop, plus higher-order solvers via `torchdiffeq`.
   - [flow_matching/utils/model_wrapper.py](flow_matching_code/flow_matching_guide/flow_matching/utils/model_wrapper.py) — the `(x, t) → output` interface your model must satisfy.

2. **Run the standalone notebook** [examples/standalone_flow_matching.ipynb](flow_matching_code/flow_matching_guide/examples/standalone_flow_matching.ipynb). It implements FM in pure PyTorch (no library) on 2D, which makes for a clean side-by-side comparison with CH5. Confirm you see the same loss shape.

3. **Run the library notebook** [examples/2d_flow_matching.ipynb](flow_matching_code/flow_matching_guide/examples/2d_flow_matching.ipynb). Same task, but now using `AffineProbPath` + `ODESolver`. Make a 4-row mapping table in a scratch note: *CH5 concept → CH5 file → official class → official file*.

4. **Skip for now:** `2d_discrete_*`, `2d_riemannian_*`, and everything under `examples/image/` and `examples/text/`. Discrete/Riemannian FM aren't on the critical path to MNIST. The image example assumes a multi-GPU setup; you'll only *read* it later, not run it.

**Exit criterion:** you can answer "where does `b'(t)` from CH5 live in the official library?" in under 30 seconds.

---

## Phase 3 — Implement FM for MNIST yourself (4–6 hours)

This is the goal. Approach: fork the CH5 code and grow it up rather than starting from scratch — you keep the simplest possible training loop and only swap in the parts MNIST actually needs.

**Critical files to create / modify** (working inside a new folder, e.g. `flow_matching_code/fm_example/mnist/`, so you don't disrupt the original):

1. **`data.py` additions** — add an MNIST loader using `torchvision.datasets.MNIST`. Normalize to roughly `[-1, 1]`. Keep the prior as standard Gaussian, now shaped `(B, 1, 28, 28)` instead of `(B, 2)`.

2. **`unet.py` (new)** — a small UNet for 28×28 grayscale. Reference, in order of helpfulness:
   - [examples/image/models/unet.py](flow_matching_code/flow_matching_guide/examples/image/models/unet.py) — production version. Read for architecture ideas, but it's heavier than you need.
   - Your own ~150-line UNet: 2 down-blocks, 1 mid-block, 2 up-blocks, base channels 32 or 64. Inject time via the same `SinusoidalTimeEmbedding` you already have in [networks.py:L8-L26](flow_matching_code/fm_example/flow_matching_ch5/networks.py#L8-L26) — added to each block as a bias (FiLM-style or simple addition).

3. **`flow_matching.py` — reuse as-is.** The CFM loss is dimension-agnostic. The only thing that must change is the broadcasting of `t` against image-shaped tensors, and the interpolant already handles that via `coefficients()` returning scalars per-batch.

4. **`interpolants.py` — reuse as-is.** Start with the linear (rectified-flow) schedule — it's the easiest to debug.

5. **`sampling.py` — reuse as-is** for first runs. Once it works, replace with the official `ODESolver` to try higher-order solvers and confirm sample quality improves.

6. **`train_mnist.py` (new)** — copy [train_cfm_toy.py](flow_matching_code/fm_example/scripts/train_cfm_toy.py), swap the data sampler for an MNIST DataLoader, swap `VelocityMLP` for your `UNet`. Suggested starting hyperparameters for CPU: batch 64, lr 2e-4, 5–10k steps, AdamW. Save a grid of 16 generated digits every 500 steps.

**Debugging strategy (do not skip):**
- First, set up an *MNIST-shaped sanity check on a single batch.* Train on 64 fixed images for 2000 steps and verify the model can overfit them. If it can't, your wiring is wrong before you spend hours on real training.
- The CPU is the real constraint. You should see recognizable digits within ~30 minutes; if not, reduce base channels or images to 14×14 to iterate faster, then scale back up.
- Common pitfall: forgetting to broadcast `t` correctly into the UNet. The interpolant returns `(B,)` shaped tensors that `affine.coefficients()` reshapes — replicate that pattern in your own code or rely on the existing `Interpolant` class.

**Optional follow-up — port to the official library.** Once your hand-written version works, redo the same MNIST training using `AffineProbPath` + `ODESolver` from the official library. This is the cleanest way to internalize what the library is buying you.

**Exit criterion:** `outputs/mnist/samples_step_5000.png` shows recognizable handwritten digits.

---

## Phase 4 — Read production image-gen code (1–2 hours, reading only)

Now that you have your own working MNIST FM, read — don't run — the CIFAR/ImageNet code to see what production scale adds:

- [examples/image/train.py](flow_matching_code/flow_matching_guide/examples/image/train.py) — distributed training, EMA, mixed precision, gradient accumulation.
- [examples/image/models/unet.py](flow_matching_code/flow_matching_guide/examples/image/models/unet.py) — full UNet with attention.
- [examples/image/training/train_loop.py](flow_matching_code/flow_matching_guide/examples/image/training/train_loop.py) — DDP, checkpointing.

For each, ask: "what would I need to change in my MNIST code to add this?" Don't implement — just understand the gap. This is what separates a working FM model from a competitive one, and is the right launching point for any future scaling work.

---

## Reading from the textbook (Light path)

Read **only these CH5 subsections** alongside the code, not the whole chapter:
- §5.1 (Neural ODE / NF context) — skim, ~30 min.
- §5.2 (Flow Matching / Conditional FM) — read carefully, ~45 min. This is where the marginal velocity = conditional expectation argument is made; pair it with [flow_matching.py](flow_matching_code/fm_example/flow_matching_ch5/flow_matching.py).
- §5.3 (Interpolants / probability paths) — read carefully, ~30 min. Pair with [interpolants.py](flow_matching_code/fm_example/flow_matching_ch5/interpolants.py).

You can skip the rest of the chapter unless something in the code is unclear.

---

## Verification — how you'll know each phase worked

| Phase | Verification |
|---|---|
| 0 | Smoke-test PNG looks right. |
| 1 | Can re-derive CFM loss from memory; the 3 interpolant ablations and the broken-`t` experiment behave as predicted. |
| 2 | Can map every CH5 file/function to its official-library counterpart in <30s. |
| 3 | Recognizable MNIST samples from your own code. Loss curve drops monotonically. Bonus: same training run using the official `AffineProbPath` + `ODESolver` produces equally good or better samples. |
| 4 | Can list 3 concrete differences between your code and `examples/image/train.py` and explain what each buys you. |

---

## Files you'll touch (quick index)

**Read in Phase 1:**
- [fm_example/flow_matching_ch5/data.py](flow_matching_code/fm_example/flow_matching_ch5/data.py)
- [fm_example/flow_matching_ch5/interpolants.py](flow_matching_code/fm_example/flow_matching_ch5/interpolants.py)
- [fm_example/flow_matching_ch5/networks.py](flow_matching_code/fm_example/flow_matching_ch5/networks.py)
- [fm_example/flow_matching_ch5/flow_matching.py](flow_matching_code/fm_example/flow_matching_ch5/flow_matching.py)
- [fm_example/flow_matching_ch5/sampling.py](flow_matching_code/fm_example/flow_matching_ch5/sampling.py)
- [fm_example/scripts/train_cfm_toy.py](flow_matching_code/fm_example/scripts/train_cfm_toy.py)

**Read in Phase 2:**
- [flow_matching_guide/flow_matching/path/path.py](flow_matching_code/flow_matching_guide/flow_matching/path/path.py)
- [flow_matching_guide/flow_matching/path/affine.py](flow_matching_code/flow_matching_guide/flow_matching/path/affine.py)
- [flow_matching_guide/flow_matching/path/scheduler/scheduler.py](flow_matching_code/flow_matching_guide/flow_matching/path/scheduler/scheduler.py)
- [flow_matching_guide/flow_matching/solver/ode_solver.py](flow_matching_code/flow_matching_guide/flow_matching/solver/ode_solver.py)
- [flow_matching_guide/flow_matching/utils/model_wrapper.py](flow_matching_code/flow_matching_guide/flow_matching/utils/model_wrapper.py)
- [flow_matching_guide/examples/standalone_flow_matching.ipynb](flow_matching_code/flow_matching_guide/examples/standalone_flow_matching.ipynb)
- [flow_matching_guide/examples/2d_flow_matching.ipynb](flow_matching_code/flow_matching_guide/examples/2d_flow_matching.ipynb)

**Write in Phase 3** (under a new `fm_example/mnist/` directory):
- `mnist/data_mnist.py`, `mnist/unet.py`, `mnist/train_mnist.py` — reusing `flow_matching.py`, `interpolants.py`, `sampling.py` unchanged from the CH5 module.

**Reference in Phase 4:**
- [flow_matching_guide/examples/image/train.py](flow_matching_code/flow_matching_guide/examples/image/train.py)
- [flow_matching_guide/examples/image/models/unet.py](flow_matching_code/flow_matching_guide/examples/image/models/unet.py)
