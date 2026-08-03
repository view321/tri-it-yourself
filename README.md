# tri-it-yourself

Pretraining a small LLM whose weights are **natively ternary** — `{-1, 0, +1}`, 2 bits each — in JAX,
with a **looped-depth** transformer and a **latent-free stochastic sign optimizer**.

The point is not to reproduce BitNet. BitNet keeps a float master copy of every weight and quantizes
it on the way into each matmul, so "1.58 bits" describes the *deployed* model while training costs
*more* memory than bf16. Here the stored weight **is** the ternary value. There is no master copy and
no straight-through estimator, because there is no quantizer in the forward pass to estimate through.
An optimizer step is a move between lattice points:

```
v = clip(w - eta * u, -1, +1)          u = per-tensor-normalized momentum direction
w = stochastic_round(v)                E[w_new] = v, so the step is unbiased
```

That is "keep a latent weight for one instant, then collapse it." `eta` is measured in lattice units
and is directly the per-step flip probability, so it warms up and decays like a learning rate.

This is a research experiment, not a reproduction of a known result. Latent-free ternary training has
not been convincingly shown to match STE at LLM scale, so the repo ships all three arms —
`bf16`, `ste`, `sign` — behind one flag and one training loop, and an Optuna harness to compare them
honestly.

## What's actually cheap, and what isn't

| Mode | Stored weight | Persistent state per weight | Notes |
|---|---|---|---|
| `bf16` | fp32 | 64 bits (weight + Muon momentum) | baseline |
| `ste` | fp32 latent | 64 bits | BitNet-style; ternary only at inference |
| `sign` + fp16 momentum | int8 ternary | **18 bits** | default |
| `sign` + int8 momentum | int8 ternary | **10 bits** | experimental |
| `sign`, stateless | int8 ternary | **2 bits** | the pure claim; noisiest |

Two things to be honest about:

- **The 2-bit number is only true without momentum.** A momentum buffer is the difference between a
  random walk that converges and one that doesn't, and it costs 8–16 bits per weight. `--sign-momentum-dtype none`
  gives you the literal 2-bit claim; the `momentum` ablation measures what you pay for it in val loss.
- **Training is not faster.** There is no ternary tensor core. Weights are held as int8 and widened to
  bf16 for every matmul, so a step runs at roughly bf16 speed. The win is memory and inference, not
  training FLOPs. Anyone who tells you 2-bit weights make pretraining fast on current hardware is
  selling something.

On disk, though, the claim is literal: checkpoints bit-pack 4 weights per byte, so the 102.8M ternary
weights of the `main` preset occupy exactly 25.7 MB.

## Architecture

`prelude → (core × n_loops) → coda`, with only the core reused. Stored parameters stay fixed while
effective depth grows with the loop count, and the token embedding is re-injected at the start of
every core pass through a zero-initialized gate.

Looping pairs naturally with ternary weights: each weight carries less information, so shifting budget
from parameters to compute-per-parameter is the right trade, and shared weights collect gradient
signal from every iteration, which stabilizes the flip statistics. The loop count is **resampled every
step** from `[loop_lo, loop_hi]`, so one model can be evaluated at any depth afterwards.

`main` preset (the recommended 2-day run on a 32 GB card):

| | |
|---|---|
| d_model / heads / seq | 1024 / 16 / 2048 |
| blocks | 2 prelude + 4 core × 3 loops + 2 coda = 16 effective |
| stored params | 136.4M (102.8M ternary, 33.7M dense) |
| compute-equivalent | 239M |
| tokens/step | 524,288 |
| FLOPs/token | 1.84 GFLOP (fwd+bwd) |

At 25–35% MFU on an RTX 5090 (209.5 dense BF16 TFLOPS) that is roughly 30–45k tokens/s, or
**5–8B tokens in 48 hours** — 30–50 tokens per parameter, past Chinchilla, enough for clean separation
between the three arms.

Only the block linears are ternary. Embeddings, the tied head, RMSNorm gains, and the per-output-channel
scales stay in float and are trained by AdamW; the 2D float matrices (the `bf16`/`ste` arms) are trained
by **Muon**. This mirrors BitNet practice and is close to mandatory for stability.

## Install

```bash
git clone https://github.com/view321/tri-it-yourself && cd tri-it-yourself
python -m venv .venv && . .venv/bin/activate
pip install -U "jax[cuda12]" && pip install -e ".[data,tune]"
```

Blackwell (sm_120) needs a CUDA 12.8+ driver. Verify with `python -c "import jax; print(jax.devices())"`.

## Run it

Smoke test — no data, no GPU, about a minute:

```bash
pytest -q && python -m tri.train --preset smoke --quant sign
```

Prepare real data (FineWeb-Edu 10BT sample, 32k byte-level BPE):

```bash
python -m tri.prepare_data --out-dir data --max-tokens 8000000000
```

The 2-day run:

```bash
python -m tri.train --preset main --quant sign --dataset bin --data-dir data --ckpt-every 1000
```

Baselines to run alongside it — without these the main run tells you nothing, because a gap could
come from ternary weights *or* from the optimizer, and you can't tell which:

```bash
python -m tri.train --preset main --quant bf16 --dataset bin --data-dir data --steps 3000
python -m tri.train --preset main --quant ste  --dataset bin --data-dir data --steps 3000
```

## Preemptible / spot instances

Checkpoints written with `--ckpt-every` hold the **full training state** — parameters, Muon and Adam
moments, the sign optimizer's momentum buffer and PRNG key, both host RNG streams, and the step
counters that every schedule reads from. Resume is therefore exact, not approximate: an interrupted
run continued with `--resume auto` produces bit-identical weights to an uninterrupted one, which is
what `tests/test_resume.py` asserts.

```bash
python -m tri.train --preset main --quant sign --dataset bin --data-dir data \
    --ckpt-every 500 --keep-last 2 --resume auto
```

Run that same command after every preemption; `auto` picks the newest checkpoint in the run
directory and starts fresh if there isn't one, so it is safe as a restart loop. Two things to know:

- **Keep `--steps` identical across segments.** Schedules are built from `total_steps`, so a segment
  that declares a different length runs a differently shaped LR and flip-rate schedule.
- `final.npz` stays params-only (that's what `tri.sample` reads). The resumable state lives in
  `ckpt_*.npz`, which are correspondingly larger.

## Running on a TPU instead

JAX is native on TPU and nothing here is CUDA-specific, so a single-chip VM works with no code
changes. A **v6e-1** (918 bf16 TFLOPS, 32 GB) is roughly 4× an RTX 5090 at the same memory, which
turns the 2-day run into something closer to twelve hours; a **v5e-1** (197 TFLOPS, 16 GB) is about
5090-equivalent on compute, so halve `--batch-size` and double `--grad-accum` to hold tokens/step.

```bash
python -m tri.train --preset main --quant sign --dataset bin --data-dir data \
    --device-tflops 918 --ckpt-every 500 --resume auto
```

`--device-tflops` only affects the reported MFU; leaving it at the 5090 default would overstate MFU
on a v6e by about 4×.

Stay on **one chip** unless you are willing to add sharding. There are no `Mesh`, `NamedSharding`, or
`pmap` constructs in this repo, so `jax.jit` places everything on `jax.devices()[0]` — an 8-chip slice
would rent eight chips and use one. Adding data parallelism is contained (mesh over devices, shard the
batch axis of `xs`/`ys`, replicate params) because gradient accumulation is already a `lax.scan` over
microbatches, but it is not written yet.

Two TPU caveats worth verifying early with `--preset smoke --quant sign --steps 60`: `optax.apply_updates`
does int8 arithmetic on the ternary weights, and `jax.nn.dot_product_attention` has no cuDNN backend on
TPU so it falls back to the XLA implementation — correct, but not a fused flash kernel, so expect lower
MFU than on a GPU. Budget a few minutes of XLA compile time, once per distinct loop count.

## Ablations

Tune the sign optimizer first — the flip-rate schedule is the most sensitive knob in the project:

```bash
python -m tri.ablate --study sign --trials 40 --preset tiny --steps 800
```

| study | question |
|---|---|
| `sign` | flip rule, step size, momentum betas, orthogonal preconditioning |
| `modes` | `bf16` vs `ste` vs `sign`, each with its own LR tuned under the same trial budget |
| `loops` | quality vs loop count, with a compute-matched control |
| `momentum` | val loss per bit of optimizer state |

`modes` deliberately gives each arm its own tuned learning rate. Comparing arms at one shared LR only
measures which arm happened to like that LR.

## What the CPU smoke test actually shows

These are from the `smoke` preset — a **0.2M-parameter model on a synthetic period-8 copy task**,
1500 steps on a laptop CPU. They exist to show the pipeline works end to end and to justify the
defaults; they say nothing about how any of this behaves at 136M parameters on real text.

Uniform baseline is 5.545 nats; the task's floor is 0.606.

| | seed 0 | seed 1 |
|---|---|---|
| `bf16` (Muon) | 1.73 | 3.22 |
| `ste` | 3.94 | 3.79 |
| `sign` (fp16 momentum) | 1.18 | 1.19 |

Three things worth taking from this, none of which is "ternary beats float":

- **Momentum is what makes the sign optimizer work.** At a matched step size, stateless updates reach
  5.12 — barely better than the uniform baseline — while int8 momentum reaches 0.98 and fp16 reaches
  1.37. The literal 2-bit configuration is the one that doesn't train. Encouragingly, **int8 momentum
  (10 bits/weight total) is competitive with fp16 (18 bits)**.
- **These comparisons are dominated by learning rate.** The `bf16` arm scored 4.86 until the embedding
  Adam LR moved from 3e-3 to 1e-2, at which point it scored 1.73 — a bigger swing than any difference
  between the arms. Seed variance for the float arms (1.73 vs 3.22) is also larger than the gaps
  between them. Any single-LR, single-seed cross-arm claim here would be noise.
- **`sign_step` has a clean optimum around 0.05** on this task (0.01 → 4.56, 0.02 → 1.73, 0.05 → 0.65,
  0.1 → 1.37, 0.2 → 1.54), which is where the default comes from.

That LR sensitivity is the entire reason `tri.ablate --study modes` tunes each arm's learning rate
under its own trial budget. Run it on `tiny` with real tokens before believing any ordering.

## Watch these, not just the loss

Ternary runs fail in ways a loss curve hides. Every step logs:

- **`flip_rate`** — fraction of weights that moved. Should track `sign_step` and decay with the
  schedule. Near zero means the model is frozen; above ~0.1 sustained means it's diffusing, not learning.
- **`dead_frac`** — fraction sitting at 0. Drifting toward 1 is collapse; pinned at 0 means the
  sparsity benefit is gone. Under `stoch_round`, weights pushed past ±1 clip and stick, so some loss of
  mobility over training is expected.
- **`val_ce_L{1..5}`** — val loss at each loop count. Because loop count is randomized during training,
  this shows the depth/quality trade the model actually learned.

## Layout

```
tri/config.py      model / train / optim configs and the four size presets
tri/quant.py       ternarization, STE, stochastic rounding, 2-bit packing
tri/model.py       looped transformer (RoPE, SwiGLU, RMSNorm, remat)
tri/muon.py        Newton-Schulz orthogonalization as an optax transform
tri/sign_opt.py    the stochastic sign optimizer
tri/optim.py       per-group optimizer assembly and schedules
tri/train.py       training loop (grad accum in-step, randomized loop count, resume)
tri/ablate.py      Optuna studies
tri/ckpt.py        checkpoints: 2-bit packing, full-state resume, rotation
tri/prepare_data.py, tri/sample.py
```

## References

- Ma et al., *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits* (2024) — the absmean
  ternarization and the STE arm.
- Helwegen et al., *Latent Weights Do Not Exist: Rethinking Binarized Neural Network Optimization*
  (2019) — the argument this repo takes literally, and the `bop` flip rule.
- Jordan et al., *Muon: An optimizer for hidden layers in neural networks* (2024).
- Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning* (2025) — looped depth with
  embedding re-injection.

MIT licensed.
