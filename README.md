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

The collapse is also the rule's weakness: the fractional part of every update is sampled away
instead of remembered, which injects a full lattice unit of noise per crossing however small the
signal.  The `ef` rule keeps that fraction in a *bounded* per-weight residual — flips fire only when
the integrated update crosses a cell boundary (with optional hysteresis):

```
v = w + e - eta * u                      e = sub-cell residual, |e| <= 0.5 + h
w = w + sign(v - w) * [|v - w| >= 0.5+h] fire on accumulated evidence, not per-step chance
e = v - w                                remainder carried exactly
```

`(w, e)` is a latent weight decomposed as lattice point + fractional position.  Because the residual
is bounded to one cell it stores in int8 with a fixed scale, so the honest accounting is 8 bits of
latent instead of STE's 32 — and the trajectory of `w + e` is momentum SGD on a clipped latent, the
same information flow that makes STE train well, at a quarter of the state.  A weight whose gradient
oscillates integrates to nothing and stops flipping; `stoch_round` would keep churning it at
`eta*|u|` per step.

This is a research experiment, not a reproduction of a known result. Latent-free ternary training has
not been convincingly shown to match STE at LLM scale, so the repo ships all three arms —
`bf16`, `ste`, `sign` — behind one flag and one training loop, and an Optuna harness to compare them
honestly.

## What's actually cheap, and what isn't

| Mode | Stored weight | Persistent state per weight | Notes |
|---|---|---|---|
| `bf16` | fp32 | 64 bits (weight + Muon momentum) | baseline |
| `ste` | fp32 latent | 64 bits | BitNet-style; ternary only at inference |
| `sign` ef + int8 momentum | int8 ternary | **18 bits** | int8 residual; STE-like dynamics |
| `sign` ef + fp16 momentum | int8 ternary | **26 bits** | int8 residual; safest ef config |
| `sign` + fp16 momentum | int8 ternary | **18 bits** | stoch_round default |
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

### The spot pipeline (`scripts/tpu_*.sh`)

Spot preemption deletes the whole TPU VM, so everything durable lives in GCS and every piece of the
loop is restartable. Checkpoint writes are atomic (tmp + rename), which makes the 5-minute GCS
mirror safe: it can only ever copy a complete file.

```bash
gsutil mb -l us-east1 gs://$PROJECT-tri          # once; same region as the TPU
bash scripts/tpu_create_spot.sh                  # spot v6e-1 (see tpu_env.sh for knobs)
gcloud compute tpus tpu-vm ssh tri-spot --zone=us-east1-d      # then, on the VM:
  git clone https://github.com/view321/tri-it-yourself && cd tri-it-yourself
  pip install -U "jax[tpu]" && pip install -e ".[data]"
  GCS_BUCKET=gs://$PROJECT-tri bash scripts/prep_to_gcs.sh     # once, ~hours; resumable
  GCS_BUCKET=gs://$PROJECT-tri bash scripts/tpu_bootstrap.sh   # starts training in tmux
```

Data prep is hours of work that a spot preemption would otherwise throw away wholesale, so
`prep_to_gcs.sh` writes the corpus in ~1.25B-token shards, mirrors progress to GCS every five
minutes, and resumes from the manifest when re-run — a preemption costs one re-run command and the
stream-skip back to position, not the night.  If you'd rather remove the risk entirely, run the same
script on a cheap **on-demand** e2 CPU VM in the bucket's region (tokenization is CPU/network-bound;
an e2-standard-32 does the whole job for a few dollars and cannot be preempted), then delete it.

From then on `scripts/tpu_babysit.sh` (run it anywhere gcloud lives and stays up — Cloud Shell
works) recreates the TPU after each preemption and re-runs the bootstrap, which pulls the newest
checkpoint from GCS and resumes exactly. Spot prices differ several-fold by region — at the time of
writing v6e was $0.65/chip-hour in us-east1/us-central1 against $1.40 in us-east5 and $1.78 in
europe-west4 - so check the Billing Catalog before picking a zone.

The `reason` preset is sized for this pipeline on a ~240 EUR budget: ~523M stored params (455M
ternary — a 114 MB packed deployment), ~1.03B compute-equivalent at 3 loops, 33B tokens of the
`reason` mix (45% FineWeb-Edu, 25% FineMath, 20% Python, 10% Cosmopedia, digit-split BPE). At the
us-east1 spot rate that is ~$230 assuming a pessimistic 20% MFU, and margin appears if MFU is
better. **Watch the logged `mfu` in the first hour**: if it sits under ~18%, stop early (a restart
at step 2k costs cents) and relaunch with fewer steps — schedules are built from `total_steps`, so
shortening a run mid-flight is not an option but restarting a young one is cheap.

## Running on a TPU instead

JAX is native on TPU and nothing here is CUDA-specific, so a single-chip VM works with no code
changes. A **v6e-1** (918 bf16 TFLOPS, 32 GB) is roughly 4× an RTX 5090 at the same memory, which
turns the 2-day run into something closer to twelve hours; a **v5e-1** (197 TFLOPS, 16 GB) is about
5090-equivalent on compute, so halve `--batch-size` and double `--grad-accum` to hold tokens/step.

Each TPU generation has its own VM image. `tpu-ubuntu2204-base` is **v4 and older** — using it on a
v5e/v6e is the easy mistake here:

| TPU | `--version` | `--accelerator-type` |
|---|---|---|
| v6e | `v2-alpha-tpuv6e` | `v6e-1` |
| v5e | `v2-alpha-tpuv5-lite` | `v5litepod-1` |
| v5p | `v2-alpha-tpuv5` | — |
| v4 and older | `tpu-ubuntu2204-base` | — |

```bash
gcloud alpha compute tpus tpu-vm create tri-v6e \
    --zone="$ZONE" --project="$PROJECT" \
    --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e

gcloud compute tpus tpu-vm ssh tri-v6e --zone="$ZONE"
pip install -U "jax[tpu]"          # stable no longer needs the libtpu find-links URL
python -c "import jax; print(jax.devices())"
pip install -e ".[data,tune]"
```

TPU VMs are managed from the Console's **TPUs** page (which has its own SSH button), not the
Compute Engine *VM instances* list. `--worker=all` runs a command on every host of a multi-host
slice; a single-chip VM has only worker 0, so you can omit it. Add `--tunnel-through-iap` if the VM
has no external IP.

Start long runs **detached** — `tmux new -s tri`, then `Ctrl-b d` — or an SSH timeout takes the job
with it. `--resume auto` makes that recoverable rather than fatal, but not needing it is better.

Confirm the accelerator string for your zone with
`gcloud compute tpus accelerator-types list --zone="$ZONE"` rather than trusting the table — the
naming has changed between generations. TPU v6e needs **JAX ≥ 0.4.37**, which this project's floor of
0.4.38 already satisfies.

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
python -m tri.ablate --study sign --trials 40 --preset tiny --steps 800 --dataset bin --data-dir data
```

**Always pass a learnable `--dataset`.** `synthetic` is uniform random tokens whose loss floor is
exactly `ln(vocab_size)`, so every trial scores the same and the search optimizes sampling noise;
`tri.ablate` now refuses to start on it. If you want to exercise the harness before the data download
finishes, use `--dataset induction` — a real signal with a known floor, and no tokens required.

| study | question |
|---|---|
| `sign` | flip rule, step size, momentum betas, orthogonal preconditioning |
| `modes` | `bf16` vs `ste` vs `sign`, each pinned with `--fix-quant` and tuned on an equal budget |
| `loops` | quality vs loop count, with a compute-matched control |
| `momentum` | val loss per bit of optimizer state |

`modes` deliberately gives each arm its own tuned learning rate. Comparing arms at one shared LR only
measures which arm happened to like that LR. Pin the arm and run one study each, then compare:

```bash
for q in bf16 ste sign; do
  python -m tri.ablate --study modes --fix-quant $q --tag $q --trials 12 --steps 3000 \
      --preset tiny --dataset bin --data-dir data
done
python -m tri.report runs/ablate/modes-{bf16,ste,sign}_summary.json --uniform 10.3972
```

Each arm searches only the knobs that are live for it: `sign` has no Muon group (every block linear is
ternary) so `muon_lr` is inert there, and `bf16`/`ste` have no sign knobs. Searching a dead parameter
wastes budget and then reads like a tuned result — `tri.ablate` prints the per-group parameter counts
at startup so an empty group is visible before you spend hours on it.

**A TPE study optimizes; it does not compare.** Once a categorical value loses a few early trials,
TPE stops spending budget on it, so it never gets a fair test — in one 40-trial run `stoch_round` was
drawn 5 times, lost all 5, and was then effectively abandoned. That is evidence it starts badly, not
evidence it is worse. To actually rank the flip rules, pin each one and give it an equal budget:

```bash
for r in stoch_round stoch_flip bop ef; do
  python -m tri.ablate --study sign --fix-rule $r --tag $r --trials 12 --steps 3000 \
      --preset tiny --dataset bin --data-dir data
done
```

Read the result with `tri.report` rather than trusting the single best trial — with eight knobs and a
few dozen trials, the winner is partly luck:

```bash
python -m tri.report runs/ablate/sign_summary.json --uniform 10.3972
```

It ranks categorical choices by best *and* mean, and for each numeric knob prints the interval the top
quartile occupies. A knob whose winners cover most of the searched range is marked `UNRESOLVED`: the
study did not determine it, however decisive the best value looks.

## When the memory saving is worth anything

`sign` and `ste` deploy identically at 2 bits and run at similar speed. The only thing `sign` buys is
training state: 18 bits/weight against 64. That is worth money **only when memory is what forces your
device count** — and often it isn't.

| ternary params whose optimizer state fits one 32 GB chip | |
|---|---|
| `ste` (fp32 latent + fp32 momentum) | 2.5B |
| `sign` (fp16 momentum) | **8.9B** |
| `sign` (int8 momentum) | **16B** |

So on a fixed small number of devices, `sign` raises the model-size ceiling ~3.6×. That is the real
claim, and it is a claim about *who can train what on what they already have*, not about cost per FLOP.

Two places it evaporates:

- **Below the ceiling.** At 305M params (`wide`), `ste` state is 2.5 GB and `sign` is 1.1 GB on a
  32 GB chip. Activations dominate; neither binds; the saving buys nothing.
- **On a cluster.** Sharding optimizer state across N devices divides `ste`'s penalty by N. For a 3B
  ternary model, `ste` needs 24 GB/chip at N=1 but only 3 GB/chip at N=8. If you already rent 8 chips
  for compute, `ste`'s memory cost is amortized to nothing and `sign` saves you nothing.

The asymmetry that matters: **memory you don't use is free, but compute is always paid.** If `sign`
needs more tokens than `ste` to reach the same loss, that is a real cost in every regime, traded
against a saving that only materializes in one. `tri.report` prints seconds per trial next to the
quality numbers so both sides are visible.

## The comparison that actually matters

Beating an unquantized `bf16` model was never the goal, and expecting it would be strange — more
precision per weight is exactly what `bf16` has. But a float model that has to fit in 2 bits per
weight does not get deployed in float, it gets quantized, and it loses something on the way. So the
honest comparison is between things that occupy the same memory at inference:

| | deploy bits/weight | training state bits/weight | role |
|---|---|---|---|
| `bf16`, unquantized | 16–32 | 64 | reference ceiling, not a competitor |
| `bf16` → PTQ ternary | 2 | 64 | the "just quantize it afterwards" baseline |
| `ste` (this *is* QAT) | 2 | 64 | the strong low-memory baseline |
| `sign` stoch_round | 2 | **18** | latent-free ternary |
| `sign` ef | 2 | **18–26** | bounded 8-bit latent; STE's information flow |

```bash
python -m tri.ptq runs/modes-bf16/t000 --dataset bin --data-dir data --uniform 10.3972
```

`sign` has to beat the **PTQ** number to justify existing at all, and be competitive with `ste` while
using a quarter of its training state. Two outcomes are worth something: matching `ste` (then the
latent weights were never necessary, and you save 46 bits per weight while training), or beating PTQ
by a wide margin (then training under the constraint really does let the model adapt to it, which is
the interesting hypothesis). Losing to `ste` by less than the `bf16`→PTQ gap is still informative.

## What the CPU smoke test actually shows

These are from the `smoke` preset — a **0.2M-parameter model on a synthetic period-8 copy task**,
1500 steps on a laptop CPU. They exist to show the pipeline works end to end and to justify the
defaults; they say nothing about how any of this behaves at 136M parameters on real text.

Uniform baseline is 5.545 nats; the task's floor is 0.606.

| | seed 0 | seed 1 | state bits/w |
|---|---|---|---|
| `bf16` (Muon) | 1.73 | 3.22 | 64 |
| `ste` | 3.94 | 3.79 | 64 |
| `sign` stoch_round (fp16 momentum) | 1.18 | 1.19 | 18 |
| `sign` ef (fp16 momentum + int8 residual) | **0.69** | **0.89** | 26 |
| `sign` ef (int8 momentum + int8 residual) | 1.00 | 1.52 | 18 |

The ef rows were measured after the others, on a machine whose re-runs of the control arms gave
1.15/1.18 (`stoch_round`) and 3.89 (`ste`) — close enough to read the table as one experiment.

Four things worth taking from this, none of which is "ternary beats float":

- **Error feedback beats imposed flips on both seeds, by more than the seed spread.** The only
  difference between the ef rows and the `stoch_round` row is whether the fractional update is
  carried in a residual or resampled away; at matched fp16 momentum it is worth ~0.4 nats here.
  At a matched 18-bit budget the comparison is a wash on this toy (1.00/1.52 vs 1.18/1.19) — the
  int8 momentum's per-tensor max scale is the fragile bit, and seed 1 shows it.

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
- Seide et al., *1-Bit Stochastic Gradient Descent* (2014) and Karimireddy et al., *Error Feedback
  Fixes SignSGD* (2019) — the error-feedback lineage behind the `ef` rule; Courbariaux et al.,
  *BinaryConnect* (2015) for clipping the latent to keep it responsive, which is what bounds the
  residual to one cell.
- Jordan et al., *Muon: An optimizer for hidden layers in neural networks* (2024).
- Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning* (2025) — looped depth with
  embedding re-injection.

MIT licensed.
