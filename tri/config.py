"""Model / training configuration and size presets."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field

QUANT_MODES = ("bf16", "ste", "sign")


@dataclass
class ModelConfig:
    """Looped-depth transformer with optionally ternary linear weights.

    Layout is prelude -> (core x n_loops) -> coda.  Only the core blocks are
    reused; every loop iteration runs the *same* parameters, so stored params
    stay fixed while compute (and effective depth) grows with ``n_loops``.
    """

    vocab_size: int = 32768
    d_model: int = 1024
    n_heads: int = 16
    n_kv_heads: int | None = None  # None -> MHA
    head_dim: int | None = None  # None -> d_model // n_heads

    n_prelude: int = 2
    n_core: int = 4
    n_coda: int = 2
    n_loops: int = 3

    seq_len: int = 2048
    mlp_hidden: int | None = None
    mlp_mult: float = 8.0 / 3.0
    mlp_round: int = 128
    rope_theta: float = 10000.0
    tie_embeddings: bool = True

    # Re-inject the token embedding at the start of every core pass.  Without
    # this the looped core is a plain weight-tied deep stack and tends to drift.
    reinject: bool = True

    quant: str = "sign"  # bf16 | ste | sign
    act_bits: int = 0  # 0 = none, 8 = int8 per-token activation quant (STE)
    p_zero_init: float = 1.0 / 3.0  # P(w=0) at init for `sign` mode
    dtype: str = "bfloat16"
    remat: bool = True
    flash_attn: bool = True
    zloss: float = 1e-4
    logit_softcap: float = 0.0  # 0 disables

    def __post_init__(self):
        if self.quant not in QUANT_MODES:
            raise ValueError(f"quant must be one of {QUANT_MODES}, got {self.quant!r}")
        if self.head_dim is None:
            if self.d_model % self.n_heads:
                raise ValueError("d_model must be divisible by n_heads when head_dim is unset")
            self.head_dim = self.d_model // self.n_heads
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.mlp_hidden is None:
            h = self.mlp_mult * self.d_model
            self.mlp_hidden = int(math.ceil(h / self.mlp_round) * self.mlp_round)
        if self.n_core == 0 and self.n_loops != 1:
            self.n_loops = 1

    # -- derived sizes -------------------------------------------------

    @property
    def n_stored_blocks(self) -> int:
        return self.n_prelude + self.n_core + self.n_coda

    def n_effective_blocks(self, n_loops: int | None = None) -> int:
        loops = self.n_loops if n_loops is None else n_loops
        return self.n_prelude + self.n_core * loops + self.n_coda

    def block_params(self) -> int:
        d, hd = self.d_model, self.head_dim
        q = d * self.n_heads * hd
        kv = 2 * d * self.n_kv_heads * hd
        o = self.n_heads * hd * d
        mlp = 3 * d * self.mlp_hidden
        return q + kv + o + mlp

    def param_counts(self, n_loops: int | None = None) -> dict:
        """Stored vs compute-equivalent parameter counts, plus memory."""
        blocks = self.block_params()
        ternary = blocks * self.n_stored_blocks if self.quant != "bf16" else 0
        emb = self.vocab_size * self.d_model
        dense = emb * (1 if self.tie_embeddings else 2)
        # norm gains (2 per block + 1 final), the re-injection gate, and
        # per-output-channel scales
        dense += self.d_model * (2 * self.n_stored_blocks + 1)
        if self.reinject and self.n_core > 0:
            dense += self.d_model
        if self.quant != "bf16":
            per_block_scales = (
                self.n_heads * self.head_dim
                + 2 * self.n_kv_heads * self.head_dim
                + self.d_model
                + 2 * self.mlp_hidden
                + self.d_model
            )
            dense += per_block_scales * self.n_stored_blocks
        stored = ternary + dense + (blocks * self.n_stored_blocks if self.quant == "bf16" else 0)
        eff_blocks = self.n_effective_blocks(n_loops)
        return {
            "stored_total": stored,
            "ternary": ternary,
            "dense": stored - ternary,
            "embedding": emb * (1 if self.tie_embeddings else 2),
            "compute_equivalent": blocks * eff_blocks + emb,
            "effective_blocks": eff_blocks,
            "ternary_bytes_2bit": ternary // 4,
            "dense_bytes_bf16": (stored - ternary) * 2,
        }

    def flops_per_token(self, n_loops: int | None = None) -> float:
        """Forward+backward FLOPs per token (6 * params_active, plus attention)."""
        eff = self.n_effective_blocks(n_loops)
        mm = 6.0 * self.block_params() * eff + 6.0 * self.vocab_size * self.d_model
        attn = 6.0 * 2.0 * self.seq_len * self.n_heads * self.head_dim * eff
        return mm + attn

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class OptimConfig:
    """Per-group optimizer settings.

    Three parameter groups exist:
      * ``sign``   - ternary weights (only in quant='sign')
      * ``matrix`` - 2D float weights (Muon by default)
      * ``vector`` / ``embed`` - norms, scales, embeddings (AdamW)
    """

    # -- stochastic sign (ternary weights) --
    sign_step: float = 0.05  # peak step size, in lattice units
    sign_b1: float = 0.9  # direction interpolation (Lion-style)
    sign_b2: float = 0.99  # momentum EMA decay
    sign_rule: str = "stoch_round"  # stoch_round | stoch_flip | bop
    sign_normalize: str = "rms"  # rms | absmean | none
    sign_precondition: str = "none"  # none | orthogonal (Newton-Schulz on momentum)
    sign_threshold: float = 0.0  # hysteresis (bop) / dead-zone (stochastic rules)
    sign_max_flip_prob: float = 1.0
    sign_momentum_dtype: str = "float16"  # float32|bfloat16|float16|int8|none
    sign_zero_bias: float = 0.0  # >0 biases updates toward w=0 (sparsity pressure)

    # -- Muon (2D float weights) --
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_weight_decay: float = 0.0

    # -- AdamW (everything else) --
    adam_lr: float = 3e-3
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.01

    grad_clip: float = 1.0
    schedule: str = "wsd"  # wsd | cosine | constant
    warmup_frac: float = 0.02
    decay_frac: float = 0.2  # WSD tail
    final_frac: float = 0.05  # end LR as fraction of peak

    # Use Muon for the latent weights in `ste` mode too (else AdamW).
    muon_on_latent: bool = True


@dataclass
class TrainConfig:
    run_name: str = "run"
    out_dir: str = "runs"
    seed: int = 0

    dataset: str = "synthetic"  # synthetic | induction | bin
    data_dir: str = "data"
    train_bin: str = ""
    val_bin: str = ""
    induction_period: int = 8  # repeat length for the `induction` toy task

    batch_size: int = 8  # microbatch (sequences)
    grad_accum: int = 1
    total_steps: int = 1000
    eval_every: int = 200
    eval_batches: int = 20
    log_every: int = 10
    ckpt_every: int = 0  # 0 disables
    keep_last: int = 1  # checkpoints to retain; <=0 keeps all
    resume: str = ""  # "" | "auto" (newest in run dir) | path to a ckpt_*.npz

    # Randomized loop count during training: uniform over [lo, hi].
    loop_lo: int = 2
    loop_hi: int = 4
    eval_loops: tuple[int, ...] = (1, 2, 3, 4, 5)

    track_oscillation: bool = False
    time_budget_s: float = 0.0  # 0 = no limit
    profile_mfu: bool = True
    # Peak dense BF16 TFLOPS of one device, used only to turn tokens/s into an
    # MFU figure.  RTX 5090 209.5 | TPU v5e 197 | TPU v6e 918 | A100 312 | H100 989.
    device_tflops: float = 209.5

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.grad_accum


PRESETS: dict[str, dict] = {
    # CPU-sized: exists so the whole pipeline can be exercised in seconds.
    "smoke": dict(
        model=dict(
            vocab_size=256, d_model=64, n_heads=4, n_prelude=1, n_core=1, n_coda=1,
            n_loops=2, seq_len=64, mlp_round=16, remat=False, flash_attn=False,
            dtype="float32",  # bf16 is emulated (and slow) on CPU
        ),
        train=dict(
            batch_size=4, grad_accum=1, total_steps=50, eval_every=25, eval_batches=4,
            log_every=10, loop_lo=1, loop_hi=2, eval_loops=(1, 2, 3),
            dataset="induction",
        ),
        # The toy task is embedding-bound at vocab=256, so the float arms need a
        # higher embedding LR than the real presets to learn at a visible rate.
        # Tuned for the smoke task only; real runs get their LRs from tri.ablate.
        optim=dict(adam_lr=1e-2),
    ),
    # ~25M ternary params: the ablation / Optuna workhorse (minutes per run).
    "tiny": dict(
        model=dict(
            vocab_size=32768, d_model=384, n_heads=6, n_prelude=2, n_core=2, n_coda=2,
            n_loops=2, seq_len=512,
        ),
        train=dict(
            batch_size=16, grad_accum=2, total_steps=2000, eval_every=250,
            loop_lo=1, loop_hi=3, eval_loops=(1, 2, 3, 4), dataset="bin",
        ),
    ),
    # ~50M: sanity bridge between ablation and the main run.
    "small": dict(
        model=dict(
            vocab_size=32768, d_model=640, n_heads=10, n_prelude=2, n_core=3, n_coda=2,
            n_loops=2, seq_len=1024,
        ),
        train=dict(batch_size=16, grad_accum=4, total_steps=6000, eval_every=500,
                   dataset="bin"),
    ),
    # Sized for a ~100-250h single-chip TPU v6e budget.  Wider than `main`
    # because at 2 bits the block linears are nearly free and the fp16
    # embedding table dominates deployment memory: growing d_model improves the
    # ratio of ternary weights to dense ones (61% embedding here vs 72% at
    # d=1024).  ~25B tokens at 0.5M/step is ~80 tokens/param, well past
    # Chinchilla - which matters more for ternary than for float, since each
    # weight carries ~1.58 bits and needs more data to place.
    "wide": dict(
        model=dict(
            vocab_size=32768, d_model=1536, n_heads=12, n_prelude=2, n_core=5,
            n_coda=2, n_loops=3, seq_len=2048,
        ),
        train=dict(
            batch_size=16, grad_accum=16, total_steps=48000, eval_every=1000,
            eval_batches=40, ckpt_every=500, keep_last=2, loop_lo=2, loop_hi=4,
            dataset="bin", device_tflops=918.0,
        ),
    ),
    # The recommended 2-day run on a single 32GB card.
    "main": dict(
        model=dict(
            vocab_size=32768, d_model=1024, n_heads=16, n_prelude=2, n_core=4, n_coda=2,
            n_loops=3, seq_len=2048,
        ),
        train=dict(
            batch_size=16, grad_accum=16, total_steps=12000, eval_every=500,
            eval_batches=40, ckpt_every=1000, loop_lo=2, loop_hi=4, dataset="bin",
        ),
    ),
}


def build_configs(
    preset: str = "main",
    model_overrides: dict | None = None,
    train_overrides: dict | None = None,
    optim_overrides: dict | None = None,
) -> tuple[ModelConfig, TrainConfig, OptimConfig]:
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; have {sorted(PRESETS)}")
    spec = PRESETS[preset]
    m = dict(spec.get("model", {}))
    t = dict(spec.get("train", {}))
    o = dict(spec.get("optim", {}))
    m.update({k: v for k, v in (model_overrides or {}).items() if v is not None})
    t.update({k: v for k, v in (train_overrides or {}).items() if v is not None})
    o.update({k: v for k, v in (optim_overrides or {}).items() if v is not None})
    return ModelConfig(**m), TrainConfig(**t), OptimConfig(**o)


def config_json(mc: ModelConfig, tc: TrainConfig, oc: OptimConfig) -> str:
    return json.dumps(
        {
            "model": dataclasses.asdict(mc),
            "train": dataclasses.asdict(tc),
            "optim": dataclasses.asdict(oc),
        },
        indent=2,
        default=str,
    )
