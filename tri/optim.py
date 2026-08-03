"""Optimizer assembly: one transform per parameter group.

  sign    ternary weights          -> stochastic sign (latent-free)
  matrix  2D float weights         -> Muon
  embed   embeddings / output head -> AdamW
  vector  norm gains, scales, gates-> AdamW, no weight decay

In ``bf16`` and ``ste`` modes the ``sign`` group is empty and the block linears
land in ``matrix``, so the same code trains all three ablation arms.
"""

from __future__ import annotations

import jax
import optax

from .config import OptimConfig, TrainConfig
from .muon import muon
from .quant import bits_per_weight
from .sign_opt import stochastic_sign

GROUPS = ("sign", "matrix", "embed", "vector")


def make_schedule(peak: float, total_steps: int, oc: OptimConfig):
    """Warmup + (WSD | cosine | constant) decay to ``final_frac * peak``."""
    warm = max(1, int(oc.warmup_frac * total_steps))
    end = peak * oc.final_frac
    if oc.schedule == "cosine":
        return optax.warmup_cosine_decay_schedule(0.0, peak, warm, total_steps, end)
    if oc.schedule == "constant":
        return optax.join_schedules(
            [optax.linear_schedule(0.0, peak, warm), optax.constant_schedule(peak)], [warm]
        )
    if oc.schedule != "wsd":
        raise ValueError(f"unknown schedule {oc.schedule!r}")
    decay = max(1, int(oc.decay_frac * total_steps))
    stable = max(1, total_steps - warm - decay)
    return optax.join_schedules(
        [
            optax.linear_schedule(0.0, peak, warm),
            optax.constant_schedule(peak),
            optax.linear_schedule(peak, end, decay),
        ],
        [warm, warm + stable],
    )


def build_optimizer(labels, oc: OptimConfig, tc: TrainConfig, seed: int = 0):
    """Assemble the multi-group optimizer for a labeled parameter tree."""
    steps = tc.total_steps
    transforms = {
        "sign": stochastic_sign(
            make_schedule(oc.sign_step, steps, oc),
            b1=oc.sign_b1,
            b2=oc.sign_b2,
            rule=oc.sign_rule,
            normalize=oc.sign_normalize,
            precondition=oc.sign_precondition,
            threshold=oc.sign_threshold,
            max_flip_prob=oc.sign_max_flip_prob,
            momentum_dtype=oc.sign_momentum_dtype,
            zero_bias=oc.sign_zero_bias,
            ns_steps=oc.muon_ns_steps,
            seed=seed,
        ),
        "matrix": muon(
            make_schedule(oc.muon_lr, steps, oc),
            momentum=oc.muon_momentum,
            nesterov=oc.muon_nesterov,
            ns_steps=oc.muon_ns_steps,
            weight_decay=oc.muon_weight_decay,
        ),
        "embed": optax.adamw(
            make_schedule(oc.adam_lr, steps, oc),
            b1=oc.adam_b1,
            b2=oc.adam_b2,
            eps=oc.adam_eps,
            weight_decay=oc.adam_weight_decay,
        ),
        "vector": optax.adamw(
            make_schedule(oc.adam_lr, steps, oc),
            b1=oc.adam_b1,
            b2=oc.adam_b2,
            eps=oc.adam_eps,
            weight_decay=0.0,
        ),
    }
    tx = optax.multi_transform(transforms, labels)
    if oc.grad_clip:
        tx = optax.chain(optax.clip_by_global_norm(oc.grad_clip), tx)
    return tx


def group_sizes(params, labels) -> dict:
    """Parameter count per optimizer group (for logging)."""
    sizes = {g: 0 for g in GROUPS}
    flat_p = jax.tree_util.tree_leaves(params)
    flat_l = jax.tree_util.tree_leaves(labels)
    for p, l in zip(flat_p, flat_l):
        sizes[l] += int(p.size)
    return sizes


def state_bits_per_weight(oc: OptimConfig, quant: str) -> float:
    """Persistent training state per block-linear weight, in bits.

    The headline number this project is chasing.  bf16 weights + Adam moments
    are 96 bits; bf16 + Muon momentum is 48; ternary + fp16 momentum is 18.
    """
    if quant == "sign":
        return bits_per_weight(oc.sign_momentum_dtype)
    if quant == "ste":
        # latent fp32 master + Muon momentum fp32 (the STE tax)
        return 32.0 + (32.0 if oc.muon_on_latent else 64.0)
    return 32.0 + 32.0


def tree_bytes(tree) -> int:
    return int(sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(tree)))
