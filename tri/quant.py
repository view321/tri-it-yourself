"""Ternary quantization primitives.

Two very different things live here, and keeping them straight is the whole
point of this repo:

``ternarize_ste``
    BitNet-b1.58 style.  A latent float weight is quantized on the fly and the
    gradient is passed straight through.  The latent weight is what the
    optimizer actually updates, so training memory is *worse* than bf16.

``stochastic_round`` (used by :mod:`tri.sign_opt`)
    Latent-free.  The stored weight already lives on the ternary lattice, and
    the optimizer moves it between lattice points.  No quantizer sits in the
    forward pass, so no STE is needed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

EPS = 1e-5


def absmean_scale(w: jnp.ndarray, per_row: bool = True) -> jnp.ndarray:
    """BitNet's gamma = mean(|W|), per output channel or per tensor.

    Weights are stored ``(in_features, out_features)``, so "row" here means a
    single output channel (a column of the stored array).
    """
    w32 = w.astype(jnp.float32)
    if per_row:
        return jnp.mean(jnp.abs(w32), axis=0, keepdims=True) + EPS
    return jnp.mean(jnp.abs(w32)) + EPS


def ternarize(w: jnp.ndarray, per_row: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
    """RoundClip(w / gamma, -1, 1) -> (ternary values in {-1,0,1}, gamma)."""
    scale = absmean_scale(w, per_row)
    q = jnp.clip(jnp.round(w.astype(jnp.float32) / scale), -1.0, 1.0)
    return q, scale


def ternarize_ste(w: jnp.ndarray, per_row: bool = True) -> jnp.ndarray:
    """Quantize with a straight-through estimator: forward quantized, backward identity."""
    q, scale = ternarize(w, per_row)
    wq = (q * scale).astype(w.dtype)
    return w + jax.lax.stop_gradient(wq - w)


def stochastic_round(x: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """Round to a neighbouring integer with probability equal to the fraction.

    Unbiased: ``E[stochastic_round(x)] == x``.  This is what turns a virtual
    continuous step into a real move on the ternary lattice.
    """
    lo = jnp.floor(x)
    frac = x - lo
    u = jax.random.uniform(key, x.shape, dtype=x.dtype)
    return lo + (u < frac).astype(x.dtype)


def quantize_act(x: jnp.ndarray, bits: int = 8) -> jnp.ndarray:
    """Per-token absmax activation quantization with STE (BitNet's 'a8')."""
    if bits <= 0:
        return x
    qmax = float(2 ** (bits - 1) - 1)
    x32 = x.astype(jnp.float32)
    scale = jnp.max(jnp.abs(x32), axis=-1, keepdims=True) + EPS
    q = jnp.clip(jnp.round(x32 * (qmax / scale)), -qmax, qmax) * (scale / qmax)
    return x + jax.lax.stop_gradient(q.astype(x.dtype) - x)


def init_ternary(key: jax.Array, shape: tuple[int, ...], p_zero: float = 1.0 / 3.0) -> jnp.ndarray:
    """Sample a ternary weight: P(0)=p_zero, P(+1)=P(-1)=(1-p_zero)/2."""
    u = jax.random.uniform(key, shape)
    w = jnp.where(u < p_zero, 0.0, jnp.where(u < p_zero + (1.0 - p_zero) / 2.0, 1.0, -1.0))
    return w.astype(jnp.int8)


def ternary_rms(p_zero: float) -> float:
    """RMS of a ternary tensor initialized by :func:`init_ternary`."""
    return max(1.0 - p_zero, 1e-6) ** 0.5


# -- 2-bit packing -----------------------------------------------------
#
# In memory we keep ternary weights as int8 because every matmul has to widen
# them anyway (there is no ternary tensor core).  On disk we pack 4 weights per
# byte, which is where the "2 bits per weight" claim becomes literally true.


def pack2(w: jnp.ndarray) -> jnp.ndarray:
    """Pack a ternary array into uint8, 4 values per byte (last axis padded)."""
    flat = jnp.asarray(w).reshape(-1).astype(jnp.int8)
    pad = (-flat.size) % 4
    if pad:
        flat = jnp.concatenate([flat, jnp.zeros((pad,), jnp.int8)])
    codes = (flat + 1).astype(jnp.uint8)  # {-1,0,1} -> {0,1,2}
    g = codes.reshape(-1, 4)
    return (g[:, 0] | (g[:, 1] << 2) | (g[:, 2] << 4) | (g[:, 3] << 6)).astype(jnp.uint8)


def unpack2(packed: jnp.ndarray, shape: tuple[int, ...]) -> jnp.ndarray:
    """Inverse of :func:`pack2`."""
    p = jnp.asarray(packed).astype(jnp.uint8)
    n = 1
    for s in shape:
        n *= s
    codes = jnp.stack(
        [(p >> 0) & 3, (p >> 2) & 3, (p >> 4) & 3, (p >> 6) & 3], axis=-1
    ).reshape(-1)
    return (codes[:n].astype(jnp.int8) - 1).reshape(shape)


def bits_per_weight(
    momentum_dtype: str,
    track_oscillation: bool = False,
    residual_dtype: str | None = None,
) -> float:
    """Persistent training state per ternary weight, in bits.

    Useful for the README table and for keeping the project honest: a
    momentum buffer is not free, and neither is the ``ef`` rule's residual
    (``residual_dtype``; None for the rules that keep no residual).
    """
    per_state = {
        "none": 0,
        "int8": 8,
        "float16": 16,
        "bfloat16": 16,
        "float32": 32,
    }
    if momentum_dtype not in per_state:
        raise ValueError(f"unknown momentum dtype {momentum_dtype!r}")
    if residual_dtype is not None and residual_dtype not in per_state:
        raise ValueError(f"unknown residual dtype {residual_dtype!r}")
    res = per_state[residual_dtype] if residual_dtype is not None else 0
    return 2.0 + per_state[momentum_dtype] + res + (8.0 if track_oscillation else 0.0)
