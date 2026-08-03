"""Muon: momentum SGD whose update is orthogonalized by Newton-Schulz.

Follows Keller Jordan's formulation.  The only convention change is the weight
layout: this repo stores linears as ``(in_features, out_features)``, so the
fan-out axis is ``-1`` and the update scale is ``sqrt(max(1, d_out / d_in))``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

# Quintic iteration coefficients: tuned so the spectrum converges fast to ~1
# rather than exactly to 1 (exact convergence is not needed for an update).
NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(g: jnp.ndarray, steps: int = 5, eps: float = 1e-7, dtype=jnp.bfloat16) -> jnp.ndarray:
    """Approximate the orthogonal factor of ``g`` (i.e. ``U @ V.T`` of its SVD)."""
    if g.ndim != 2:
        raise ValueError(f"newton_schulz expects a 2D matrix, got shape {g.shape}")
    a, b, c = NS_COEFFS
    x = g.astype(dtype)
    transposed = x.shape[0] > x.shape[1]
    if transposed:  # iterate on the wide orientation: fewer flops, same result
        x = x.T
    x = x / (jnp.linalg.norm(x.astype(jnp.float32)) + eps).astype(dtype)
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


def update_scale(shape: tuple[int, ...]) -> float:
    """RMS-matching factor for an ``(in, out)`` weight."""
    d_in, d_out = shape[-2], shape[-1]
    return max(1.0, d_out / d_in) ** 0.5


class MuonState(NamedTuple):
    count: jnp.ndarray
    mu: Any


def scale_by_muon(
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    ns_dtype=jnp.bfloat16,
) -> optax.GradientTransformation:
    def init_fn(params):
        return MuonState(
            count=jnp.zeros([], jnp.int32),
            mu=jax.tree.map(lambda p: jnp.zeros_like(p, jnp.float32), params),
        )

    def update_fn(updates, state, params=None):
        del params
        mu = jax.tree.map(
            lambda m, g: momentum * m + (1.0 - momentum) * g.astype(jnp.float32),
            state.mu,
            updates,
        )
        if nesterov:
            direction = jax.tree.map(
                lambda m, g: (1.0 - momentum) * g.astype(jnp.float32) + momentum * m, mu, updates
            )
        else:
            direction = mu

        def orth(d, g):
            if d.ndim != 2:
                raise ValueError(
                    f"Muon received a {d.ndim}D parameter of shape {d.shape}; route "
                    "non-matrix parameters to AdamW instead."
                )
            return (newton_schulz(d, ns_steps, dtype=ns_dtype) * update_scale(d.shape)).astype(g.dtype)

        out = jax.tree.map(orth, direction, updates)
        return out, MuonState(count=optax.safe_int32_increment(state.count), mu=mu)

    return optax.GradientTransformation(init_fn, update_fn)


def muon(
    learning_rate,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    weight_decay: float = 0.0,
    ns_dtype=jnp.bfloat16,
) -> optax.GradientTransformation:
    chain = [scale_by_muon(momentum, nesterov, ns_steps, ns_dtype)]
    if weight_decay:
        chain.append(optax.add_decayed_weights(weight_decay))
    chain.append(optax.scale_by_learning_rate(learning_rate))
    return optax.chain(*chain)
