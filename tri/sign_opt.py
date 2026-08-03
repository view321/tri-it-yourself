"""Stochastic sign optimizer: latent-free training on the ternary lattice.

The stored weight *is* the ternary value.  There is no master copy to update,
so an update is a (possibly random) move between lattice points.  The default
rule is stochastic rounding of a virtual continuous step:

    v = clip(w - eta * u, -1, +1)
    w_new = stochastic_round(v)              E[w_new] = v

which is exactly "keep a latent weight for one instant, then collapse it".
``u`` is a momentum-smoothed, per-tensor-normalized direction, so ``eta`` is
measured in lattice units and means "typical fraction of a step per update" -
it schedules like a learning rate and directly controls the flip rate.

Alternative rules:

``stoch_flip``  explicit Bernoulli flip with p = clip(eta*|u|, 0, p_max)
                toward ``-sign(u)``.  Same expectation, one flip max per step.
``bop``         deterministic threshold flip (Helwegen et al. 2019, "Latent
                weights do not exist"), generalized to ternary by clipping.

Momentum is the honest cost of this method: at ``momentum_dtype='none'`` the
persistent state really is 2 bits/weight, but the walk is noisy.  fp16 momentum
costs 16 more bits and is still ~4.5x cheaper than bf16 weights + Adam moments.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from .muon import newton_schulz
from .quant import stochastic_round

RULES = ("stoch_round", "stoch_flip", "bop")
NORMALIZERS = ("rms", "absmean", "none")
MOMENTUM_DTYPES = ("none", "int8", "float16", "bfloat16", "float32")


class SignState(NamedTuple):
    count: jnp.ndarray
    key: jax.Array
    mu: Any  # momentum buffer (empty dict-free tree of None when disabled)
    mu_scale: Any  # per-tensor scale for int8 momentum


def _norm_dir(d: jnp.ndarray, how: str, eps: float = 1e-8) -> jnp.ndarray:
    if how == "rms":
        return d / (jnp.sqrt(jnp.mean(jnp.square(d))) + eps)
    if how == "absmean":
        return d / (jnp.mean(jnp.abs(d)) + eps)
    return d


def _read_mom(mu, scale, dtype: str) -> jnp.ndarray | float:
    if dtype == "none":
        return 0.0
    if dtype == "int8":
        return mu.astype(jnp.float32) * scale
    return mu.astype(jnp.float32)


def _write_mom(m: jnp.ndarray, dtype: str, key: jax.Array):
    """Store the momentum buffer, stochastically quantizing when int8."""
    if dtype == "none":
        return jnp.zeros((), jnp.int8), jnp.zeros((), jnp.float32)
    if dtype == "int8":
        scale = jnp.maximum(jnp.max(jnp.abs(m)), 1e-12) / 127.0
        q = stochastic_round(jnp.clip(m / scale, -127.0, 127.0), key)
        return q.astype(jnp.int8), scale
    return m.astype(jnp.dtype(dtype)), jnp.zeros((), jnp.float32)


def stochastic_sign(
    step_size,
    b1: float = 0.9,
    b2: float = 0.99,
    rule: str = "stoch_round",
    normalize: str = "rms",
    precondition: str = "none",
    threshold: float = 0.0,
    max_flip_prob: float = 1.0,
    momentum_dtype: str = "float16",
    zero_bias: float = 0.0,
    ns_steps: int = 5,
    seed: int = 0,
) -> optax.GradientTransformation:
    """Ternary weight updates driven by the sign (and magnitude) of the gradient.

    Args:
      step_size: float or optax schedule.  Lattice units per step.
      b1: interpolation between the momentum buffer and the fresh gradient when
        forming the update direction (Lion-style; 0 disables smoothing).
      b2: EMA decay of the momentum buffer itself.
      rule: one of ``stoch_round`` | ``stoch_flip`` | ``bop``.
      normalize: per-tensor normalization of the direction, so ``step_size`` is
        comparable across layers and over training.
      precondition: ``orthogonal`` runs Newton-Schulz on the direction first,
        i.e. Muon-style whitening before the flip decision.
      threshold: dead zone (stochastic rules) / flip threshold (``bop``).
      max_flip_prob: cap on the per-weight per-step flip probability.
      momentum_dtype: persistent state precision, or ``none`` for stateless.
      zero_bias: >0 pulls weights toward 0, trading accuracy for sparsity.

    Returns an optax transformation whose updates are *deltas on the lattice*;
    with int8 params, ``optax.apply_updates`` keeps everything in int8.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    if normalize not in NORMALIZERS:
        raise ValueError(f"normalize must be one of {NORMALIZERS}, got {normalize!r}")
    if momentum_dtype not in MOMENTUM_DTYPES:
        raise ValueError(f"momentum_dtype must be one of {MOMENTUM_DTYPES}, got {momentum_dtype!r}")
    if precondition not in ("none", "orthogonal"):
        raise ValueError(f"precondition must be 'none' or 'orthogonal', got {precondition!r}")

    step_fn = step_size if callable(step_size) else (lambda _c: step_size)
    stateless = momentum_dtype == "none"

    def init_fn(params):
        if stateless:
            mu = jax.tree.map(lambda p: jnp.zeros((), jnp.int8), params)
            mu_scale = jax.tree.map(lambda p: jnp.zeros((), jnp.float32), params)
        elif momentum_dtype == "int8":
            mu = jax.tree.map(lambda p: jnp.zeros(p.shape, jnp.int8), params)
            mu_scale = jax.tree.map(lambda p: jnp.zeros((), jnp.float32), params)
        else:
            dt = jnp.dtype(momentum_dtype)
            mu = jax.tree.map(lambda p: jnp.zeros(p.shape, dt), params)
            mu_scale = jax.tree.map(lambda p: jnp.zeros((), jnp.float32), params)
        return SignState(
            count=jnp.zeros([], jnp.int32),
            key=jax.random.PRNGKey(seed),
            mu=mu,
            mu_scale=mu_scale,
        )

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("stochastic_sign needs params (the current ternary weights)")
        eta = jnp.asarray(step_fn(state.count), jnp.float32)
        key, sub = jax.random.split(state.key)
        g_leaves, treedef = jax.tree_util.tree_flatten(updates)
        w_leaves = jax.tree_util.tree_leaves(params)
        mu_leaves = jax.tree_util.tree_leaves(state.mu)
        ms_leaves = jax.tree_util.tree_leaves(state.mu_scale)
        keys = jax.random.split(sub, max(len(g_leaves), 1))

        def per_leaf(g, w, mu, mu_s, k):
            k_round, k_mom = jax.random.split(k)
            g32 = g.astype(jnp.float32)
            m = _read_mom(mu, mu_s, momentum_dtype)
            direction = g32 if stateless else (b1 * m + (1.0 - b1) * g32)
            m_new = g32 if stateless else (b2 * m + (1.0 - b2) * g32)

            if precondition == "orthogonal" and direction.ndim == 2:
                # Whitening reshapes the direction's spectrum; the RMS step
                # afterwards restores scale, so `eta` still means lattice units.
                u = newton_schulz(direction, ns_steps, dtype=jnp.float32)
                u = _norm_dir(u, "rms")
            else:
                u = _norm_dir(direction, normalize)

            if threshold and rule != "bop":
                u = jnp.where(jnp.abs(u) < threshold, 0.0, u)

            w32 = w.astype(jnp.float32)
            if rule == "stoch_round":
                v = w32 - eta * u
                if zero_bias:
                    v = v - eta * zero_bias * jnp.sign(w32)
                v = jnp.clip(v, -1.0, 1.0)
                if max_flip_prob < 1.0:
                    # Cap how far the virtual position may leave the current
                    # lattice point, which caps P(flip) at max_flip_prob.
                    v = jnp.clip(v, w32 - max_flip_prob, w32 + max_flip_prob)
                w_new = stochastic_round(v, k_round)
            elif rule == "stoch_flip":
                p = jnp.clip(eta * jnp.abs(u), 0.0, max_flip_prob)
                if zero_bias:
                    # sign(u) == sign(w) means this flip heads toward 0
                    toward_zero = jnp.sign(u) == jnp.sign(w32)
                    boosted = jnp.clip(p * (1.0 + zero_bias), 0.0, max_flip_prob)
                    p = jnp.where(toward_zero, boosted, p)
                flip = (jax.random.uniform(k_round, w32.shape) < p).astype(jnp.float32)
                w_new = jnp.clip(w32 - flip * jnp.sign(u), -1.0, 1.0)
            else:  # bop
                move = (eta * jnp.abs(u) > threshold).astype(jnp.float32) * jnp.sign(u)
                w_new = jnp.clip(w32 - move, -1.0, 1.0)

            delta = (w_new - w32).astype(w.dtype)
            mu_q, mu_s_new = _write_mom(m_new, momentum_dtype, k_mom)
            return delta, mu_q, mu_s_new

        out = [per_leaf(*args) for args in zip(g_leaves, w_leaves, mu_leaves, ms_leaves, keys)]
        deltas = jax.tree_util.tree_unflatten(treedef, [o[0] for o in out])
        mu_new = jax.tree_util.tree_unflatten(treedef, [o[1] for o in out])
        mu_scale_new = jax.tree_util.tree_unflatten(treedef, [o[2] for o in out])
        return deltas, SignState(
            count=optax.safe_int32_increment(state.count),
            key=key,
            mu=mu_new,
            mu_scale=mu_scale_new,
        )

    return optax.GradientTransformation(init_fn, update_fn)


def flip_stats(deltas, params_before) -> dict:
    """Health metrics for a ternary run, from one step's deltas.

    ``params_before`` must be the weights *before* ``apply_updates``; the
    post-update state is reconstructed as ``before + delta``.

    ``flip_rate`` is the fraction of weights that moved this step; it should
    start near ``step_size`` and decay with the schedule.  ``dead_frac`` is the
    fraction sitting at 0 afterwards - if it runs away toward 1 the model is
    collapsing, if it pins at 0 the sparsity benefit is gone.
    """
    d_leaves = jax.tree_util.tree_leaves(deltas)
    p_leaves = jax.tree_util.tree_leaves(params_before)
    if not d_leaves:
        return {}
    n = float(sum(x.size for x in d_leaves))
    flips = sum(jnp.sum(d != 0) for d in d_leaves)
    after = [p.astype(jnp.int32) + d.astype(jnp.int32) for p, d in zip(p_leaves, d_leaves)]
    zeros = sum(jnp.sum(a == 0) for a in after)
    to_zero = sum(jnp.sum((a == 0) & (p != 0)) for a, p in zip(after, p_leaves))
    return {
        "flip_rate": flips / n,
        "dead_frac": zeros / n,
        "to_zero_rate": to_zero / n,
    }
