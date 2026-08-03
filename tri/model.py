"""Looped-depth transformer with ternary-capable linear layers.

The parameter tree is a plain nested dict.  Leaf *names* carry the optimizer
grouping, so no path-guessing is needed downstream:

  ``w``        linear weight, ``(in_features, out_features)``.  int8 ternary in
               ``sign`` mode, float32 latent in ``ste``/``bf16``.
  ``s``        per-output-channel scale (ternary modes only)
  ``g``        RMSNorm gain
  ``tok_emb``  token embedding (also the output head when tied)
  ``gate``     re-injection gate

The forward pass never quantizes in ``sign`` mode: the stored weight already
*is* ternary, so gradients w.r.t. it are exact and the optimizer is the only
thing that knows about the lattice.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp

from .config import ModelConfig
from .quant import init_ternary, quantize_act, ternarize_ste, ternary_rms

_HAS_SDPA = hasattr(jax.nn, "dot_product_attention")


def _dtype(cfg: ModelConfig):
    return jnp.dtype(cfg.dtype)


# -- init --------------------------------------------------------------


def _init_linear(key, d_in, d_out, std, cfg: ModelConfig) -> dict:
    if cfg.quant == "sign":
        kw, _ = jax.random.split(key)
        w = init_ternary(kw, (d_in, d_out), cfg.p_zero_init)
        s = jnp.full((d_out,), std / ternary_rms(cfg.p_zero_init), jnp.float32)
        return {"w": w, "s": s}
    w = jax.random.normal(key, (d_in, d_out), jnp.float32) * std
    return {"w": w}


def _init_block(key, cfg: ModelConfig, resid_std_div: float) -> dict:
    d, hd = cfg.d_model, cfg.head_dim
    qd, kvd = cfg.n_heads * hd, cfg.n_kv_heads * hd
    base = 1.0 / math.sqrt(d)
    k = jax.random.split(key, 7)
    return {
        "n1": {"g": jnp.ones((d,), jnp.float32)},
        "attn": {
            "q": _init_linear(k[0], d, qd, base, cfg),
            "k": _init_linear(k[1], d, kvd, base, cfg),
            "v": _init_linear(k[2], d, kvd, base, cfg),
            "o": _init_linear(k[3], qd, d, (1.0 / math.sqrt(qd)) / resid_std_div, cfg),
        },
        "n2": {"g": jnp.ones((d,), jnp.float32)},
        "mlp": {
            "gate": _init_linear(k[4], d, cfg.mlp_hidden, base, cfg),
            "up": _init_linear(k[5], d, cfg.mlp_hidden, base, cfg),
            "down": _init_linear(
                k[6], cfg.mlp_hidden, d, (1.0 / math.sqrt(cfg.mlp_hidden)) / resid_std_div, cfg
            ),
        },
    }


def init_params(key: jax.Array, cfg: ModelConfig) -> dict:
    """Initialize the parameter tree."""
    # Output projections are scaled down by the *effective* depth, which for a
    # looped core is larger than the number of stored blocks.
    resid_div = math.sqrt(2.0 * cfg.n_effective_blocks())
    keys = jax.random.split(key, 4 + cfg.n_stored_blocks)
    params: dict = {
        "tok_emb": jax.random.normal(keys[0], (cfg.vocab_size, cfg.d_model), jnp.float32) * 0.02,
        "final_norm": {"g": jnp.ones((cfg.d_model,), jnp.float32)},
    }
    if not cfg.tie_embeddings:
        params["head_w"] = jnp.zeros((cfg.d_model, cfg.vocab_size), jnp.float32)
    if cfg.reinject and cfg.n_core > 0:
        # Zero-init gate: at step 0 the looped core is exactly a weight-tied
        # deep stack, and the model learns how much embedding to re-inject.
        params["gate"] = jnp.zeros((cfg.d_model,), jnp.float32)

    i = 4
    for name, n in (("prelude", cfg.n_prelude), ("core", cfg.n_core), ("coda", cfg.n_coda)):
        group = {}
        for j in range(n):
            group[str(j)] = _init_block(keys[i], cfg, resid_div)
            i += 1
        params[name] = group
    return params


# -- primitives --------------------------------------------------------


def rms_norm(x: jnp.ndarray, g: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    x32 = x.astype(jnp.float32)
    v = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return (x32 * jax.lax.rsqrt(v + eps) * g).astype(x.dtype)


def rope_tables(seq: int, head_dim: int, theta: float, dtype):
    half = head_dim // 2
    inv = 1.0 / (theta ** (jnp.arange(half, dtype=jnp.float32) * 2.0 / head_dim))
    t = jnp.arange(seq, dtype=jnp.float32)
    f = t[:, None] * inv[None, :]
    return jnp.cos(f).astype(dtype), jnp.sin(f).astype(dtype)


def apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    # x: (B, S, N, H); split-half convention (llama style)
    x1, x2 = jnp.split(x, 2, axis=-1)
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    return jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


def linear(p: dict, x: jnp.ndarray, cfg: ModelConfig, dtype) -> jnp.ndarray:
    w = p["w"]
    if cfg.quant == "ste":
        w = ternarize_ste(w.astype(jnp.float32)).astype(dtype)
    else:
        # `sign` mode: w is already on the lattice (int8 or a float view of it).
        w = w.astype(dtype)
    if cfg.act_bits:
        x = quantize_act(x, cfg.act_bits)
    y = x @ w
    if "s" in p:
        y = y * p["s"].astype(dtype)
    return y


def attention(p: dict, x: jnp.ndarray, cos, sin, cfg: ModelConfig, dtype) -> jnp.ndarray:
    B, S, _ = x.shape
    nh, nkv, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    q = linear(p["q"], x, cfg, dtype).reshape(B, S, nh, hd)
    k = linear(p["k"], x, cfg, dtype).reshape(B, S, nkv, hd)
    v = linear(p["v"], x, cfg, dtype).reshape(B, S, nkv, hd)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    if cfg.flash_attn and _HAS_SDPA:
        o = jax.nn.dot_product_attention(q, k, v, is_causal=True)
    else:
        if nkv != nh:
            k = jnp.repeat(k, nh // nkv, axis=2)
            v = jnp.repeat(v, nh // nkv, axis=2)
        qt, kt, vt = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        scores = jnp.einsum("bnqh,bnkh->bnqk", qt, kt).astype(jnp.float32) / math.sqrt(hd)
        causal = jnp.tril(jnp.ones((S, S), bool))[None, None]
        scores = jnp.where(causal, scores, -1e30)
        w = jax.nn.softmax(scores, axis=-1).astype(dtype)
        o = jnp.einsum("bnqk,bnkh->bnqh", w, vt).transpose(0, 2, 1, 3)

    return linear(p["o"], o.reshape(B, S, nh * hd), cfg, dtype)


def mlp(p: dict, x: jnp.ndarray, cfg: ModelConfig, dtype) -> jnp.ndarray:
    return linear(p["down"], jax.nn.silu(linear(p["gate"], x, cfg, dtype)) * linear(p["up"], x, cfg, dtype), cfg, dtype)


def block(p: dict, x: jnp.ndarray, cos, sin, cfg: ModelConfig, dtype) -> jnp.ndarray:
    x = x + attention(p["attn"], rms_norm(x, p["n1"]["g"]), cos, sin, cfg, dtype)
    return x + mlp(p["mlp"], rms_norm(x, p["n2"]["g"]), cfg, dtype)


# -- forward -----------------------------------------------------------


def forward(params: dict, tokens: jnp.ndarray, cfg: ModelConfig, n_loops: int | None = None) -> jnp.ndarray:
    """Token ids ``(B, S)`` -> logits ``(B, S, V)``.

    ``n_loops`` must be a Python int (it controls an unrolled loop, so it is a
    static argument to any surrounding ``jit``).
    """
    loops = cfg.n_loops if n_loops is None else int(n_loops)
    dtype = _dtype(cfg)
    B, S = tokens.shape
    cos, sin = rope_tables(S, cfg.head_dim, cfg.rope_theta, dtype)

    blk = partial(block, cfg=cfg, dtype=dtype)
    if cfg.remat:
        blk = jax.checkpoint(blk)

    emb = params["tok_emb"].astype(dtype)[tokens]
    h = emb

    for j in range(cfg.n_prelude):
        h = blk(params["prelude"][str(j)], h, cos, sin)

    if cfg.n_core > 0:
        gate = params.get("gate")
        for _ in range(loops):
            if gate is not None:
                h = h + emb * gate.astype(dtype)
            for j in range(cfg.n_core):
                h = blk(params["core"][str(j)], h, cos, sin)

    for j in range(cfg.n_coda):
        h = blk(params["coda"][str(j)], h, cos, sin)

    h = rms_norm(h, params["final_norm"]["g"])
    head = params["tok_emb"].T if cfg.tie_embeddings else params["head_w"]
    logits = (h @ head.astype(dtype)).astype(jnp.float32)
    if cfg.logit_softcap:
        c = cfg.logit_softcap
        logits = c * jnp.tanh(logits / c)
    return logits


def loss_fn(
    params: dict,
    batch: tuple[jnp.ndarray, jnp.ndarray],
    cfg: ModelConfig,
    n_loops: int | None = None,
) -> tuple[jnp.ndarray, dict]:
    """Mean next-token cross entropy (+ optional z-loss), and metrics."""
    x, y = batch
    logits = forward(params, x, cfg, n_loops)
    logz = jax.nn.logsumexp(logits, axis=-1)
    true = jnp.take_along_axis(logits, y[..., None], axis=-1)[..., 0]
    ce = jnp.mean(logz - true)
    loss = ce + cfg.zloss * jnp.mean(jnp.square(logz)) if cfg.zloss else ce
    return loss, {"ce": ce, "logz": jnp.mean(logz)}


# -- helpers -----------------------------------------------------------


def materialize(params: dict, cfg: ModelConfig, dtype=None):
    """Cast int8 ternary leaves to a float view so autodiff can reach them.

    This is the one place the ``sign`` mode pays for not having latent weights:
    the float view is transient (recomputed every step, and under ``remat``
    re-materialized in the backward pass) rather than persistent optimizer state.
    """
    dt = _dtype(cfg) if dtype is None else dtype
    return jax.tree.map(lambda a: a.astype(dt) if a.dtype == jnp.int8 else a, params)


def param_labels(params: dict, cfg: ModelConfig, muon_on_latent: bool = True) -> dict:
    """Map each leaf to an optimizer group: sign | matrix | embed | vector."""

    def label(path, leaf):
        name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
        if name == "w":
            if cfg.quant == "sign":
                return "sign"
            if leaf.ndim == 2 and muon_on_latent:
                return "matrix"
            return "vector"
        if name in ("tok_emb", "head_w"):
            return "embed"
        if name in ("s", "g", "gate"):
            return "vector"
        raise ValueError(f"unlabeled parameter leaf: {path}")

    return jax.tree_util.tree_map_with_path(label, params)


def count_params(params: dict) -> dict:
    leaves = jax.tree_util.tree_leaves(params)
    total = sum(int(a.size) for a in leaves)
    tern = sum(int(a.size) for a in leaves if a.dtype == jnp.int8)
    return {"total": total, "ternary": tern, "dense": total - tern}
