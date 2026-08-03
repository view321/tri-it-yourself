"""Sampling from a trained checkpoint.

No KV cache: the model is re-run over a fixed-length buffer for every new
token, so generation compiles exactly once.  That is slow per token but it
keeps the sampler honest about the looped core (which a naive cache would get
wrong anyway, since every loop iteration revisits the same positions).
"""

from __future__ import annotations

import argparse
import json
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import ckpt as ckpt_io
from .config import ModelConfig
from .model import forward, init_params, materialize


def _make_logits_fn(cfg: ModelConfig):
    """ModelConfig is mutable (hence unhashable), so it rides in via closure."""

    @partial(jax.jit, static_argnames=("n_loops",))
    def logits_at(params, buf, pos, n_loops: int):
        logits = forward(materialize(params, cfg), buf, cfg, n_loops)
        return jnp.take(logits, pos, axis=1)

    return logits_at


def generate(
    params,
    cfg: ModelConfig,
    prompt: np.ndarray,
    n_new: int = 64,
    temperature: float = 0.8,
    top_k: int = 40,
    n_loops: int | None = None,
    key: jax.Array | None = None,
) -> np.ndarray:
    """Continue ``prompt`` ``(B, T)`` by ``n_new`` tokens."""
    loops = cfg.n_loops if n_loops is None else n_loops
    key = jax.random.PRNGKey(0) if key is None else key
    B, T = prompt.shape
    total = min(T + n_new, cfg.seq_len)
    buf = np.zeros((B, cfg.seq_len), np.int32)
    buf[:, :T] = prompt[:, : cfg.seq_len]
    logits_at = _make_logits_fn(cfg)

    for t in range(T, total):
        # pos stays a traced value so the whole loop shares one compilation
        logits = np.asarray(logits_at(params, jnp.asarray(buf), jnp.asarray(t - 1), loops))
        if temperature <= 0:
            nxt = logits.argmax(-1)
        else:
            logits = logits / temperature
            if top_k:
                k = min(top_k, logits.shape[-1])
                cut = np.partition(logits, -k, axis=-1)[:, -k][:, None]
                logits = np.where(logits < cut, -1e30, logits)
            p = np.exp(logits - logits.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            key, sub = jax.random.split(key)
            u = np.asarray(jax.random.uniform(sub, (B, 1)))
            nxt = (p.cumsum(-1) < u).sum(-1).clip(0, logits.shape[-1] - 1)
        buf[:, t] = nxt
    return buf[:, :total]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sample from a tri-it-yourself checkpoint")
    ap.add_argument("run_dir", help="run directory (containing config.json and final.npz)")
    ap.add_argument("--ckpt", default="final.npz")
    ap.add_argument("--n-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--n-loops", type=int, default=None)
    ap.add_argument("--prompt-tokens", default="", help="comma-separated token ids")
    args = ap.parse_args(argv)

    with open(os.path.join(args.run_dir, "config.json")) as f:
        cfg = ModelConfig(**json.load(f)["model"])
    template = init_params(jax.random.PRNGKey(0), cfg)
    params = ckpt_io.load(os.path.join(args.run_dir, args.ckpt), template)

    ids = [int(t) for t in args.prompt_tokens.split(",") if t.strip()] or [1, 2, 3, 4]
    out = generate(
        params, cfg, np.array([ids], np.int32), args.n_new,
        args.temperature, args.top_k, args.n_loops,
    )
    print(",".join(str(int(t)) for t in out[0]))


if __name__ == "__main__":
    main()
