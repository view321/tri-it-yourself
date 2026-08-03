import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tri.config import ModelConfig
from tri.model import count_params, forward, init_params, loss_fn, materialize, param_labels


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(
        vocab_size=64, d_model=32, n_heads=4, n_prelude=1, n_core=1, n_coda=1,
        n_loops=2, seq_len=16, mlp_round=8, dtype="float32", remat=False,
    )
    base.update(kw)
    return ModelConfig(**base)


@pytest.mark.parametrize("quant", ["bf16", "ste", "sign"])
def test_forward_shapes(quant):
    cfg = tiny_cfg(quant=quant)
    p = init_params(jax.random.PRNGKey(0), cfg)
    toks = jnp.zeros((2, cfg.seq_len), jnp.int32)
    out = forward(materialize(p, cfg), toks, cfg)
    assert out.shape == (2, cfg.seq_len, cfg.vocab_size)
    assert jnp.all(jnp.isfinite(out))


def test_sign_mode_stores_int8_ternary():
    cfg = tiny_cfg(quant="sign")
    p = init_params(jax.random.PRNGKey(0), cfg)
    w = p["core"]["0"]["attn"]["q"]["w"]
    assert w.dtype == jnp.int8
    assert set(np.unique(np.asarray(w)).tolist()) <= {-1, 0, 1}
    assert "s" in p["core"]["0"]["attn"]["q"]  # per-channel scale rides alongside
    assert count_params(p)["ternary"] > 0


def test_bf16_mode_has_no_ternary_leaves():
    cfg = tiny_cfg(quant="bf16")
    p = init_params(jax.random.PRNGKey(0), cfg)
    assert count_params(p)["ternary"] == 0
    assert "s" not in p["core"]["0"]["attn"]["q"]


def test_causality():
    cfg = tiny_cfg(quant="sign")
    p = materialize(init_params(jax.random.PRNGKey(0), cfg), cfg)
    toks = jax.random.randint(jax.random.PRNGKey(1), (1, cfg.seq_len), 0, cfg.vocab_size)
    a = forward(p, toks, cfg)
    changed = toks.at[0, -1].set((toks[0, -1] + 1) % cfg.vocab_size)
    b = forward(p, changed, cfg)
    # everything strictly before the edited position must be untouched
    np.testing.assert_allclose(np.asarray(a[:, :-1]), np.asarray(b[:, :-1]), atol=1e-5)
    assert not np.allclose(np.asarray(a[:, -1]), np.asarray(b[:, -1]))


def test_flash_and_manual_attention_agree():
    cfg_f = tiny_cfg(quant="sign", flash_attn=True)
    cfg_m = tiny_cfg(quant="sign", flash_attn=False)
    p = materialize(init_params(jax.random.PRNGKey(0), cfg_f), cfg_f)
    toks = jax.random.randint(jax.random.PRNGKey(1), (2, cfg_f.seq_len), 0, cfg_f.vocab_size)
    np.testing.assert_allclose(
        np.asarray(forward(p, toks, cfg_f)), np.asarray(forward(p, toks, cfg_m)), atol=2e-4
    )


def test_loop_count_changes_output_but_not_params():
    cfg = tiny_cfg(quant="sign")
    p = materialize(init_params(jax.random.PRNGKey(0), cfg), cfg)
    toks = jax.random.randint(jax.random.PRNGKey(1), (1, cfg.seq_len), 0, cfg.vocab_size)
    o2 = forward(p, toks, cfg, n_loops=2)
    o4 = forward(p, toks, cfg, n_loops=4)
    assert not np.allclose(np.asarray(o2), np.asarray(o4))
    assert cfg.n_effective_blocks(4) > cfg.n_effective_blocks(2)


def test_gate_is_zero_init_so_reinjection_starts_neutral():
    cfg = tiny_cfg(quant="sign", reinject=True)
    p = init_params(jax.random.PRNGKey(0), cfg)
    np.testing.assert_array_equal(np.asarray(p["gate"]), np.zeros(cfg.d_model))


@pytest.mark.parametrize("quant", ["bf16", "ste", "sign"])
def test_every_leaf_gets_a_label(quant):
    cfg = tiny_cfg(quant=quant)
    p = init_params(jax.random.PRNGKey(0), cfg)
    labels = param_labels(p, cfg)
    leaves = jax.tree_util.tree_leaves(labels)
    assert len(leaves) == len(jax.tree_util.tree_leaves(p))
    assert set(leaves) <= {"sign", "matrix", "embed", "vector"}
    assert ("sign" in leaves) == (quant == "sign")


def test_param_count_matches_config_formula():
    cfg = tiny_cfg(quant="sign")
    p = init_params(jax.random.PRNGKey(0), cfg)
    assert count_params(p)["total"] == cfg.param_counts()["stored_total"]
    assert count_params(p)["ternary"] == cfg.param_counts()["ternary"]


def test_gradients_reach_ternary_weights_without_ste():
    """The point of `sign` mode: no quantizer in the forward, so grads are exact."""
    cfg = tiny_cfg(quant="sign")
    p = materialize(init_params(jax.random.PRNGKey(0), cfg), cfg)
    toks = jax.random.randint(jax.random.PRNGKey(1), (2, cfg.seq_len), 0, cfg.vocab_size)
    g = jax.grad(lambda q: loss_fn(q, (toks, toks), cfg)[0])(p)
    gw = g["core"]["0"]["mlp"]["down"]["w"]
    assert jnp.all(jnp.isfinite(gw))
    assert float(jnp.max(jnp.abs(gw))) > 0


def test_loss_is_near_uniform_at_init():
    cfg = tiny_cfg(quant="sign", zloss=0.0)
    p = materialize(init_params(jax.random.PRNGKey(0), cfg), cfg)
    toks = jax.random.randint(jax.random.PRNGKey(1), (4, cfg.seq_len), 0, cfg.vocab_size)
    loss, _ = loss_fn(p, (toks, toks), cfg)
    assert abs(float(loss) - np.log(cfg.vocab_size)) < 0.5


def test_gqa_shapes():
    cfg = tiny_cfg(quant="sign", n_heads=4, n_kv_heads=2, flash_attn=False)
    p = materialize(init_params(jax.random.PRNGKey(0), cfg), cfg)
    toks = jnp.zeros((1, cfg.seq_len), jnp.int32)
    assert forward(p, toks, cfg).shape == (1, cfg.seq_len, cfg.vocab_size)
