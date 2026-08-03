import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tri.config import build_configs
from tri.model import forward, init_params, materialize
from tri.ptq import evaluate_both, load_run, sparsity, to_ternary
from tri.train import train


def bf16_run(tmp_path, steps=40):
    mc, tc, oc = build_configs(
        "smoke", {"quant": "bf16"},
        dict(out_dir=str(tmp_path), run_name="fp", total_steps=steps,
             eval_every=0, eval_batches=2, log_every=100), {},
    )
    train(mc, tc, oc, verbose=False)
    return str(tmp_path / "fp"), tc


def test_ptq_produces_the_sign_mode_structure(tmp_path):
    """The quantized tree must be exactly what a sign-mode model expects."""
    run_dir, _ = bf16_run(tmp_path)
    cfg, params = load_run(run_dir)
    q, qcfg = to_ternary(params, cfg)

    assert qcfg.quant == "sign"
    reference = init_params(jax.random.PRNGKey(0), qcfg)
    assert jax.tree_util.tree_structure(q) == jax.tree_util.tree_structure(reference)
    for a, b in zip(jax.tree_util.tree_leaves(q), jax.tree_util.tree_leaves(reference)):
        assert a.shape == b.shape and a.dtype == b.dtype


def test_quantized_weights_are_on_the_lattice(tmp_path):
    run_dir, _ = bf16_run(tmp_path)
    cfg, params = load_run(run_dir)
    q, _ = to_ternary(params, cfg)
    tern = [x for x in jax.tree_util.tree_leaves(q) if x.dtype == jnp.int8]
    assert tern
    for x in tern:
        assert set(np.unique(np.asarray(x)).tolist()) <= {-1, 0, 1}
    assert 0.0 <= sparsity(q) <= 1.0


def test_embeddings_and_norms_stay_float(tmp_path):
    """Only block linears are ternary in any arm; quantizing more would compare
    a different model."""
    run_dir, _ = bf16_run(tmp_path)
    cfg, params = load_run(run_dir)
    q, _ = to_ternary(params, cfg)
    assert q["tok_emb"].dtype == jnp.float32
    assert q["final_norm"]["g"].dtype == jnp.float32
    assert q["core"]["0"]["attn"]["q"]["w"].dtype == jnp.int8


def test_ptq_degrades_quality_but_stays_finite(tmp_path):
    run_dir, tc = bf16_run(tmp_path, steps=120)
    r = evaluate_both(run_dir, tc)
    assert np.isfinite(r["fp_val_ce"]) and np.isfinite(r["ptq_val_ce"])
    # quantizing a model that never saw quantization should cost something
    assert r["degradation"] > 0
    assert r["ptq_val_ce"] == pytest.approx(r["fp_val_ce"] + r["degradation"])


def test_ptq_forward_runs_at_every_loop_count(tmp_path):
    run_dir, tc = bf16_run(tmp_path)
    r = evaluate_both(run_dir, tc)
    assert set(r["per_loops_ptq"]) == {f"val_ce_L{L}" for L in tc.eval_loops}


def test_refuses_a_checkpoint_that_is_already_ternary(tmp_path):
    mc, tc, oc = build_configs(
        "smoke", {"quant": "sign"},
        dict(out_dir=str(tmp_path), run_name="tern", total_steps=3,
             eval_every=0, log_every=100), {},
    )
    train(mc, tc, oc, verbose=False)
    cfg, params = load_run(str(tmp_path / "tern"))
    with pytest.raises(ValueError, match="float checkpoint"):
        to_ternary(params, cfg)


def test_quantized_model_produces_finite_logits(tmp_path):
    run_dir, _ = bf16_run(tmp_path)
    cfg, params = load_run(run_dir)
    q, qcfg = to_ternary(params, cfg)
    toks = jnp.zeros((1, qcfg.seq_len), jnp.int32)
    out = forward(materialize(q, qcfg), toks, qcfg)
    assert jnp.all(jnp.isfinite(out))
