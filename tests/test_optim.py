import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from tri.config import OptimConfig, TrainConfig
from tri.muon import muon, newton_schulz, scale_by_muon, update_scale
from tri.optim import make_schedule, state_bits_per_weight
from tri.sign_opt import flip_stats, stochastic_sign


# -- Muon --------------------------------------------------------------


@pytest.mark.parametrize("shape", [(32, 16), (16, 32), (24, 24)])
def test_newton_schulz_whitens_the_spectrum(shape):
    g = jax.random.normal(jax.random.PRNGKey(0), shape)
    o = newton_schulz(g, steps=6, dtype=jnp.float32)
    assert o.shape == g.shape
    s_out = np.linalg.svd(np.asarray(o), compute_uv=False)
    # Keller's quintic is tuned for speed, not exact convergence: it lands the
    # whole spectrum near 1 (roughly [0.68, 1.13]) rather than exactly on it.
    assert 0.6 < s_out.min() and s_out.max() < 1.4


def test_newton_schulz_fixes_ill_conditioning():
    """The actual job: a gradient with a 1000:1 spectrum comes out ~flat."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    u, _ = jnp.linalg.qr(jax.random.normal(k1, (32, 32)))
    v, _ = jnp.linalg.qr(jax.random.normal(k2, (32, 32)))
    s = jnp.logspace(0, -3, 32)
    g = (u * s) @ v.T
    assert np.linalg.cond(np.asarray(g)) > 500

    o = newton_schulz(g, steps=6, dtype=jnp.float32)
    assert np.linalg.cond(np.asarray(o)) < 2.0


def test_newton_schulz_rejects_non_matrices():
    with pytest.raises(ValueError):
        newton_schulz(jnp.ones((8,)))


def test_update_scale_matches_fan_ratio():
    assert update_scale((100, 100)) == 1.0
    assert update_scale((100, 400)) == 2.0
    assert update_scale((400, 100)) == 1.0  # never shrinks below 1


def test_muon_step_moves_params_downhill():
    params = {"w": jax.random.normal(jax.random.PRNGKey(0), (16, 8))}
    tx = muon(0.05)
    state = tx.init(params)
    loss = lambda p: jnp.sum(jnp.square(p["w"]))
    before = float(loss(params))
    for _ in range(5):
        g = jax.grad(loss)(params)
        upd, state = tx.update(g, state, params)
        params = optax.apply_updates(params, upd)
    assert float(loss(params)) < before


def test_muon_rejects_non_matrix_params():
    params = {"b": jnp.ones((8,))}
    tx = scale_by_muon()
    state = tx.init(params)
    with pytest.raises(ValueError, match="route"):
        tx.update({"b": jnp.ones((8,))}, state, params)


# -- schedules ---------------------------------------------------------


def test_wsd_schedule_shape():
    oc = OptimConfig(schedule="wsd", warmup_frac=0.1, decay_frac=0.2, final_frac=0.0)
    s = make_schedule(1.0, 100, oc)
    assert float(s(0)) == 0.0
    assert float(s(10)) == pytest.approx(1.0)
    assert float(s(50)) == pytest.approx(1.0)  # stable phase
    assert float(s(99)) < 0.2  # decayed


def test_state_bits_accounting_favours_sign():
    oc = OptimConfig(sign_momentum_dtype="float16")
    assert state_bits_per_weight(oc, "sign") == 18.0
    assert state_bits_per_weight(oc, "ste") > state_bits_per_weight(oc, "sign")
    assert state_bits_per_weight(oc, "bf16") == 64.0


# -- stochastic sign ---------------------------------------------------


def ternary_params(shape=(32, 16), seed=0):
    from tri.quant import init_ternary

    return {"w": init_ternary(jax.random.PRNGKey(seed), shape)}


@pytest.mark.parametrize("rule", ["stoch_round", "stoch_flip", "bop"])
def test_updates_stay_on_the_lattice(rule):
    params = ternary_params()
    tx = stochastic_sign(0.2, rule=rule, threshold=0.5 if rule == "bop" else 0.0)
    state = tx.init(params)
    for i in range(10):
        g = {"w": jax.random.normal(jax.random.PRNGKey(i), params["w"].shape)}
        upd, state = tx.update(g, state, params)
        params = optax.apply_updates(params, upd)
        assert params["w"].dtype == jnp.int8
        assert set(np.unique(np.asarray(params["w"])).tolist()) <= {-1, 0, 1}


def test_zero_gradient_causes_no_flips():
    params = ternary_params()
    tx = stochastic_sign(0.5)
    state = tx.init(params)
    before = np.asarray(params["w"]).copy()
    for _ in range(3):
        upd, state = tx.update({"w": jnp.zeros_like(params["w"], jnp.float32)}, state, params)
        params = optax.apply_updates(params, upd)
    np.testing.assert_array_equal(np.asarray(params["w"]), before)


def test_consistent_gradient_drives_weights_against_it():
    """A steady positive gradient should push weights toward -1."""
    params = ternary_params((64, 64))
    tx = stochastic_sign(0.2, b1=0.9)
    state = tx.init(params)
    g = {"w": jnp.ones((64, 64), jnp.float32)}
    start = float(jnp.mean(params["w"]))
    for _ in range(30):
        upd, state = tx.update(g, state, params)
        params = optax.apply_updates(params, upd)
    end = float(jnp.mean(params["w"]))
    assert end < start
    assert end < -0.5  # most weights have walked to -1


def test_flip_rate_scales_with_step_size():
    def rate(step):
        params = ternary_params((64, 64), seed=1)
        tx = stochastic_sign(step, b1=0.0, momentum_dtype="none")
        state = tx.init(params)
        g = {"w": jax.random.normal(jax.random.PRNGKey(7), (64, 64))}
        upd, _ = tx.update(g, state, params)
        return float(flip_stats(upd, params)["flip_rate"])

    assert rate(0.01) < rate(0.1) < rate(0.5)


def test_max_flip_prob_caps_the_flip_rate():
    params = ternary_params((128, 128))
    tx = stochastic_sign(5.0, b1=0.0, momentum_dtype="none", max_flip_prob=0.05)
    state = tx.init(params)
    g = {"w": jax.random.normal(jax.random.PRNGKey(2), (128, 128))}
    upd, _ = tx.update(g, state, params)
    assert float(flip_stats(upd, params)["flip_rate"]) <= 0.06


@pytest.mark.parametrize("dtype", ["none", "int8", "float16", "bfloat16", "float32"])
def test_all_momentum_dtypes_train(dtype):
    params = ternary_params((32, 32))
    tx = stochastic_sign(0.1, momentum_dtype=dtype)
    state = tx.init(params)
    g = {"w": jnp.ones((32, 32), jnp.float32)}
    for _ in range(5):
        upd, state = tx.update(g, state, params)
        params = optax.apply_updates(params, upd)
    assert float(jnp.mean(params["w"])) < 0.0


def test_orthogonal_preconditioning_runs_and_flips():
    params = ternary_params((32, 16))
    tx = stochastic_sign(0.2, precondition="orthogonal")
    state = tx.init(params)
    g = {"w": jax.random.normal(jax.random.PRNGKey(3), (32, 16))}
    upd, _ = tx.update(g, state, params)
    assert float(flip_stats(upd, params)["flip_rate"]) > 0.0


def test_zero_bias_increases_sparsity():
    def dead_frac(bias):
        params = ternary_params((64, 64), seed=2)
        tx = stochastic_sign(0.15, zero_bias=bias)
        state = tx.init(params)
        for i in range(25):
            g = {"w": 0.01 * jax.random.normal(jax.random.PRNGKey(i), (64, 64))}
            upd, state = tx.update(g, state, params)
            params = optax.apply_updates(params, upd)
        return float(jnp.mean(params["w"] == 0))

    assert dead_frac(1.0) > dead_frac(0.0)


def test_flip_stats_describe_the_post_update_state():
    """Stats take pre-update weights and must report where they landed."""
    params = {"w": jnp.ones((64, 64), jnp.int8)}
    tx = stochastic_sign(1.0, b1=0.0, momentum_dtype="none")
    state = tx.init(params)
    upd, _ = tx.update({"w": jnp.ones((64, 64), jnp.float32)}, state, params)

    st = flip_stats(upd, params)
    after = np.asarray(params["w"]) + np.asarray(upd["w"])
    assert float(st["dead_frac"]) == pytest.approx(float(np.mean(after == 0)))
    assert float(st["to_zero_rate"]) == pytest.approx(float(np.mean(after == 0)))
    # a full-size step against a uniform gradient walks every +1 down to 0
    assert float(st["flip_rate"]) == pytest.approx(1.0)


def test_stateless_mode_keeps_no_buffer():
    params = ternary_params((16, 16))
    tx = stochastic_sign(0.1, momentum_dtype="none")
    state = tx.init(params)
    mu_size = sum(x.size for x in jax.tree_util.tree_leaves(state.mu))
    assert mu_size <= 1  # a scalar placeholder, not a per-weight buffer


def test_sign_optimizer_requires_params():
    tx = stochastic_sign(0.1)
    params = ternary_params((8, 8))
    state = tx.init(params)
    with pytest.raises(ValueError, match="params"):
        tx.update({"w": jnp.zeros((8, 8))}, state, None)


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        stochastic_sign(0.1, rule="nope")
    with pytest.raises(ValueError):
        stochastic_sign(0.1, momentum_dtype="float8")
