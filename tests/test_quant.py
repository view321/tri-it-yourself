import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tri.quant import (
    absmean_scale,
    bits_per_weight,
    init_ternary,
    pack2,
    stochastic_round,
    ternarize,
    ternarize_ste,
    unpack2,
)


def test_ternarize_is_on_the_lattice():
    w = jax.random.normal(jax.random.PRNGKey(0), (64, 32))
    q, scale = ternarize(w)
    assert set(np.unique(np.asarray(q)).tolist()) <= {-1.0, 0.0, 1.0}
    assert scale.shape == (1, 32)  # one gamma per output channel


def test_ternarize_per_tensor_vs_per_row():
    w = jax.random.normal(jax.random.PRNGKey(1), (16, 8))
    assert absmean_scale(w, per_row=False).shape == ()
    assert absmean_scale(w, per_row=True).shape == (1, 8)


def test_ste_forward_is_quantized_backward_is_identity():
    w = jax.random.normal(jax.random.PRNGKey(2), (16, 8))
    q, scale = ternarize(w)
    np.testing.assert_allclose(np.asarray(ternarize_ste(w)), np.asarray(q * scale), rtol=1e-5)

    g = jax.grad(lambda x: jnp.sum(ternarize_ste(x)))(w)
    np.testing.assert_allclose(np.asarray(g), np.ones_like(np.asarray(w)), atol=1e-6)


def test_stochastic_round_is_unbiased_and_integral():
    x = jnp.full((20000,), 0.3)
    r = stochastic_round(x, jax.random.PRNGKey(3))
    vals = np.unique(np.asarray(r))
    assert set(vals.tolist()) <= {0.0, 1.0}
    assert abs(float(jnp.mean(r)) - 0.3) < 0.02


def test_stochastic_round_exact_on_integers():
    x = jnp.array([-1.0, 0.0, 1.0])
    for seed in range(5):
        r = stochastic_round(x, jax.random.PRNGKey(seed))
        np.testing.assert_array_equal(np.asarray(r), np.asarray(x))


@pytest.mark.parametrize("p_zero", [0.0, 1 / 3, 0.6])
def test_init_ternary_distribution(p_zero):
    w = init_ternary(jax.random.PRNGKey(4), (200, 200), p_zero)
    assert w.dtype == jnp.int8
    assert set(np.unique(np.asarray(w)).tolist()) <= {-1, 0, 1}
    assert abs(float(jnp.mean(w == 0)) - p_zero) < 0.02
    # +1 and -1 balanced
    assert abs(float(jnp.mean(w == 1)) - float(jnp.mean(w == -1))) < 0.02


def test_pack_roundtrip_and_size():
    w = init_ternary(jax.random.PRNGKey(5), (37, 11))
    packed = pack2(w)
    assert packed.dtype == jnp.uint8
    assert packed.size == int(np.ceil(w.size / 4))  # exactly 2 bits per weight
    np.testing.assert_array_equal(np.asarray(unpack2(packed, w.shape)), np.asarray(w))


def test_bits_per_weight_accounting():
    assert bits_per_weight("none") == 2.0
    assert bits_per_weight("int8") == 10.0
    assert bits_per_weight("float16") == 18.0
    with pytest.raises(ValueError):
        bits_per_weight("float8")
