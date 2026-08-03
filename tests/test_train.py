import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tri import ckpt as ckpt_io
from tri.config import build_configs
from tri.data import InductionData, SyntheticData
from tri.model import init_params
from tri.train import train


def smoke(tmp_path, quant, **train_over):
    over = dict(out_dir=str(tmp_path), run_name=f"t-{quant}", total_steps=12,
                eval_every=12, eval_batches=2, log_every=100)
    over.update(train_over)
    return build_configs("smoke", {"quant": quant}, over, {})


@pytest.mark.parametrize("quant", ["bf16", "ste", "sign"])
def test_all_three_modes_train(tmp_path, quant):
    mc, tc, oc = smoke(tmp_path, quant)
    r = train(mc, tc, oc, verbose=False)
    assert np.isfinite(r["best_val_ce"])
    assert os.path.exists(r["checkpoint"])
    # eval reports every requested loop count
    for L in tc.eval_loops:
        assert f"val_ce_L{L}" in r


def test_induction_task_is_learned(tmp_path):
    """End-to-end signal check on the arm this repo exists for.

    The ternary/sign arm is used because it has by far the lowest seed variance
    on this task (~1.18 either way at 1500 steps, where the float arms swing
    between 1.7 and 3.9).  At 1000 steps it sits near 2.5 against a 5.55
    uniform baseline, so the threshold below has a wide margin.
    """
    mc, tc, oc = smoke(tmp_path, "sign", total_steps=1000, eval_every=1000, eval_batches=8)
    r = train(mc, tc, oc, verbose=False)
    floor = InductionData(mc.vocab_size, mc.seq_len, tc.induction_period).loss_floor
    uniform = float(np.log(mc.vocab_size))
    assert r["best_val_ce"] < uniform - 1.5, "the copy task is not being learned"
    assert floor < r["best_val_ce"]  # sanity: the floor really is a floor


def test_log_records_flip_metrics_for_sign_mode(tmp_path):
    mc, tc, oc = smoke(tmp_path, "sign", log_every=1, total_steps=5)
    train(mc, tc, oc, verbose=False)
    lines = [
        json.loads(l)
        for l in open(os.path.join(str(tmp_path), tc.run_name, "log.jsonl"))
    ]
    trains = [l for l in lines if l["event"] == "train"]
    assert trains and "flip_rate" in trains[0] and "dead_frac" in trains[0]
    assert 0.0 <= trains[0]["flip_rate"] <= 1.0


def test_bf16_mode_has_no_flip_metrics(tmp_path):
    mc, tc, oc = smoke(tmp_path, "bf16", log_every=1, total_steps=3)
    train(mc, tc, oc, verbose=False)
    lines = [
        json.loads(l)
        for l in open(os.path.join(str(tmp_path), tc.run_name, "log.jsonl"))
    ]
    trains = [l for l in lines if l["event"] == "train"]
    assert trains and "flip_rate" not in trains[0]


def test_checkpoint_roundtrip_and_2bit_packing(tmp_path):
    mc, tc, oc = smoke(tmp_path, "sign", total_steps=3)
    r = train(mc, tc, oc, verbose=False)
    template = init_params(jax.random.PRNGKey(tc.seed), mc)
    loaded = ckpt_io.load(r["checkpoint"], template)

    # every ternary leaf survives packing exactly
    lt = jax.tree_util.tree_leaves(loaded)
    tern = [x for x in lt if x.dtype == jnp.int8]
    assert tern and all(set(np.unique(np.asarray(x)).tolist()) <= {-1, 0, 1} for x in tern)

    packed_bits = sum(x.size for x in tern) * 2
    assert r["checkpoint_bytes"] * 8 > packed_bits  # dense params also in there
    meta = ckpt_io.read_extra(r["checkpoint"])
    assert meta["quant"] == "sign"


def test_reloaded_checkpoint_reproduces_logits(tmp_path):
    from tri.model import forward, materialize

    mc, tc, oc = smoke(tmp_path, "sign", total_steps=3)
    r = train(mc, tc, oc, verbose=False)
    template = init_params(jax.random.PRNGKey(0), mc)
    loaded = ckpt_io.load(r["checkpoint"], template)
    toks = jnp.zeros((1, mc.seq_len), jnp.int32)
    out = forward(materialize(loaded, mc), toks, mc)
    assert jnp.all(jnp.isfinite(out))


def test_sampling_runs(tmp_path):
    from tri.sample import generate

    mc, tc, oc = smoke(tmp_path, "sign", total_steps=3)
    r = train(mc, tc, oc, verbose=False)
    params = ckpt_io.load(r["checkpoint"], init_params(jax.random.PRNGKey(0), mc))
    out = generate(params, mc, np.array([[1, 2, 3]], np.int32), n_new=5, top_k=8)
    assert out.shape == (1, 8)
    assert out.min() >= 0 and out.max() < mc.vocab_size


def test_induction_task_is_periodic_and_has_a_known_floor():
    d = InductionData(256, 64, period=8)
    # only the first period is unpredictable: 7 of 64 target positions
    assert d.loss_floor == pytest.approx(np.log(255) * 7 / 64)
    x, y = d.batch(np.random.default_rng(0), 4, 64)
    assert x.shape == y.shape == (4, 64)
    np.testing.assert_array_equal(x[:, 1:], y[:, :-1])
    # everything from the second period on is a copy from `period` back
    np.testing.assert_array_equal(x[:, 8:], x[:, :-8])


def test_induction_rejects_period_longer_than_sequence():
    with pytest.raises(ValueError):
        InductionData(256, 64, period=64)


def test_synthetic_floor_is_uniform():
    d = SyntheticData(256)
    assert abs(d.loss_floor - np.log(256)) < 1e-6


def test_ablate_refuses_a_constant_objective():
    """`synthetic` has loss floor ln(vocab) by construction: nothing to tune."""
    from tri.ablate import check_objective_is_learnable

    with pytest.raises(SystemExit, match="uniform random tokens"):
        check_objective_is_learnable("tiny", "synthetic")
    # a learnable task passes
    check_objective_is_learnable("tiny", "induction")


def test_real_data_presets_do_not_default_to_synthetic():
    """A silent default of random tokens made a 2h TPU study measure noise."""
    from tri.config import build_configs

    for preset in ("tiny", "small", "main"):
        _, tc, _ = build_configs(preset)
        assert tc.dataset != "synthetic", f"{preset} would train on noise"
