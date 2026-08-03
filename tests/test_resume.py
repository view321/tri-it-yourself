import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tri import ckpt as ckpt_io
from tri.config import build_configs
from tri.model import init_params, param_labels
from tri.optim import build_optimizer
from tri.quant import pack2, unpack2
from tri.train import train


def cfgs(tmp_path, name, **over):
    base = dict(out_dir=str(tmp_path), run_name=name, eval_every=0, eval_batches=2,
                log_every=100, seed=3)
    base.update(over)
    return build_configs("smoke", {"quant": "sign"}, base, {})


def final_weights(run_dir, mc):
    template = init_params(jax.random.PRNGKey(0), mc)
    return ckpt_io.load(os.path.join(run_dir, "final.npz"), template)


def test_resumed_run_matches_uninterrupted_run_exactly(tmp_path):
    """The whole point: killing and resuming must change nothing.

    Params, optimizer moments, the sign optimizer's PRNG key and both host RNG
    streams are all restored, so the two runs should agree bit for bit.

    Both segments must declare the same ``total_steps``, because that is what
    the LR/flip-rate schedules are built from - a first segment claiming 10
    total steps would run a differently shaped schedule over those 10 steps and
    diverge for reasons that have nothing to do with resume.
    """
    STEPS = 20
    mc_a, tc_a, oc_a = cfgs(tmp_path, "straight", total_steps=STEPS, eval_every=10)
    train(mc_a, tc_a, oc_a, verbose=False)

    # interrupt at step 10 while still declaring the full run length
    mc_b, tc_b, oc_b = cfgs(tmp_path, "split", total_steps=STEPS, ckpt_every=10, eval_every=10)
    train(mc_b, tc_b, oc_b, verbose=False, report_fn=lambda s, v: s >= 10)
    assert ckpt_io.latest_checkpoint(os.path.join(str(tmp_path), "split")) is not None

    mc_c, tc_c, oc_c = cfgs(tmp_path, "split", total_steps=STEPS, ckpt_every=10,
                            eval_every=10, resume="auto")
    train(mc_c, tc_c, oc_c, verbose=False)

    a = final_weights(os.path.join(str(tmp_path), "straight"), mc_a)
    b = final_weights(os.path.join(str(tmp_path), "split"), mc_c)
    for pa, pb in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_array_equal(np.asarray(pa), np.asarray(pb))


def test_checkpoint_carries_optimizer_state(tmp_path):
    """Params alone would silently restart every LR schedule from warmup."""
    mc, tc, oc = cfgs(tmp_path, "opt", total_steps=6, ckpt_every=6)
    train(mc, tc, oc, verbose=False)
    path = ckpt_io.latest_checkpoint(os.path.join(str(tmp_path), "opt"))
    assert path is not None

    params = init_params(jax.random.PRNGKey(tc.seed), mc)
    tx = build_optimizer(param_labels(params, mc, oc.muon_on_latent), oc, tc, seed=tc.seed)
    p2, opt2, meta = ckpt_io.load_train_state(path, params, tx.init(params))
    assert meta["step"] == 6
    assert meta["quant"] == "sign"
    # the sign optimizer's step counter survived, so schedules continue
    counts = [x for x in jax.tree_util.tree_leaves(opt2) if x.dtype == jnp.int32 and x.shape == ()]
    assert any(int(c) == 6 for c in counts)
    assert "data_rng" in meta and "loop_rng" in meta


def test_resume_restores_the_data_stream(tmp_path):
    """Without RNG restore a resumed run replays batches it already saw."""
    mc, tc, oc = cfgs(tmp_path, "rng", total_steps=5, ckpt_every=5)
    train(mc, tc, oc, verbose=False)
    path = ckpt_io.latest_checkpoint(os.path.join(str(tmp_path), "rng"))
    meta = ckpt_io.read_extra(path)

    def restored_rng():  # a fresh generator each time; drawing advances state
        g = np.random.default_rng(0)
        g.bit_generator.state = meta["data_rng"]
        return g

    # the saved stream is not where a fresh run would start
    assert restored_rng().integers(0, 10_000) != np.random.default_rng(tc.seed).integers(0, 10_000)

    # it is exactly where a stream that consumed 5 steps of batches would be
    from tri.data import InductionData

    src = InductionData(mc.vocab_size, mc.seq_len, tc.induction_period)
    replay = np.random.default_rng(tc.seed)
    for _ in range(5 * tc.grad_accum):
        src.batch(replay, tc.batch_size, mc.seq_len)
    assert restored_rng().bit_generator.state["state"] == replay.bit_generator.state["state"]


def test_resume_auto_with_no_checkpoint_starts_fresh(tmp_path):
    mc, tc, oc = cfgs(tmp_path, "none", total_steps=3, resume="auto")
    r = train(mc, tc, oc, verbose=False)
    assert np.isfinite(r["best_val_ce"])


def test_resume_rejects_a_different_architecture(tmp_path):
    mc, tc, oc = cfgs(tmp_path, "arch", total_steps=4, ckpt_every=4)
    train(mc, tc, oc, verbose=False)
    path = ckpt_io.latest_checkpoint(os.path.join(str(tmp_path), "arch"))

    mc2, tc2, oc2 = build_configs(
        "smoke", {"quant": "ste"},
        dict(out_dir=str(tmp_path), run_name="arch2", total_steps=8, resume=path,
             eval_every=0, log_every=100, seed=3), {},
    )
    with pytest.raises(ValueError, match="matching architecture"):
        train(mc2, tc2, oc2, verbose=False)


def test_checkpoint_rotation_keeps_only_the_newest(tmp_path):
    mc, tc, oc = cfgs(tmp_path, "rot", total_steps=9, ckpt_every=3, keep_last=2)
    train(mc, tc, oc, verbose=False)
    run_dir = os.path.join(str(tmp_path), "rot")
    kept = sorted(f for f in os.listdir(run_dir) if f.startswith("ckpt_"))
    assert kept == ["ckpt_0000006.npz", "ckpt_0000009.npz"]


def test_keep_all_when_keep_last_is_zero(tmp_path):
    mc, tc, oc = cfgs(tmp_path, "keepall", total_steps=6, ckpt_every=2, keep_last=0)
    train(mc, tc, oc, verbose=False)
    run_dir = os.path.join(str(tmp_path), "keepall")
    assert len([f for f in os.listdir(run_dir) if f.startswith("ckpt_")]) == 3


# -- the packing hazard ------------------------------------------------


def test_int8_momentum_is_not_bit_packed(tmp_path):
    """int8 momentum spans [-127,127]; 2-bit packing would truncate it."""
    mc, tc, oc = build_configs(
        "smoke", {"quant": "sign"},
        dict(out_dir=str(tmp_path), run_name="i8", total_steps=6, ckpt_every=6,
             eval_every=0, log_every=100),
        {"sign_momentum_dtype": "int8", "sign_step": 0.3},
    )
    train(mc, tc, oc, verbose=False)
    path = ckpt_io.latest_checkpoint(os.path.join(str(tmp_path), "i8"))

    params = init_params(jax.random.PRNGKey(tc.seed), mc)
    tx = build_optimizer(param_labels(params, mc, oc.muon_on_latent), oc, tc, seed=tc.seed)
    _, opt2, _ = ckpt_io.load_train_state(path, params, tx.init(params))

    buffers = [
        x for x in jax.tree_util.tree_leaves(opt2)
        if x.dtype == jnp.int8 and x.ndim == 2
    ]
    assert buffers, "expected int8 momentum buffers in the optimizer state"
    # a real buffer uses the full int8 range, not just {-1,0,1}
    assert max(int(jnp.max(jnp.abs(b))) for b in buffers) > 1


def test_pack_roundtrip_would_corrupt_out_of_range_values():
    """Documents why save() range-checks instead of packing every int8 leaf."""
    wide = jnp.array([[100, -100, 3, 0]], jnp.int8)
    assert not np.array_equal(np.asarray(unpack2(pack2(wide), wide.shape)), np.asarray(wide))
    lattice = jnp.array([[1, -1, 0, 1]], jnp.int8)
    np.testing.assert_array_equal(
        np.asarray(unpack2(pack2(lattice), lattice.shape)), np.asarray(lattice)
    )
