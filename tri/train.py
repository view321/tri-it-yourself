"""Training loop.

Gradient accumulation runs inside one jitted step (``lax.scan`` over
microbatches), so a step compiles once per loop count.  The loop count itself
is resampled every step from ``[loop_lo, loop_hi]``, which both regularizes the
core and lets you trade depth for quality at eval time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import ModelConfig, OptimConfig, TrainConfig, build_configs, config_json
from .data import build_data, loss_floor
from .model import count_params, init_params, loss_fn, materialize, param_labels
from .optim import build_optimizer, group_sizes, state_bits_per_weight, tree_bytes
from .sign_opt import flip_stats
from . import ckpt as ckpt_io


def _sign_leaves(tree, label_leaves):
    return [x for x, l in zip(jax.tree_util.tree_leaves(tree), label_leaves) if l == "sign"]


def _resume_meta(step, best, mc: ModelConfig, counts: dict, rng, loop_rng) -> dict:
    """Everything needed to continue a run that was killed mid-flight."""
    return {
        "step": int(step),
        "best": None if best == math.inf else float(best),
        "quant": mc.quant,
        "params_total": counts["total"],
        "data_rng": rng.bit_generator.state,
        "loop_rng": loop_rng.bit_generator.state,
    }


def _check_resume_compatible(meta: dict, mc: ModelConfig, params) -> None:
    """Catch a resume into a different architecture before it trains garbage.

    A structural mismatch already fails in the loader, but same-shape/different-
    hyperparameter resumes (e.g. switching quant mode) would otherwise proceed
    silently with meaningless optimizer state.
    """
    want = {"quant": mc.quant, "params_total": count_params(params)["total"]}
    for k, v in want.items():
        if k in meta and meta[k] != v:
            raise ValueError(
                f"checkpoint has {k}={meta[k]!r} but this config wants {v!r}; "
                "resume requires a matching architecture"
            )


def make_train_step(mc: ModelConfig, tx, label_leaves, donate: bool = True):
    def step(params, opt_state, xs, ys, n_loops: int):
        fparams = materialize(params, mc)

        def micro(fp, x, y):
            return loss_fn(fp, (x, y), mc, n_loops)

        grad_fn = jax.value_and_grad(micro, has_aux=True)
        zeros = jax.tree.map(lambda p: jnp.zeros(p.shape, jnp.float32), fparams)

        def body(carry, mb):
            gacc, lacc, ceacc = carry
            (l, aux), g = grad_fn(fparams, mb[0], mb[1])
            gacc = jax.tree.map(lambda a, b: a + b.astype(jnp.float32), gacc, g)
            return (gacc, lacc + l, ceacc + aux["ce"]), None

        (grads, tot_loss, tot_ce), _ = jax.lax.scan(body, (zeros, 0.0, 0.0), (xs, ys))
        n = xs.shape[0]
        grads = jax.tree.map(lambda g: g / n, grads)

        updates, opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        metrics = {
            "loss": tot_loss / n,
            "ce": tot_ce / n,
            "grad_norm": optax.tree.norm(grads),
        }
        sd = _sign_leaves(updates, label_leaves)
        if sd:
            # stats want the pre-update weights, so read `params`, not `new_params`
            metrics.update(flip_stats(sd, _sign_leaves(params, label_leaves)))
        return new_params, opt_state, metrics

    kw = {"static_argnames": ("n_loops",)}
    if donate:
        kw["donate_argnames"] = ("params", "opt_state")
    return jax.jit(step, **kw)


def make_eval_step(mc: ModelConfig):
    @partial(jax.jit, static_argnames=("n_loops",))
    def ev(params, x, y, n_loops: int):
        _, aux = loss_fn(materialize(params, mc), (x, y), mc, n_loops)
        return aux["ce"]

    return ev


def evaluate(ev, params, source, tc: TrainConfig, mc: ModelConfig, seed: int = 1234) -> dict:
    """Mean CE at each eval loop count, on a fixed (seeded) set of batches."""
    out = {}
    for L in tc.eval_loops:
        rng = np.random.default_rng(seed)
        tot = 0.0
        for _ in range(tc.eval_batches):
            x, y = source.batch(rng, tc.batch_size, mc.seq_len)
            tot += float(ev(params, jnp.asarray(x), jnp.asarray(y), L))
        out[f"val_ce_L{L}"] = tot / tc.eval_batches
    trained = mc.n_loops if mc.n_loops in tc.eval_loops else tc.eval_loops[-1]
    out["val_ce"] = out[f"val_ce_L{trained}"]
    return out


def train(
    mc: ModelConfig,
    tc: TrainConfig,
    oc: OptimConfig,
    log_fn=None,
    report_fn=None,
    verbose: bool = True,
) -> dict:
    """Run training.  Returns final metrics.

    ``report_fn(step, val_ce)`` is called at each eval; return ``True`` from it
    to stop early (Optuna pruning uses this).
    """
    t_start = time.time()
    run_dir = os.path.join(tc.out_dir, tc.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        f.write(config_json(mc, tc, oc))
    log_path = os.path.join(run_dir, "log.jsonl")
    log_file = open(log_path, "a")

    def emit(rec: dict):
        log_file.write(json.dumps(rec) + "\n")
        log_file.flush()
        if log_fn:
            log_fn(rec)

    key = jax.random.PRNGKey(tc.seed)
    params = init_params(key, mc)
    labels = param_labels(params, mc, oc.muon_on_latent)
    label_leaves = jax.tree_util.tree_leaves(labels)
    tx = build_optimizer(labels, oc, tc, seed=tc.seed)
    opt_state = tx.init(params)

    rng = np.random.default_rng(tc.seed)
    loop_rng = np.random.default_rng(tc.seed + 1)
    start_step = 1
    best = math.inf
    resumed_from = None
    if tc.resume:
        path = ckpt_io.latest_checkpoint(run_dir) if tc.resume == "auto" else tc.resume
        if path is None:
            if verbose:
                print(f"[{tc.run_name}] --resume auto found no checkpoint; starting fresh")
        else:
            # Check the cheap metadata before unflattening the tree, so a
            # mismatched resume reports why instead of a missing-leaf KeyError.
            _check_resume_compatible(ckpt_io.read_extra(path), mc, params)
            params, opt_state, meta = ckpt_io.load_train_state(path, params, opt_state)
            start_step = int(meta["step"]) + 1
            saved_best = meta.get("best")  # stored as null when still infinite
            best = math.inf if saved_best is None else float(saved_best)
            # Restoring the data streams matters: without it a resumed run
            # replays the same batches it already trained on.
            rng.bit_generator.state = meta["data_rng"]
            loop_rng.bit_generator.state = meta["loop_rng"]
            resumed_from = path
            if verbose:
                print(f"[{tc.run_name}] resumed from {path} at step {start_step}", flush=True)

    train_src, val_src = build_data(tc, mc)
    donate = jax.default_backend() != "cpu"
    step_fn = make_train_step(mc, tx, label_leaves, donate=donate)
    ev = make_eval_step(mc)

    counts = count_params(params)
    sizes = group_sizes(params, labels)
    tokens_per_step = tc.batch_size * tc.grad_accum * mc.seq_len
    header = {
        "event": "start",
        "quant": mc.quant,
        "params_total": counts["total"],
        "params_ternary": counts["ternary"],
        "groups": sizes,
        "compute_equivalent": mc.param_counts()["compute_equivalent"],
        "state_bits_per_ternary_weight": state_bits_per_weight(oc, mc.quant),
        "param_bytes": tree_bytes(params),
        "opt_state_bytes": tree_bytes(opt_state),
        "tokens_per_step": tokens_per_step,
        "loss_floor": loss_floor(val_src),
        "backend": jax.default_backend(),
        "device_tflops": tc.device_tflops,
        "resumed_from": resumed_from,
        "start_step": start_step,
    }
    emit(header)
    if verbose:
        print(
            f"[{tc.run_name}] quant={mc.quant} params={counts['total']/1e6:.1f}M "
            f"(ternary {counts['ternary']/1e6:.1f}M) "
            f"compute-equiv={header['compute_equivalent']/1e6:.1f}M "
            f"state={header['state_bits_per_ternary_weight']:.0f} bits/w "
            f"opt_state={header['opt_state_bytes']/1e6:.1f}MB "
            f"tok/step={tokens_per_step} backend={header['backend']}",
            flush=True,
        )

    final: dict = {}
    t_last = time.time()
    tokens_since = 0

    for step in range(start_step, tc.total_steps + 1):
        n_loops = int(loop_rng.integers(tc.loop_lo, tc.loop_hi + 1)) if mc.n_core else 1
        xb, yb = [], []
        for _ in range(tc.grad_accum):
            x, y = train_src.batch(rng, tc.batch_size, mc.seq_len)
            xb.append(x)
            yb.append(y)
        xs = jnp.asarray(np.stack(xb))
        ys = jnp.asarray(np.stack(yb))

        params, opt_state, m = step_fn(params, opt_state, xs, ys, n_loops)
        tokens_since += tokens_per_step

        if step % tc.log_every == 0 or step == 1:
            m = {k: float(v) for k, v in m.items()}
            dt = time.time() - t_last
            tps = tokens_since / max(dt, 1e-9)
            rec = {
                "event": "train",
                "step": step,
                "n_loops": n_loops,
                "tokens_per_s": tps,
                "elapsed": time.time() - t_start,
                **m,
            }
            if tc.profile_mfu and tc.device_tflops > 0:
                rec["mfu"] = tps * mc.flops_per_token(n_loops) / (tc.device_tflops * 1e12)
            emit(rec)
            if verbose:
                extra = ""
                if "flip_rate" in m:
                    extra = f" flip={m['flip_rate']:.4f} dead={m['dead_frac']:.3f}"
                print(
                    f"  step {step:>6} loss {m['loss']:.4f} ce {m['ce']:.4f} "
                    f"gnorm {m['grad_norm']:.3f}{extra} {tps:,.0f} tok/s",
                    flush=True,
                )
            t_last = time.time()
            tokens_since = 0

        # Checkpoint before evaluating: on a preemptible instance the step that
        # just finished is what we want on disk, and an early stop below must
        # not discard it.
        if tc.ckpt_every and step % tc.ckpt_every == 0:
            ckpt_io.save_train_state(
                os.path.join(run_dir, f"ckpt_{step:07d}.npz"),
                params,
                opt_state,
                _resume_meta(step, best, mc, counts, rng, loop_rng),
            )
            ckpt_io.rotate_checkpoints(run_dir, tc.keep_last)

        do_eval = tc.eval_every and (step % tc.eval_every == 0 or step == tc.total_steps)
        if do_eval:
            ev_metrics = evaluate(ev, params, val_src, tc, mc)
            best = min(best, ev_metrics["val_ce"])
            rec = {"event": "eval", "step": step, "best": best, **ev_metrics}
            emit(rec)
            final = ev_metrics
            if verbose:
                per_loop = " ".join(
                    f"L{L}={ev_metrics[f'val_ce_L{L}']:.4f}" for L in tc.eval_loops
                )
                print(f"  eval  {step:>6} val_ce {ev_metrics['val_ce']:.4f} | {per_loop}", flush=True)
            if report_fn is not None and report_fn(step, ev_metrics["val_ce"]):
                emit({"event": "pruned", "step": step})
                break
            t_last = time.time()
            tokens_since = 0

        if tc.time_budget_s and (time.time() - t_start) > tc.time_budget_s:
            emit({"event": "time_budget_reached", "step": step})
            if verbose:
                print(f"  stopping: time budget {tc.time_budget_s}s reached at step {step}")
            break

    if not final:
        final = evaluate(ev, params, val_src, tc, mc)
        best = min(best, final["val_ce"])
    path = ckpt_io.save(
        os.path.join(run_dir, "final.npz"), params, pack=True,
        extra={"quant": mc.quant, "steps": tc.total_steps},
    )
    result = {
        **final,
        "best_val_ce": best,
        "wall_s": time.time() - t_start,
        "checkpoint": path,
        "checkpoint_bytes": ckpt_io.checkpoint_bytes(path),
        "params_total": counts["total"],
        "params_ternary": counts["ternary"],
    }
    emit({"event": "done", **result})
    log_file.close()
    if verbose:
        print(
            f"[{tc.run_name}] done in {result['wall_s']:.1f}s  best val_ce {best:.4f}  "
            f"ckpt {result['checkpoint_bytes']/1e6:.2f}MB",
            flush=True,
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a ternary-weight looped transformer")
    p.add_argument("--preset", default="main", help="smoke | tiny | small | main")
    p.add_argument("--quant", default=None, choices=["bf16", "ste", "sign"])
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--dataset", default=None, choices=["synthetic", "induction", "bin"])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--dtype", default=None, choices=["float32", "bfloat16", "float16"])
    p.add_argument("--n-loops", type=int, default=None)
    p.add_argument("--loop-lo", type=int, default=None)
    p.add_argument("--loop-hi", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--keep-last", type=int, default=None,
                   help="checkpoints to retain (<=0 keeps all)")
    p.add_argument("--resume", default=None,
                   help="'auto' for the newest checkpoint in the run dir, or a path")
    p.add_argument("--device-tflops", type=float, default=None,
                   help="peak dense BF16 TFLOPS for MFU logging "
                        "(5090 209.5 | TPU v5e 197 | TPU v6e 918 | A100 312 | H100 989)")
    p.add_argument("--time-budget-s", type=float, default=None)
    p.add_argument("--no-remat", action="store_true")
    p.add_argument("--act-bits", type=int, default=None)
    # optimizer
    p.add_argument("--sign-step", type=float, default=None)
    p.add_argument("--sign-b1", type=float, default=None)
    p.add_argument("--sign-b2", type=float, default=None)
    p.add_argument("--sign-rule", default=None,
                   choices=["stoch_round", "stoch_flip", "bop", "ef"])
    p.add_argument("--sign-normalize", default=None, choices=["rms", "absmean", "none"])
    p.add_argument("--sign-precondition", default=None, choices=["none", "orthogonal"])
    p.add_argument("--sign-threshold", type=float, default=None)
    p.add_argument("--sign-momentum-dtype", default=None,
                   choices=["none", "int8", "float16", "bfloat16", "float32"])
    p.add_argument("--sign-residual-dtype", default=None,
                   choices=["int8", "float16", "float32"],
                   help="storage precision of the ef rule's residual")
    p.add_argument("--sign-zero-bias", type=float, default=None)
    p.add_argument("--muon-lr", type=float, default=None)
    p.add_argument("--adam-lr", type=float, default=None)
    p.add_argument("--schedule", default=None, choices=["wsd", "cosine", "constant"])
    return p


def configs_from_args(args) -> tuple[ModelConfig, TrainConfig, OptimConfig]:
    model = {
        "quant": args.quant,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "n_loops": args.n_loops,
        "act_bits": args.act_bits,
    }
    if args.no_remat:
        model["remat"] = False
    train_o = {
        "run_name": args.run_name or (f"{args.preset}-{args.quant or 'sign'}"),
        "out_dir": args.out_dir,
        "seed": args.seed,
        "total_steps": args.steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "loop_lo": args.loop_lo,
        "loop_hi": args.loop_hi,
        "eval_every": args.eval_every,
        "ckpt_every": args.ckpt_every,
        "keep_last": args.keep_last,
        "resume": args.resume,
        "device_tflops": args.device_tflops,
        "time_budget_s": args.time_budget_s,
    }
    optim = {
        "sign_step": args.sign_step,
        "sign_b1": args.sign_b1,
        "sign_b2": args.sign_b2,
        "sign_rule": args.sign_rule,
        "sign_normalize": args.sign_normalize,
        "sign_precondition": args.sign_precondition,
        "sign_threshold": args.sign_threshold,
        "sign_momentum_dtype": args.sign_momentum_dtype,
        "sign_residual_dtype": args.sign_residual_dtype,
        "sign_zero_bias": args.sign_zero_bias,
        "muon_lr": args.muon_lr,
        "adam_lr": args.adam_lr,
        "schedule": args.schedule,
    }
    return build_configs(args.preset, model, train_o, optim)


def main(argv=None):
    args = build_parser().parse_args(argv)
    mc, tc, oc = configs_from_args(args)
    train(mc, tc, oc)


if __name__ == "__main__":
    main()
