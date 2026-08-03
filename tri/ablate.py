"""Ablations and hyperparameter search.

Four studies, all small enough to run on one card between the smoke test and
the real run:

``sign``      TPE over the stochastic sign optimizer's knobs.  Start here: the
              flip-rate schedule is the single most sensitive thing in the
              whole project.
``modes``     bf16 vs STE-latent vs latent-free sign, each with its own
              learning rate tuned under the same trial budget.  Comparing arms
              at one shared LR would just measure which arm liked that LR.
``loops``     quality vs loop count at fixed stored parameters, plus a
              compute-matched control that spends the same FLOPs on depth.
``momentum``  what the momentum buffer actually buys, in val loss per bit.

Example:
    python -m tri.ablate --study sign --trials 40 --preset tiny --steps 800 \
        --dataset bin --data-dir data
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

from .config import build_configs
from .optim import state_bits_per_weight
from .train import train


def check_objective_is_learnable(preset: str, dataset: str | None) -> None:
    """Refuse to tune against a constant objective.

    ``synthetic`` is uniform random tokens: its loss floor is exactly
    ln(vocab_size), so every trial returns the same number and the search
    optimizes sampling noise.  It exists to measure throughput, not to train on.
    """
    _, tc, _ = build_configs(preset, {}, {"dataset": dataset}, {})
    if tc.dataset == "synthetic":
        raise SystemExit(
            "refusing to run a study on --dataset synthetic: those are uniform "
            "random tokens, so val loss is ln(vocab_size) for every trial no "
            "matter the hyperparameters.\n"
            "  real tokens:  --dataset bin --data-dir data   (see tri.prepare_data)\n"
            "  toy signal:   --dataset induction             (no download needed)"
        )


def run_trial(
    preset: str,
    model_over: dict,
    train_over: dict,
    optim_over: dict,
    trial=None,
    verbose: bool = False,
) -> dict:
    """One training run; reports intermediate val loss to Optuna for pruning."""
    mc, tc, oc = build_configs(preset, model_over, train_over, optim_over)
    state = {"pruned": False}

    def report(step, val_ce):
        if trial is None:
            return False
        if not math.isfinite(val_ce):
            state["pruned"] = True
            return True
        trial.report(val_ce, step)
        if trial.should_prune():
            state["pruned"] = True
            return True
        return False

    result = train(mc, tc, oc, report_fn=report, verbose=verbose)
    result["pruned"] = state["pruned"]
    result["state_bits"] = state_bits_per_weight(oc, mc.quant)
    return result


# -- search spaces -----------------------------------------------------


def _sign_space(trial) -> dict:
    rule = trial.suggest_categorical("sign_rule", ["stoch_round", "stoch_flip", "bop"])
    space = {
        "sign_rule": rule,
        "sign_step": trial.suggest_float("sign_step", 3e-3, 5e-1, log=True),
        "sign_b1": trial.suggest_float("sign_b1", 0.0, 0.98),
        # Widened downward: at `tiny`/3000 steps the winning b2 values pinned
        # against a 0.9 lower bound, so the optimum was outside the interval.
        # Short momentum memory suits a noisy flip process.
        "sign_b2": trial.suggest_float("sign_b2", 0.5, 0.999, log=False),
        "sign_normalize": trial.suggest_categorical("sign_normalize", ["rms", "absmean"]),
        "sign_precondition": trial.suggest_categorical("sign_precondition", ["none", "orthogonal"]),
        "adam_lr": trial.suggest_float("adam_lr", 3e-4, 1e-2, log=True),
        "muon_lr": trial.suggest_float("muon_lr", 2e-3, 1e-1, log=True),
    }
    if rule == "bop":
        space["sign_threshold"] = trial.suggest_float("sign_threshold", 1e-3, 1.0, log=True)
    return space


def _modes_space(trial) -> tuple[dict, dict]:
    quant = trial.suggest_categorical("quant", ["bf16", "ste", "sign"])
    optim = {"adam_lr": trial.suggest_float("adam_lr", 3e-4, 1e-2, log=True)}
    if quant == "sign":
        optim["sign_step"] = trial.suggest_float("sign_step", 3e-3, 5e-1, log=True)
        optim["sign_b1"] = trial.suggest_float("sign_b1", 0.0, 0.98)
    else:
        optim["muon_lr"] = trial.suggest_float("muon_lr", 2e-3, 1e-1, log=True)
    return {"quant": quant}, optim


def _study_fn(study: str, trial):
    """Return (model_overrides, train_overrides, optim_overrides) for a trial."""
    if study == "sign":
        return {"quant": "sign"}, {}, _sign_space(trial)
    if study == "modes":
        m, o = _modes_space(trial)
        return m, {}, o
    if study == "loops":
        loops = trial.suggest_categorical("n_loops", [1, 2, 3, 4])
        matched = trial.suggest_categorical("compute_matched", [False, True])
        model = {"quant": "sign", "n_loops": loops}
        if matched:
            # Same FLOPs as loops=1 by shrinking the core instead of looping.
            model["n_core"] = max(1, 4 // loops)
            model["n_loops"] = loops
        return model, {"loop_lo": loops, "loop_hi": loops}, {}
    if study == "momentum":
        dt = trial.suggest_categorical(
            "sign_momentum_dtype", ["none", "int8", "float16", "float32"]
        )
        return (
            {"quant": "sign"},
            {},
            {
                "sign_momentum_dtype": dt,
                "sign_step": trial.suggest_float("sign_step", 3e-3, 5e-1, log=True),
                "sign_b1": trial.suggest_float("sign_b1", 0.0, 0.98),
            },
        )
    raise ValueError(f"unknown study {study!r}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Ablations for tri-it-yourself")
    ap.add_argument("--study", default="sign", choices=["sign", "modes", "loops", "momentum"])
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default=None, choices=["synthetic", "induction", "bin"])
    ap.add_argument("--data-dir", default=None, help="directory holding train.bin / val.bin")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--device-tflops", type=float, default=None,
                    help="peak dense BF16 TFLOPS for MFU logging "
                         "(5090 209.5 | TPU v5e 197 | TPU v6e 918 | A100 312 | H100 989)")
    ap.add_argument("--out-dir", default="runs/ablate")
    ap.add_argument("--storage", default=None, help="e.g. sqlite:///runs/ablate/study.db")
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--time-budget-s", type=float, default=0.0, help="per trial")
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    try:
        import optuna
    except ImportError as e:  # pragma: no cover
        raise SystemExit("`pip install optuna` to run ablations") from e

    check_objective_is_learnable(args.preset, args.dataset)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.makedirs(args.out_dir, exist_ok=True)
    pruner = (
        optuna.pruners.NopPruner()
        if args.no_prune
        else optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1)
    )
    study = optuna.create_study(
        study_name=f"{args.study}-{args.preset}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=pruner,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )

    rows: list[dict] = []
    # Record each knob's true search bounds and scale.  Inferring them from the
    # sampled values misreads a linear range whose minimum happens to land near
    # zero as log-scaled, which then reports a healthy knob as pinned to an edge.
    dists: dict = {}
    t0 = time.time()

    def objective(trial):
        model_o, train_o, optim_o = _study_fn(args.study, trial)
        train_o = dict(train_o)
        train_o.update(
            {
                "run_name": f"{args.study}/t{trial.number:03d}",
                "out_dir": args.out_dir,
                "total_steps": args.steps,
                "seed": args.seed,
                "dataset": args.dataset,
                "data_dir": args.data_dir,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "device_tflops": args.device_tflops,
                "eval_every": args.eval_every or max(1, args.steps // 6),
                "time_budget_s": args.time_budget_s,
            }
        )
        for name, d in trial.distributions.items():
            if hasattr(d, "low") and hasattr(d, "high"):
                dists[name] = {
                    "low": float(d.low),
                    "high": float(d.high),
                    "log": bool(getattr(d, "log", False)),
                }
        r = run_trial(args.preset, model_o, train_o, optim_o, trial, verbose=args.verbose)
        rows.append(
            {
                "trial": trial.number,
                "params": dict(trial.params),
                "val_ce": r["best_val_ce"],
                "state_bits": r["state_bits"],
                "wall_s": r["wall_s"],
                "pruned": r["pruned"],
            }
        )
        print(
            f"trial {trial.number:>3}  val_ce {r['best_val_ce']:.4f}  "
            f"{r['wall_s']:>6.1f}s  {trial.params}",
            flush=True,
        )
        if r["pruned"]:
            raise optuna.TrialPruned()
        return r["best_val_ce"]

    study.optimize(objective, n_trials=args.trials, catch=(RuntimeError,))

    summary = {
        "study": args.study,
        "preset": args.preset,
        "steps": args.steps,
        "trials": rows,
        "distributions": dists,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "wall_s": time.time() - t0,
    }
    path = os.path.join(args.out_dir, f"{args.study}_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nbest val_ce {study.best_value:.4f} with {study.best_params}")
    print(f"summary -> {path}")

    if args.study in ("modes", "loops", "momentum"):
        key = {"modes": "quant", "loops": "n_loops", "momentum": "sign_momentum_dtype"}[args.study]
        best: dict = {}
        for r in rows:
            k = r["params"].get(key)
            if k is not None and (k not in best or r["val_ce"] < best[k]["val_ce"]):
                best[k] = r
        print(f"\nbest per {key}:")
        for k, r in sorted(best.items(), key=lambda kv: kv[1]["val_ce"]):
            print(f"  {str(k):>10}  val_ce {r['val_ce']:.4f}  state {r['state_bits']:.0f} bits/w")


if __name__ == "__main__":
    main()
