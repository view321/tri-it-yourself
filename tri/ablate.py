"""Ablations and hyperparameter search.

Four studies, all small enough to run on one card between the smoke test and
the real run:

``sign``      TPE over the stochastic sign optimizer's knobs.  Start here: the
              flip-rate schedule is the single most sensitive thing in the
              whole project.
``modes``     bf16 vs STE-latent vs latent-free sign, each with its own
              learning rate tuned under the same trial budget.  Comparing arms
              at one shared LR would just measure which arm liked that LR.
``loops``     is looping worth its FLOPs?  FLOP-matched, so a cheaper loop
              count buys proportionally more steps and the winner is the better
              use of a fixed budget - which is the question a compute budget
              actually poses.
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


def report_active_groups(preset: str, quant: str | None) -> dict:
    """Print how many parameters each optimizer group actually owns.

    A group with zero parameters makes its whole family of hyperparameters
    inert: in `sign` mode every block linear is ternary, so the Muon group is
    empty and muon_lr does nothing.  Searching it anyway costs budget and the
    result then reads like a tuned value.
    """
    import jax

    from .model import init_params, param_labels
    from .optim import group_sizes

    mc, _, oc = build_configs(preset, {"quant": quant}, {}, {})
    params = init_params(jax.random.PRNGKey(0), mc)
    sizes = group_sizes(params, param_labels(params, mc, oc.muon_on_latent))
    live = {g: n for g, n in sizes.items() if n}
    dead = [g for g, n in sizes.items() if not n]
    print(
        f"optimizer groups for quant={mc.quant}: "
        + ", ".join(f"{g}={n/1e6:.2f}M" for g, n in live.items())
        + (f"  |  empty: {', '.join(dead)}" if dead else "")
    )
    return sizes


def _preserve_existing(path: str) -> str | None:
    """Move an existing summary aside instead of overwriting it.

    Re-running a study with a changed search space is the normal workflow, and
    silently destroying the result you are comparing against is not acceptable.
    """
    if not os.path.exists(path):
        return None
    stem, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{stem}.prev{n}{ext}"):
        n += 1
    dest = f"{stem}.prev{n}{ext}"
    os.replace(path, dest)
    return dest


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


def _sign_space(trial, fix_rule: str | None = None) -> dict:
    # TPE allocates samples to whatever is winning, so a rule that loses its
    # first few draws gets tried less and never receives a fair test.  Pinning
    # the rule and giving each one its own study is the only way to compare
    # them on equal budget.
    rule = fix_rule or trial.suggest_categorical(
        "sign_rule", ["stoch_round", "stoch_flip", "bop", "ef"]
    )
    space = {
        "sign_rule": rule,
        "sign_b1": trial.suggest_float("sign_b1", 0.0, 0.98),
        # [0, 1) is the natural domain of an EMA decay and there is no principled
        # reason to exclude the short-memory end; b2=0 simply means "no memory".
        # An earlier 0.9 floor had the winners pinned against it.
        "sign_b2": trial.suggest_float("sign_b2", 0.0, 0.999, log=False),
        "sign_precondition": trial.suggest_categorical("sign_precondition", ["none", "orthogonal"]),
        # No muon_lr here.  In `sign` mode every block linear is ternary, so the
        # Muon group is empty and muon_lr cannot affect the objective - an
        # earlier study searched it anyway and burned an eighth of its budget on
        # a parameter with no effect.
        "adam_lr": trial.suggest_float("adam_lr", 3e-4, 2e-2, log=True),
    }
    # `normalize` only executes on the non-orthogonal path: with orthogonal
    # preconditioning every 2D tensor (i.e. every sign-group param) is
    # Newton-Schulz-whitened and then rms-rescaled, so the knob is inert - an
    # ef study searched it anyway and its winners split on it at random.
    if space["sign_precondition"] == "none":
        space["sign_normalize"] = trial.suggest_categorical(
            "sign_normalize", ["rms", "absmean"]
        )
    if rule == "bop":
        # bop flips when eta*|u| > threshold, so eta and threshold are
        # degenerate: only their ratio changes behaviour, and searching both
        # burns a dimension while making neither value interpretable.  Pin eta
        # at 1.0 - it still carries the schedule shape, annealing the flip rate
        # over training - and search the |u| cutoff directly.
        space["sign_step"] = 1.0
        space["sign_threshold"] = trial.suggest_float("sign_threshold", 1e-2, 5e1, log=True)
    else:
        # A brief detour widened this to 3e-4 on the theory that `stoch_round`
        # needs a much smaller step because it can move two lattice points at
        # once; the observed optima for both rules landed near 0.011, so the
        # theory was wrong and the extra decade only diluted the search.
        space["sign_step"] = trial.suggest_float("sign_step", 3e-3, 5e-1, log=True)
    if rule == "ef":
        # For ef the threshold is a hysteresis half-width in lattice units, not
        # a |u| cutoff: 0 is legal (pure integrate-and-fire) and the useful
        # range is a fraction of a cell, so search it linearly, not log.
        space["sign_threshold"] = trial.suggest_float("sign_threshold", 0.0, 0.2)
    return space


def _modes_space(trial, fix_quant: str | None = None) -> tuple[dict, dict]:
    """Each arm searches only the knobs that are live for it.

    `sign` mode makes every block linear ternary, so its Muon group is empty and
    muon_lr is inert; `bf16`/`ste` keep float matrices and have no sign knobs.
    Searching a dead parameter wastes budget and then reads as a tuned result.
    """
    quant = fix_quant or trial.suggest_categorical("quant", ["bf16", "ste", "sign"])
    optim = {"adam_lr": trial.suggest_float("adam_lr", 3e-4, 2e-2, log=True)}
    if quant == "sign":
        optim["sign_step"] = trial.suggest_float("sign_step", 3e-3, 5e-1, log=True)
        optim["sign_b1"] = trial.suggest_float("sign_b1", 0.0, 0.98)
        # b2 earned its place: its winners pinned against every bound it was given.
        optim["sign_b2"] = trial.suggest_float("sign_b2", 0.0, 0.999)
    else:
        optim["muon_lr"] = trial.suggest_float("muon_lr", 2e-3, 1e-1, log=True)
    return {"quant": quant}, optim


def flop_matched_steps(preset: str, loops: int, ref_steps: int, ref_loops: int | None = None) -> int:
    """Steps that spend the same FLOPs at `loops` as `ref_steps` does at the preset's default."""
    mc, _, _ = build_configs(preset)
    ref = ref_loops if ref_loops is not None else mc.n_loops
    return max(1, round(ref_steps * mc.flops_per_token(ref) / mc.flops_per_token(loops)))


def _study_fn(study: str, trial, fix_rule: str | None = None, fix_quant: str | None = None,
              fix_loops: int | None = None):
    """Return (model_overrides, train_overrides, optim_overrides) for a trial."""
    if study == "sign":
        return {"quant": "sign"}, {}, _sign_space(trial, fix_rule)
    if study == "modes":
        m, o = _modes_space(trial, fix_quant)
        return m, {}, o
    if study == "loops":
        loops = fix_loops or trial.suggest_categorical("n_loops", [1, 2, 3, 4])
        # FLOP-matched, which is the decision this study actually informs: a
        # fixed budget buys either more depth per token or more tokens, not
        # both.  Cheaper loop counts therefore get proportionally more steps,
        # so every arm spends the same compute and the winner is the better use
        # of it.  Comparing at equal *steps* would just confirm that more
        # compute helps, which needs no experiment.
        return ({"quant": "sign", "n_loops": loops},
                {"loop_lo": loops, "loop_hi": loops, "_flop_matched_loops": loops},
                {})
    if study == "momentum":
        dt = trial.suggest_categorical(
            "sign_momentum_dtype", ["none", "int8", "float16", "float32"]
        )
        return (
            {"quant": "sign"},
            {},
            {
                # fix_rule lets the loss-per-bit question be asked of any rule;
                # for `ef` the state accounting picks up the residual bits.
                "sign_rule": fix_rule,
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
    ap.add_argument("--tag", default="", help="suffix separating this study's outputs from earlier ones")
    ap.add_argument("--fix-rule", default=None,
                    choices=["stoch_round", "stoch_flip", "bop", "ef"],
                    help="pin sign_rule so each rule can be tuned on an equal budget; "
                         "TPE otherwise starves whichever rule loses early")
    ap.add_argument("--fix-loops", type=int, default=None, choices=[1, 2, 3, 4, 5],
                    help="pin n_loops for a FLOP-matched `loops` study")
    ap.add_argument("--fix-quant", default=None, choices=["bf16", "ste", "sign"],
                    help="pin the weight mode for a `modes` study; run one study per "
                         "arm so each gets an equal budget rather than TPE's allocation")
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
    report_active_groups(args.preset, args.fix_quant or ("sign" if args.study != "modes" else None))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"-{args.tag}" if args.tag else ""
    pruner = (
        optuna.pruners.NopPruner()
        if args.no_prune
        else optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1)
    )
    study = optuna.create_study(
        study_name=f"{args.study}{tag}-{args.preset}",
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
        model_o, train_o, optim_o = _study_fn(
            args.study, trial, args.fix_rule, args.fix_quant, args.fix_loops)
        train_o = dict(train_o)
        # A FLOP-matched study sets its own step count; everything else
        # takes --steps directly.
        matched = train_o.pop("_flop_matched_loops", None)
        steps = flop_matched_steps(args.preset, matched, args.steps) if matched else args.steps
        train_o.update(
            {
                # Tagging the directory keeps a re-run's per-trial log.jsonl from
                # being appended to the previous study's, which would interleave
                # two different search spaces in one file.
                "run_name": f"{args.study}{tag}/t{trial.number:03d}",
                "out_dir": args.out_dir,
                "total_steps": steps,
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
            if hasattr(d, "choices"):
                # Recording the choices lets the report distinguish "never
                # sampled" from "sampled but never survived".
                dists[name] = {"choices": [str(c) for c in d.choices]}
            elif hasattr(d, "low") and hasattr(d, "high"):
                dists[name] = {
                    "low": float(d.low),
                    "high": float(d.high),
                    "log": bool(getattr(d, "log", False)),
                }
        try:
            r = run_trial(args.preset, model_o, train_o, optim_o, trial, verbose=args.verbose)
        except Exception as e:
            # study.optimize(catch=...) would otherwise drop this trial from
            # both the summary and stdout, making a configuration that always
            # crashes indistinguishable from one that was never sampled.
            rows.append({
                "trial": trial.number,
                "params": dict(trial.params),
                "val_ce": None,
                "pruned": False,
                "failed": True,
                "error": f"{type(e).__name__}: {e}",
            })
            print(f"trial {trial.number:>3}  FAILED  {type(e).__name__}: {e}", flush=True)
            raise
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
    path = os.path.join(args.out_dir, f"{args.study}{tag}_summary.json")
    preserved = _preserve_existing(path)
    if preserved:
        print(f"previous summary kept at {preserved}")
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
