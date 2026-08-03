"""Summarize an Optuna study written by :mod:`tri.ablate`.

The single best trial of a few dozen in an eight-dimensional space is partly
luck.  This reports the *distribution* of good trials instead: which categorical
choices win on average, and what range of each numeric knob the top trials
actually occupy.  A hyperparameter whose best value looks decisive but whose
top-quartile range spans the whole search interval was not really determined by
the study.

    python -m tri.report runs/ablate/sign_summary.json
"""

from __future__ import annotations

import argparse
import json
import math


def _completed(trials: list[dict]) -> list[dict]:
    return [t for t in trials if not t.get("pruned") and math.isfinite(t.get("val_ce", math.inf))]


def _split_params(trials: list[dict]) -> tuple[list[str], list[str]]:
    cat, num = set(), set()
    for t in trials:
        for k, v in t["params"].items():
            (num if isinstance(v, (int, float)) and not isinstance(v, bool) else cat).add(k)
    return sorted(cat), sorted(num)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def report(path: str, top: int = 8, uniform: float | None = None) -> dict:
    with open(path) as f:
        s = json.load(f)
    trials = s.get("trials", [])
    dists = s.get("distributions", {})
    done = _completed(trials)
    if not done:
        raise SystemExit(f"no completed trials in {path}")
    done.sort(key=lambda t: t["val_ce"])

    print(f"study={s.get('study')} preset={s.get('preset')} steps={s.get('steps')}")
    print(f"{len(trials)} trials, {len(done)} completed, {len(trials)-len(done)} pruned")
    best = done[0]
    line = f"best val_ce {best['val_ce']:.4f}"
    if uniform:
        line += f"  ({uniform - best['val_ce']:.2f} nats below uniform, ppl {math.exp(best['val_ce']):.0f})"
    print(line)

    cat, num = _split_params(done)
    # The analysis slice is a top quartile, independent of how many trials the
    # caller wants printed.  Tying it to `top` would make the "best" set most of
    # the study whenever few trials completed, and nothing would look determined.
    k = min(len(done), max(3, len(done) // 4))
    head = done[:k]

    print(f"\ntop {min(top, len(done))} trials")
    for t in done[:top]:
        ps = " ".join(f"{a}={_fmt(b)}" for a, b in sorted(t["params"].items()))
        print(f"  {t['val_ce']:.4f}  {ps}")

    if cat:
        print("\ncategorical choices (best / mean over completed trials)")
        for p in cat:
            groups: dict = {}
            pruned: dict = {}
            for t in done:
                if p in t["params"]:
                    groups.setdefault(t["params"][p], []).append(t["val_ce"])
            for t in trials:
                if t.get("pruned") and p in t.get("params", {}):
                    pruned[t["params"][p]] = pruned.get(t["params"][p], 0) + 1
            # A value that was sampled but never survived is a finding, not an
            # absence; without this it silently vanishes from the table.
            for v in pruned:
                groups.setdefault(v, [])
            if len(groups) < 2:
                continue
            print(f"  {p}")
            for v, xs in sorted(groups.items(), key=lambda kv: min(kv[1]) if kv[1] else math.inf):
                share = sum(1 for t in head if t["params"].get(p) == v)
                npr = pruned.get(v, 0)
                if not xs:
                    print(f"    {str(v):<12} n=0   ALL {npr} SAMPLED TRIALS PRUNED")
                    continue
                print(
                    f"    {str(v):<12} n={len(xs):<3} best={min(xs):.4f} "
                    f"mean={sum(xs)/len(xs):.4f}  {share}/{k} of top  ({npr} pruned)"
                )

    if num:
        print(f"\nnumeric knobs: range occupied by the top {k} trials")
        for p in num:
            vals = sorted(t["params"][p] for t in head if p in t["params"])
            # Range searched includes pruned trials: they still sampled the space,
            # and excluding them would understate how wide the search really was.
            allv = sorted(t["params"][p] for t in trials if p in t.get("params", {}))
            if not vals or not allv:
                continue
            # `cover` is the fraction of the searched interval the top trials
            # occupy: winners spread across the whole range mean the study never
            # pinned the knob down, however decisive the single best value looks.
            # Prefer the recorded search space over the sampled extremes, which
            # older summaries are all we have.
            info = dists.get(p)
            if info:
                b_lo, b_hi, logscale = info["low"], info["high"], info["log"]
            else:
                b_lo, b_hi = allv[0], allv[-1]
                logscale = b_lo > 0 and b_hi / b_lo > 20
            f = (lambda x: math.log(x)) if logscale and b_lo > 0 else (lambda x: x)
            lo, hi = f(b_lo), f(b_hi)
            span = hi - lo
            cover = (f(vals[-1]) - f(vals[0])) / span if span > 0 else 0.0
            verdict = "determined" if cover < 0.5 else "UNRESOLVED"
            if span > 0 and cover < 0.5:
                # Winners bunched against a bound mean the optimum may lie
                # outside the interval, so "determined" would be misleading.
                pos = (f(vals[len(vals) // 2]) - lo) / span
                if pos < 0.20:
                    verdict = "AT LOWER EDGE - widen the search range downward"
                elif pos > 0.80:
                    verdict = "AT UPPER EDGE - widen the search range upward"
            scale = "log" if logscale else "lin"
            print(
                f"  {p:<18} top: {vals[0]:.4g} .. {vals[-1]:.4g} (median {vals[len(vals)//2]:.4g})"
                f"  searched {b_lo:.4g} .. {b_hi:.4g} [{scale}]  covers {cover:>4.0%}  {verdict}"
            )
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description="Summarize a tri.ablate study")
    ap.add_argument("summary", help="path to <study>_summary.json")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--uniform", type=float, default=None,
                    help="ln(vocab_size) to report nats-below-uniform, e.g. 10.3972")
    args = ap.parse_args(argv)
    report(args.summary, args.top, args.uniform)


if __name__ == "__main__":
    main()
