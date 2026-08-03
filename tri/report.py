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
            for t in done:
                if p in t["params"]:
                    groups.setdefault(t["params"][p], []).append(t["val_ce"])
            if len(groups) < 2:
                continue
            print(f"  {p}")
            for v, xs in sorted(groups.items(), key=lambda kv: min(kv[1])):
                share = sum(1 for t in head if t["params"].get(p) == v)
                print(
                    f"    {str(v):<12} n={len(xs):<3} best={min(xs):.4f} "
                    f"mean={sum(xs)/len(xs):.4f}  {share}/{k} of top"
                )

    if num:
        print(f"\nnumeric knobs: range occupied by the top {k} trials")
        for p in num:
            vals = sorted(t["params"][p] for t in head if p in t["params"])
            allv = sorted(t["params"][p] for t in done if p in t["params"])
            if not vals or not allv:
                continue
            # Fraction of the searched interval that the top trials occupy.  A
            # knob whose winners are spread across the whole range was not
            # actually pinned down by the study, however decisive the single
            # best value looks.
            width = allv[-1] - allv[0]
            cover = (vals[-1] - vals[0]) / width if width > 0 else 0.0
            verdict = "determined" if cover < 0.5 else "UNRESOLVED"
            print(
                f"  {p:<18} top: {vals[0]:.4g} .. {vals[-1]:.4g} (median {vals[len(vals)//2]:.4g})"
                f"   covers {cover:>4.0%} of searched range   {verdict}"
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
