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
    out = []
    for t in trials:
        if t.get("pruned") or t.get("failed"):
            continue
        v = t.get("val_ce")
        if isinstance(v, (int, float)) and math.isfinite(v):
            out.append(t)
    return out


def _split_params(trials: list[dict]) -> tuple[list[str], list[str]]:
    cat, num = set(), set()
    for t in trials:
        for k, v in t["params"].items():
            (num if isinstance(v, (int, float)) and not isinstance(v, bool) else cat).add(k)
    return sorted(cat), sorted(num)


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for m in range(i, j + 1):
            out[order[m]] = avg
        i = j + 1
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation, computed without scipy.

    Reported alongside the top-quartile spread because that spread is partly a
    measure of how tightly TPE converged: once it locks onto a basin the best
    trials are near-duplicates and *every* knob looks "determined".  A
    correlation over all completed trials is a weaker but less circular signal
    of whether a knob actually moves the objective.
    """
    n = len(xs)
    if n < 5:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx > 0 and dy > 0 else None


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _cross_table(done: list[dict], a: str, b: str) -> None:
    """Best val_ce per (a, b) cell, with counts.

    Marginal tables hide interactions: a setting can look bad on average while
    being the best choice for one particular arm.  Counts are printed because
    most cells in a small study hold one or two trials.
    """
    cells: dict = {}
    for t in done:
        ka, kb = t["params"].get(a), t["params"].get(b)
        if ka is None or kb is None:
            continue
        cells.setdefault((str(ka), str(kb)), []).append(t["val_ce"])
    if not cells:
        print(f"\n(no completed trials carry both {a} and {b})")
        return
    rows = sorted({k[0] for k in cells})
    cols = sorted({k[1] for k in cells})
    w = max(len(r) for r in rows)
    print(f"\nbest val_ce by {a} x {b}   (n in parentheses)")
    print("  " + " " * w + "  " + "  ".join(f"{c:>16}" for c in cols))
    for r in rows:
        out = []
        for c in cols:
            xs = cells.get((r, c))
            out.append(f"{min(xs):>10.4f} ({len(xs):>2})" if xs else f"{'-':>16}")
        print(f"  {r:<{w}}  " + "  ".join(out))


def report(path: str, top: int = 8, uniform: float | None = None,
           cross: tuple[str, str] | None = None) -> dict:
    with open(path) as f:
        s = json.load(f)
    trials = s.get("trials", [])
    dists = s.get("distributions", {})
    done = _completed(trials)
    if not done:
        raise SystemExit(f"no completed trials in {path}")
    done.sort(key=lambda t: t["val_ce"])

    print(f"study={s.get('study')} preset={s.get('preset')} steps={s.get('steps')}")
    npruned = sum(1 for t in trials if t.get("pruned"))
    nfailed = sum(1 for t in trials if t.get("failed"))
    print(f"{len(trials)} trials, {len(done)} completed, {npruned} pruned, {nfailed} failed")
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
            failed: dict = {}
            for t in done:
                if p in t["params"]:
                    groups.setdefault(t["params"][p], []).append(t["val_ce"])
            for t in trials:
                v = t.get("params", {}).get(p)
                if v is None:
                    continue
                if t.get("failed"):
                    failed[v] = failed.get(v, 0) + 1
                elif t.get("pruned"):
                    pruned[v] = pruned.get(v, 0) + 1
            # A value sampled but never surviving is a finding, not an absence,
            # and one never sampled at all is a different finding again.  Both
            # vanish silently if only completed trials are tabulated.
            for v in list(pruned) + list(failed) + dists.get(p, {}).get("choices", []):
                groups.setdefault(v, [])
            if len(groups) < 2:
                continue
            print(f"  {p}")
            for v, xs in sorted(groups.items(), key=lambda kv: min(kv[1]) if kv[1] else math.inf):
                share = sum(1 for t in head if t["params"].get(p) == v)
                npr, nfa = pruned.get(v, 0), failed.get(v, 0)
                if not xs:
                    if npr or nfa:
                        why = f"{npr} pruned" + (f", {nfa} failed" if nfa else "")
                        print(f"    {str(v):<12} 0 survived of {npr + nfa} sampled ({why})")
                    else:
                        print(f"    {str(v):<12} NEVER SAMPLED in {len(trials)} trials")
                    continue
                tail = f"  ({npr} pruned)" if npr else ""
                tail += f"  ({nfa} FAILED)" if nfa else ""
                print(
                    f"    {str(v):<12} n={len(xs):<3} best={min(xs):.4f} "
                    f"mean={sum(xs)/len(xs):.4f}  {share}/{k} of top{tail}"
                )

    if cross:
        _cross_table(done, cross[0], cross[1])

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
            pairs = [(t["params"][p], t["val_ce"]) for t in done if p in t["params"]]
            rho = spearman([f(x) for x, _ in pairs], [y for _, y in pairs])
            rho_s = f"rho={rho:+.2f}" if rho is not None else "rho=n/a"
            print(
                f"  {p:<18} top: {vals[0]:.4g} .. {vals[-1]:.4g} (median {vals[len(vals)//2]:.4g})"
                f"  searched {b_lo:.4g} .. {b_hi:.4g} [{scale}]  covers {cover:>4.0%}"
                f"  {rho_s}  {verdict}"
            )
        print(
            "  rho = rank correlation with val_ce over all completed trials;"
            " |rho| near 0 means\n  the knob did not move the objective, however"
            " tightly the top trials cluster."
        )
    return s


def compare(paths: list[str], uniform: float | None = None) -> list[dict]:
    """Rank several studies side by side, e.g. one per --fix-quant arm.

    Reports the top-quartile median next to the best, because a single best
    trial is the statistic most likely to be luck - and two studies of the same
    task here have already disagreed on their winner.
    """
    import os

    rows = []
    for p in paths:
        with open(p) as f:
            s = json.load(f)
        done = _completed(s.get("trials", []))
        if not done:
            print(f"  {os.path.basename(p)}: no completed trials")
            continue
        done.sort(key=lambda t: t["val_ce"])
        k = min(len(done), max(3, len(done) // 4))
        rows.append({
            "name": os.path.basename(p).replace("_summary.json", ""),
            "best": done[0]["val_ce"],
            "top_median": done[k // 2]["val_ce"],
            "completed": len(done),
            "total": len(s.get("trials", [])),
            "params": done[0]["params"],
        })
    rows.sort(key=lambda r: r["best"])
    w = max((len(r["name"]) for r in rows), default=4)
    print(f"  {'study':<{w}}  {'best':>8}  {'topQ med':>9}  {'trials':>12}")
    for r in rows:
        extra = ""
        if uniform:
            extra = f"   ({uniform - r['best']:.2f} nats below uniform)"
        print(f"  {r['name']:<{w}}  {r['best']:>8.4f}  {r['top_median']:>9.4f}  "
              f"{r['completed']:>4}/{r['total']:<7}{extra}")
    if len(rows) > 1:
        gap = rows[1]["best"] - rows[0]["best"]
        spread = max(r["top_median"] - r["best"] for r in rows)
        print(f"\n  gap between the best two: {gap:.4f}")
        print(f"  widest best-to-top-quartile spread within one study: {spread:.4f}")
        if gap < spread:
            print("  the arms are separated by less than the within-study noise:"
                  " NOT a ranking")
    for r in rows:
        print(f"\n  {r['name']} best params: {r['params']}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Summarize a tri.ablate study")
    ap.add_argument("summary", nargs="+", help="path(s) to <study>_summary.json")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--uniform", type=float, default=None,
                    help="ln(vocab_size) to report nats-below-uniform, e.g. 10.3972")
    ap.add_argument("--cross", nargs=2, metavar=("A", "B"), default=None,
                    help="best val_ce per cell of two categoricals, e.g. "
                         "--cross sign_rule sign_precondition")
    args = ap.parse_args(argv)
    if len(args.summary) > 1:
        compare(args.summary, args.uniform)
    else:
        report(args.summary[0], args.top, args.uniform, args.cross)


if __name__ == "__main__":
    main()
