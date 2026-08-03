import json

import pytest

from tri.report import _completed, _split_params, report


def make_summary(tmp_path, rows):
    p = tmp_path / "sign_summary.json"
    p.write_text(json.dumps({"study": "sign", "preset": "tiny", "steps": 3000,
                             "trials": rows, "best_value": 1.0, "best_params": {}}))
    return str(p)


def trial(i, ce, step, rule="stoch_flip", pruned=False):
    return {"trial": i, "val_ce": ce, "pruned": pruned, "state_bits": 18, "wall_s": 1,
            "params": {"sign_step": step, "sign_rule": rule}}


def test_pruned_and_nonfinite_trials_are_excluded():
    rows = [trial(0, 5.0, 0.01), trial(1, 4.0, 0.02, pruned=True),
            {"trial": 2, "val_ce": float("inf"), "pruned": False, "params": {}}]
    assert [t["trial"] for t in _completed(rows)] == [0]


def test_params_split_by_type():
    cat, num = _split_params([trial(0, 5.0, 0.01)])
    assert cat == ["sign_rule"] and num == ["sign_step"]


def test_knob_is_unresolved_when_winners_span_the_search_range(tmp_path, capsys):
    """The point of the report: a lucky best value is not a determined one.

    val_ce here is deliberately uncorrelated with sign_step - the three best
    trials sit at both ends and the middle of the range, so the study says
    nothing about the knob even though one of them is nominally "best".
    """
    rows = [
        trial(0, 4.0, 0.001), trial(1, 4.1, 0.351), trial(2, 4.2, 0.151),
        trial(3, 9.0, 0.05), trial(4, 9.1, 0.10), trial(5, 9.2, 0.20),
        trial(6, 9.3, 0.25), trial(7, 9.4, 0.30),
    ]
    report(make_summary(tmp_path, rows))
    out = capsys.readouterr().out
    assert "UNRESOLVED" in out
    assert "covers 100%" in out


def test_knob_is_determined_when_winners_cluster(tmp_path, capsys):
    """Winners tightly grouped at one end: the study really did locate it."""
    good = [trial(i, 4.0 + 0.001 * i, 0.010 + 0.0001 * i) for i in range(4)]
    bad = [trial(10 + i, 9.0 + i, 0.5 + 0.1 * i) for i in range(8)]
    report(make_summary(tmp_path, good + bad))
    out = capsys.readouterr().out
    assert "determined" in out and "UNRESOLVED" not in out


def test_errors_when_every_trial_was_pruned(tmp_path):
    with pytest.raises(SystemExit, match="no completed trials"):
        report(make_summary(tmp_path, [trial(0, 5.0, 0.01, pruned=True)]))
