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
    good = [trial(i, 4.0 + 0.001 * i, 0.20 + 0.0001 * i) for i in range(4)]
    bad = [trial(10 + i, 9.0 + i, 0.05 + 0.2 * i) for i in range(6)]
    report(make_summary(tmp_path, good + bad))
    out = capsys.readouterr().out
    assert "determined" in out and "UNRESOLVED" not in out


def test_errors_when_every_trial_was_pruned(tmp_path):
    with pytest.raises(SystemExit, match="no completed trials"):
        report(make_summary(tmp_path, [trial(0, 5.0, 0.01, pruned=True)]))


def test_a_value_whose_trials_all_pruned_is_reported_not_dropped(tmp_path, capsys):
    """An arm that never survives is a finding; it must not vanish silently."""
    rows = [trial(i, 5.0 + 0.01 * i, 0.01, rule="stoch_flip") for i in range(4)]
    rows += [trial(10 + i, 9.0, 0.02, rule="stoch_round", pruned=True) for i in range(5)]
    report(make_summary(tmp_path, rows))
    out = capsys.readouterr().out
    assert "stoch_round" in out and "0 survived of 5 sampled (5 pruned)" in out


def test_winners_pinned_to_a_search_bound_are_flagged(tmp_path, capsys):
    """sign_b2 hugging its lower bound means the optimum is outside the range."""
    good = [trial(i, 4.0 + 0.01 * i, 0.901 + 0.001 * i) for i in range(4)]
    bad = [trial(10 + i, 9.0 + i, 0.95 + 0.01 * i) for i in range(6)]
    report(make_summary(tmp_path, good + bad))
    out = capsys.readouterr().out
    assert "AT LOWER EDGE" in out


def test_recorded_distributions_beat_guessing_the_scale(tmp_path, capsys):
    """A linear knob whose sampled min lands near zero must not read as log.

    sign_b1 is searched linearly on [0, 0.98]; inferring log scale from the
    sampled ratio would put a median of 0.70 at 92% of the range and wrongly
    flag a perfectly interior knob as pinned to the upper edge.
    """
    rows = [
        {"trial": i, "val_ce": 4.0 + 0.01 * i, "pruned": False,
         "params": {"sign_b1": v}}
        for i, v in enumerate([0.70, 0.69, 0.71, 0.90, 0.95, 0.017, 0.30, 0.55])
    ]
    payload = {"study": "sign", "preset": "tiny", "steps": 3000, "trials": rows,
               "distributions": {"sign_b1": {"low": 0.0, "high": 0.98, "log": False}},
               "best_value": 4.0, "best_params": {}}
    p = tmp_path / "sign_summary.json"
    p.write_text(json.dumps(payload))
    report(str(p))
    out = capsys.readouterr().out
    assert "[lin]" in out
    assert "EDGE" not in out


def test_existing_summary_is_preserved_not_overwritten(tmp_path):
    """Re-running a study must not destroy the result you compare against."""
    from tri.ablate import _preserve_existing

    p = tmp_path / "sign_summary.json"
    p.write_text('{"study": "first"}')
    first = _preserve_existing(str(p))
    assert first and json.loads(open(first).read())["study"] == "first"
    assert not p.exists()

    p.write_text('{"study": "second"}')
    second = _preserve_existing(str(p))
    assert second != first  # does not clobber the first backup either
    assert json.loads(open(second).read())["study"] == "second"
    assert json.loads(open(first).read())["study"] == "first"


def test_preserve_is_a_noop_when_nothing_exists(tmp_path):
    from tri.ablate import _preserve_existing

    assert _preserve_existing(str(tmp_path / "absent.json")) is None


def test_never_sampled_choice_is_distinguished_from_always_pruned(tmp_path, capsys):
    """The question this answers: absent because untried, or because it lost?"""
    rows = [trial(i, 5.0 + 0.01 * i, 0.01, rule="stoch_flip") for i in range(4)]
    rows += [trial(10 + i, 9.0, 0.02, rule="bop", pruned=True) for i in range(3)]
    payload = {"study": "sign", "preset": "tiny", "steps": 3000, "trials": rows,
               "distributions": {"sign_rule": {
                   "choices": ["stoch_round", "stoch_flip", "bop"]}},
               "best_value": 5.0, "best_params": {}}
    p = tmp_path / "sign_summary.json"
    p.write_text(json.dumps(payload))
    report(str(p))
    out = capsys.readouterr().out
    assert "stoch_round" in out and "NEVER SAMPLED in 7 trials" in out
    assert "bop" in out and "0 survived of 3 sampled" in out


def test_failed_trials_are_counted_not_silently_dropped(tmp_path, capsys):
    rows = [trial(i, 5.0 + 0.01 * i, 0.01, rule="stoch_flip") for i in range(4)]
    rows += [{"trial": 10, "val_ce": None, "pruned": False, "failed": True,
              "error": "RuntimeError: boom", "params": {"sign_step": 0.02,
                                                        "sign_rule": "bop"}}]
    report(make_summary(tmp_path, rows))
    out = capsys.readouterr().out
    assert "1 failed" in out
    assert "0 survived of 1 sampled" in out and "failed" in out


def test_fix_rule_pins_the_rule_and_removes_it_from_the_search(tmp_path):
    """Equal-budget comparison needs the rule fixed, not sampled by TPE."""
    optuna = pytest.importorskip("optuna")
    from tri.ablate import _sign_space

    seen = {}

    def objective(trial):
        seen[trial.number] = _sign_space(trial, fix_rule="stoch_round")
        return 1.0

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=4)

    assert all(s["sign_rule"] == "stoch_round" for s in seen.values())
    # the rule must not be a searched dimension when pinned
    assert all("sign_rule" not in t.params for t in study.trials)
    # the rest of the space is still explored
    assert len({s["sign_step"] for s in seen.values()}) > 1


def test_unpinned_sign_space_still_searches_the_rule():
    optuna = pytest.importorskip("optuna")
    from tri.ablate import _sign_space

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(lambda t: (_sign_space(t), 1.0)[1], n_trials=6)
    assert all("sign_rule" in t.params for t in study.trials)
