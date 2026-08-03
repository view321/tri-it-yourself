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


def test_cross_table_exposes_an_interaction_a_marginal_hides(tmp_path, capsys):
    """orthogonal can be bad on average yet best for one rule specifically."""
    def t(i, ce, rule, pre):
        return {"trial": i, "val_ce": ce, "pruned": False,
                "params": {"sign_rule": rule, "sign_precondition": pre, "sign_step": 0.01}}

    rows = [
        t(0, 5.31, "stoch_round", "orthogonal"), t(1, 5.37, "stoch_round", "orthogonal"),
        t(2, 6.40, "stoch_round", "none"),
        t(3, 4.94, "stoch_flip", "none"), t(4, 5.04, "stoch_flip", "none"),
        t(5, 6.20, "stoch_flip", "orthogonal"),
    ]
    report(make_summary(tmp_path, rows), cross=("sign_rule", "sign_precondition"))
    out = capsys.readouterr().out
    assert "best val_ce by sign_rule x sign_precondition" in out
    # orthogonal wins for stoch_round, loses for stoch_flip: an interaction
    assert "5.3100" in out and "4.9400" in out
    assert "( 2)" in out  # cell counts are shown (right-aligned to width 2)


def test_cross_table_handles_a_missing_param(tmp_path, capsys):
    report(make_summary(tmp_path, [trial(0, 5.0, 0.01)]), cross=("sign_rule", "nope"))
    assert "no completed trials carry both" in capsys.readouterr().out


def test_spearman_matches_known_cases():
    from tri.report import spearman

    assert spearman([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3], [1, 2, 3]) is None  # too few points to be meaningful
    assert spearman([1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6]) is None  # no variance


def test_rho_flags_a_knob_that_does_not_move_the_objective(tmp_path, capsys):
    """A converged TPE makes every knob look 'determined'; rho should not."""
    # sign_step is irrelevant here: val_ce is driven purely by trial index
    rows = [
        {"trial": i, "val_ce": 5.0 + 0.01 * i, "pruned": False,
         "params": {"sign_step": s}}
        for i, s in enumerate([0.01, 0.4, 0.02, 0.3, 0.011, 0.25, 0.012, 0.2])
    ]
    report(make_summary(tmp_path, rows))
    out = capsys.readouterr().out
    assert "rho=" in out
    assert "rank correlation with val_ce" in out


def test_bop_pins_the_step_and_searches_the_cutoff(tmp_path):
    """eta and threshold are degenerate for bop, so only one may be searched."""
    optuna = pytest.importorskip("optuna")
    from tri.ablate import _sign_space

    seen = []

    def objective(trial):
        seen.append(_sign_space(trial, fix_rule="bop"))
        return 1.0

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=5)

    assert all(s["sign_step"] == 1.0 for s in seen)
    assert all("sign_step" not in t.params for t in study.trials)
    assert all("sign_threshold" in t.params for t in study.trials)
    assert len({s["sign_threshold"] for s in seen}) > 1


def test_non_bop_rules_still_search_the_step(tmp_path):
    optuna = pytest.importorskip("optuna")
    from tri.ablate import _sign_space

    seen = []
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(lambda t: (seen.append(_sign_space(t, fix_rule="stoch_flip")), 1.0)[1],
                   n_trials=5)
    assert len({s["sign_step"] for s in seen}) > 1
    assert all("sign_threshold" not in s for s in seen)


def test_modes_arms_only_search_their_live_knobs():
    """muon_lr is inert in sign mode (empty Muon group); sign knobs are inert otherwise."""
    optuna = pytest.importorskip("optuna")
    from tri.ablate import _modes_space

    for quant, want, unwanted in [
        ("sign", {"sign_step", "sign_b1", "sign_b2"}, {"muon_lr"}),
        ("bf16", {"muon_lr"}, {"sign_step", "sign_b1", "sign_b2"}),
        ("ste", {"muon_lr"}, {"sign_step", "sign_b1", "sign_b2"}),
    ]:
        seen = []
        st = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
        st.optimize(lambda t: (seen.append(_modes_space(t, fix_quant=quant)), 1.0)[1], n_trials=3)
        keys = set(seen[0][1])
        assert want <= keys, (quant, keys)
        assert not (unwanted & keys), (quant, keys)
        assert all("quant" not in t.params for t in st.trials)  # pinned, not searched


def test_sign_space_no_longer_searches_the_inert_muon_lr():
    optuna = pytest.importorskip("optuna")
    from tri.ablate import _sign_space

    seen = []
    st = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    st.optimize(lambda t: (seen.append(_sign_space(t)), 1.0)[1], n_trials=3)
    assert all("muon_lr" not in s for s in seen)


def test_compare_calls_out_arms_closer_than_the_noise(tmp_path, capsys):
    """Two arms separated by less than within-study spread are not a ranking."""
    from tri.report import compare

    def write(name, ces):
        rows = [{"trial": i, "val_ce": c, "pruned": False,
                 "params": {"adam_lr": 0.001}} for i, c in enumerate(ces)]
        p = tmp_path / f"{name}_summary.json"
        p.write_text(json.dumps({"study": "modes", "trials": rows,
                                 "best_value": min(ces), "best_params": {}}))
        return str(p)

    a = write("sign", [5.10, 5.40, 5.60, 5.90])   # spread 0.30 within arm
    b = write("bf16", [5.12, 5.45, 5.70, 6.00])   # gap between arms only 0.02
    compare([a, b])
    out = capsys.readouterr().out
    assert "NOT a ranking" in out


def test_compare_stays_quiet_when_the_gap_is_real(tmp_path, capsys):
    from tri.report import compare

    def write(name, ces):
        rows = [{"trial": i, "val_ce": c, "pruned": False, "params": {}} for i, c in enumerate(ces)]
        p = tmp_path / f"{name}_summary.json"
        p.write_text(json.dumps({"study": "modes", "trials": rows,
                                 "best_value": min(ces), "best_params": {}}))
        return str(p)

    a = write("sign", [4.00, 4.02, 4.03, 4.05])
    b = write("bf16", [6.00, 6.02, 6.03, 6.05])
    compare([a, b])
    assert "NOT a ranking" not in capsys.readouterr().out


def _study_with_logs(tmp_path, name, trajectory, best_ce, wall):
    """Write a summary plus the per-trial log.jsonl that curves() reads."""
    import os

    rows = [{"trial": 0, "val_ce": best_ce, "pruned": False, "wall_s": wall,
             "params": {"adam_lr": 0.001}}]
    d = tmp_path / name / "t000"
    os.makedirs(d, exist_ok=True)
    with open(d / "log.jsonl", "w") as f:
        for step, ce in trajectory:
            f.write(json.dumps({"event": "eval", "step": step, "val_ce": ce}) + "\n")
        f.write(json.dumps({"event": "done", "wall_s": wall}) + "\n")
    p = tmp_path / f"{name}_summary.json"
    p.write_text(json.dumps({"study": "modes", "trials": rows,
                             "best_value": best_ce, "best_params": {}}))
    return str(p)


def test_curves_align_trajectories_across_arms(tmp_path, capsys):
    """A final number cannot say which arm got there faster; the curve can."""
    from tri.report import curves

    a = _study_with_logs(tmp_path, "modes-sign", [(500, 7.0), (1000, 5.8), (1500, 5.3)], 5.3, 210)
    b = _study_with_logs(tmp_path, "modes-bf16", [(500, 8.2), (1000, 6.4), (1500, 5.2)], 5.2, 180)
    out_series = curves([a, b])
    out = capsys.readouterr().out

    assert "modes-sign" in out and "modes-bf16" in out
    assert "7.0000" in out and "8.2000" in out  # sign leads early
    assert "5.3000" in out and "5.2000" in out  # bf16 ends lower
    assert set(out_series) == {"modes-sign", "modes-bf16"}
    assert "wall-clock is not" in out


def test_curves_reports_a_missing_log_instead_of_crashing(tmp_path, capsys):
    from tri.report import curves

    rows = [{"trial": 0, "val_ce": 5.0, "pruned": False, "params": {}}]
    p = tmp_path / "modes-ste_summary.json"
    p.write_text(json.dumps({"study": "modes", "trials": rows,
                             "best_value": 5.0, "best_params": {}}))
    curves([str(p)])
    assert "no log at" in capsys.readouterr().out


def test_compare_shows_wall_clock_since_only_steps_are_matched(tmp_path, capsys):
    from tri.report import compare

    a = _study_with_logs(tmp_path, "modes-sign", [(500, 7.0)], 5.30, 260)
    b = _study_with_logs(tmp_path, "modes-bf16", [(500, 8.2)], 5.20, 180)
    compare([a, b])
    out = capsys.readouterr().out
    assert "s/trial" in out and "260" in out and "180" in out


def test_flop_matched_steps_equalise_compute_across_loop_counts():
    """A fixed budget buys depth per token or more tokens, never both."""
    from tri.ablate import flop_matched_steps
    from tri.config import build_configs

    mc, _, _ = build_configs("wide")
    ref_loops, ref_steps = mc.n_loops, 4000
    for loops in (1, 2, 3, 4):
        steps = flop_matched_steps("wide", loops, ref_steps)
        spent = steps * mc.flops_per_token(loops)
        budget = ref_steps * mc.flops_per_token(ref_loops)
        assert spent == pytest.approx(budget, rel=0.01), loops
    # cheaper loop counts must buy strictly more steps
    assert flop_matched_steps("wide", 1, 4000) > flop_matched_steps("wide", 4, 4000)


def test_loops_study_requests_flop_matching(tmp_path):
    optuna = pytest.importorskip("optuna")
    from tri.ablate import _study_fn

    st = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    seen = []
    st.optimize(lambda t: (seen.append(_study_fn("loops", t, fix_loops=2)), 1.0)[1], n_trials=2)
    for model_o, train_o, _ in seen:
        assert model_o["n_loops"] == 2
        assert train_o["loop_lo"] == train_o["loop_hi"] == 2
        assert train_o["_flop_matched_loops"] == 2
    assert all("n_loops" not in t.params for t in st.trials)  # pinned, not searched
