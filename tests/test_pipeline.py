"""End-to-end behaviour: tracking, managed runs, reproduction, holdouts and the hygiene rules."""

import numpy as np
import pandas as pd
import pytest

import qros
from qros.registry.models import Periods
from qros.validation.report import VERDICTS
from helpers import make_returns

STRATEGIES = "strategies_fixture"


def _codes(reg, exp_id):
    return {w["code"] for w in reg.warnings(experiment=exp_id)}


def test_track_records_everything(reg):
    h = reg.add_hypothesis("Drift", "The series has a positive drift", register=True)
    with qros.track("drift", registry=reg, hypothesis=h.id, params={"k": 1}, seed=3, train="2010:2014",
                    test="2015:2019") as run:
        exp = run.log_result(make_returns(n=2600, mu=0.0008))
    assert exp.status == "completed" and exp.result_hash and exp.code["file_sha256"]
    assert {"is", "oos", "full"} <= set(exp.metrics)
    report = reg.latest_validation(exp.id)
    assert report["verdict"] in VERDICTS
    assert {g["key"] for g in report["scorecard"]} >= {"preregistered", "dsr", "oos_significance", "hygiene"}
    assert reg.verify()["ok"]


def test_tracked_block_failure_marks_experiment_failed(reg):
    with pytest.raises(RuntimeError):
        with qros.track("boom", registry=reg) as run:
            raise RuntimeError("strategy crashed")
    exp = reg.get_experiment(run.id)
    assert exp.status == "failed" and "strategy crashed" in exp.error


def test_strong_preregistered_effect_is_robust(reg):
    h = reg.add_hypothesis("Edge", "A planted strong edge", register=True)
    ho = reg.add_holdout("final", "2020-01-01", "2021-12-31", max_touches=1)
    r = make_returns(seed=5, n=3100, mu=0.0012, sd=0.01)
    with qros.track("edge", registry=reg, hypothesis=h.id, seed=1, train="2010:2014", test="2015:2019",
                    holdout=ho.id, open_holdout=True, holdout_purpose="final test") as run:
        exp = run.log_result(r)
    report = reg.latest_validation(exp.id)
    assert report["verdict"] == "robust", report["verdict_reason"]


def test_noise_is_not_robust(reg):
    h = reg.add_hypothesis("Noise", "No edge", register=True)
    for i in range(6):
        with qros.track(f"noise {i}", registry=reg, hypothesis=h.id, params={"i": i}, seed=i, train="2010:2014",
                        test="2015:2019") as run:
            run.log_result(make_returns(seed=100 + i, n=2600, mu=0.0, sd=0.01))
    verdicts = {e.verdict for e in reg.list_experiments(family=h.id)}
    assert "robust" not in verdicts
    assert reg.latest_validation(reg.list_experiments(family=h.id)[-1].id)["pbo"] is not None


def test_managed_run_reproduces_exactly(reg, prices):
    reg.add_dataset("prices", adjusted=True, survivorship_free=True)
    v = reg.snapshot("prices", prices)
    h = reg.add_hypothesis("Trend", "Trend", register=True)
    exp = qros.run_experiment(reg, f"{STRATEGIES}:trend", params={"lookback": 20}, seed=1, data={"prices": v.id},
                              periods=Periods.of("2010:2014", "2015:2019"), hypothesis=h.id)
    assert exp.entry_point.endswith(":trend") and exp.data == {"prices": v.id}
    rec = qros.reproduce(reg, exp.id)
    assert rec["status"] == "exact"
    gates = {g["key"]: g["status"] for g in reg.latest_validation(exp.id)["scorecard"]}
    assert gates["reproducible"] == "pass"


def test_unseeded_randomness_is_caught(reg, prices):
    reg.add_dataset("prices")
    v = reg.snapshot("prices", prices)
    exp = qros.run_experiment(reg, f"{STRATEGIES}:noisy", params={}, seed=1, data={"prices": v.id},
                              periods=Periods.of("2010:2014", "2015:2019"))
    rec = qros.reproduce(reg, exp.id)
    assert rec["status"] == "mismatch"
    assert "NONREPRODUCIBLE" in _codes(reg, exp.id)
    assert reg.get_experiment(exp.id).verdict == "invalid"


def test_sealed_holdout_is_hidden_from_managed_runs(reg, prices):
    reg.add_dataset("prices")
    v = reg.snapshot("prices", prices)
    ho = reg.add_holdout("final", "2018-01-01", "2019-12-31")
    exp = qros.run_experiment(reg, f"{STRATEGIES}:trend", params={"lookback": 20}, seed=1, data={"prices": v.id},
                              periods=Periods.of("2010:2014", "2015:2017"), holdout=ho.id)
    frame = reg.result_frame(exp.id)
    assert frame.index.max() < pd.Timestamp("2018-01-01")
    assert reg.holdout_accesses(ho.id) == []
    opened = qros.run_experiment(reg, f"{STRATEGIES}:trend", params={"lookback": 20}, seed=1, data={"prices": v.id},
                                 periods=Periods.of("2010:2014", "2015:2017"), holdout=ho.id, open_holdout=True,
                                 holdout_purpose="final")
    assert reg.result_frame(opened.id).index.max() > pd.Timestamp("2018-01-01")
    assert "holdout" in reg.get_experiment(opened.id).metrics
    assert len(reg.holdout_accesses(ho.id)) == 1


def test_undeclared_holdout_use_is_invalid(reg):
    ho = reg.add_holdout("final", "2018-01-01", "2019-12-31")
    with qros.track("peek", registry=reg, train="2010:2014", test="2015:2017", holdout=ho.id) as run:
        exp = run.log_result(make_returns(n=2600))   # runs through 2019: reaches into the holdout
    assert "HOLDOUT_IMPLICIT_ACCESS" in _codes(reg, exp.id)
    assert exp.verdict == "invalid"
    assert reg.holdout_accesses(ho.id)[0]["purpose"].startswith("implicit")


def test_holdout_overuse_is_flagged_from_the_second_look(reg):
    ho = reg.add_holdout("final", "2018-01-01", "2019-12-31", max_touches=1)
    ids = []
    for i in range(2):
        with qros.track(f"look {i}", registry=reg, params={"i": i}, train="2010:2014", test="2015:2017",
                        holdout=ho.id, open_holdout=True, holdout_purpose="check") as run:
            ids.append(run.log_result(make_returns(seed=i, n=2600)).id)
    assert "HOLDOUT_OVERUSE" not in _codes(reg, ids[0])
    assert "HOLDOUT_OVERUSE" in _codes(reg, ids[1])
    assert reg.get_experiment(ids[1]).verdict == "invalid"


def test_harking_is_detected(reg):
    h = reg.add_hypothesis("Continuation", "Winners keep winning", prediction={"direction": "positive"}, register=True)
    with qros.track("first test", registry=reg, hypothesis=h.id, train="2010:2014", test="2015:2019") as run:
        run.log_result(make_returns(seed=1, n=2600, mu=-0.0005))
    reg.amend_hypothesis(h.id, reason="results show the opposite", statement="Losers keep losing",
                         prediction={"direction": "negative"})
    codes = {w["code"] for w in reg.warnings(hypothesis=h.id)}
    assert {"HYPOTHESIS_AMENDED_AFTER_RESULTS", "PREDICTION_REVERSED"} <= codes
    with qros.track("confirm on same data", registry=reg, hypothesis=h.id, train="2010:2014", test="2015:2019") as run:
        exp = run.log_result(make_returns(seed=2, n=2600, mu=0.0008))
    assert "AMENDED_TESTED_ON_SEEN_DATA" in _codes(reg, exp.id)
    assert exp.verdict == "invalid"


def test_not_preregistered_and_survivorship_flags(reg, prices):
    h = reg.add_hypothesis("Late", "registered after the run")
    reg.add_dataset("universe", survivorship_free=False)
    u = reg.snapshot("universe", pd.DataFrame({"ticker": ["A0", "A1"]}))
    with qros.track("early", registry=reg, hypothesis=h.id, data={"universe": u.id}, train="2010:2014", test="2015:2019") as run:
        exp = run.log_result(make_returns(n=2600))
    reg.register_hypothesis(h.id)
    qros.validate_experiment(reg, exp.id)
    codes = _codes(reg, exp.id)
    assert {"NOT_PREREGISTERED", "SURVIVORSHIP_BIAS"} <= codes
    assert reg.get_experiment(exp.id).verdict == "invalid"


def test_lookahead_in_managed_run_is_invalid(reg):
    reg.add_dataset("prices")
    reg.add_dataset("fund", entity_col="ticker", time_col="period_end", available_col="available_at")
    idx = pd.bdate_range("2010-01-01", "2019-12-31")
    px = pd.DataFrame(100 + np.cumsum(np.random.default_rng(0).normal(0, 1, (len(idx), 2)), axis=0), index=idx, columns=["A", "B"])
    qe = pd.date_range("2010-03-31", "2019-09-30", freq="QE")
    fund = pd.DataFrame({"ticker": np.repeat(["A", "B"], len(qe)), "period_end": np.tile(qe, 2),
                         "available_at": np.tile(qe + pd.Timedelta(days=45), 2), "profit": np.arange(2 * len(qe), dtype=float)})
    vp, vf = reg.snapshot("prices", px), reg.snapshot("fund", fund)
    exp = qros.run_experiment(reg, f"{STRATEGIES}:fundamental", params={"point_in_time": False}, seed=1,
                              data={"prices": vp.id, "fund": vf.id}, periods=Periods.of("2010:2014", "2015:2019"))
    assert "PIT_VIOLATION" in _codes(reg, exp.id) and exp.verdict == "invalid"
    ok = qros.run_experiment(reg, f"{STRATEGIES}:fundamental", params={"point_in_time": True}, seed=1,
                             data={"prices": vp.id, "fund": vf.id}, periods=Periods.of("2010:2014", "2015:2019"))
    assert "PIT_VIOLATION" not in _codes(reg, ok.id)


def test_lookahead_detected_with_csv_storage(reg, monkeypatch):
    """The same audit must work when frames are stored as qcsv (no pyarrow installed)."""
    import qros.artifacts
    monkeypatch.setattr(qros.artifacts, "_has_pyarrow", lambda: False)
    test_lookahead_in_managed_run_is_invalid(reg)
    assert all(v.storage == "qcsv" for v in reg.list_versions())


def test_metrics_only_experiment(reg):
    exp = reg.create_experiment("sim run", family="sim", source="ingested")
    done = reg.complete_experiment(exp.id, None, metrics={"pnl": 12.5})
    assert done.status == "completed" and done.verdict == "insufficient"
    assert done.metric("pnl", "custom") == 12.5


def test_family_revalidation_keeps_history(reg):
    h = reg.add_hypothesis("Sweep", "s", register=True)
    first = None
    for i in range(5):
        with qros.track(f"cfg {i}", registry=reg, hypothesis=h.id, params={"p": i}, train="2010:2014", test="2015:2019") as run:
            e = run.log_result(make_returns(seed=i, n=2600, mu=0.0003))
            first = first or e.id
    before = reg.validation_history(first)
    qros.revalidate_family(reg, h.id)
    after = reg.validation_history(first)
    assert len(after) == len(before) + 1
    assert after[0]["n_trials"] == 1 and after[-1]["n_trials"] == 5
