import math

import numpy as np
import pandas as pd
import pytest

from qros.validation.bootstrap import bootstrap_sharpe, paired_sharpe_difference, stationary_bootstrap_indices
from qros.validation.metrics import infer_periods_per_year, newey_west_tstat, performance_summary
from qros.validation.multiple_testing import adjust_pvalues, haircut_sharpe
from qros.validation.pbo import probability_of_backtest_overfitting
from qros.validation.regimes import regime_performance, volatility_regimes
from qros.validation.sensitivity import cost_sensitivity, drop_best, grid_matrix, neighborhood
from qros.validation.sharpe import (
    deflated_sharpe_ratio, effective_number_of_trials, expected_max_sharpe, min_track_record_length,
    probabilistic_sharpe_ratio, sharpe_standard_error,
)
from qros.validation.splits import PurgedKFold, walk_forward_by_years, walk_forward_optimize, walk_forward_splits


# ----------------------------------------------------------------------------- Sharpe inference

def test_deflated_sharpe_matches_bailey_lopez_de_prado_example():
    # Bailey & Lopez de Prado (2014): annual SR 2.5 over 5 years of daily data (T = 1250, 250 days a year),
    # skewness -3, kurtosis 10, N = 100 trials whose annualised Sharpe ratios have variance 0.5.
    sr = 2.5 / math.sqrt(250)
    sr0 = expected_max_sharpe(100, 0.5 / 250)
    assert sr0 == pytest.approx(0.1132, abs=5e-4)
    res = deflated_sharpe_ratio(sr, 1250, -3.0, 10.0, 100, 0.5 / 250)
    assert res["dsr"] == pytest.approx(0.9004, abs=2e-3)


def test_psr_basic_properties():
    assert probabilistic_sharpe_ratio(0.05, 500, 0.0, 3.0, 0.05) == pytest.approx(0.5)
    hi = probabilistic_sharpe_ratio(0.08, 1000, 0.0, 3.0)
    lo_skew = probabilistic_sharpe_ratio(0.08, 1000, -2.0, 3.0)
    fat = probabilistic_sharpe_ratio(0.08, 1000, 0.0, 12.0)
    assert hi > lo_skew and hi > fat
    # standard error equals the PSR denominator (Mertens / Lo)
    se = sharpe_standard_error(0.1, 101, 0.0, 3.0)
    assert se == pytest.approx(math.sqrt((1 + 0.5 * 0.01) / 100))


def test_expected_max_sharpe_grows_with_trials():
    v = 0.001
    assert expected_max_sharpe(1, v) == 0.0
    vals = [expected_max_sharpe(n, v) for n in (2, 10, 100, 1000)]
    assert vals == sorted(vals)


def test_min_track_record_length_formula():
    sr = 1.0 / math.sqrt(252)
    expected = 1 + (1 + 0.5 * sr ** 2) * (1.6448536269514722 / sr) ** 2
    assert min_track_record_length(sr) == pytest.approx(expected, rel=1e-9)
    assert min_track_record_length(-0.1) == float("inf")


def test_effective_number_of_trials():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1000, 1))
    identical = pd.DataFrame(np.repeat(base, 5, axis=1))
    independent = pd.DataFrame(rng.normal(size=(1000, 5)))
    assert effective_number_of_trials(identical)["n_eff_corr"] == pytest.approx(1.0, abs=1e-6)
    eff = effective_number_of_trials(independent)
    assert 4.5 < eff["n_eff_corr"] <= 5 and 4.5 < eff["n_eff_eig"] <= 5


# ----------------------------------------------------------------------------- multiple testing (reference: statsmodels 0.15)

P = [0.01, 0.04, 0.03, 0.005, 0.20, 0.5]
REF = {
    "bonferroni": [0.06, 0.24, 0.18, 0.03, 1.0, 1.0],
    "sidak": [0.058519850599, 0.217242210304, 0.167027995071, 0.029627490644, 0.737856, 0.984375],
    "holm": [0.05, 0.12, 0.12, 0.03, 0.4, 0.5],
    "hochberg": [0.05, 0.12, 0.12, 0.03, 0.4, 0.5],
    "bh": [0.03, 0.06, 0.06, 0.03, 0.24, 0.5],
    "by": [0.0735, 0.147, 0.147, 0.0735, 0.588, 1.0],
}


@pytest.mark.parametrize("method", sorted(REF))
def test_adjust_pvalues_matches_statsmodels(method):
    np.testing.assert_allclose(adjust_pvalues(P, method), REF[method], rtol=0, atol=1e-10)


def test_adjust_pvalues_ignores_nan():
    out = adjust_pvalues([0.01, float("nan"), 0.04], "bonferroni")
    assert out[0] == pytest.approx(0.02) and math.isnan(out[1]) and out[2] == pytest.approx(0.08)


def test_haircut_sharpe():
    one = haircut_sharpe(1.0, 2520, 252, 1)
    many = haircut_sharpe(1.0, 2520, 252, 50)
    assert one["haircut"] == pytest.approx(0.0, abs=1e-9)
    assert 0 < many["haircut"] < 1 and many["sharpe_adjusted"] < 1.0
    huge = haircut_sharpe(8.0, 2520, 252, 10)
    assert huge["sharpe_adjusted"] == pytest.approx(8.0)


# ----------------------------------------------------------------------------- metrics

def test_newey_west_matches_statsmodels():
    rng = np.random.default_rng(42)
    e = rng.standard_normal(500)
    x = np.empty(500)
    x[0] = e[0]
    for t in range(1, 500):
        x[t] = 0.3 * x[t - 1] + e[t]
    x = x * 0.01 + 0.001
    if x[0] != 0.004047170797544314:
        pytest.skip("numpy changed its normal generator; reference series differs")
    for lags, ref in ((0, 1.7990750893925374), (5, 1.3872856572406749), (10, 1.438607228908713)):
        assert newey_west_tstat(x, lags)[0] == pytest.approx(ref, rel=1e-10)


def test_performance_summary_values():
    idx = pd.bdate_range("2020-01-01", periods=4)
    r = pd.Series([0.10, -0.05, 0.02, 0.0], index=idx)
    s = performance_summary(r, 252)
    assert s["total_return"] == pytest.approx(1.1 * 0.95 * 1.02 - 1)
    assert s["max_drawdown"] == pytest.approx(-0.05)
    assert s["hit_rate"] == pytest.approx(2 / 3)
    assert s["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252))


def test_infer_periods_per_year():
    assert infer_periods_per_year(pd.bdate_range("2020-01-01", periods=600)) == 252
    assert infer_periods_per_year(pd.date_range("2000-01-31", periods=120, freq="ME")) == 12
    assert infer_periods_per_year(pd.date_range("2020-01-01", periods=800, freq="D")) == 365


# ----------------------------------------------------------------------------- bootstrap

def test_stationary_bootstrap_block_structure():
    rng = np.random.default_rng(3)
    idx = stationary_bootstrap_indices(1000, 20, 50, rng)
    assert idx.shape == (50, 1000) and idx.min() >= 0 and idx.max() < 1000
    continuation = np.mean(idx[:, 1:] == (idx[:, :-1] + 1) % 1000)
    assert continuation == pytest.approx(1 - 1 / 20, abs=0.01)


def test_bootstrap_sharpe_is_deterministic_and_brackets_the_estimate():
    r = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 2000))
    a = bootstrap_sharpe(r, 252, n_boot=400, seed=5)
    b = bootstrap_sharpe(r, 252, n_boot=400, seed=5)
    assert a == b
    assert a["lo"] < a["sharpe"] < a["hi"]


def test_paired_sharpe_difference_detects_identical_strategies():
    r = pd.Series(np.random.default_rng(2).normal(0.0005, 0.01, 1500), index=pd.bdate_range("2015-01-01", periods=1500))
    res = paired_sharpe_difference(r, r.copy(), n_boot=200)
    assert res["diff"] == pytest.approx(0.0) and res["lo"] == pytest.approx(0.0) and res["hi"] == pytest.approx(0.0)


# ----------------------------------------------------------------------------- PBO

def test_pbo_noise_vs_real_edge():
    rng = np.random.default_rng(11)
    noise = pd.DataFrame(rng.normal(0, 0.01, (2000, 20)))
    pbo_noise = probability_of_backtest_overfitting(noise, 16)["pbo"]
    assert 0.25 < pbo_noise < 0.75
    edge = noise.copy()
    edge[0] = edge[0] + 0.003
    res = probability_of_backtest_overfitting(edge, 16)
    assert res["pbo"] < 0.05 and res["n_combinations"] == 12870


def test_pbo_needs_enough_data():
    assert probability_of_backtest_overfitting(pd.DataFrame(np.zeros((20, 3))), 16) is None
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(pd.DataFrame(np.zeros((200, 3))), 15)


# ----------------------------------------------------------------------------- splits

def test_walk_forward_splits_are_ordered_and_disjoint():
    folds = walk_forward_splits(100, train=40, test=10, gap=2, expanding=False)
    assert len(folds) == 5
    for f in folds:
        assert f.train.max() + 2 < f.test.min()
        assert len(f.train) == 40 and len(f.test) == 10
    tests = np.concatenate([f.test for f in folds])
    assert len(tests) == len(set(tests))


def test_walk_forward_by_years():
    idx = pd.bdate_range("2005-01-01", "2010-12-31")
    folds = walk_forward_by_years(idx, train_years=3)
    assert [f.test_range[0][:4] for f in folds] == ["2008", "2009", "2010"]
    assert all(idx[f.train].max() < idx[f.test].min() for f in folds)


def test_purged_kfold_purges_overlaps_and_embargoes():
    t0 = pd.bdate_range("2020-01-01", periods=200)
    t1 = t0 + pd.Timedelta(days=7)
    kf = PurgedKFold(n_splits=5, embargo=0.02)
    for train, test in kf.split(t0, t1):
        lo, hi = t0[test].min(), t1[test].max()
        assert not ((t0[train] <= hi) & (t1[train] >= lo)).any()
        after = np.arange(test.max() + 1, min(200, test.max() + 1 + 4))
        assert not np.isin(after, train).any()
        assert len(np.intersect1d(train, test)) == 0


def test_walk_forward_optimize_reports_selection_stability():
    idx = pd.bdate_range("2005-01-01", "2012-12-31")
    rng = np.random.default_rng(4)
    good = pd.Series(rng.normal(0.001, 0.01, len(idx)), index=idx)
    bad = pd.Series(rng.normal(-0.001, 0.01, len(idx)), index=idx)
    folds = walk_forward_by_years(idx, train_years=3)
    res = walk_forward_optimize({"good": good, "bad": bad}, folds, idx)
    assert res["selections"] == ["good"] * len(folds)
    assert res["switch_rate"] == 0 and res["oos_returns"].index.min().year == 2008


# ----------------------------------------------------------------------------- sensitivity and regimes

def test_neighbourhood_plateau_vs_spike():
    params = [{"a": a, "b": b} for a in (1, 2, 3) for b in (10, 20)]
    plateau = [1.0, 0.9, 1.1, 1.0, 0.95, 1.05]
    spike = [0.0, 0.1, 2.0, 0.0, -0.1, 0.05]
    target = 2  # a=2, b=10
    assert neighborhood(params, plateau, target)["plateau_ratio"] > 0.8
    res = neighborhood(params, spike, target)
    assert res["plateau_ratio"] < 0.1 and res["n_neighbours"] == 3


def test_grid_matrix_and_costs_and_trimming():
    params = [{"f": f, "s": s} for f in (5, 10) for s in (50, 100)]
    gm = grid_matrix(params, [1, 2, 3, 4], "f", "s")
    assert gm["xs"] == [5, 10] and gm["ys"] == [50, 100] and gm["z"] == [[1.0, 3.0], [2.0, 4.0]]
    idx = pd.bdate_range("2020-01-01", periods=500)
    gross = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 500), index=idx)
    costs = pd.Series(0.0005, index=idx)
    cs = cost_sensitivity(gross, costs, 252)
    assert cs["0x"] > cs["1x"] > cs["2x"] > cs["3x"]
    lottery = pd.Series(np.r_[np.full(495, -0.001), np.full(5, 0.5)], index=idx)
    db = drop_best(lottery, 252, 0.01)
    assert db["sharpe"] > 0 and db["sharpe_trimmed"] < 0


def test_volatility_regimes_have_no_look_ahead():
    idx = pd.bdate_range("2010-01-01", periods=1500)
    mkt = pd.Series(np.random.default_rng(9).normal(0, 0.01, 1500), index=idx)
    labels = volatility_regimes(mkt)
    future_shock = mkt.copy()
    future_shock.iloc[1200:] *= 10
    labels2 = volatility_regimes(future_shock)
    pd.testing.assert_series_equal(labels.iloc[:1200], labels2.iloc[:1200])
    perf = regime_performance(mkt, labels, 252)
    assert perf["n_regimes"] == 3
