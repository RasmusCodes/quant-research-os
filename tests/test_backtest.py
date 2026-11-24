import numpy as np
import pandas as pd
import pytest

from qros.backtest import BacktestResult, ResultError, as_result, backtest_signal, backtest_weights, long_short_weights
from qros.backtest.costs import LinearCost, SquareRootImpact
from qros.backtest.risk import exposure_summary, tail_risk


def _rets():
    idx = pd.bdate_range("2020-01-01", periods=6)
    return pd.DataFrame({"A": [0.0, 0.01, -0.02, 0.03, 0.0, 0.01], "B": [0.0, -0.01, 0.01, 0.0, 0.02, -0.01]}, index=idx)


def test_weights_are_applied_with_a_lag():
    r = _rets()
    w = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0], "B": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]}, index=r.index)
    res = backtest_weights(w, r, cost=0.0, lag=1)
    expected = [0.0, 0.01, -0.02, 0.0, 0.02, -0.01]  # day t earns weights decided at t-1
    np.testing.assert_allclose(res.gross_returns.to_numpy(), expected)


def test_turnover_and_costs():
    r = _rets()
    w = pd.DataFrame({"A": [1.0, 1.0, -1.0, -1.0, 0.0, 0.0], "B": 0.0}, index=r.index)
    res = backtest_weights(w, r, cost=10.0, lag=1)
    np.testing.assert_allclose(res.turnover.to_numpy(), [0.0, 1.0, 0.0, 2.0, 0.0, 1.0])
    np.testing.assert_allclose(res.costs.to_numpy(), res.turnover.to_numpy() * 0.001)
    np.testing.assert_allclose(res.returns.to_numpy(), (res.gross_returns - res.costs).to_numpy())


def test_same_bar_trading_is_refused_by_default():
    r = _rets()
    w = pd.DataFrame(1.0, index=r.index, columns=r.columns)
    with pytest.raises(ValueError):
        backtest_weights(w, r, lag=0)
    backtest_weights(w, r, lag=0, allow_same_bar=True)


def test_perfect_foresight_signal_only_pays_without_lag():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 300), index=idx)
    same_bar = np.sign(r).to_frame("x")  # computed from the very return it would trade
    honest = backtest_weights(same_bar, r.to_frame("x"), cost=0.0, lag=1).returns
    cheat = backtest_weights(same_bar, r.to_frame("x"), cost=0.0, lag=0, allow_same_bar=True).returns
    assert abs(honest.mean()) < 0.002
    assert cheat.mean() > 0.005


def test_long_short_weights_are_dollar_neutral():
    scores = pd.DataFrame(np.arange(20.0).reshape(2, 10), index=pd.bdate_range("2020-01-01", periods=2))
    w = long_short_weights(scores, 0.2)
    np.testing.assert_allclose(w.sum(axis=1).to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(w.abs().sum(axis=1).to_numpy(), 1.0)
    assert (w.iloc[0, -2:] > 0).all() and (w.iloc[0, :2] < 0).all()


def test_result_validation_and_coercion():
    idx = pd.bdate_range("2020-01-01", periods=3)
    with pytest.raises(ResultError):
        BacktestResult(pd.Series([0.1, np.nan, 0.0], index=idx)).validate()
    with pytest.raises(ResultError):
        BacktestResult(pd.Series([0.1, -1.0, 0.0], index=idx)).validate()
    with pytest.raises(ResultError):
        BacktestResult(pd.Series([0.1, 0.1, 0.0], index=idx), gross_returns=pd.Series([0.2, 0.1, 0.0], index=idx),
                       costs=pd.Series([0.0, 0.0, 0.0], index=idx)).validate()
    res = as_result(pd.DataFrame({"net": [0.01, 0.02, 0.0], "gross": [0.011, 0.02, 0.0], "cost": [0.001, 0.0, 0.0]}, index=idx))
    assert list(res.frame().columns) == ["net", "gross", "cost"]
    pnl = BacktestResult.from_pnl(pd.Series([100.0, -50.0], index=idx[:2]), capital=10_000)
    np.testing.assert_allclose(pnl.returns.to_numpy(), [0.01, -0.005])


def test_cost_models_and_risk():
    t = pd.Series([0.0, 1.0, 4.0])
    np.testing.assert_allclose(LinearCost(10).cost(t).to_numpy(), [0.0, 0.001, 0.004])
    sq = SquareRootImpact(5, 10, 1.0).cost(t).to_numpy()
    assert sq[2] > 4 * sq[1] / 1.0 * 0.5 and sq[0] == 0
    r = _rets()
    w = pd.DataFrame({"A": 0.5, "B": -0.5}, index=r.index)
    res = backtest_weights(w, r, cost=0)
    ex = exposure_summary(res)
    assert ex["avg_gross"] == pytest.approx(5 / 6) and ex["max_weight"] == pytest.approx(0.5)
    tr = tail_risk(pd.Series(np.linspace(-0.05, 0.05, 101)))
    assert tr["var_95"] == pytest.approx(0.045)
