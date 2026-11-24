"""Strategy entry points used by the demo research project (all run on the synthetic market).

Each function takes a :class:`qros.RunContext` and returns a :class:`qros.BacktestResult`. Data come
only through ``ctx.data(role)`` / ``ctx.pit(role)``, so the research OS knows which dataset versions
produced each result, hides the sealed holdout, and audits point-in-time access.

Roles: ``prices`` (adjusted closes, dates x tickers), ``prices_raw`` (unadjusted closes),
``universe`` (ticker, first_date, last_date, status), ``fundamentals`` (point-in-time table),
``index`` (equal-weighted market index returns), ``actions`` (corporate actions).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qros.backtest.engine import backtest_signal, backtest_weights, long_short_weights
from qros.backtest.interface import BacktestResult
from qros.data.corporate_actions import split_check_summary
from qros.validation.splits import walk_forward_by_years, walk_forward_optimize


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().to_numpy())


def week_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    iso = index.isocalendar()
    return pd.DatetimeIndex(s.groupby([iso.year.to_numpy(), iso.week.to_numpy()]).max().to_numpy())


def eligibility(universe: pd.DataFrame, dates: pd.DatetimeIndex, tickers: pd.Index) -> pd.DataFrame:
    u = universe.set_index("ticker").reindex(tickers)
    first = pd.to_datetime(u["first_date"]).to_numpy()
    last = pd.to_datetime(u["last_date"]).to_numpy()
    d = dates.to_numpy()[:, None]
    mask = (d >= first[None, :]) & (d <= last[None, :])
    return pd.DataFrame(mask, index=dates, columns=tickers)


def equal_weight_index(returns: pd.DataFrame, eligible: pd.DataFrame) -> pd.Series:
    r = returns.where(eligible)
    return r.mean(axis=1).fillna(0.0).rename("index")


def _daily(weights_at: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    return weights_at.reindex(index).ffill().fillna(0.0)


def _vol_scale(weights: pd.DataFrame, returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    vol = returns.rolling(window, min_periods=20).std().reindex(weights.index)
    raw = weights / vol.replace(0, np.nan)
    raw = raw.fillna(0.0)
    gross = raw.abs().sum(axis=1).replace(0, np.nan)
    return raw.div(gross, axis=0).fillna(0.0)


def momentum(ctx: Any) -> BacktestResult:
    lookback = int(ctx.param("lookback_months", 12))
    skip = int(ctx.param("skip_months", 1))
    q = float(ctx.param("quantile", 0.1))
    px = ctx.data("prices")
    uni = ctx.data("universe")
    rets = px.pct_change(fill_method=None)
    elig = eligibility(uni, px.index, px.columns)
    me = month_ends(px.index)
    px_me = px.reindex(me)
    score = (px_me.shift(skip) / px_me.shift(lookback) - 1.0).where(elig.reindex(me))
    if ctx.param("unseeded_tiebreak", False):
        # A deliberate bug for the demo: an unseeded random generator, so no two runs agree.
        noise = np.random.default_rng().normal(0.0, 0.02, score.shape)
        score = score + noise
    w = long_short_weights(score, q)
    if ctx.param("vol_scaled", False):
        w = _vol_scale(w, rets)
    bench = equal_weight_index(rets, elig)
    return backtest_weights(_daily(w, px.index), rets, cost=float(ctx.param("cost_bps", 10.0)), lag=1,
                            benchmark=bench, periods_per_year=252, keep_positions=False,
                            meta={"signal": f"{lookback}-{skip} month momentum, top/bottom {q:.0%}"})


def quality(ctx: Any) -> BacktestResult:
    q = float(ctx.param("quantile", 0.2))
    point_in_time = bool(ctx.param("point_in_time", True))
    px = ctx.data("prices")
    uni = ctx.data("universe")
    fund = ctx.pit("fundamentals")
    rets = px.pct_change(fill_method=None)
    elig = eligibility(uni, px.index, px.columns)
    me = month_ends(px.index)
    key = None if point_in_time else fund.time       # the classic bug: join on the period end
    panel = fund.panel(me, "profitability", key=key, entities=list(px.columns))
    score = panel.where(elig.reindex(me))
    w = long_short_weights(score, q)
    bench = equal_weight_index(rets, elig)
    return backtest_weights(_daily(w, px.index), rets, cost=float(ctx.param("cost_bps", 10.0)), lag=1,
                            benchmark=bench, periods_per_year=252, keep_positions=False,
                            meta={"signal": "profitability, " + ("as published" if point_in_time else "at period end")})


def past_losers(ctx: Any) -> BacktestResult:
    """Long the past year's losers, short its winners (a long-term reversal bet)."""
    lookback = int(ctx.param("lookback_days", 252))
    q = float(ctx.param("quantile", 0.1))
    px = ctx.data("prices")
    uni = ctx.data("universe")
    rets = px.pct_change(fill_method=None)
    universe_tickers = [t for t in px.columns if t in set(uni["ticker"])]
    px, rets = px[universe_tickers], rets[universe_tickers]
    elig = eligibility(uni, px.index, px.columns)
    me = month_ends(px.index)
    past = (px / px.shift(lookback) - 1.0).reindex(me)
    score = (-past).where(elig.reindex(me))
    w = long_short_weights(score, q)
    bench = equal_weight_index(rets, elig)
    return backtest_weights(_daily(w, px.index), rets, cost=float(ctx.param("cost_bps", 10.0)), lag=1,
                            benchmark=bench, periods_per_year=252, keep_positions=False,
                            meta={"signal": f"minus {lookback}-day return"})


def reversal(ctx: Any) -> BacktestResult:
    """Weekly short-term reversal (long last week's losers) or, with direction='continuation', the opposite."""
    lookback = int(ctx.param("lookback_days", 5))
    q = float(ctx.param("quantile", 0.1))
    raw = bool(ctx.param("raw_prices", False))
    sign = 1.0 if ctx.param("direction", "reversal") == "continuation" else -1.0
    px = ctx.data("prices_raw" if raw else "prices")
    uni = ctx.data("universe")
    rets = px.pct_change(fill_method=None)
    elig = eligibility(uni, px.index, px.columns)
    we = week_ends(px.index)
    score = (sign * (px / px.shift(lookback) - 1.0)).reindex(we).where(elig.reindex(we))
    w = long_short_weights(score, q)
    bench = equal_weight_index(rets, elig)
    meta: dict[str, Any] = {"signal": f"minus {lookback}-day return, weekly"}
    if "actions" in ctx.data_versions:
        meta["data_checks"] = {"split_check": split_check_summary(px, ctx.data("actions"))}
    return backtest_weights(_daily(w, px.index), rets, cost=float(ctx.param("cost_bps", 10.0)), lag=1,
                            benchmark=bench, periods_per_year=252, keep_positions=False, meta=meta)


def ma_timing(ctx: Any) -> BacktestResult:
    fast = int(ctx.param("fast", 20))
    slow = int(ctx.param("slow", 200))
    idx = ctx.data("index")
    r = idx["ret"].astype(float)
    level = (1.0 + r).cumprod()
    signal = np.sign(level.rolling(fast).mean() - level.rolling(slow).mean()).fillna(0.0)
    if ctx.param("mode", "long_short") == "long_flat":
        signal = signal.clip(lower=0.0)
    return backtest_signal(signal, r, cost=float(ctx.param("cost_bps", 5.0)), lag=1, benchmark=r, periods_per_year=252,
                           meta={"signal": f"MA({fast}) vs MA({slow}) on the index"})


def ma_grid() -> list[dict[str, int]]:
    fasts = [5, 10, 20, 30, 50, 75, 100]
    slows = [50, 100, 150, 200, 250, 300, 400]
    return [{"fast": f, "slow": s} for f in fasts for s in slows if f < s]


def ma_walk_forward(ctx: Any) -> BacktestResult:
    """Pick the best MA pair on all prior years, trade it next year; stitch the out-of-sample years."""
    idx = ctx.data("index")
    r = idx["ret"].astype(float)
    level = (1.0 + r).cumprod()
    cost = float(ctx.param("cost_bps", 5.0))
    candidates, params, gross_c, cost_c = {}, {}, {}, {}
    for p in ma_grid():
        sig = np.sign(level.rolling(p["fast"]).mean() - level.rolling(p["slow"]).mean()).fillna(0.0)
        res = backtest_signal(sig, r, cost=cost, lag=1)
        key = f"{p['fast']}/{p['slow']}"
        candidates[key], params[key] = res.returns, p
        gross_c[key], cost_c[key] = res.gross_returns, res.costs
    folds = walk_forward_by_years(r.index, train_years=int(ctx.param("train_years", 3)), expanding=True)
    wf = walk_forward_optimize(candidates, folds, r.index, params=params)
    oos = wf["oos_returns"]
    gross = pd.concat([gross_c[row["selected"]].loc[row["test_range"][0]:row["test_range"][1]] for row in wf["folds"]])
    costs = pd.concat([cost_c[row["selected"]].loc[row["test_range"][0]:row["test_range"][1]] for row in wf["folds"]])
    return BacktestResult(returns=oos, gross_returns=gross, costs=costs, benchmark=r.reindex(oos.index),
                          periods_per_year=252, meta={"walk_forward": wf, "signal": "walk-forward selected MA pair"})
