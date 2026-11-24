"""A small, transparent vectorised backtester for weight-based strategies.

Conventions (stated, because they decide whether a backtest can see the future):

* ``weights`` at date t are decided with information available at the close of t;
* they are held over the NEXT period: the position during period t is ``weights[t - lag]`` with
  ``lag >= 1`` (``lag=0`` is refused unless explicitly allowed);
* ``asset_returns`` at t are simple returns from the close of t-1 to the close of t;
* weights are held constant between rebalances (no drift), turnover is sum |w_t - w_{t-1}| and costs
  are turnover times the cost model's rate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qros.backtest.costs import CostModel, LinearCost
from qros.backtest.interface import BacktestResult


def backtest_weights(weights: pd.DataFrame, asset_returns: pd.DataFrame, *, cost: CostModel | float = 10.0,
                     lag: int = 1, benchmark: pd.Series | None = None, allow_same_bar: bool = False,
                     periods_per_year: float | None = None, meta: dict[str, Any] | None = None,
                     keep_positions: bool = True) -> BacktestResult:
    if lag < 1 and not allow_same_bar:
        raise ValueError("lag=0 trades on the same bar the signal was computed from (look-ahead); "
                         "pass allow_same_bar=True only if the signal truly predates the bar")
    cost_model = cost if isinstance(cost, CostModel) else LinearCost(float(cost))
    rets = asset_returns.astype(float)
    w = weights.reindex(index=rets.index, columns=rets.columns).ffill().fillna(0.0)
    held = w.shift(lag).fillna(0.0)
    gross = (held * rets.fillna(0.0)).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()
    costs = cost_model.cost(turnover, held)
    net = gross - costs
    exposure = pd.DataFrame({
        "gross": held.abs().sum(axis=1), "net": held.sum(axis=1),
        "long": held.clip(lower=0).sum(axis=1), "short": held.clip(upper=0).sum(axis=1),
        "n_long": (held > 0).sum(axis=1).astype(float), "n_short": (held < 0).sum(axis=1).astype(float),
    })
    return BacktestResult(
        returns=net, gross_returns=gross, costs=costs, turnover=turnover,
        positions=held if keep_positions else None, exposure=exposure,
        benchmark=benchmark.reindex(rets.index) if benchmark is not None else None,
        periods_per_year=periods_per_year,
        meta={**(meta or {}), "engine": "qros.backtest.backtest_weights", "lag": lag,
              "cost_model": cost_model.describe()},
    )


def backtest_signal(signal: pd.Series, asset_returns: pd.Series, *, cost: CostModel | float = 5.0, lag: int = 1,
                    benchmark: pd.Series | None = None, periods_per_year: float | None = None,
                    meta: dict[str, Any] | None = None) -> BacktestResult:
    """Single-asset version: ``signal`` is the target position (e.g. -1, 0, +1) decided at t."""
    w = signal.astype(float).to_frame("asset")
    r = asset_returns.astype(float).to_frame("asset")
    return backtest_weights(w, r, cost=cost, lag=lag, benchmark=benchmark, periods_per_year=periods_per_year,
                            meta=meta, keep_positions=True)


def long_short_weights(scores: pd.DataFrame, quantile: float = 0.1, *, min_names: int = 5,
                       gross: float = 1.0) -> pd.DataFrame:
    """Equal-weight top-minus-bottom quantile portfolio from a score panel (NaN scores excluded)."""
    ranks = scores.rank(axis=1, pct=True)
    n = scores.notna().sum(axis=1)
    long = (ranks > 1 - quantile) & n.ge(min_names).to_numpy()[:, None]
    short = (ranks <= quantile) & n.ge(min_names).to_numpy()[:, None]
    nl = long.sum(axis=1).replace(0, np.nan)
    ns = short.sum(axis=1).replace(0, np.nan)
    w = long.astype(float).div(nl, axis=0).fillna(0.0) * (gross / 2) - \
        short.astype(float).div(ns, axis=0).fillna(0.0) * (gross / 2)
    return w
