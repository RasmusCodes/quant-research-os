"""Regime analysis: does a strategy's performance depend on one market state?

Regime labels are built without look-ahead: a volatility regime at t compares trailing realised
volatility with quantiles of its own history up to t-1 only, and a trend regime uses the trailing
return up to t-1.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from qros.validation.metrics import clean_returns


def volatility_regimes(market: pd.Series, window: int = 63, min_history: int = 252) -> pd.Series:
    r = clean_returns(market)
    rv = r.rolling(window).std(ddof=1)
    q1 = rv.expanding(min_periods=min_history).quantile(1 / 3).shift(1)
    q2 = rv.expanding(min_periods=min_history).quantile(2 / 3).shift(1)
    labels = pd.Series(np.where(rv <= q1, "low vol", np.where(rv > q2, "high vol", "mid vol")), index=r.index)
    return labels.where(q1.notna() & rv.notna())


def trend_regimes(market: pd.Series, window: int = 252) -> pd.Series:
    r = clean_returns(market)
    trailing = (1.0 + r).rolling(window).apply(np.prod, raw=True).shift(1) - 1.0
    labels = pd.Series(np.where(trailing >= 0, "up market", "down market"), index=r.index)
    return labels.where(trailing.notna())


def calendar_regimes(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index.year.astype(str), index=index)


def regime_performance(returns: pd.Series, labels: pd.Series, periods_per_year: float,
                       min_obs: int = 20) -> dict[str, Any]:
    r = clean_returns(returns)
    lab = labels.reindex(r.index)
    rows = []
    total_pnl = float(r.sum())
    for name, grp in r.groupby(lab):
        if len(grp) < 2:
            continue
        sd = grp.std(ddof=1)
        sr = float(grp.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else float("nan")
        rows.append({
            "regime": str(name), "n_obs": int(len(grp)), "ann_return": float(grp.mean() * periods_per_year),
            "sharpe": sr, "t_stat": float(grp.mean() / (sd / math.sqrt(len(grp)))) if sd > 0 else float("nan"),
            "share_of_pnl": float(grp.sum() / total_pnl) if total_pnl != 0 else float("nan"),
        })
    usable = [row for row in rows if row["n_obs"] >= min_obs and math.isfinite(row["sharpe"])]
    positive = [row for row in usable if row["sharpe"] > 0]
    return {"regimes": rows, "n_regimes": len(usable),
            "share_positive": len(positive) / len(usable) if usable else float("nan"),
            "worst": min((row["sharpe"] for row in usable), default=float("nan"))}
