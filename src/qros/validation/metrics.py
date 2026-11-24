"""Consistent performance metrics for any per-period return series.

Every strategy, whatever produced it, is summarised by the same function, so comparisons across
experiments and repositories are like-for-like. Returns are simple (not log) per-period returns.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

_N = NormalDist()
STANDARD_FREQUENCIES = (1, 4, 12, 52, 252, 365)


def clean_returns(returns: Any) -> pd.Series:
    r = pd.Series(returns, dtype="float64") if not isinstance(returns, pd.Series) else returns.astype("float64")
    return r.replace([np.inf, -np.inf], np.nan).dropna()


def infer_periods_per_year(index: pd.Index, default: float = 252.0) -> float:
    """Observations per year from a DatetimeIndex, snapped to 1/4/12/52/252/365 when within 15%."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return float(default)
    span_years = (index[-1] - index[0]).days / 365.25
    if span_years <= 0:
        return float(default)
    raw = (len(index) - 1) / span_years
    best = min(STANDARD_FREQUENCIES, key=lambda f: abs(math.log(raw / f)))
    return float(best) if abs(raw / best - 1) <= 0.15 else float(raw)


def moments(r: np.ndarray) -> tuple[float, float, float, float]:
    """Mean, sample std (ddof=1), skewness and (non-excess) kurtosis from population moments."""
    n = len(r)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(r.mean())
    dev = r - mean
    m2 = float(np.mean(dev ** 2))
    std = float(r.std(ddof=1))
    if m2 <= 0:
        return mean, std, 0.0, 3.0
    skew = float(np.mean(dev ** 3) / m2 ** 1.5)
    kurt = float(np.mean(dev ** 4) / m2 ** 2)
    return mean, std, skew, kurt


def newey_west_tstat(x: Any, lags: int | None = None) -> tuple[float, int]:
    """t-statistic of the mean with Newey-West (Bartlett) standard errors; returns (t, lags)."""
    a = np.asarray(clean_returns(x), dtype=float)
    n = len(a)
    if n < 3:
        return float("nan"), 0
    if lags is None:
        lags = int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, n - 1))
    mean = a.mean()
    d = a - mean
    s = float(d @ d) / n
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        s += 2.0 * w * float(d[lag:] @ d[:-lag]) / n
    if s <= 0:
        return float("nan"), lags
    return float(mean / math.sqrt(s / n)), lags


def drawdown_series(returns: Any) -> pd.Series:
    r = clean_returns(returns)
    equity = (1.0 + r).cumprod()
    peak = np.maximum(equity.cummax(), 1.0)
    return equity / peak - 1.0


def _max_underwater(dd: pd.Series) -> int:
    longest = current = 0
    for v in (dd.to_numpy() < -1e-12):
        current = current + 1 if v else 0
        longest = max(longest, current)
    return int(longest)


def sharpe(returns: Any, periods_per_year: float = 252.0) -> float:
    r = np.asarray(clean_returns(returns), dtype=float)
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else float("nan")


def performance_summary(returns: Any, periods_per_year: float | None = None, *, turnover: Any = None,
                        costs: Any = None, gross: Any = None, gross_exposure: Any = None, net_exposure: Any = None,
                        benchmark: Any = None) -> dict[str, Any]:
    """Standard metrics. Annualised figures use ``periods_per_year`` (inferred when omitted)."""
    r = clean_returns(returns)
    ppy = float(periods_per_year or infer_periods_per_year(r.index))
    out: dict[str, Any] = {"n_obs": int(len(r)), "periods_per_year": ppy}
    if isinstance(r.index, pd.DatetimeIndex) and len(r):
        out["start"] = r.index[0].strftime("%Y-%m-%d")
        out["end"] = r.index[-1].strftime("%Y-%m-%d")
    if len(r) < 2:
        return out
    a = r.to_numpy()
    mean, std, skew, kurt = moments(a)
    years = len(a) / ppy
    equity = np.cumprod(1.0 + a)
    final = float(equity[-1])
    dd = drawdown_series(r)
    max_dd = float(dd.min())
    downside = math.sqrt(float(np.mean(np.minimum(a, 0.0) ** 2)))
    nonzero = a[a != 0]
    sr = mean / std if std > 0 else float("nan")
    nw_t, nw_lags = newey_west_tstat(a)
    cagr = final ** (1.0 / years) - 1.0 if final > 0 and years > 0 else -1.0
    q05 = float(np.quantile(a, 0.05))
    out.update({
        "years": years,
        "total_return": final - 1.0,
        "cagr": cagr,
        "ann_return": mean * ppy,
        "ann_vol": std * math.sqrt(ppy),
        "sharpe": sr * math.sqrt(ppy) if std > 0 else float("nan"),
        "sharpe_per_period": sr,
        "sortino": mean / downside * math.sqrt(ppy) if downside > 0 else float("nan"),
        "max_drawdown": max_dd,
        "max_dd_periods": _max_underwater(dd),
        "calmar": cagr / abs(max_dd) if max_dd < 0 else float("nan"),
        "hit_rate": float(np.mean(nonzero > 0)) if len(nonzero) else float("nan"),
        "skew": skew,
        "kurtosis": kurt,
        "t_stat": mean / (std / math.sqrt(len(a))) if std > 0 else float("nan"),
        "nw_t_stat": nw_t,
        "nw_lags": nw_lags,
        "p_value": 1.0 - _N.cdf(nw_t) if nw_t == nw_t else float("nan"),
        "best": float(a.max()),
        "worst": float(a.min()),
        "var_95": -q05,
        "cvar_95": -float(a[a <= q05].mean()),
    })
    if std > 0:
        from qros.validation.sharpe import probabilistic_sharpe_ratio
        out["psr0"] = probabilistic_sharpe_ratio(sr, len(a), skew, kurt, 0.0)
    for name, series in (("turnover", turnover), ("cost", costs), ("gross_exposure", gross_exposure),
                         ("net_exposure", net_exposure)):
        if series is None:
            continue
        s = pd.Series(series, dtype="float64").reindex(r.index) if isinstance(series, pd.Series) else \
            pd.Series(np.asarray(series, dtype=float)[: len(r)], index=r.index)
        s = s.dropna()
        if len(s) == 0:
            continue
        out[f"avg_{name}"] = float(s.mean())
        if name in ("turnover", "cost"):
            out[f"ann_{name}"] = float(s.mean() * ppy)
    if gross is not None:
        g = pd.Series(gross, dtype="float64").reindex(r.index).dropna() if isinstance(gross, pd.Series) else None
        if g is not None and len(g) > 1:
            out["gross_sharpe"] = sharpe(g, ppy)
            out["gross_ann_return"] = float(g.mean() * ppy)
    if benchmark is not None:
        b = pd.Series(benchmark, dtype="float64").reindex(r.index)
        joined = pd.concat([r, b], axis=1).dropna()
        if len(joined) > 2 and joined.iloc[:, 1].var() > 0:
            x, y = joined.iloc[:, 1].to_numpy(), joined.iloc[:, 0].to_numpy()
            beta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
            resid = y - beta * x
            out["beta"] = beta
            out["alpha_ann"] = float(resid.mean() * ppy)
            out["correlation"] = float(np.corrcoef(x, y)[0, 1])
            rs = resid.std(ddof=1)
            out["information_ratio"] = float(resid.mean() / rs * math.sqrt(ppy)) if rs > 0 else float("nan")
    return out
