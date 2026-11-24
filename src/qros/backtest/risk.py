"""Risk and exposure statistics computed the same way for every strategy."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from qros.backtest.interface import BacktestResult


def exposure_summary(result: BacktestResult) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if result.exposure is not None and len(result.exposure):
        e = result.exposure
        for col in e.columns:
            out[f"avg_{col}"] = float(e[col].mean())
            out[f"max_{col}"] = float(e[col].abs().max())
    if result.positions is not None and len(result.positions):
        w = result.positions.to_numpy(dtype=float)
        sq = (w ** 2).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            enp = np.where(sq > 0, (np.abs(w).sum(axis=1) ** 2) / sq, np.nan)
        out["effective_positions"] = float(np.nanmean(enp)) if np.isfinite(enp).any() else float("nan")
        out["max_weight"] = float(np.nanmax(np.abs(w))) if w.size else float("nan")
    return out


def tail_risk(returns: pd.Series, levels: tuple[float, ...] = (0.95, 0.99)) -> dict[str, float]:
    r = returns.dropna().to_numpy(dtype=float)
    out = {}
    for level in levels:
        q = float(np.quantile(r, 1 - level))
        out[f"var_{int(level * 100)}"] = -q
        tail = r[r <= q]
        out[f"cvar_{int(level * 100)}"] = -float(tail.mean()) if len(tail) else float("nan")
    return out


def rolling_volatility(returns: pd.Series, window: int = 63, periods_per_year: float = 252.0) -> pd.Series:
    return returns.rolling(window).std(ddof=1) * math.sqrt(periods_per_year)
