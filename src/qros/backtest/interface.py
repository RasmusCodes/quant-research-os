"""The standard result every strategy hands to the research OS.

Only ``returns`` is required. Everything else is optional but unlocks more checks: gross returns and
costs enable cost-sensitivity tests, positions enable exposure and turnover statistics, a benchmark
enables beta, information ratio and market-regime analysis, and ``meta`` carries audit records such
as the point-in-time access log or walk-forward selections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


class ResultError(ValueError):
    pass


def _series(x: Any, name: str) -> pd.Series | None:
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ResultError(f"{name} must be a single column")
        x = x.iloc[:, 0]
    s = pd.Series(x).astype("float64")
    s.name = name
    return s


@dataclass
class BacktestResult:
    returns: pd.Series
    gross_returns: pd.Series | None = None
    costs: pd.Series | None = None
    turnover: pd.Series | None = None
    positions: pd.DataFrame | None = None
    signals: pd.DataFrame | None = None
    exposure: pd.DataFrame | None = None
    benchmark: pd.Series | None = None
    periods_per_year: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.returns = _series(self.returns, "net")
        self.gross_returns = _series(self.gross_returns, "gross")
        self.costs = _series(self.costs, "cost")
        self.turnover = _series(self.turnover, "turnover")
        self.benchmark = _series(self.benchmark, "benchmark")

    # ---------------------------------------------------------------- constructors

    @classmethod
    def from_returns(cls, returns: Any, **kwargs: Any) -> "BacktestResult":
        return cls(returns=returns, **kwargs)

    @classmethod
    def from_pnl(cls, pnl: pd.Series, capital: float, **kwargs: Any) -> "BacktestResult":
        """Per-period P&L in currency on a constant capital base."""
        if capital <= 0:
            raise ResultError("capital must be positive")
        return cls(returns=pd.Series(pnl, dtype=float) / capital, **kwargs)

    @classmethod
    def from_equity(cls, equity: pd.Series, **kwargs: Any) -> "BacktestResult":
        e = pd.Series(equity, dtype=float)
        return cls(returns=e.pct_change().dropna(), **kwargs)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, **kwargs: Any) -> "BacktestResult":
        """A frame with a ``net`` column and optional gross/cost/turnover/benchmark/exposure columns."""
        if "net" not in frame.columns:
            raise ResultError("frame needs a 'net' column")
        exposure_cols = [c for c in ("gross_exposure", "net_exposure", "long_exposure", "short_exposure")
                         if c in frame.columns]
        return cls(
            returns=frame["net"], gross_returns=frame.get("gross"), costs=frame.get("cost"),
            turnover=frame.get("turnover"), benchmark=frame.get("benchmark"),
            exposure=frame[exposure_cols].rename(columns=lambda c: c.replace("_exposure", "")) if exposure_cols else None,
            **kwargs)

    # ---------------------------------------------------------------- checks and views

    def validate(self) -> "BacktestResult":
        r = self.returns
        if r is None or len(r) == 0:
            raise ResultError("returns are empty")
        if r.index.has_duplicates:
            raise ResultError("returns have duplicate timestamps")
        if isinstance(r.index, pd.DatetimeIndex):
            if not r.index.is_monotonic_increasing:
                raise ResultError("returns must be sorted by time")
        if not np.isfinite(r.to_numpy()).all():
            raise ResultError("returns contain NaN or infinite values")
        if (r <= -1.0).any():
            raise ResultError("a per-period return of -100% or worse: check units (returns, not percent)")
        if r.abs().max() > 1.0:
            self.meta.setdefault("warnings", []).append("per-period returns above 100%: are these in percent?")
        for name in ("gross_returns", "costs", "turnover", "benchmark"):
            s = getattr(self, name)
            if s is not None and not s.index.equals(r.index):
                aligned = s.reindex(r.index)
                setattr(self, name, aligned)
        if self.gross_returns is not None and self.costs is not None:
            diff = (self.gross_returns - self.costs - r).abs().max()
            if np.isfinite(diff) and diff > 1e-9:
                raise ResultError(f"net != gross - costs (max difference {diff:.3g})")
        return self

    def frame(self) -> pd.DataFrame:
        cols: dict[str, pd.Series] = {"net": self.returns}
        if self.gross_returns is not None:
            cols["gross"] = self.gross_returns
        if self.costs is not None:
            cols["cost"] = self.costs
        if self.turnover is not None:
            cols["turnover"] = self.turnover
        if self.benchmark is not None:
            cols["benchmark"] = self.benchmark
        if self.exposure is not None:
            for c in self.exposure.columns:
                cols[f"{c}_exposure"] = self.exposure[c].astype(float)
        frame = pd.DataFrame(cols)
        frame.index.name = "date" if isinstance(frame.index, pd.DatetimeIndex) else frame.index.name
        return frame


def as_result(obj: Any, **kwargs: Any) -> BacktestResult:
    """Coerce a BacktestResult, a return Series or a result frame into a BacktestResult."""
    if isinstance(obj, BacktestResult):
        return obj
    if isinstance(obj, pd.DataFrame):
        return BacktestResult.from_frame(obj, **kwargs)
    if isinstance(obj, (pd.Series, np.ndarray, list)):
        return BacktestResult.from_returns(obj, **kwargs)
    if isinstance(obj, dict) and "returns" in obj:
        return BacktestResult(**obj)
    raise ResultError(f"Cannot interpret {type(obj).__name__} as a backtest result")
