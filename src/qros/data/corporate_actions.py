"""Corporate-action adjustments and checks for unadjusted prices.

``actions`` is a long table with columns ``entity``, ``date`` (ex-date), ``split`` (new shares per old
share, 1 when none) and ``dividend`` (cash per post-split share, 0 when none).

The backward adjustment factor for an event on ex-date e is f = P_e / ((P_e + D) * k), applied to all
prices before e. With it, adjusted-price returns equal holding-period total returns
(P_e + D) * k / P_{e-1} - 1 exactly, i.e. dividends reinvested at the ex-date close.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

SPLIT_RATIOS = (2.0, 3.0, 4.0, 5.0, 10.0, 1.5, 1.25, 0.5, 1 / 3, 0.25, 0.1)


def _event_table(actions: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    a = actions.copy()
    a["date"] = pd.to_datetime(a["date"]).astype("datetime64[ns]")
    a["split"] = a.get("split", 1.0)
    a["dividend"] = a.get("dividend", 0.0)
    a["split"] = a["split"].fillna(1.0).astype(float)
    a["dividend"] = a["dividend"].fillna(0.0).astype(float)
    a = a.groupby(["entity", "date"], as_index=False).agg(split=("split", "prod"), dividend=("dividend", "sum"))
    a = a[a["entity"].isin(close.columns) & a["date"].isin(close.index)]
    return a


def adjustment_factors(close: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Cumulative backward factors F so that adjusted = close * F."""
    factors = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    events = _event_table(actions, close)
    positions = {d: i for i, d in enumerate(close.index)}
    for row in events.itertuples(index=False):
        i = positions[row.date]
        p_e = close.at[row.date, row.entity]
        if not np.isfinite(p_e) or p_e <= 0 or i == 0:
            continue
        f = p_e / ((p_e + row.dividend) * row.split)
        factors.iloc[:i, factors.columns.get_loc(row.entity)] *= f
    return factors


def adjust_prices(close: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    return close * adjustment_factors(close, actions)


def total_returns(close: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Holding-period returns including splits and dividends: (P_t + D_t) * k_t / P_{t-1} - 1."""
    events = _event_table(actions, close)
    split = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    div = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for row in events.itertuples(index=False):
        split.at[row.date, row.entity] *= row.split
        div.at[row.date, row.entity] += row.dividend
    return (close + div) * split / close.shift(1) - 1.0


def detect_unadjusted_splits(close: pd.DataFrame, actions: pd.DataFrame | None = None, *,
                             tolerance: float = 0.04, min_move: float = 0.2) -> list[dict[str, Any]]:
    """Find one-day price moves that look like an unadjusted split (price divides by ~2, 3, 1/2 ...).

    When ``actions`` is given, each jump is matched against the recorded splits on that date; matched
    jumps in a price series prove the series is NOT split-adjusted.
    """
    ratio = close.shift(1) / close
    move = (close / close.shift(1) - 1.0).abs()
    recorded = set()
    if actions is not None and len(actions):
        ev = _event_table(actions, close)
        recorded = {(r.entity, r.date) for r in ev.itertuples(index=False) if r.split != 1.0}
    found = []
    stacked = ratio.where(move >= min_move).stack()
    for (date, entity), value in stacked.items():
        for k in SPLIT_RATIOS:
            if abs(value / k - 1.0) <= tolerance:
                found.append({"entity": str(entity), "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                              "price_ratio": float(value), "implied_split": float(k),
                              "recorded_split": (entity, pd.Timestamp(date)) in recorded})
                break
    return found


def split_check_summary(close: pd.DataFrame, actions: pd.DataFrame | None) -> dict[str, Any]:
    jumps = detect_unadjusted_splits(close, actions)
    matched = [j for j in jumps if j["recorded_split"]]
    n_recorded = 0
    if actions is not None and len(actions):
        n_recorded = int((_event_table(actions, close)["split"] != 1.0).sum())
    return {"split_like_jumps": len(jumps), "matching_recorded_splits": len(matched), "recorded_splits": n_recorded,
            "looks_unadjusted": bool(matched) and len(matched) >= max(1, n_recorded // 2),
            "examples": jumps[:5]}
