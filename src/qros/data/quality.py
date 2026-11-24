"""Data-quality profile recorded with every dataset snapshot."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PRICE_LIKE = ("close", "price", "open", "high", "low", "adj_close")


def quality_report(df: pd.DataFrame, *, entity_col: str | None = None, time_col: str | None = None) -> dict[str, Any]:
    n_rows, n_cols = df.shape
    report: dict[str, Any] = {"rows": int(n_rows), "cols": int(n_cols)}
    if n_rows == 0:
        return report
    na = df.isna().mean()
    report["missing"] = {str(k): round(float(v), 6) for k, v in na.items() if v > 0}
    report["missing_cells"] = round(float(df.isna().to_numpy().mean()), 6)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1]:
        arr = numeric.to_numpy(dtype=float)
        report["infinite"] = int(np.isinf(arr).sum())
        bad_price = 0
        for col in numeric.columns:
            if str(col).lower() in PRICE_LIKE:
                bad_price += int((numeric[col] <= 0).sum())
        report["nonpositive_prices"] = bad_price
    if entity_col and time_col and {entity_col, time_col} <= set(df.columns):
        keys = df[[entity_col, time_col]]
        report["duplicate_keys"] = int(keys.duplicated().sum())
        report["entities"] = int(df[entity_col].nunique())
        t = pd.to_datetime(df[time_col], errors="coerce")
        last = t.groupby(df[entity_col]).max()
        first = t.groupby(df[entity_col]).min()
        report["entities_ending_early"] = int((last < t.max()).sum())
        report["entities_starting_late"] = int((first > t.min()).sum())
        report["obs_per_entity_median"] = float(df.groupby(entity_col).size().median())
    elif isinstance(df.index, pd.DatetimeIndex):
        report["duplicate_index"] = int(df.index.duplicated().sum())
        if df.shape[1] > 1 and numeric.shape[1] == df.shape[1]:
            # Wide panel (dates x entities): columns that stop early are candidates for delistings.
            last_valid = df.apply(lambda s: s.last_valid_index())
            first_valid = df.apply(lambda s: s.first_valid_index())
            report["entities"] = int(df.shape[1])
            report["entities_ending_early"] = int((last_valid < df.index.max()).sum())
            report["entities_starting_late"] = int((first_valid > df.index.min()).sum())
            stale = (df.diff() == 0) & df.notna()
            report["stale_share"] = round(float(stale.to_numpy().mean()), 6)
    return report
