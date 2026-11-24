"""Out-of-sample evaluation schemes: walk-forward windows and purged k-fold cross-validation.

* Walk-forward: train on a window, test on the next, roll forward (expanding or rolling train window,
  optional gap between train and test).
* Purged k-fold with embargo (Lopez de Prado 2018, *Advances in Financial Machine Learning*, ch. 7):
  each observation i has a label interval [t0_i, t1_i]. Training observations whose interval overlaps
  the test fold's span are purged, and the ``embargo`` fraction of observations right after the test
  fold is dropped from training, so labels that look into the test period cannot leak into training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import pandas as pd

from qros.util.hashing import canonical_json


@dataclass(frozen=True)
class Fold:
    index: int
    train: np.ndarray
    test: np.ndarray
    train_range: tuple[str, str] | None = None
    test_range: tuple[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "n_train": int(len(self.train)), "n_test": int(len(self.test)),
                "train_range": self.train_range, "test_range": self.test_range}


def _label(index: pd.Index, positions: np.ndarray) -> tuple[str, str] | None:
    if not len(positions) or not isinstance(index, pd.DatetimeIndex):
        return None
    return (index[positions[0]].strftime("%Y-%m-%d"), index[positions[-1]].strftime("%Y-%m-%d"))


def walk_forward_splits(index: pd.Index | int, *, train: int, test: int, step: int | None = None,
                        expanding: bool = True, gap: int = 0) -> list[Fold]:
    """Folds over positions 0..n-1: train windows of ``train`` observations, then ``gap``, then ``test``."""
    idx = pd.RangeIndex(index) if isinstance(index, int) else index
    n = len(idx)
    step = step or test
    if train < 1 or test < 1 or gap < 0:
        raise ValueError("train and test must be positive and gap non-negative")
    folds, start = [], 0
    while True:
        train_end = start + train
        test_start = train_end + gap
        test_end = test_start + test
        if test_end > n:
            break
        tr = np.arange(0 if expanding else start, train_end)
        te = np.arange(test_start, test_end)
        folds.append(Fold(len(folds), tr, te, _label(idx, tr), _label(idx, te)))
        start += step
    return folds


def walk_forward_by_years(index: pd.DatetimeIndex, *, train_years: int, test_years: int = 1,
                          expanding: bool = True, first_test_year: int | None = None) -> list[Fold]:
    """Calendar walk-forward: test on whole calendar years, train on the preceding years."""
    years = sorted(set(index.year))
    folds = []
    start_year = first_test_year or years[0] + train_years
    for y in range(start_year, years[-1] + 1, test_years):
        te_mask = (index.year >= y) & (index.year < y + test_years)
        lo = years[0] if expanding else y - train_years
        tr_mask = (index.year >= lo) & (index.year < y)
        tr, te = np.flatnonzero(tr_mask), np.flatnonzero(te_mask)
        if len(tr) == 0 or len(te) == 0:
            continue
        folds.append(Fold(len(folds), tr, te, _label(index, tr), _label(index, te)))
    return folds


class PurgedKFold:
    """k-fold CV over time-ordered observations with purging and an embargo."""

    def __init__(self, n_splits: int = 5, embargo: float = 0.0):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not 0.0 <= embargo < 1.0:
            raise ValueError("embargo is a fraction of the sample in [0, 1)")
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, t0: Sequence[Any], t1: Sequence[Any] | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        start = pd.DatetimeIndex(t0) if not isinstance(t0, pd.DatetimeIndex) else t0
        if not start.is_monotonic_increasing:
            raise ValueError("observations must be sorted by start time")
        end = start if t1 is None else pd.DatetimeIndex(t1)
        if len(end) != len(start) or (end < start).any():
            raise ValueError("t1 must have one end time >= t0 per observation")
        n = len(start)
        s = start.values.astype("datetime64[ns]").view("i8")
        e = end.values.astype("datetime64[ns]").view("i8")
        embargo_n = int(math.ceil(n * self.embargo))
        for block in np.array_split(np.arange(n), self.n_splits):
            if len(block) == 0:
                continue
            i0, i1 = block[0], block[-1]
            test_lo, test_hi = s[i0], e[block].max()
            keep = np.ones(n, dtype=bool)
            keep[block] = False
            keep &= ~((s <= test_hi) & (e >= test_lo))          # purge overlapping labels
            keep[i1 + 1: min(n, i1 + 1 + embargo_n)] = False      # embargo after the test fold
            yield np.flatnonzero(keep), block


def walk_forward_optimize(candidates: dict[str, pd.Series], folds: Sequence[Fold], index: pd.Index, *,
                          score: Callable[[pd.Series], float] | None = None,
                          params: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Select the best candidate on each train window and stitch its test-window returns.

    ``candidates`` maps a configuration key to its full-sample return series. This is valid for causal
    strategies, whose return at t only uses data up to t, so evaluating every configuration once and
    slicing by window equals re-running them per fold. The stitched series is a genuine out-of-sample
    record of the *selection procedure*.
    """
    if score is None:
        def score(r: pd.Series) -> float:
            sd = r.std(ddof=1)
            return float(r.mean() / sd) if sd and sd > 0 else -np.inf
    frame = pd.DataFrame(candidates).reindex(index)
    pieces, rows, chosen = [], [], []
    for fold in folds:
        train = frame.iloc[fold.train]
        scores = {k: score(train[k].dropna()) for k in frame.columns}
        best = max(scores, key=lambda k: (scores[k], k))
        chosen.append(best)
        test_returns = frame.iloc[fold.test][best]
        pieces.append(test_returns)
        rows.append({**fold.to_dict(), "selected": best, "train_score": scores[best],
                     "selected_params": (params or {}).get(best)})
    oos = pd.concat(pieces) if pieces else pd.Series(dtype=float)
    switches = sum(1 for a, b in zip(chosen, chosen[1:]) if a != b)
    modal = max(set(chosen), key=chosen.count) if chosen else None
    return {
        "oos_returns": oos,
        "folds": rows,
        "selections": chosen,
        "switch_rate": switches / max(1, len(chosen) - 1),
        "modal_selection": modal,
        "modal_share": chosen.count(modal) / len(chosen) if chosen else float("nan"),
        "distinct_selections": len(set(chosen)),
        "candidates": len(frame.columns),
        "selection_key": canonical_json(chosen),
    }
