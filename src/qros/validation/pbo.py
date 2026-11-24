"""Probability of backtest overfitting via combinatorially symmetric cross-validation (CSCV).

Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability of Backtest Overfitting":

1. Stack the per-period returns of N configurations into a T x N matrix and cut its rows into S
   contiguous blocks (S even).
2. For every one of the C(S, S/2) ways to pick half of the blocks as the in-sample set J (the rest,
   J-bar, is out-of-sample): find the configuration n* with the best in-sample Sharpe ratio and its
   relative rank w in the out-of-sample Sharpe ratios, w = rank / (N + 1).
3. The logit l = ln(w / (1 - w)) is <= 0 when the in-sample winner is at or below the out-of-sample
   median. PBO is the fraction of combinations with l <= 0.

Block sums and sums of squares are precomputed, so every combination costs one matrix product.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd


def probability_of_backtest_overfitting(returns: pd.DataFrame, n_splits: int = 16,
                                        periods_per_year: float = 252.0) -> dict[str, Any] | None:
    """CSCV PBO for the configurations in ``returns`` (columns). ``None`` if there is too little data."""
    if n_splits % 2 or n_splits < 2:
        raise ValueError("n_splits must be an even number >= 2")
    x = returns.dropna(how="any").to_numpy(dtype=float)
    t, n = x.shape
    if n < 2 or t < n_splits * 5:
        return None
    blocks = np.array_split(np.arange(t), n_splits)
    sums = np.stack([x[b].sum(axis=0) for b in blocks])
    sqs = np.stack([(x[b] ** 2).sum(axis=0) for b in blocks])
    counts = np.array([len(b) for b in blocks], dtype=float)
    combos = list(itertools.combinations(range(n_splits), n_splits // 2))
    member = np.zeros((len(combos), n_splits))
    for i, combo in enumerate(combos):
        member[i, list(combo)] = 1.0

    def sharpe(s: np.ndarray, q: np.ndarray, c: np.ndarray) -> np.ndarray:
        mean = s / c[:, None]
        var = (q - c[:, None] * mean ** 2) / (c[:, None] - 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            sr = np.where(var > 0, mean / np.sqrt(np.maximum(var, 1e-300)), 0.0)
        return sr * math.sqrt(periods_per_year)

    is_sr = sharpe(member @ sums, member @ sqs, member @ counts)
    oos_member = 1.0 - member
    oos_sr = sharpe(oos_member @ sums, oos_member @ sqs, oos_member @ counts)
    best = np.argmax(is_sr, axis=1)
    rows = np.arange(len(combos))
    is_best = is_sr[rows, best]
    oos_best = oos_sr[rows, best]
    below = (oos_sr < oos_best[:, None]).sum(axis=1)
    ties = (oos_sr == oos_best[:, None]).sum(axis=1)
    rank = below + (ties + 1) / 2.0
    w = rank / (n + 1.0)
    logits = np.log(w / (1.0 - w))
    slope, intercept = (np.polyfit(is_best, oos_best, 1) if np.ptp(is_best) > 0 else (float("nan"), float("nan")))
    hist, edges = np.histogram(logits, bins=24)
    return {
        "pbo": float(np.mean(logits <= 0)),
        "n_configs": int(n),
        "n_obs": int(t),
        "n_splits": int(n_splits),
        "n_combinations": int(len(combos)),
        "logit_median": float(np.median(logits)),
        "is_best_sharpe_median": float(np.median(is_best)),
        "oos_of_best_sharpe_median": float(np.median(oos_best)),
        "prob_oos_loss": float(np.mean(oos_best < 0)),
        "degradation_slope": float(slope),
        "degradation_intercept": float(intercept),
        "logit_hist": {"counts": hist.tolist(), "edges": [float(e) for e in edges]},
        "selected_counts": {str(k): int(v) for k, v in zip(*np.unique(best, return_counts=True))},
    }
