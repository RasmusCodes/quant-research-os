"""Stationary bootstrap (Politis & Romano 1994) for time-series statistics.

Blocks start at random positions and have geometric lengths with mean ``mean_block``, which keeps
serial dependence (volatility clustering, autocorrelated returns) inside each resample. The default
mean block length is n^(1/3), a common rule of thumb; pass your own for strongly persistent series.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

from qros.validation.metrics import clean_returns


def default_block_length(n: int) -> int:
    return max(1, int(round(n ** (1.0 / 3.0))))


def stationary_bootstrap_indices(n: int, mean_block: float, n_boot: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """An (n_boot, n) array of resampled positions."""
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    restart = rng.random((n_boot, n)) < p
    starts = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(restart[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def _resampled_stat(a: np.ndarray, stat: Callable[[np.ndarray], np.ndarray], n_boot: int, mean_block: float,
                    seed: int, chunk: int = 250) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        idx = stationary_bootstrap_indices(len(a), mean_block, k, rng)
        out[done:done + k] = stat(a[idx])
        done += k
    return out


def _sharpe_rows(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, x.mean(axis=1) / sd, np.nan)


def bootstrap_sharpe(returns: pd.Series | np.ndarray, periods_per_year: float = 252.0, *, n_boot: int = 1000,
                     mean_block: float | None = None, alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    """Percentile confidence interval for the annualised Sharpe ratio and P(SR <= 0)."""
    a = np.asarray(clean_returns(returns), dtype=float)
    n = len(a)
    if n < 10 or a.std(ddof=1) == 0:
        return {"n_obs": n, "sharpe": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "se": float("nan"), "prob_le_0": float("nan")}
    block = mean_block or default_block_length(n)
    boot = _resampled_stat(a, _sharpe_rows, n_boot, block, seed) * math.sqrt(periods_per_year)
    boot = boot[np.isfinite(boot)]
    point = a.mean() / a.std(ddof=1) * math.sqrt(periods_per_year)
    return {
        "n_obs": n, "sharpe": float(point), "lo": float(np.quantile(boot, alpha / 2)),
        "hi": float(np.quantile(boot, 1 - alpha / 2)), "se": float(boot.std(ddof=1)),
        "prob_le_0": float(np.mean(boot <= 0)), "mean_block": float(block), "n_boot": int(len(boot)),
        "alpha": alpha,
    }


def bootstrap_statistic(returns: pd.Series | np.ndarray, stat: Callable[[np.ndarray], float], *,
                        n_boot: int = 1000, mean_block: float | None = None, alpha: float = 0.05,
                        seed: int = 0) -> dict[str, float]:
    """Percentile interval for any scalar statistic of a return series."""
    a = np.asarray(clean_returns(returns), dtype=float)
    block = mean_block or default_block_length(len(a))
    vec = lambda x: np.apply_along_axis(stat, 1, x)  # noqa: E731
    boot = _resampled_stat(a, vec, n_boot, block, seed)
    boot = boot[np.isfinite(boot)]
    return {"value": float(stat(a)), "lo": float(np.quantile(boot, alpha / 2)),
            "hi": float(np.quantile(boot, 1 - alpha / 2)), "se": float(boot.std(ddof=1))}


def paired_sharpe_difference(a: pd.Series, b: pd.Series, periods_per_year: float = 252.0, *, n_boot: int = 1000,
                             mean_block: float | None = None, alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    """Bootstrap the difference of two strategies' Sharpe ratios on their common dates (paired)."""
    joined = pd.concat([clean_returns(a), clean_returns(b)], axis=1, join="inner").dropna()
    x = joined.to_numpy()
    n = len(x)
    if n < 20:
        return {"n_obs": n, "diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_value": float("nan")}
    block = mean_block or default_block_length(n)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    done = 0
    while done < n_boot:
        k = min(250, n_boot - done)
        idx = stationary_bootstrap_indices(n, block, k, rng)
        diffs[done:done + k] = _sharpe_rows(x[:, 0][idx]) - _sharpe_rows(x[:, 1][idx])
        done += k
    diffs = diffs[np.isfinite(diffs)] * math.sqrt(periods_per_year)
    point = (x[:, 0].mean() / x[:, 0].std(ddof=1) - x[:, 1].mean() / x[:, 1].std(ddof=1)) * math.sqrt(
        periods_per_year)
    p_two = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {"n_obs": n, "diff": float(point), "lo": float(np.quantile(diffs, alpha / 2)),
            "hi": float(np.quantile(diffs, 1 - alpha / 2)), "p_value": float(min(1.0, p_two)),
            "correlation": float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])}
