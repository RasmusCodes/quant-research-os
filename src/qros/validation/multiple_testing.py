"""Multiple-testing corrections and the Sharpe-ratio haircut.

``adjust_pvalues`` implements Bonferroni, Sidak, Holm (step-down FWER), Hochberg (step-up FWER),
Benjamini-Hochberg (FDR) and Benjamini-Yekutieli (FDR under arbitrary dependence). Missing p-values
are ignored and returned as NaN.

``haircut_sharpe`` follows the idea of Harvey & Liu (2015, "Backtesting"): convert the Sharpe ratio to
a two-sided p-value, adjust it for the number of tests, and convert the adjusted p-value back into a
Sharpe ratio. The haircut is the fraction of the Sharpe ratio that the correction removes.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Sequence

import numpy as np

_N = NormalDist()
METHODS = ("bonferroni", "sidak", "holm", "hochberg", "bh", "by")


def adjust_pvalues(pvalues: Sequence[float], method: str = "holm") -> np.ndarray:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    p_all = np.asarray(pvalues, dtype=float)
    out = np.full(p_all.shape, np.nan)
    ok = np.isfinite(p_all)
    p = np.clip(p_all[ok], 0.0, 1.0)
    m = len(p)
    if m == 0:
        return out
    if method == "bonferroni":
        adj = np.minimum(p * m, 1.0)
    elif method == "sidak":
        adj = 1.0 - (1.0 - p) ** m
    else:
        order = np.argsort(p, kind="mergesort")
        ps = p[order]
        ranks = np.arange(1, m + 1)
        if method == "holm":
            stepped = np.maximum.accumulate((m - ranks + 1) * ps)
        elif method == "hochberg":
            stepped = np.minimum.accumulate(((m - ranks + 1) * ps)[::-1])[::-1]
        else:
            factor = 1.0 if method == "bh" else float(np.sum(1.0 / ranks))
            stepped = np.minimum.accumulate((ps * m * factor / ranks)[::-1])[::-1]
        adj = np.empty(m)
        adj[order] = np.minimum(stepped, 1.0)
    out[ok] = adj
    return out


def sharpe_pvalue(sharpe_annual: float, n_obs: int, periods_per_year: float, two_sided: bool = True) -> float:
    """p-value of H0: Sharpe = 0 from t = SR_per_period * sqrt(T) (normal approximation)."""
    if n_obs < 2 or sharpe_annual != sharpe_annual:
        return float("nan")
    t = sharpe_annual / math.sqrt(periods_per_year) * math.sqrt(n_obs)
    if two_sided:
        return 2.0 * (1.0 - _N.cdf(abs(t)))
    return 1.0 - _N.cdf(t)


def haircut_sharpe(sharpe_annual: float, n_obs: int, periods_per_year: float, n_tests: int,
                   method: str = "bonferroni", other_pvalues: Sequence[float] | None = None) -> dict[str, float]:
    """Adjusted Sharpe ratio after accounting for ``n_tests`` tests.

    With ``other_pvalues`` (the remaining tests' p-values) Holm and BHY use the full set; without them
    only Bonferroni is possible and is used.
    """
    p = sharpe_pvalue(sharpe_annual, n_obs, periods_per_year)
    if p != p:
        return {"p_value": p, "p_adjusted": p, "sharpe_adjusted": float("nan"), "haircut": float("nan")}
    if other_pvalues is not None and method in ("holm", "by", "bh", "hochberg"):
        allp = np.concatenate([[p], np.asarray(other_pvalues, dtype=float)])
        p_adj = float(adjust_pvalues(allp, method)[0])
    else:
        p_adj = min(1.0, p * max(1, n_tests))
    if p_adj >= 1.0:
        sr_adj = 0.0
    elif p_adj < 1e-15:
        # Beyond double precision of the normal tail: the correction is negligible.
        sr_adj = sharpe_annual
    else:
        t_adj = _N.inv_cdf(1.0 - p_adj / 2.0)
        sr_adj = math.copysign(t_adj / math.sqrt(n_obs) * math.sqrt(periods_per_year), sharpe_annual)
    haircut = 1.0 - sr_adj / sharpe_annual if sharpe_annual not in (0.0,) else float("nan")
    return {"p_value": p, "p_adjusted": p_adj, "sharpe_adjusted": sr_adj, "haircut": haircut}
