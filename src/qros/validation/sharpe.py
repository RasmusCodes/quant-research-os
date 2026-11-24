"""Inference on Sharpe ratios under non-normal returns and multiple trials.

Formulas (all Sharpe ratios here are PER PERIOD, i.e. not annualised):

* Variance of the estimated Sharpe ratio (Mertens 2002; as used by Bailey & Lopez de Prado 2012)::

      Var[SR] ~= (1 - g3*SR + (g4 - 1)/4 * SR^2) / (T - 1)

  with g3 the skewness and g4 the (non-excess) kurtosis of returns.
* Probabilistic Sharpe ratio (Bailey & Lopez de Prado 2012, "The Sharpe Ratio Efficient Frontier")::

      PSR(SR*) = Phi( (SR - SR*) * sqrt(T - 1) / sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2) )

* Expected maximum Sharpe ratio of N independent unskilled trials whose estimated Sharpe ratios
  have cross-sectional variance V (Bailey & Lopez de Prado 2014, "The Deflated Sharpe Ratio")::

      SR0 = sqrt(V) * ( (1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N e)) )

  with gamma the Euler-Mascheroni constant. The deflated Sharpe ratio is DSR = PSR(SR0).
* Minimum track record length (Bailey & Lopez de Prado 2012)::

      MinTRL = 1 + (1 - g3*SR + (g4 - 1)/4 * SR^2) * (Phi^-1(1 - alpha) / (SR - SR*))^2
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


def norm_cdf(x: float) -> float:
    return _N.cdf(x)


def norm_ppf(p: float) -> float:
    return _N.inv_cdf(min(max(p, 1e-16), 1 - 1e-16))


def _denominator(sr: float, skew: float, kurt: float) -> float:
    d = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    return max(d, 1e-12)


def sharpe_standard_error(sr: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    if n_obs < 2:
        return float("nan")
    return math.sqrt(_denominator(sr, skew, kurt) / (n_obs - 1))


def probabilistic_sharpe_ratio(sr: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0,
                               sr_benchmark: float = 0.0) -> float:
    """Probability that the true per-period Sharpe exceeds ``sr_benchmark``."""
    if n_obs < 2 or sr != sr:
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(_denominator(sr, skew, kurt))
    return norm_cdf(z)


def expected_max_sharpe(n_trials: float, var_sharpe: float, mean_sharpe: float = 0.0) -> float:
    """Expected maximum per-period Sharpe among ``n_trials`` unskilled trials."""
    if n_trials <= 1 or var_sharpe <= 0:
        return mean_sharpe
    n = float(n_trials)
    z = (1.0 - EULER_GAMMA) * norm_ppf(1.0 - 1.0 / n) + EULER_GAMMA * norm_ppf(1.0 - 1.0 / (n * math.e))
    return mean_sharpe + math.sqrt(var_sharpe) * z


def deflated_sharpe_ratio(sr: float, n_obs: int, skew: float, kurt: float, n_trials: float,
                          var_trials: float) -> dict[str, float]:
    """DSR = PSR(SR0); returns the probability together with the hurdle SR0 (per period)."""
    sr0 = expected_max_sharpe(n_trials, var_trials)
    return {"dsr": probabilistic_sharpe_ratio(sr, n_obs, skew, kurt, sr0), "sr0": sr0}


def min_track_record_length(sr: float, skew: float = 0.0, kurt: float = 3.0, sr_benchmark: float = 0.0,
                            alpha: float = 0.05) -> float:
    """Observations needed for PSR(sr_benchmark) to reach 1 - alpha (inf if sr <= sr_benchmark)."""
    if sr != sr or sr <= sr_benchmark:
        return float("inf")
    return 1.0 + _denominator(sr, skew, kurt) * (norm_ppf(1.0 - alpha) / (sr - sr_benchmark)) ** 2


def effective_number_of_trials(returns: pd.DataFrame, min_overlap: int = 60) -> dict[str, Any]:
    """Two estimates of how many independent trials a set of correlated backtests amounts to.

    * ``n_eff_corr`` = rho + (1 - rho) * N, with rho the average pairwise correlation (the
      approximation Bailey & Lopez de Prado use in the DSR paper; 1 when all trials are identical,
      N when they are uncorrelated);
    * ``n_eff_eig`` = (sum lambda)^2 / sum lambda^2 over the eigenvalues of the correlation matrix
      (participation ratio).
    """
    n = returns.shape[1]
    if n <= 1:
        return {"n": n, "avg_corr": None, "n_eff_corr": float(n), "n_eff_eig": float(n)}
    corr = returns.corr(min_periods=min_overlap).to_numpy()
    mask = ~np.eye(n, dtype=bool)
    off = corr[mask]
    off = off[np.isfinite(off)]
    rho = float(off.mean()) if len(off) else 0.0
    rho_c = min(max(rho, 0.0), 1.0)
    n_eff_corr = rho_c + (1.0 - rho_c) * n
    filled = np.where(np.isfinite(corr), corr, 0.0)
    np.fill_diagonal(filled, 1.0)
    eig = np.clip(np.linalg.eigvalsh((filled + filled.T) / 2.0), 0.0, None)
    n_eff_eig = float(eig.sum() ** 2 / (eig ** 2).sum()) if (eig ** 2).sum() > 0 else float(n)
    return {"n": n, "avg_corr": rho, "n_eff_corr": float(min(max(n_eff_corr, 1.0), n)),
            "n_eff_eig": float(min(max(n_eff_eig, 1.0), n))}
