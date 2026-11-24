"""Parameter sensitivity and robustness perturbations.

A robust configuration sits on a plateau: its neighbours in parameter space perform about as well. A
fragile one is an isolated spike that is more likely to be noise selected by the search.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from qros.validation.metrics import clean_returns, sharpe


def _numeric(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def varying_keys(params_list: Sequence[dict[str, Any]]) -> list[str]:
    keys = sorted({k for p in params_list for k in p})
    out = []
    for k in keys:
        values = {repr(p.get(k)) for p in params_list}
        if len(values) > 1:
            out.append(k)
    return out


def neighborhood(params_list: Sequence[dict[str, Any]], values: Sequence[float], target: int) -> dict[str, Any]:
    """Compare configuration ``target`` with its grid neighbours.

    Neighbours differ from the target in exactly one numeric parameter, by one step on that
    parameter's grid of tried values, and agree on every other varying parameter.
    """
    keys = varying_keys(params_list)
    vals = np.asarray(values, dtype=float)
    target_params = params_list[target]
    grids = {k: sorted({p[k] for p in params_list if k in p and _numeric(p[k])}) for k in keys}
    neighbours = []
    for i, p in enumerate(params_list):
        if i == target:
            continue
        diff = [k for k in keys if repr(p.get(k)) != repr(target_params.get(k))]
        if len(diff) != 1:
            continue
        k = diff[0]
        grid = grids.get(k) or []
        a, b = target_params.get(k), p.get(k)
        if not (_numeric(a) and _numeric(b)) or a not in grid or b not in grid:
            continue
        if abs(grid.index(a) - grid.index(b)) == 1:
            neighbours.append({"index": i, "param": k, "value": b, "metric": float(vals[i])})
    own = float(vals[target])
    finite = vals[np.isfinite(vals)]
    nb = [n["metric"] for n in neighbours if math.isfinite(n["metric"])]
    nb_mean = float(np.mean(nb)) if nb else float("nan")
    return {
        "keys": keys,
        "metric": own,
        "n_neighbours": len(nb),
        "neighbours": neighbours,
        "neighbour_mean": nb_mean,
        "neighbour_min": float(np.min(nb)) if nb else float("nan"),
        "plateau_ratio": nb_mean / own if nb and own > 0 else float("nan"),
        "rank": int((finite > own).sum() + 1),
        "n_configs": int(len(finite)),
        "share_positive": float(np.mean(finite > 0)) if len(finite) else float("nan"),
        "family_median": float(np.median(finite)) if len(finite) else float("nan"),
    }


def grid_matrix(params_list: Sequence[dict[str, Any]], values: Sequence[float], x: str, y: str,
                fixed: dict[str, Any] | None = None) -> dict[str, Any]:
    """A heatmap of ``values`` over two parameters (other parameters fixed or averaged)."""
    rows = []
    for p, v in zip(params_list, values):
        if fixed and any(repr(p.get(k)) != repr(val) for k, val in fixed.items()):
            continue
        if x in p and y in p:
            rows.append((p[x], p[y], float(v)))
    if not rows:
        return {"x": x, "y": y, "xs": [], "ys": [], "z": []}
    df = pd.DataFrame(rows, columns=["x", "y", "v"]).groupby(["y", "x"])["v"].mean().unstack()
    df = df.sort_index().sort_index(axis=1)
    z = [[None if not np.isfinite(val) else float(val) for val in row] for row in df.to_numpy()]
    return {"x": x, "y": y, "xs": [v.item() if hasattr(v, "item") else v for v in df.columns],
            "ys": [v.item() if hasattr(v, "item") else v for v in df.index], "z": z}


def cost_sensitivity(gross: pd.Series, costs: pd.Series, periods_per_year: float,
                     multipliers: Sequence[float] = (0.0, 1.0, 2.0, 3.0)) -> dict[str, float]:
    g, c = clean_returns(gross), clean_returns(costs)
    joined = pd.concat([g, c], axis=1, join="inner").fillna(0.0)
    out = {}
    for m in multipliers:
        out[f"{m:g}x"] = sharpe(joined.iloc[:, 0] - m * joined.iloc[:, 1], periods_per_year)
    total_cost = float(joined.iloc[:, 1].sum())
    if total_cost > 0:
        mean_g = float(joined.iloc[:, 0].mean())
        mean_c = float(joined.iloc[:, 1].mean())
        out["breakeven_multiplier"] = mean_g / mean_c if mean_c > 0 else float("inf")
    return out


def drop_best(returns: pd.Series, periods_per_year: float, fraction: float = 0.01) -> dict[str, float]:
    """Sharpe ratio after trimming the best AND worst ``fraction`` of periods (outlier dependence).

    Trimming both tails keeps the test fair: removing only the best days lowers the Sharpe ratio of
    any strategy, skilled or not, whereas a symmetric trim only hurts results that rest on a few
    unusually large gains. ``sharpe_drop_best`` (best days only) is reported for information.
    """
    r = clean_returns(returns)
    k = max(1, int(math.ceil(len(r) * fraction)))
    best = r.nlargest(k).index
    trimmed = r.drop(best.union(r.nsmallest(k).index))
    return {"removed": k, "sharpe": sharpe(r, periods_per_year), "sharpe_trimmed": sharpe(trimmed, periods_per_year),
            "sharpe_drop_best": sharpe(r.drop(best), periods_per_year)}


def rolling_sharpe(returns: pd.Series, window: int, periods_per_year: float) -> pd.Series:
    r = clean_returns(returns)
    mean = r.rolling(window).mean()
    sd = r.rolling(window).std(ddof=1)
    return (mean / sd * math.sqrt(periods_per_year)).dropna()
