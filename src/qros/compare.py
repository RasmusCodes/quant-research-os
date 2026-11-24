"""Side-by-side comparison of experiments: metrics, the robustness scorecard and paired tests."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

from qros.validation.bootstrap import paired_sharpe_difference

if TYPE_CHECKING:  # pragma: no cover
    from qros.registry.core import Registry

ROWS = [
    ("verdict", "Verdict"),
    ("is.sharpe", "Sharpe, in-sample"),
    ("oos.sharpe", "Sharpe, out-of-sample"),
    ("holdout.sharpe", "Sharpe, holdout"),
    ("is.ann_return", "Annual return, in-sample"),
    ("oos.max_drawdown", "Max drawdown, out-of-sample"),
    ("sig.dsr", "Deflated Sharpe ratio"),
    ("sig.n_trials", "Trials in family"),
    ("sig.oos_p_holm", "OOS p-value (Holm)"),
    ("pbo", "PBO (family)"),
    ("gate.plateau", "Parameter plateau"),
    ("gate.costs", "Sharpe at 2x costs"),
    ("gate.regimes", "Regime consistency"),
    ("gate.outliers", "Trimmed Sharpe"),
    ("gate.preregistered", "Pre-registered"),
    ("warnings.critical", "Critical warnings"),
    ("data", "Data versions"),
    ("spec_hash", "Spec hash"),
]


def compare_experiments(reg: "Registry", ids: Sequence[str]) -> dict[str, Any]:
    if len(ids) < 2:
        raise ValueError("compare needs at least two experiments")
    exps = [reg.get_experiment(i) for i in ids]
    reports = {e.id: reg.latest_validation(e.id) or {} for e in exps}
    table: dict[str, dict[str, Any]] = {}
    for e in exps:
        rep = reports[e.id]
        gates = {g["key"]: g for g in rep.get("scorecard", [])}
        crit = [w for w in reg.warnings(experiment=e.id) if w["severity"] == "critical"]
        col: dict[str, Any] = {}
        for key, _ in ROWS:
            if key == "verdict":
                col[key] = e.verdict
            elif key.startswith("sig."):
                col[key] = (rep.get("significance") or {}).get(key[4:])
            elif key == "pbo":
                col[key] = (rep.get("pbo") or {}).get("pbo")
            elif key.startswith("gate."):
                g = gates.get(key[5:])
                col[key] = None if not g else f"{g['status']}" + (f" ({g['value']:.2f})" if isinstance(g.get("value"), float) else "")
            elif key == "warnings.critical":
                col[key] = len(crit)
            elif key == "data":
                col[key] = ", ".join(f"{r}={v}" for r, v in e.data.items())
            elif key == "spec_hash":
                col[key] = (e.spec_hash or "")[:12]
            else:
                scope, name = key.split(".", 1)
                col[key] = e.metric(name, scope)
        table[e.id] = col
    frame = pd.DataFrame(table).reindex([k for k, _ in ROWS])
    frame.index = [label for _, label in ROWS]

    returns = {}
    for e in exps:
        if e.status == "completed" and not (e.summary or {}).get("metrics_only"):
            returns[e.id] = reg.returns(e.id)
    corr = pd.DataFrame(returns).corr() if len(returns) >= 2 else None
    pairs = []
    ids_with = list(returns)
    for i in range(len(ids_with)):
        for j in range(i + 1, len(ids_with)):
            a, b = ids_with[i], ids_with[j]
            ea, eb = reg.get_experiment(a), reg.get_experiment(b)
            window = None
            if ea.test_start and eb.test_start:
                start, end = max(ea.test_start, eb.test_start), min(ea.test_end, eb.test_end)
                if start < end:
                    window = (start, end)
            ra, rb = returns[a], returns[b]
            if window:
                ra, rb = ra.loc[window[0]:window[1]], rb.loc[window[0]:window[1]]
            test = paired_sharpe_difference(ra, rb, ea.periods_per_year or 252)
            pairs.append({"a": a, "b": b, "window": window or "common dates", **test})
    return {"table": frame, "correlation": corr, "pairs": pairs}
