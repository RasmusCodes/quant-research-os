"""Freeze a finished experiment and (re)compute its validation report.

``finalize`` turns a raw result into a completed, immutable record: standard metrics per window,
artifacts, the result hash and the holdout check. ``validate_experiment`` builds the trial-family
context, the statistics, the hygiene findings and the scorecard, and stores a new validation report
(earlier reports are kept, so you can see a deflated Sharpe ratio fall as a family grows).
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from qros.backtest.interface import as_result
from qros.hygiene.rules import (
    apply_findings, experiment_findings, family_findings, is_preregistered, lint_holdouts, lint_hypothesis,
)
from qros.registry import db
from qros.util.hashing import canonical_json, fingerprint_series, normalize
from qros.validation.metrics import infer_periods_per_year, moments, performance_summary, sharpe
from qros.validation.pbo import probability_of_backtest_overfitting
from qros.validation.report import FamilyContext, ValidationConfig, build_report, compute_statistics
from qros.validation.sharpe import effective_number_of_trials, probabilistic_sharpe_ratio

if TYPE_CHECKING:  # pragma: no cover
    from qros.registry.core import Registry
    from qros.registry.models import Experiment

MAX_POSITION_CELLS = 3_000_000


def _window_mask(index: pd.Index, start: str | None, end: str | None) -> np.ndarray:
    if not start:
        return np.zeros(len(index), dtype=bool)
    return np.asarray((index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end)))


def _jsonable_meta(meta: dict[str, Any], non_holdout: pd.Index, ppy: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if key == "variants" and isinstance(value, dict):
            out["variants_sharpe"] = {
                str(k): sharpe(pd.Series(v).reindex(non_holdout).dropna(), ppy) for k, v in value.items()
                if isinstance(v, pd.Series)}
            continue
        if key == "walk_forward" and isinstance(value, dict):
            out[key] = {k: v for k, v in value.items() if not isinstance(v, (pd.Series, pd.DataFrame))}
            continue
        if isinstance(value, (pd.Series, pd.DataFrame)):
            continue
        try:
            out[key] = normalize(value)
        except TypeError:
            out[key] = repr(value)
    return out


def finalize_metrics_only(reg: "Registry", experiment_id: str, metrics: dict[str, float], *,
                          finished_at: str | None = None, digest: str | None = None) -> "Experiment":
    """Complete an experiment that produces scalar results only (a forecast score, one simulated P&L).

    Trial accounting, hypothesis tracking, provenance and reproduction still apply; return-based
    statistics do not, so the verdict is ``insufficient``.
    """
    exp = reg.get_experiment(experiment_id)
    if exp.status != "running":
        raise RuntimeError(f"{experiment_id} is {exp.status}; only running experiments can be completed")
    reg.log_metrics(experiment_id, metrics, "custom")
    result_hash = digest or canonical_json(normalize(metrics))
    if not digest:
        from qros.util.hashing import sha256_text
        result_hash = sha256_text(result_hash)
    summary = {"metrics_only": True, "headline": {"custom": {k: v for k, v in list(metrics.items())[:12]}}}
    with db.transaction(reg.conn):
        now = finished_at or reg.now()
        reg.conn.execute(
            "UPDATE experiments SET status = 'completed', finished_at = ?, result_hash = ?, summary_json = ?"
            " WHERE id = ? AND status = 'running'", (now, result_hash, canonical_json(summary), experiment_id))
        reg._event("experiment.completed", "experiment", experiment_id, {"result_hash": result_hash,
                                                                        "metrics_only": True}, at=now)
    exp = reg.get_experiment(experiment_id)
    findings = experiment_findings(reg, exp, {"periods_per_year": 0}, {})
    findings = [f for f in findings if f.code not in ("NO_OOS",)]
    apply_findings(reg, exp.id, findings)
    preregistered = is_preregistered(reg, exp)
    critical =[f.code for f in findings if f.severity == "critical"]
    report = {
        "version": 1, "metrics_only": True, "n_trials": reg.family_trial_count(exp.family)["total"],
        "verdict": "invalid" if any(c in ("PIT_VIOLATION", "NONREPRODUCIBLE", "DATA_INTEGRITY") for c in critical)
        else "insufficient",
        "verdict_reason": "Scalar results only: no return series to test out of sample.",
        "scorecard": [
            {"key": "preregistered", "label": "Hypothesis pre-registered",
             "status": "na" if preregistered is None else ("pass" if preregistered else "fail"),
             "value": preregistered, "threshold": "registered before run", "note": "", "core": True},
            {"key": "hygiene", "label": "No critical research-hygiene warnings", "status": "fail" if critical else "pass",
             "value": len(critical), "threshold": "0", "note": ", ".join(critical) or "None.", "core": True},
        ],
        "findings": [{"code": f.code, "severity": f.severity, "message": f.message} for f in findings],
        "significance": {"n_trials": reg.family_trial_count(exp.family)["total"]},
    }
    reg.store_validation(experiment_id, report)
    return reg.get_experiment(experiment_id)


def finalize(reg: "Registry", experiment_id: str, result: Any, *, extra_metrics: dict[str, float] | None = None,
             validate: bool = True, finished_at: str | None = None,
             config: ValidationConfig | None = None) -> "Experiment":
    if result is None:
        return finalize_metrics_only(reg, experiment_id, extra_metrics or {}, finished_at=finished_at)
    exp = reg.get_experiment(experiment_id)
    if exp.status != "running":
        raise RuntimeError(f"{experiment_id} is {exp.status}; only running experiments can be completed")
    res = as_result(result).validate()
    if not isinstance(res.returns.index, pd.DatetimeIndex):
        raise ValueError("returns need a DatetimeIndex so windows, holdouts and regimes can be applied")
    ppy = float(res.periods_per_year or exp.periods_per_year or infer_periods_per_year(res.returns.index))
    frame = res.frame()
    idx = frame.index
    holdout = reg.get_holdout(exp.holdout_id) if exp.holdout_id else None
    in_ho = _window_mask(idx, holdout.start, holdout.end) if holdout else np.zeros(len(idx), dtype=bool)
    implicit = bool(in_ho.any() and not exp.holdout_opened)
    if implicit:
        reg.touch_holdout(holdout.id, experiment_id, "implicit: the result reaches into the sealed holdout")
    non_ho = ~in_ho
    scopes = {"full": non_ho}
    if exp.train_start:
        scopes["is"] = non_ho & _window_mask(idx, exp.train_start, exp.train_end)
    if exp.test_start:
        scopes["oos"] = non_ho & _window_mask(idx, exp.test_start, exp.test_end)
    if in_ho.any():
        scopes["holdout"] = in_ho

    def col(name: str) -> pd.Series | None:
        return frame[name] if name in frame.columns else None

    summaries = {}
    for scope, mask in scopes.items():
        if mask.sum() < 2:
            continue
        sel = lambda s: None if s is None else s[mask]  # noqa: E731
        summaries[scope] = performance_summary(
            frame["net"][mask], ppy, turnover=sel(col("turnover")), costs=sel(col("cost")), gross=sel(col("gross")),
            gross_exposure=sel(col("gross_exposure")), net_exposure=sel(col("net_exposure")),
            benchmark=sel(col("benchmark")))
    for scope, summ in summaries.items():
        reg.log_metrics(experiment_id, {k: v for k, v in summ.items()
                                        if isinstance(v, (int, float)) and not isinstance(v, bool)}, scope)
    if extra_metrics:
        reg.log_metrics(experiment_id, extra_metrics, "custom")

    reg.log_artifact(experiment_id, "result", frame, kind="returns")
    for name, table in (("positions", res.positions), ("signals", res.signals)):
        if table is not None and table.size <= MAX_POSITION_CELLS:
            reg.log_artifact(experiment_id, name, table.astype(float), kind=name)
    meta = _jsonable_meta(res.meta, idx[non_ho], ppy)
    reg.log_artifact(experiment_id, "meta", meta, kind="report")

    result_hash = fingerprint_series(frame["net"])
    headline = {s: {k: summaries[s].get(k) for k in ("sharpe", "ann_return", "ann_vol", "max_drawdown", "n_obs",
                                                     "start", "end", "cagr")} for s in summaries}
    summary = {"periods_per_year": ppy, "headline": headline, "implicit_holdout": implicit,
               "start": idx[0].strftime("%Y-%m-%d"), "end": idx[-1].strftime("%Y-%m-%d"),
               "columns": list(frame.columns), "warnings": res.meta.get("warnings", [])}
    with db.transaction(reg.conn):
        now = finished_at or reg.now()
        cur = reg.conn.execute(
            "UPDATE experiments SET status = 'completed', finished_at = ?, result_hash = ?, periods_per_year = ?,"
            " summary_json = ? WHERE id = ? AND status = 'running'",
            (now, result_hash, ppy, canonical_json(summary), experiment_id))
        if cur.rowcount != 1:
            raise RuntimeError(f"{experiment_id} changed state while completing")
        reg._event("experiment.completed", "experiment", experiment_id, {
            "result_hash": result_hash, "headline": headline, "implicit_holdout": implicit}, at=now)
    reg._returns_cache[(experiment_id, "net")] = frame["net"].rename(experiment_id)
    if validate:
        validate_experiment(reg, experiment_id, meta=meta, config=config)
    return reg.get_experiment(experiment_id)


# ----------------------------------------------------------------------------- family context


def _member_windows(m: "Experiment", r: pd.Series, reg: "Registry") -> tuple[pd.Series, pd.Series, pd.Series]:
    ho = reg.get_holdout(m.holdout_id) if m.holdout_id else None
    if ho:
        r = r[~_window_mask(r.index, ho.start, ho.end)]
    if m.train_start:
        r_is = r[_window_mask(r.index, m.train_start, m.train_end)]
    elif m.test_start:
        r_is = r[r.index < pd.Timestamp(m.test_start)]
    else:
        r_is = r
    r_oos = r[_window_mask(r.index, m.test_start, m.test_end)] if m.test_start else r.iloc[0:0]
    return r, r_is, r_oos


def build_family_context(reg: "Registry", exp: "Experiment", config: ValidationConfig | None = None) -> FamilyContext:
    cfg = config or ValidationConfig()
    members = [m for m in reg.family_members(exp.family) if not (m.summary or {}).get("metrics_only")]
    if exp.id not in [m.id for m in members]:
        members.append(exp)
    groups: dict[str, "Experiment"] = {}
    for m in members:
        groups.setdefault(m.spec_hash or m.id, m)
    unique = list(groups.values())
    target_key = exp.spec_hash or exp.id
    target = [i for i, m in enumerate(unique) if (m.spec_hash or m.id) == target_key][0]
    ids, params, sr, sr_pp, n_is, oos_p, series = [], [], [], [], [], [], {}
    ppy = exp.periods_per_year or 252.0
    for m in unique:
        r = reg.returns(m.id)
        r_all, r_is, r_oos = _member_windows(m, r, reg)
        ids.append(m.id)
        params.append(m.params)
        if len(r_is) > 2 and r_is.std(ddof=1) > 0:
            s_pp = float(r_is.mean() / r_is.std(ddof=1))
        else:
            s_pp = float("nan")
        sr_pp.append(s_pp)
        sr.append(s_pp * math.sqrt(m.periods_per_year or ppy) if s_pp == s_pp else float("nan"))
        n_is.append(int(len(r_is)))
        if len(r_oos) > 2 and r_oos.std(ddof=1) > 0:
            mean, sd, sk, ku = moments(r_oos.to_numpy())
            oos_p.append(1.0 - probabilistic_sharpe_ratio(mean / sd, len(r_oos), sk, ku, 0.0))
        else:
            oos_p.append(float("nan"))
        series[m.id] = r_all
    matrix = None
    if len(series) >= 2:
        joined = pd.concat(series, axis=1).dropna(how="any")
        if len(joined) >= 100:
            matrix = joined
    fam = reg.get_family(exp.family)
    pbo = n_eff = None
    if matrix is not None:
        n_eff = effective_number_of_trials(matrix)
        if matrix.shape[1] >= cfg.min_family_for_pbo:
            pbo = probability_of_backtest_overfitting(matrix, cfg.pbo_splits, ppy)
    return FamilyContext(name=exp.family, member_ids=ids, params=params, is_sharpe=sr, is_sharpe_pp=sr_pp,
                         is_n_obs=n_is, oos_pvalues=oos_p, returns=matrix,
                         external_trials=int(fam.get("external_trials", 0)), target=target, pbo=pbo, n_eff=n_eff)


# ----------------------------------------------------------------------------- validation


def validate_experiment(reg: "Registry", experiment_id: str, *, meta: dict[str, Any] | None = None,
                        config: ValidationConfig | None = None, family: FamilyContext | None = None) -> dict[str, Any]:
    cfg = config or ValidationConfig()
    exp = reg.get_experiment(experiment_id)
    if exp.status not in ("completed", "invalidated"):
        raise RuntimeError(f"{experiment_id} is {exp.status}; validate completed experiments only")
    returns = reg.returns(experiment_id)
    frame = reg.result_frame(experiment_id)
    if meta is None:
        try:
            meta = reg.load_artifact(experiment_id, "meta")
        except Exception:
            meta = {}
    ctx = family or build_family_context(reg, exp, cfg)
    if family is not None:
        key = exp.spec_hash or exp.id
        matches = [i for i, mid in enumerate(ctx.member_ids)
                   if (reg.get_experiment(mid).spec_hash or mid) == key]
        ctx = dataclasses.replace(ctx, target=matches[0] if matches else ctx.target)
    ho = reg.get_holdout(exp.holdout_id) if exp.holdout_id else None
    ppy = exp.periods_per_year or infer_periods_per_year(returns.index)
    stats = compute_statistics(
        returns=returns, periods_per_year=ppy,
        is_window=(exp.train_start, exp.train_end) if exp.train_start else None,
        oos_window=(exp.test_start, exp.test_end) if exp.test_start else None,
        holdout_window=(ho.start, ho.end) if ho else None, result_frame=frame, family=ctx,
        variants=(meta or {}).get("variants_sharpe"), walk_forward=(meta or {}).get("walk_forward"), config=cfg)

    exp_f = experiment_findings(reg, exp, stats, meta)
    apply_findings(reg, exp.id, exp_f)
    fam_f = family_findings(reg, exp.family, ctx, ppy)
    apply_findings(reg, f"family:{exp.family}", fam_f)
    hyp_f = lint_hypothesis(reg, exp.hypothesis_id) if exp.hypothesis_id else []
    lint_holdouts(reg)
    relevant = exp_f + fam_f + (hyp_f if (exp.hypothesis_version or 1) > 1 else [])

    preregistered = is_preregistered(reg, exp)
    reps = reg.reproductions(experiment_id)
    look = limit = None
    if ho:
        accesses = reg.holdout_accesses(ho.id)
        mine = [i for i, a in enumerate(accesses, start=1) if a["experiment_id"] == experiment_id]
        look = mine[0] if mine else None
        limit = ho.max_touches
    report = build_report(stats, findings=[f.to_dict() for f in relevant], preregistered=preregistered,
                          reproduction=reps[-1]["status"] if reps else None, holdout_touch=look,
                          holdout_max=limit, config=cfg)
    report["family_members"] = len(ctx.member_ids)
    report["family_selected_rank"] = None
    finite = [s for s in ctx.is_sharpe if s == s]
    if finite and ctx.is_sharpe[ctx.target] == ctx.is_sharpe[ctx.target]:
        report["family_selected_rank"] = int(sum(1 for s in finite if s > ctx.is_sharpe[ctx.target]) + 1)
    reg.store_validation(experiment_id, report)
    return report


def revalidate_family(reg: "Registry", family: str, config: ValidationConfig | None = None) -> dict[str, str]:
    """Re-score every completed member of a family against the family as it stands now."""
    members = [m for m in reg.family_members(family) if not (m.summary or {}).get("metrics_only")]
    if not members:
        return {}
    ctx = build_family_context(reg, members[-1], config)
    out = {}
    for m in members:
        report = validate_experiment(reg, m.id, config=config, family=ctx)
        out[m.id] = report["verdict"]
    return out


def revalidate_all(reg: "Registry", config: ValidationConfig | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for fam in reg.list_families():
        out.update(revalidate_family(reg, fam["name"], config))
    return out
