"""Export a registry to the static dashboard: an index file, one file per experiment, and the web app.

The same app serves both uses: ``qros dashboard`` for a local registry and the public demo on the
website (exported from the synthetic demo registry). Everything is static JSON; nothing is executed
on a server. Equity curves are sampled weekly to keep the files small; every statistic shown is
computed by qros from the full-resolution series.
"""

from __future__ import annotations

import hashlib
import json
import math
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from qros._version import __version__
from qros.hygiene.rules import RULES
from qros.util.hashing import normalize
from qros.validation.sensitivity import grid_matrix, rolling_sharpe, varying_keys

if TYPE_CHECKING:  # pragma: no cover
    from qros.registry.core import Registry

STATIC_FILES = ("index.html", "app.js", "app.css", "charts.js", "toy.js")


def _r(x: Any, nd: int = 4) -> Any:
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        return round(x, nd)
    return x


def _clean(obj: Any, nd: int = 4) -> Any:
    obj = normalize(obj)
    if isinstance(obj, dict):
        return {k: _clean(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v, nd) for v in obj]
    return _r(obj, nd)


def _weekly(series: pd.Series, nd: int = 5) -> dict[str, list]:
    if series is None or len(series) == 0:
        return {"d": [], "v": []}
    w = series.resample("W-FRI").last().dropna()
    return {"d": w.index.strftime("%Y-%m-%d").tolist(), "v": [round(float(v), nd) for v in w.to_numpy()]}


def _metric_block(exp: Any, scope: str) -> dict[str, Any] | None:
    m = exp.metrics.get(scope)
    if not m:
        return None
    keys = ("sharpe", "ann_return", "ann_vol", "max_drawdown", "n_obs", "cagr", "sortino", "hit_rate", "psr0",
            "nw_t_stat", "ann_turnover", "ann_cost", "skew", "kurtosis", "calmar", "beta", "total_return")
    return {k: _r(m.get(k)) for k in keys if m.get(k) is not None}


def _warning_counts(warnings: list[dict[str, Any]]) -> dict[str, int]:
    out = {"critical": 0, "warning": 0, "info": 0}
    for w in warnings:
        out[w["severity"]] = out.get(w["severity"], 0) + 1
    return out


def _gate_map(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    return {g["key"]: [g["status"], _r(g["value"]) if isinstance(g["value"], float) else g["value"]]
            for g in report.get("scorecard", [])}


def export_site(reg: "Registry", out: str | Path, *, title: str | None = None, include_static: bool = True,
                extra: dict[str, Any] | None = None) -> Path:
    out = Path(out)
    (out / "data" / "experiments").mkdir(parents=True, exist_ok=True)
    demo_raw = None
    try:
        from qros.registry import db
        demo_raw = db.get_meta(reg.conn, "demo")
    except Exception:  # pragma: no cover
        demo_raw = None
    demo = json.loads(demo_raw) if demo_raw else None
    truth = (demo or {}).get("experiments", {})

    experiments = reg.list_experiments()
    all_warnings = reg.warnings(active=True)
    by_exp: dict[str, list] = {}
    for w in all_warnings:
        if w["experiment_id"]:
            by_exp.setdefault(w["experiment_id"], []).append(w)
    index_exps = []
    toy = []
    reports = {}
    for exp in experiments:
        report = reg.latest_validation(exp.id) if exp.status in ("completed", "invalidated") else None
        reports[exp.id] = report
        sig = (report or {}).get("significance", {})
        pbo = (report or {}).get("pbo")
        fam_warn = [w for w in all_warnings if w["scope_key"] == f"family:{exp.family}"]
        entry = {
            "id": exp.id, "name": exp.name, "project": exp.project, "family": exp.family,
            "hypothesis": exp.hypothesis_id, "hypothesis_version": exp.hypothesis_version, "model": exp.model,
            "model_version": exp.model_version, "parent": exp.parent_id, "status": exp.status, "source": exp.source,
            "verdict": exp.verdict, "verdict_reason": (report or {}).get("verdict_reason"),
            "started": exp.started_at, "finished": exp.finished_at, "seed": exp.seed, "params": _clean(exp.params),
            "tags": exp.tags, "data": exp.data,
            "periods": {"train": [exp.train_start, exp.train_end] if exp.train_start else None,
                        "test": [exp.test_start, exp.test_end] if exp.test_start else None},
            "holdout": exp.holdout_id, "holdout_opened": exp.holdout_opened,
            "spec_hash": exp.spec_hash, "result_hash": exp.result_hash, "code_hash": exp.code_hash,
            "metrics": {s: _metric_block(exp, s) for s in ("is", "oos", "holdout", "full") if exp.metrics.get(s)},
            "custom": _clean(exp.metrics.get("custom") or {}),
            "sig": _clean({k: sig.get(k) for k in ("dsr", "sr0_annual", "n_trials", "n_eff", "is_psr0", "oos_p",
                                                   "oos_p_holm", "oos_p_bh", "oos_tests", "min_trl_years")}),
            "pbo": _r(pbo["pbo"]) if pbo else None,
            "gates": _gate_map(report),
            "warnings": _warning_counts(by_exp.get(exp.id, []) + fam_warn),
            "truth": truth.get(exp.id),
            "metrics_only": bool((exp.summary or {}).get("metrics_only")),
        }
        index_exps.append(entry)
        if "browser-reproducible" in exp.tags:
            try:
                meta = reg.load_artifact(exp.id, "meta")
                if meta.get("toy"):
                    toy.append({"id": exp.id, "name": exp.name, **meta["toy"], "result_hash": exp.result_hash})
            except Exception:
                pass
        _export_detail(reg, exp, report, by_exp.get(exp.id, []), out)

    hypotheses = []
    for h in reg.list_hypotheses():
        exps = [e for e in experiments if e.hypothesis_id == h.id]
        finished = [e.finished_at for e in exps if e.finished_at]
        hypotheses.append(_clean({
            **h.to_dict(), "experiments": [e.id for e in exps], "first_result_at": min(finished) if finished else None,
            "notes": reg.notes(hypothesis=h.id), "warnings": [w for w in all_warnings if w["hypothesis_id"] == h.id
                                                               and w["scope_key"].startswith("hypothesis:")],
            "timeline": [{"at": ev["at"], "action": ev["action"], "payload": ev["payload"]}
                         for ev in reg.events(entity="hypothesis", entity_id=h.id)],
        }))

    families = []
    for fam in reg.list_families():
        members = [e for e in index_exps if e["family"] == fam["name"] and e["status"] == "completed"]
        if not members:
            continue
        counts = reg.family_trial_count(fam["name"])
        params = [m["params"] for m in members]
        keys = [k for k in varying_keys(params)
                if all(isinstance(p.get(k), (int, float)) and not isinstance(p.get(k), bool) for p in params if k in p)]
        heat = None
        if len(keys) >= 2:
            is_vals = [((m["metrics"].get("is") or {}).get("sharpe")) for m in members]
            oos_vals = [((m["metrics"].get("oos") or {}).get("sharpe")) for m in members]
            pairs = [(p, a, b) for p, a, b in zip(params, is_vals, oos_vals) if a is not None and b is not None
                     and all(k in p for k in keys[:2])]
            if len(pairs) >= 4:
                heat = {"is": grid_matrix([p for p, _, _ in pairs], [a for _, a, _ in pairs], keys[0], keys[1]),
                        "oos": grid_matrix([p for p, _, _ in pairs], [b for _, _, b in pairs], keys[0], keys[1])}
        report = next((reports[m["id"]] for m in members if reports.get(m["id"])), None)
        fam_warn = [w for w in all_warnings if w["scope_key"] == f"family:{fam['name']}"]
        families.append(_clean({
            "name": fam["name"], "description": fam.get("description"), "counts": counts,
            "members": [m["id"] for m in members], "keys": keys, "heat": heat,
            "pbo": (report or {}).get("pbo"), "n_eff": ((report or {}).get("significance") or {}).get("n_eff"),
            "avg_corr": ((report or {}).get("significance") or {}).get("avg_corr"),
            "sr0_annual": ((report or {}).get("significance") or {}).get("sr0_annual"),
            "warnings": fam_warn,
        }))

    datasets = reg.list_datasets()
    versions = [v.to_dict() for v in reg.list_versions()]
    used_by: dict[str, list[str]] = {}
    for e in experiments:
        for vid in e.data.values():
            used_by.setdefault(vid, []).append(e.id)
    for v in versions:
        v["used_by"] = used_by.get(v["id"], [])
    holdouts = []
    for ho in reg.list_holdouts():
        holdouts.append({**ho.to_dict(), "accesses": reg.holdout_accesses(ho.id)})
    chain = reg.verify(deep=False)["audit_chain"]
    events = reg.events()
    index = {
        "title": title or ("qros demo research project" if demo else "qros registry"),
        "generated_by": f"quant-research-os {__version__}",
        "registry": reg.stats(),
        "demo": _clean(demo) if demo else None,
        "experiments": index_exps,
        "hypotheses": hypotheses,
        "families": families,
        "sources": reg.list_sources(),
        "datasets": datasets,
        "versions": _clean(versions),
        "transforms": reg.list_transforms(),
        "lineage": reg.list_lineage_edges(),
        "holdouts": holdouts,
        "warnings": _clean(all_warnings),
        "rules": {k: {"severity": v[0], "title": v[1], "why": v[2]} for k, v in RULES.items()},
        "audit": {"ok": chain["ok"], "events": chain["events"], "head": chain["head"],
                  "recent": [{"seq": e["seq"], "at": e["at"], "action": e["action"], "entity": e["entity"],
                              "id": e["entity_id"], "hash": e["hash"][:16]} for e in events[-60:]]},
        "reproductions": reg.reproductions(),
        "toy": toy,
        **(extra or {}),
    }
    (out / "data" / "index.json").write_text(json.dumps(_clean(index), separators=(",", ":")), encoding="utf-8")
    # The full hash chain, so a browser can re-verify every event (payloads kept as their exact canonical text).
    rows = reg.conn.execute("SELECT seq, at, actor, action, entity, entity_id, payload_json, prev_hash, hash"
                            " FROM events ORDER BY seq").fetchall()
    (out / "data" / "audit.json").write_text(json.dumps([dict(r) for r in rows], separators=(",", ":"),
                                                        ensure_ascii=False), encoding="utf-8")
    if include_static:
        static = resources.files("qros.dashboard").joinpath("static")
        assets = {}
        for name in STATIC_FILES:
            src = static.joinpath(name)
            if src.is_file():
                assets[name] = src.read_bytes()
        # Stamp asset URLs with a content hash so browsers and CDNs pick up new versions.
        stamp = hashlib.sha256(b"".join(v for k, v in sorted(assets.items()) if k != "index.html")).hexdigest()[:10]
        html = assets.pop("index.html").decode("utf-8")
        for name in assets:
            html = html.replace(f'"{name}"', f'"{name}?v={stamp}"')
        (out / "index.html").write_text(html, encoding="utf-8")
        for name, payload in assets.items():
            (out / name).write_bytes(payload)
    return out


def _export_detail(reg: "Registry", exp: Any, report: dict[str, Any] | None, warnings: list[dict[str, Any]],
                   out: Path) -> None:
    detail: dict[str, Any] = {"id": exp.id, "report": None, "series": None}
    if exp.status in ("completed", "invalidated") and not (exp.summary or {}).get("metrics_only"):
        frame = reg.result_frame(exp.id)
        net = frame["net"]
        equity = (1.0 + net).cumprod()
        peak = np.maximum(equity.cummax(), 1.0)
        dd = equity / peak - 1.0
        # Weekly samples on one shared date axis; drawdowns keep each week's worst point.
        eq_w = equity.resample("W-FRI").last().dropna()
        rnd = lambda s, nd=5: [None if not np.isfinite(v) else round(float(v), nd) for v in s.to_numpy()]  # noqa: E731
        series: dict[str, Any] = {"d": eq_w.index.strftime("%Y-%m-%d").tolist(), "equity": rnd(eq_w),
                                  "drawdown": rnd(dd.resample("W-FRI").min().reindex(eq_w.index))}
        if "gross" in frame.columns:
            series["gross_equity"] = rnd((1.0 + frame["gross"]).cumprod().resample("W-FRI").last().reindex(eq_w.index))
        ppy = exp.periods_per_year or 252
        if len(net) > ppy + 10:
            series["rolling_sharpe"] = _weekly(rolling_sharpe(net, int(ppy), ppy), nd=3)
        detail["series"] = series
    detail["report"] = _clean(report) if report else None
    detail["history"] = _clean(reg.validation_history(exp.id))
    detail["notes"] = reg.notes(experiment=exp.id)
    detail["warnings"] = _clean(warnings)
    detail["artifacts"] = [{k: a[k] for k in ("name", "kind", "sha256", "size", "media_type")} for a in reg.artifacts(exp.id)]
    detail["environment"] = exp.environment
    detail["code"] = exp.code
    detail["metrics"] = _clean(exp.metrics)
    detail["reproductions"] = reg.reproductions(exp.id)
    detail["events"] = [{"seq": e["seq"], "at": e["at"], "action": e["action"], "hash": e["hash"][:16],
                         "payload": _clean(e["payload"])} for e in reg.events(entity_id=exp.id)]
    detail["data"] = [{"role": role, **reg.get_version(vid).to_dict()} for role, vid in exp.data.items()]
    for d in detail["data"]:
        d.pop("quality", None)
    (out / "data" / "experiments" / f"{exp.id}.json").write_text(
        json.dumps(_clean(detail), separators=(",", ":")), encoding="utf-8")


def serve(directory: str | Path, port: int = 8765, open_browser: bool = False) -> None:  # pragma: no cover - interactive
    import functools
    import http.server
    import webbrowser
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"qros dashboard at {url} (Ctrl+C to stop)")
        if open_browser:
            webbrowser.open(url)
        httpd.serve_forever()
