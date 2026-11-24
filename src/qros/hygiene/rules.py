"""Research-hygiene rules: warnings for practices that make backtests look better than they are.

Each rule reads the registry (hypothesis history, holdout ledger, data flags, the family of trials)
and the experiment's validation statistics. Findings are stored as warnings on one scope: an
experiment, a trial family, a hypothesis or a holdout. Re-running the rules updates warnings in place
and clears the ones that no longer apply.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from qros.registry.core import Registry
    from qros.registry.models import Experiment

# code: (default severity, title, why it matters)
RULES: dict[str, tuple[str, str, str]] = {
    "NOT_PREREGISTERED": ("warning", "Hypothesis not pre-registered",
                          "Without a hypothesis locked before the run, the question can be chosen after the answer."),
    "PIT_VIOLATION": ("critical", "Look-ahead: data used before it was available",
                      "Rows whose availability time is after the decision time leak future information."),
    "SURVIVORSHIP_BIAS": ("warning", "Survivorship-biased dataset",
                          "A universe of today's survivors omits the losers that were delisted."),
    "UNADJUSTED_PRICES": ("warning", "Prices not adjusted for corporate actions",
                          "Splits and dividends create fake returns in unadjusted prices."),
    "HOLDOUT_IMPLICIT_ACCESS": ("critical", "Undeclared use of the sealed holdout",
                                "Results reach into the holdout window without opening it in the ledger."),
    "HOLDOUT_OVERUSE": ("critical", "Holdout evaluated more often than allowed",
                        "Every look at the holdout turns it a little more into training data."),
    "HYPOTHESIS_AMENDED_AFTER_RESULTS": ("critical", "Hypothesis changed after results were seen",
                                         "Rewriting the hypothesis to fit the data (HARKing) makes it unfalsifiable on that data."),
    "PREDICTION_REVERSED": ("critical", "Predicted direction reversed after results",
                            "Flipping the sign of a prediction after seeing the result is the strongest form of HARKing."),
    "AMENDED_TESTED_ON_SEEN_DATA": ("critical", "Amended hypothesis confirmed on data already seen",
                                    "An amended hypothesis needs fresh data; the windows it was formed on cannot confirm it."),
    "EXCESSIVE_TUNING": ("warning", "Many configurations tried",
                         "The best of many trials is biased upwards; significance must be deflated by the trial count."),
    "SIGNIFICANCE_LOST_AFTER_DEFLATION": ("warning", "Significant only before deflation",
                                          "The Sharpe ratio is significant alone but not after accounting for the trials."),
    "HIGH_PBO": ("warning", "High probability of backtest overfitting",
                 "The in-sample winner tends to rank below the median out of sample."),
    "TEST_WINDOW_REUSE": ("info", "Test window evaluated by many trials",
                          "Each trial evaluated on the same test window is another test of it; p-values are adjusted."),
    "SHORT_OOS": ("warning", "Short out-of-sample window",
                  "Less than a year of out-of-sample data cannot separate skill from luck."),
    "NO_OOS": ("warning", "No out-of-sample evaluation", "Everything was evaluated on the data used to build it."),
    "IS_OOS_DEGRADATION": ("warning", "Out-of-sample performance collapses",
                           "An out-of-sample Sharpe under half the in-sample one suggests overfitting."),
    "FRAGILE_PEAK": ("warning", "Isolated parameter peak",
                     "Neighbouring parameter values perform much worse: the optimum may be noise."),
    "COST_SENSITIVE": ("warning", "Edge disappears with higher costs", "Doubling costs removes the profit."),
    "OUTLIER_DEPENDENCE": ("warning", "Driven by a few periods",
                           "Trimming the 1% best and worst periods removes the profit."),
    "IMPLAUSIBLE_SHARPE": ("warning", "Implausibly high Sharpe ratio",
                           "Sharpe ratios above 3 in daily backtests usually mean leakage or an error."),
    "SINGLE_REGIME": ("info", "Profitable in one regime only", "Performance depends on a single market state."),
    "CODE_NOT_VERSIONED": ("info", "Code version not pinned", "Uncommitted or unhashed code cannot be re-run exactly."),
    "MISSING_SEED": ("info", "No random seed recorded", "Stochastic steps without a seed are not reproducible."),
    "NONREPRODUCIBLE": ("critical", "Re-run did not reproduce the result",
                        "The recorded result could not be regenerated from the recorded inputs."),
    "DATA_INTEGRITY": ("critical", "Dataset no longer matches its hash", "The data behind the result changed."),
    "LICENSED_DATA": ("info", "Licensed data", "Vendor terms may forbid publishing row-level data or derived series."),
}


@dataclass
class Finding:
    code: str
    severity: str
    message: str
    scope_key: str
    details: dict[str, Any] = field(default_factory=dict)
    experiment_id: str | None = None
    hypothesis_id: str | None = None
    family: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["title"] = RULES.get(self.code, ("", self.code, ""))[1]
        return d


def _f(code: str, message: str, scope: str, severity: str | None = None, **kw: Any) -> Finding:
    return Finding(code=code, severity=severity or RULES[code][0], message=message, scope_key=scope, **kw)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _overlap(a: tuple[str, str], b: tuple[str, str]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def run_started_at(reg: "Registry", exp: "Experiment") -> str | None:
    """When the run demonstrably started, for the pre-registration check.

    Managed and tracked runs start under the registry's own clock. For ingested or imported runs the
    registry only saw the result, so the run's declared start time is used, and None (unknown) when the
    source declared none: a hypothesis registered at import time cannot pre-date results already known.
    """
    if exp.source != "ingested":
        return exp.started_at
    try:
        manifest = reg.load_artifact(exp.id, "manifest")
    except Exception:
        return None
    declared = (manifest or {}).get("started_at") if isinstance(manifest, dict) else None
    if not declared:
        return None
    from qros.util.timeutil import parse_iso, to_iso
    try:
        return to_iso(parse_iso(str(declared)))
    except ValueError:
        return None


def is_preregistered(reg: "Registry", exp: "Experiment") -> bool | None:
    """None without a hypothesis; otherwise whether it was registered before the run started."""
    if not exp.hypothesis_id:
        return None
    hyp = reg.get_hypothesis(exp.hypothesis_id)
    start = run_started_at(reg, exp)
    return bool(hyp.registered_at and start and hyp.registered_at <= start)


def evaluated_windows(exp: "Experiment", holdout: tuple[str, str] | None) -> list[tuple[str, str]]:
    out = []
    if exp.train_start:
        out.append((exp.train_start, exp.train_end))
    if exp.test_start:
        out.append((exp.test_start, exp.test_end))
    if holdout and exp.holdout_opened:
        out.append(holdout)
    return out


# ----------------------------------------------------------------------------- experiment scope


def experiment_findings(reg: "Registry", exp: "Experiment", stats: dict[str, Any],
                        meta: dict[str, Any] | None) -> list[Finding]:
    meta = meta or {}
    scope = exp.id
    kw = {"experiment_id": exp.id, "family": exp.family}
    out: list[Finding] = []

    # pre-registration
    if not exp.hypothesis_id:
        out.append(_f("NOT_PREREGISTERED", "No hypothesis is linked to this experiment.", scope, **kw))
    elif not is_preregistered(reg, exp):
        hyp = reg.get_hypothesis(exp.hypothesis_id)
        start = run_started_at(reg, exp)
        if start is None:
            msg = (f"{hyp.id} cannot be shown to pre-date this run: the run was imported without a declared start "
                   "time, so its results may have been known when the hypothesis was registered.")
        else:
            msg = (f"{hyp.id} was registered {hyp.registered_at or 'never'}, after this experiment started "
                   f"({start}).")
        out.append(_f("NOT_PREREGISTERED", msg, scope, details={"registered_at": hyp.registered_at,
                                                               "run_started_at": start}, **kw))

    # point-in-time audit
    audits = meta.get("pit_audit") or []
    violations = sum(int(a.get("violations", 0)) for a in audits)
    if violations:
        worst = max(audits, key=lambda a: a.get("violations", 0))
        out.append(_f("PIT_VIOLATION", f"{violations:,} data rows were used before they became available "
                      f"(worst: {worst.get('dataset')}, up to {worst.get('max_lead_days', 0):.0f} days early, joined on "
                      f"{', '.join(worst.get('keys_used', []))}).", scope, details={"audits": audits}, **kw))

    # dataset flags and integrity (stored bytes must still match their address)
    licensed = []
    for role, vid in exp.data.items():
        version = reg.get_version(vid)
        ds = reg.get_dataset(version.dataset_id)
        if version.object_sha256 and not reg.objects.verify(version.object_sha256):
            out.append(_f("DATA_INTEGRITY", f"{vid} ({ds['name']}, role {role}) no longer matches its stored hash.",
                          scope, details={"version": vid}, **kw))
        if ds.get("survivorship_free") is False:
            sev = "critical" if role in ("universe",) else "warning"
            out.append(_f("SURVIVORSHIP_BIAS", f"{ds['name']} ({role}) is flagged as not survivorship-free.", scope,
                          severity=sev, details={"dataset": ds["id"], "role": role}, **kw))
        if ds.get("adjusted") is False and role in ("prices", "returns"):
            out.append(_f("UNADJUSTED_PRICES", f"{ds['name']} is used as {role} but is not adjusted for corporate "
                          "actions.", scope, details={"dataset": ds["id"], "role": role}, **kw))
        if ds.get("licensed"):
            licensed.append(ds["name"])
    split_check = (meta.get("data_checks") or {}).get("split_check")
    if split_check and split_check.get("looks_unadjusted"):
        out.append(_f("UNADJUSTED_PRICES", f"{split_check['matching_recorded_splits']} one-day moves in the price "
                      "series match recorded splits: the returns were computed from unadjusted prices.", scope,
                      severity="critical", details=split_check, **kw))
    if licensed:
        out.append(_f("LICENSED_DATA", "Uses licensed data: " + ", ".join(sorted(set(licensed))) + ".", scope, **kw))

    # holdout ledger
    summary = exp.summary or {}
    if summary.get("implicit_holdout"):
        out.append(_f("HOLDOUT_IMPLICIT_ACCESS", "The result reaches into the sealed holdout "
                      f"{exp.holdout_id} without declaring it.", scope, **kw))
    if exp.holdout_id:
        ho = reg.get_holdout(exp.holdout_id)
        accesses = reg.holdout_accesses(ho.id)
        mine = [i for i, a in enumerate(accesses, start=1) if a["experiment_id"] == exp.id]
        if mine and max(mine) > ho.max_touches:
            out.append(_f("HOLDOUT_OVERUSE", f"This experiment is look #{max(mine)} at holdout {ho.name}, which "
                          f"allows {ho.max_touches}.", scope, details={"touch": max(mine), "limit": ho.max_touches}, **kw))

    # hypothesis amended, then "confirmed" on windows already seen
    if exp.hypothesis_id and (exp.hypothesis_version or 1) > 1:
        hyp = reg.get_hypothesis(exp.hypothesis_id)
        version = next(v for v in hyp.versions if v.version == exp.hypothesis_version)
        ho_window = None
        if exp.holdout_id:
            h = reg.get_holdout(exp.holdout_id)
            ho_window = (h.start, h.end)
        seen: list[tuple[str, str]] = []
        earlier = [e for e in reg.list_experiments(hypothesis=exp.hypothesis_id, status="completed")
                   if e.id != exp.id and (e.finished_at or "") <= version.created_at]
        for e in earlier:
            seen.extend(evaluated_windows(e, ho_window))
        confirm = [ho_window] if (ho_window and exp.holdout_opened) else (
            [(exp.test_start, exp.test_end)] if exp.test_start else evaluated_windows(exp, ho_window))
        reused = [w for w in confirm if any(_overlap(w, s) for s in seen)]
        if reused:
            out.append(_f("AMENDED_TESTED_ON_SEEN_DATA", f"Tests version {exp.hypothesis_version} of "
                          f"{exp.hypothesis_id}, written after results, on windows already examined "
                          f"({', '.join(f'{a}..{b}' for a, b in reused)}).", scope,
                          details={"reused": reused, "earlier_experiments": [e.id for e in earlier]},
                          hypothesis_id=exp.hypothesis_id, experiment_id=exp.id, family=exp.family))

    # statistics-based rules
    ppy = stats.get("periods_per_year", 252)
    is_d, oos_d, ho_d = stats.get("is", {}), stats.get("oos", {}), stats.get("holdout", {})
    sig, sens, reg_stats = stats.get("significance", {}), stats.get("sensitivity", {}), stats.get("regimes", {})
    if not oos_d.get("n_obs") and not ho_d.get("n_obs"):
        out.append(_f("NO_OOS", "No test window or opened holdout: every number is in-sample.", scope, **kw))
    elif oos_d.get("n_obs") and oos_d["n_obs"] < ppy:
        out.append(_f("SHORT_OOS", f"The test window has {oos_d['n_obs']} observations (under a year).", scope, **kw))
    if _finite(is_d.get("sharpe")) and _finite(oos_d.get("sharpe")) and is_d["sharpe"] > 0.5 \
            and oos_d["sharpe"] < 0.5 * is_d["sharpe"]:
        out.append(_f("IS_OOS_DEGRADATION", f"Sharpe {is_d['sharpe']:.2f} in-sample but {oos_d['sharpe']:.2f} out of "
                      "sample.", scope, **kw))
    if _finite(sig.get("is_psr0")) and _finite(sig.get("dsr")) and sig["is_psr0"] >= 0.95 and sig["dsr"] < 0.95:
        out.append(_f("SIGNIFICANCE_LOST_AFTER_DEFLATION", f"PSR {sig['is_psr0']:.3f} alone, deflated Sharpe "
                      f"{sig['dsr']:.3f} after {sig.get('n_trials')} trials (effective {sig.get('n_eff', 0):.1f}).",
                      scope, **kw))
    nb = sens.get("neighborhood")
    if nb and nb.get("n_neighbours", 0) >= 2 and _finite(nb.get("plateau_ratio")) and nb["plateau_ratio"] < 0.5:
        out.append(_f("FRAGILE_PEAK", f"Neighbouring configurations average {nb['neighbour_mean']:.2f} against "
                      f"{nb['metric']:.2f} here.", scope, details={"plateau_ratio": nb["plateau_ratio"]}, **kw))
    costs = sens.get("costs") or {}
    if _finite(costs.get("1x")) and costs["1x"] > 0 and _finite(costs.get("2x")) and costs["2x"] <= 0:
        out.append(_f("COST_SENSITIVE", f"Sharpe {costs['1x']:.2f} at modelled costs, {costs['2x']:.2f} at double.",
                      scope, **kw))
    db = sens.get("drop_best") or {}
    if _finite(db.get("sharpe")) and db["sharpe"] > 0 and _finite(db.get("sharpe_trimmed")) and db["sharpe_trimmed"] <= 0:
        out.append(_f("OUTLIER_DEPENDENCE", f"Sharpe {db['sharpe']:.2f} falls to {db['sharpe_trimmed']:.2f} when the "
                      f"{db['removed']} best and {db['removed']} worst periods are trimmed.", scope, **kw))
    if ppy >= 250 and _finite(is_d.get("sharpe")) and is_d["sharpe"] > 3.0:
        out.append(_f("IMPLAUSIBLE_SHARPE", f"In-sample Sharpe {is_d['sharpe']:.2f} on daily data.", scope, **kw))
    vol = reg_stats.get("volatility")
    if vol and vol.get("n_regimes", 0) >= 2 and _finite(vol.get("share_positive")) and vol["share_positive"] < 0.5:
        out.append(_f("SINGLE_REGIME", "Positive Sharpe in fewer than half of the volatility regimes.", scope, **kw))

    # code and seed
    code = exp.code or {}
    git = code.get("git") or {}
    if not code or (not code.get("file_sha256") and not code.get("commit") and not git.get("commit")
                    and not code.get("source_sha256") and not code.get("code_hash")):
        out.append(_f("CODE_NOT_VERSIONED", "No code hash or commit was recorded.", scope, **kw))
    elif git.get("dirty") or code.get("dirty"):
        out.append(_f("CODE_NOT_VERSIONED", "The repository had uncommitted changes when this ran.", scope, **kw))
    if exp.source == "managed" and exp.seed is None:
        out.append(_f("MISSING_SEED", "Managed run without a random seed.", scope, **kw))

    reps = reg.reproductions(exp.id)
    if reps and reps[-1]["status"] in ("mismatch", "error"):
        out.append(_f("NONREPRODUCIBLE", f"Latest reproduction ({reps[-1]['run_at']}): {reps[-1]['status']}.", scope,
                      details={"max_abs_diff": reps[-1]["max_abs_diff"]}, **kw))
    return out


# ----------------------------------------------------------------------------- family, hypothesis, holdout


def family_findings(reg: "Registry", family: str, ctx: Any, periods_per_year: float = 252.0) -> list[Finding]:
    scope = f"family:{family}"
    out: list[Finding] = []
    if ctx is None:
        return out
    n = ctx.n_trials
    if n >= 20:
        severity, extra = "warning", ""
        best = [s for s in ctx.is_sharpe if _finite(s)]
        per_period = [s for s in ctx.is_sharpe_pp if _finite(s)]
        if best and len(per_period) >= 2:
            from qros.validation.sharpe import expected_max_sharpe
            n_eff = float((ctx.n_eff or {}).get("n_eff_corr", ctx.n_registered)) + float(ctx.external_trials)
            var = float(np.var(per_period, ddof=1))
            sr0 = expected_max_sharpe(max(1.0, n_eff), var) * math.sqrt(periods_per_year)
            if max(best) <= sr0:
                severity = "critical"
                extra = (f" The best in-sample Sharpe ({max(best):.2f}) is no higher than the expected best of "
                         f"{n_eff:.0f} effectively independent noise trials ({sr0:.2f}).")
        out.append(_f("EXCESSIVE_TUNING", f"{n} configurations tried in family '{family}'.{extra}", scope,
                      severity=severity, family=family, details={"n_trials": n}))
    pbo = ctx.pbo
    if pbo and pbo.get("pbo", 0) > 0.25:
        sev = "critical" if pbo["pbo"] > 0.5 else "warning"
        out.append(_f("HIGH_PBO", f"PBO = {pbo['pbo']:.2f} across {pbo['n_configs']} configurations: the in-sample "
                      "winner lands below the out-of-sample median in "
                      f"{pbo['pbo']:.0%} of the {pbo['n_combinations']:,} splits.", scope, severity=sev, family=family,
                      details={"pbo": pbo["pbo"]}))
    tested = [p for p in ctx.oos_pvalues if _finite(p)]
    if len(tested) > 1:
        sev = "warning" if len(tested) > 10 else "info"
        out.append(_f("TEST_WINDOW_REUSE", f"{len(tested)} trials in '{family}' were evaluated on their test window; "
                      "out-of-sample p-values are Holm-adjusted for that.", scope, severity=sev, family=family,
                      details={"n_tested": len(tested)}))
    return out


def hypothesis_findings(reg: "Registry", hid: str) -> list[Finding]:
    hyp = reg.get_hypothesis(hid)
    scope = f"hypothesis:{hid}"
    out: list[Finding] = []
    done = [e for e in reg.list_experiments(hypothesis=hid, status="completed") if e.finished_at]
    if not done or len(hyp.versions) < 2:
        return out
    first_result = min(e.finished_at for e in done)
    late = [v for v in hyp.versions[1:] if v.created_at > first_result]
    if late:
        out.append(_f("HYPOTHESIS_AMENDED_AFTER_RESULTS", f"{hid} was amended {len(late)} time(s) after its first "
                      f"result ({first_result}): " + "; ".join(f"v{v.version} on {v.created_at} ({v.reason})"
                                                               for v in late), scope, hypothesis_id=hid,
                      details={"first_result_at": first_result, "versions": [v.version for v in late]}))
        directions = [(v.prediction or {}).get("direction") for v in hyp.versions]
        directions = [d for d in directions if d]
        if len(set(directions)) > 1:
            out.append(_f("PREDICTION_REVERSED", f"The predicted direction of {hid} changed: "
                          f"{' -> '.join(directions)}.", scope, hypothesis_id=hid))
    return out


def holdout_findings(reg: "Registry") -> list[tuple[str, list[Finding]]]:
    out = []
    for ho in reg.list_holdouts():
        scope = f"holdout:{ho.id}"
        n = len(reg.holdout_accesses(ho.id))
        items = []
        if n > ho.max_touches:
            items.append(_f("HOLDOUT_OVERUSE", f"Holdout {ho.name} ({ho.start}..{ho.end}) was evaluated {n} times; "
                            f"the limit is {ho.max_touches}.", scope, details={"touches": n, "limit": ho.max_touches}))
        out.append((scope, items))
    return out


# ----------------------------------------------------------------------------- applying


def apply_findings(reg: "Registry", scope_key: str, findings: Iterable[Finding]) -> list[str]:
    ids, codes = [], []
    for f in findings:
        if f.scope_key != scope_key:
            continue
        ids.append(reg.upsert_warning(scope_key, f.code, f.severity, f.message, details=f.details,
                                      experiment_id=f.experiment_id, hypothesis_id=f.hypothesis_id, family=f.family))
        codes.append(f.code)
    reg.deactivate_warnings(scope_key, codes)
    return ids


def lint_hypothesis(reg: "Registry", hid: str) -> list[Finding]:
    findings = hypothesis_findings(reg, hid)
    apply_findings(reg, f"hypothesis:{hid}", findings)
    return findings


def lint_holdouts(reg: "Registry") -> list[Finding]:
    found = []
    for scope, items in holdout_findings(reg):
        apply_findings(reg, scope, items)
        found.extend(items)
    return found
