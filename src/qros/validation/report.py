"""Assemble every validation statistic for one experiment and grade it with a transparent scorecard.

The verdict is a rule, not a model, so anyone can check it:

* ``invalid``      - an integrity problem (look-ahead, undeclared or excessive holdout use, a hypothesis
                     confirmed on data already seen, a failed reproduction, corrupted data);
* ``robust``       - every applicable core check passes, including out-of-sample significance after
                     multiple-testing correction, the deflated Sharpe ratio and pre-registration;
* ``fragile``      - looks attractive in-sample but fails at least one core check;
* ``no-edge``      - not attractive even in-sample, and not robust;
* ``insufficient`` - no out-of-sample or holdout window long enough to judge.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from qros.validation.bootstrap import bootstrap_sharpe
from qros.validation.metrics import clean_returns, moments, performance_summary
from qros.validation.multiple_testing import adjust_pvalues, haircut_sharpe, sharpe_pvalue
from qros.validation.regimes import calendar_regimes, regime_performance, trend_regimes, volatility_regimes
from qros.validation.sensitivity import cost_sensitivity, drop_best, neighborhood
from qros.validation.sharpe import (
    deflated_sharpe_ratio, effective_number_of_trials, min_track_record_length, probabilistic_sharpe_ratio,
)

REPORT_VERSION = 1
VERDICTS = ("robust", "fragile", "no-edge", "invalid", "insufficient")


@dataclass
class ValidationConfig:
    dsr_threshold: float = 0.95
    oos_alpha: float = 0.05
    pbo_threshold: float = 0.25
    degradation_floor: float = 0.5
    plateau_floor: float = 0.7
    cost_multiplier: float = 2.0
    regime_share: float = 2.0 / 3.0
    outlier_fraction: float = 0.01
    attractive_sharpe: float = 0.5
    min_oos_years: float = 1.0
    n_boot: int = 1000
    bootstrap_seed: int = 20260101
    pbo_splits: int = 16
    min_family_for_pbo: int = 4
    require_preregistration: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FamilyContext:
    """The experiment's trial family as it stands when the report is computed."""

    name: str
    member_ids: list[str]
    params: list[dict[str, Any]]
    is_sharpe: list[float]            # annualised in-sample Sharpe per member
    is_sharpe_pp: list[float]         # per-period in-sample Sharpe per member
    is_n_obs: list[int]
    oos_pvalues: list[float]          # one-sided, 1 - PSR(0) on each member's test window
    returns: pd.DataFrame | None      # aligned non-holdout returns, one column per member
    external_trials: int = 0
    target: int = 0
    pbo: dict[str, Any] | None = None
    n_eff: dict[str, Any] | None = None

    @property
    def n_registered(self) -> int:
        return len(self.member_ids)

    @property
    def n_trials(self) -> int:
        return len(self.member_ids) + int(self.external_trials)


@dataclass
class Gate:
    key: str
    label: str
    status: str            # pass | fail | na | info
    value: Any = None
    threshold: Any = None
    note: str = ""
    core: bool = True


def _window(r: pd.Series, window: Sequence[str] | None) -> pd.Series:
    if not window:
        return r.iloc[0:0]
    start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    return r[(r.index >= start) & (r.index <= end)]


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _fmt(x: Any, nd: int = 2) -> str:
    return f"{x:.{nd}f}" if _finite(x) else "n/a"


def compute_statistics(*, returns: pd.Series, periods_per_year: float, is_window: Sequence[str] | None,
                       oos_window: Sequence[str] | None, holdout_window: Sequence[str] | None,
                       result_frame: pd.DataFrame | None, family: FamilyContext | None,
                       variants: dict[str, float] | None = None, walk_forward: dict[str, Any] | None = None,
                       config: ValidationConfig | None = None) -> dict[str, Any]:
    cfg = config or ValidationConfig()
    ppy = float(periods_per_year)
    r = clean_returns(returns)
    holdout = _window(r, holdout_window) if holdout_window else r.iloc[0:0]
    non_holdout = r.drop(holdout.index) if len(holdout) else r
    r_is = _window(non_holdout, is_window) if is_window else non_holdout
    r_oos = _window(non_holdout, oos_window) if oos_window else r.iloc[0:0]
    if is_window is None and oos_window is not None:
        r_is = non_holdout[non_holdout.index < pd.Timestamp(oos_window[0])]

    def describe(x: pd.Series) -> dict[str, Any]:
        if len(x) < 2:
            return {"n_obs": int(len(x))}
        mean, sd, sk, ku = moments(x.to_numpy())
        out = {"n_obs": int(len(x)), "start": x.index[0].strftime("%Y-%m-%d"), "end": x.index[-1].strftime("%Y-%m-%d"),
               "sharpe": mean / sd * math.sqrt(ppy) if sd > 0 else float("nan"),
               "sharpe_pp": mean / sd if sd > 0 else float("nan"), "skew": sk, "kurtosis": ku,
               "ann_return": mean * ppy, "ann_vol": sd * math.sqrt(ppy)}
        out["psr0"] = probabilistic_sharpe_ratio(out["sharpe_pp"], len(x), sk, ku, 0.0) if sd > 0 else float("nan")
        return out

    stats: dict[str, Any] = {"version": REPORT_VERSION, "periods_per_year": ppy, "config": cfg.to_dict(),
                             "is": describe(r_is), "oos": describe(r_oos), "holdout": describe(holdout),
                             "full": describe(non_holdout)}

    # ------------------------------------------------------------ significance
    sig: dict[str, Any] = {}
    is_d, oos_d = stats["is"], stats["oos"]
    n_trials = family.n_trials if family else 1
    sig["n_trials"] = n_trials
    if family and family.returns is not None and family.returns.shape[1] >= 2:
        eff = family.n_eff or effective_number_of_trials(family.returns)
        sig["avg_corr"] = eff["avg_corr"]
        sig["n_eff"] = float(eff["n_eff_corr"]) + float(family.external_trials)
        sig["n_eff_eig"] = float(eff["n_eff_eig"]) + float(family.external_trials)
    else:
        sig["n_eff"] = float(n_trials)
    if family and len(family.is_sharpe_pp) >= 2:
        vals = np.asarray([v for v in family.is_sharpe_pp if _finite(v)], dtype=float)
        sig["var_trials_pp"] = float(np.var(vals, ddof=1)) if len(vals) >= 2 else 0.0
    else:
        sig["var_trials_pp"] = 0.0
    if _finite(is_d.get("sharpe_pp")):
        d = deflated_sharpe_ratio(is_d["sharpe_pp"], is_d["n_obs"], is_d["skew"], is_d["kurtosis"],
                                  max(1.0, sig["n_eff"]), sig["var_trials_pp"])
        sig["dsr"] = d["dsr"]
        sig["sr0_annual"] = d["sr0"] * math.sqrt(ppy)
        sig["is_psr0"] = is_d["psr0"]
        mintrl = min_track_record_length(is_d["sharpe_pp"], is_d["skew"], is_d["kurtosis"], 0.0, 0.05)
        sig["min_trl_years"] = mintrl / ppy if math.isfinite(mintrl) else float("inf")
        others = None
        if family and len(family.is_sharpe) > 1:
            others = [sharpe_pvalue(s, n, ppy) for i, (s, n) in enumerate(zip(family.is_sharpe, family.is_n_obs))
                      if i != family.target]
        sig["haircut"] = haircut_sharpe(is_d["sharpe"], is_d["n_obs"], ppy, n_trials, "holm", others)
    if _finite(oos_d.get("psr0")):
        p_oos = 1.0 - oos_d["psr0"]
        sig["oos_p"] = p_oos
        if family and len(family.oos_pvalues) > 1:
            pv = list(family.oos_pvalues)
            pv[family.target] = p_oos
            sig["oos_p_holm"] = float(adjust_pvalues(pv, "holm")[family.target])
            sig["oos_p_bh"] = float(adjust_pvalues(pv, "bh")[family.target])
            sig["oos_tests"] = int(np.isfinite(np.asarray(pv, dtype=float)).sum())
        else:
            sig["oos_p_holm"] = p_oos
            sig["oos_p_bh"] = p_oos
            sig["oos_tests"] = 1
        summ = performance_summary(r_oos, ppy)
        sig["oos_nw_t"] = summ.get("nw_t_stat")
    if _finite(stats["holdout"].get("psr0")):
        sig["holdout_p"] = 1.0 - stats["holdout"]["psr0"]
    stats["significance"] = sig

    # ------------------------------------------------------------ resampling
    boot = {}
    for name, series in (("is", r_is), ("oos", r_oos), ("holdout", holdout)):
        if len(series) >= 30:
            boot[name] = bootstrap_sharpe(series, ppy, n_boot=cfg.n_boot, seed=cfg.bootstrap_seed)
    stats["bootstrap"] = boot

    # ------------------------------------------------------------ overfitting across the family
    stats["pbo"] = family.pbo if family else None
    stats["family"] = None
    if family:
        stats["family"] = {"name": family.name, "registered": family.n_registered,
                           "external_trials": family.external_trials, "n_trials": family.n_trials}

    # ------------------------------------------------------------ sensitivity
    sens: dict[str, Any] = {}
    if family and len(family.params) >= 2:
        sens["neighborhood"] = neighborhood(family.params, family.is_sharpe, family.target)
    if result_frame is not None and {"gross", "cost"}.issubset(result_frame.columns):
        rf = result_frame.loc[non_holdout.index.intersection(result_frame.index)]
        if rf["cost"].abs().sum() > 0:
            sens["costs"] = cost_sensitivity(rf["gross"], rf["cost"], ppy)
    if len(non_holdout) >= 50:
        sens["drop_best"] = drop_best(non_holdout, ppy, cfg.outlier_fraction)
    if variants:
        sens["variants"] = variants
    stats["sensitivity"] = sens

    # ------------------------------------------------------------ regimes
    reg: dict[str, Any] = {}
    if result_frame is not None and "benchmark" in result_frame.columns and result_frame["benchmark"].notna().sum() > 300:
        bench = result_frame["benchmark"].dropna()
        reg["volatility"] = regime_performance(non_holdout, volatility_regimes(bench), ppy)
        reg["trend"] = regime_performance(non_holdout, trend_regimes(bench), ppy)
    if isinstance(non_holdout.index, pd.DatetimeIndex) and len(non_holdout) > 0:
        reg["years"] = regime_performance(non_holdout, calendar_regimes(non_holdout.index), ppy,
                                          min_obs=int(ppy / 4))
    stats["regimes"] = reg
    if walk_forward:
        stats["walk_forward"] = {k: v for k, v in walk_forward.items() if k != "oos_returns"}
    return stats


INTEGRITY_CODES = {
    "PIT_VIOLATION", "HOLDOUT_IMPLICIT_ACCESS", "HOLDOUT_OVERUSE", "AMENDED_TESTED_ON_SEEN_DATA",
    "NONREPRODUCIBLE", "DATA_INTEGRITY", "SURVIVORSHIP_BIAS", "UNADJUSTED_PRICES",
}


def scorecard(stats: dict[str, Any], *, findings: Sequence[dict[str, Any]], preregistered: bool | None,
              reproduction: str | None, holdout_touch: int | None, holdout_max: int | None,
              config: ValidationConfig | None = None) -> list[Gate]:
    """Grade one experiment. ``holdout_touch`` is this experiment's look number at its holdout (1 = first).

    When the experiment evaluated the sealed holdout, the holdout test is the confirmatory test: its
    p-value is Bonferroni-adjusted for the number of looks so far, and the test-window significance
    (a window every trial in the family has seen) is reported but no longer decides the verdict.
    """
    cfg = config or ValidationConfig()
    ppy = stats["periods_per_year"]
    sig, sens, reg = stats.get("significance", {}), stats.get("sensitivity", {}), stats.get("regimes", {})
    is_d, oos_d, ho_d = stats.get("is", {}), stats.get("oos", {}), stats.get("holdout", {})
    gates: list[Gate] = []

    if preregistered is None:
        gates.append(Gate("preregistered", "Hypothesis pre-registered", "fail", None, "registered before run",
                          "No hypothesis is linked to this experiment.", core=cfg.require_preregistration))
    else:
        gates.append(Gate("preregistered", "Hypothesis pre-registered", "pass" if preregistered else "fail",
                          preregistered, "registered before run",
                          "Registered before the experiment started." if preregistered else
                          "The hypothesis was not registered before this experiment started.",
                          core=cfg.require_preregistration))

    min_oos = cfg.min_oos_years * ppy
    holdout_evaluated = ho_d.get("n_obs", 0) > 0
    if oos_d.get("n_obs", 0) >= min_oos and "oos_p_holm" in sig:
        note = (f"p = {_fmt(sig.get('oos_p'), 3)} raw, adjusted for {sig.get('oos_tests', 1)} trials "
                "evaluated on the test window.")
        if holdout_evaluated:
            note += " Informational here: the sealed holdout is this experiment's confirmatory test."
        gates.append(Gate("oos_significance", "Out-of-sample significance (Holm-adjusted)",
                          "pass" if sig["oos_p_holm"] <= cfg.oos_alpha else "fail", sig["oos_p_holm"],
                          f"<= {cfg.oos_alpha}", note, core=not holdout_evaluated))
    else:
        gates.append(Gate("oos_significance", "Out-of-sample significance (Holm-adjusted)", "na", None,
                          f"<= {cfg.oos_alpha}", "No test window of at least "
                          f"{cfg.min_oos_years:g} year(s)."))

    if _finite(sig.get("dsr")):
        gates.append(Gate("dsr", "Deflated Sharpe ratio (in-sample)", "pass" if sig["dsr"] >= cfg.dsr_threshold else "fail",
                          sig["dsr"], f">= {cfg.dsr_threshold}",
                          f"In-sample Sharpe {_fmt(is_d.get('sharpe'))} against an expected best-of-"
                          f"{_fmt(sig.get('n_eff'), 1)} noise Sharpe of {_fmt(sig.get('sr0_annual'))}."))
    else:
        gates.append(Gate("dsr", "Deflated Sharpe ratio (in-sample)", "na", None, f">= {cfg.dsr_threshold}",
                          "Too few in-sample observations."))

    pbo = stats.get("pbo")
    if pbo:
        gates.append(Gate("pbo", "Probability of backtest overfitting (family)",
                          "pass" if pbo["pbo"] <= cfg.pbo_threshold else "fail", pbo["pbo"],
                          f"<= {cfg.pbo_threshold}",
                          f"CSCV over {pbo['n_configs']} configurations, {pbo['n_combinations']} splits."))
    else:
        gates.append(Gate("pbo", "Probability of backtest overfitting (family)", "na", None, f"<= {cfg.pbo_threshold}",
                          f"Needs at least {cfg.min_family_for_pbo} configurations in the family."))

    if _finite(is_d.get("sharpe")) and _finite(oos_d.get("sharpe")) and is_d["sharpe"] > 0 and oos_d.get("n_obs", 0) >= min_oos:
        ratio = oos_d["sharpe"] / is_d["sharpe"]
        gates.append(Gate("degradation", "Out-of-sample / in-sample Sharpe", "pass" if ratio >= cfg.degradation_floor else "fail",
                          ratio, f">= {cfg.degradation_floor}",
                          f"In-sample {_fmt(is_d['sharpe'])}, out-of-sample {_fmt(oos_d['sharpe'])}."))
    else:
        gates.append(Gate("degradation", "Out-of-sample / in-sample Sharpe", "na", None, f">= {cfg.degradation_floor}",
                          "Needs a positive in-sample Sharpe and a test window."))

    nb = sens.get("neighborhood")
    if nb and nb.get("n_neighbours", 0) >= 1 and _finite(nb.get("plateau_ratio")):
        gates.append(Gate("plateau", "Parameter plateau (neighbours / selected)",
                          "pass" if nb["plateau_ratio"] >= cfg.plateau_floor else "fail", nb["plateau_ratio"],
                          f">= {cfg.plateau_floor}",
                          f"{nb['n_neighbours']} neighbouring configurations average "
                          f"{_fmt(nb['neighbour_mean'])} vs {_fmt(nb['metric'])}."))
    else:
        gates.append(Gate("plateau", "Parameter plateau (neighbours / selected)", "na", None, f">= {cfg.plateau_floor}",
                          "No neighbouring configurations were tried (or the in-sample Sharpe is not positive)."))

    costs = sens.get("costs")
    key = f"{cfg.cost_multiplier:g}x"
    if costs and key in costs:
        gates.append(Gate("costs", f"Sharpe with {cfg.cost_multiplier:g}x trading costs",
                          "pass" if _finite(costs[key]) and costs[key] > 0 else "fail", costs[key], "> 0",
                          f"Base costs {_fmt(costs.get('1x'))}, doubled {_fmt(costs[key])}."))
    else:
        gates.append(Gate("costs", f"Sharpe with {cfg.cost_multiplier:g}x trading costs", "na", None, "> 0",
                          "The result has no cost series."))

    vol = reg.get("volatility")
    years = reg.get("years")
    if vol and vol.get("n_regimes", 0) >= 2:
        gates.append(Gate("regimes", "Positive in volatility regimes", "pass" if vol["share_positive"] >= cfg.regime_share else "fail",
                          vol["share_positive"], f">= {cfg.regime_share:.2f}",
                          "Share of low/mid/high volatility regimes with a positive Sharpe."))
    elif years and years.get("n_regimes", 0) >= 3:
        gates.append(Gate("regimes", "Positive in calendar years", "pass" if years["share_positive"] >= 0.5 else "fail",
                          years["share_positive"], ">= 0.50", "Share of calendar years with a positive Sharpe."))
    else:
        gates.append(Gate("regimes", "Regime consistency", "na", None, None, "Not enough history."))

    db = sens.get("drop_best")
    if db and _finite(db.get("sharpe")):
        ok = _finite(db.get("sharpe_trimmed")) and db["sharpe_trimmed"] > 0
        gates.append(Gate("outliers", f"Sharpe with the {cfg.outlier_fraction:.0%} best and worst periods trimmed",
                          "pass" if ok else "fail", db.get("sharpe_trimmed"), "> 0",
                          f"{db['removed']} best and {db['removed']} worst periods removed."))
    else:
        gates.append(Gate("outliers", f"Sharpe with the {cfg.outlier_fraction:.0%} best and worst periods trimmed", "na"))

    if holdout_evaluated:
        look = max(1, int(holdout_touch or 1))
        limit = int(holdout_max or 1)
        p_raw = sig.get("holdout_p")
        p_adj = min(1.0, p_raw * look) if _finite(p_raw) else float("nan")
        ok = look <= limit and _finite(p_adj) and p_adj <= cfg.oos_alpha
        gates.append(Gate("holdout", "Sealed holdout test (Bonferroni over looks)", "pass" if ok else "fail", p_adj,
                          f"<= {cfg.oos_alpha}, look <= {limit}",
                          f"Holdout Sharpe {_fmt(ho_d.get('sharpe'))}, p = {_fmt(p_raw, 3)}; this was look {look} of "
                          f"{limit} allowed, so p x {look} = {_fmt(p_adj, 3)}."))
    else:
        gates.append(Gate("holdout", "Sealed holdout test (Bonferroni over looks)", "na", None, None,
                          "Holdout still sealed (not evaluated)."))

    if reproduction:
        gates.append(Gate("reproducible", "Reproduced from the registry", "pass" if reproduction in ("exact", "numerical") else "fail",
                          reproduction, "exact or numerical", "Latest reproduction run."))
    else:
        gates.append(Gate("reproducible", "Reproduced from the registry", "na", None, None, "Not re-run yet.", core=False))

    critical = [f["code"] for f in findings if f.get("severity") == "critical"]
    gates.append(Gate("hygiene", "No critical research-hygiene warnings", "fail" if critical else "pass",
                      len(critical), "0", ", ".join(sorted(set(critical))) or "None."))
    return gates


def verdict(stats: dict[str, Any], gates: Sequence[Gate], findings: Sequence[dict[str, Any]],
            config: ValidationConfig | None = None) -> tuple[str, str]:
    cfg = config or ValidationConfig()
    integrity = sorted({f["code"] for f in findings if f["code"] in INTEGRITY_CODES and f.get("severity") == "critical"})
    if integrity:
        return "invalid", "Integrity problem: " + ", ".join(integrity) + "."
    by_key = {g.key: g for g in gates}
    oos_ok = by_key["oos_significance"].status
    holdout_ok = by_key["holdout"].status
    if oos_ok == "na" and holdout_ok == "na":
        is_d = stats.get("is", {})
        if _finite(is_d.get("sharpe")) and is_d["sharpe"] < cfg.attractive_sharpe:
            return "no-edge", "No out-of-sample test, and the in-sample Sharpe is below the attractiveness bar."
        return "insufficient", "No out-of-sample or holdout window long enough to judge."
    failed = [g for g in gates if g.core and g.status == "fail"]
    if not failed and by_key["dsr"].status == "pass" and (oos_ok == "pass" or holdout_ok == "pass"):
        return "robust", "Every applicable core check passes."
    is_d = stats.get("is", {})
    attractive = (_finite(is_d.get("sharpe")) and is_d["sharpe"] >= cfg.attractive_sharpe) or \
        (_finite(is_d.get("psr0")) and is_d["psr0"] >= 0.95)
    names = ", ".join(g.label.lower() for g in failed) or "core checks not passed"
    if attractive:
        return "fragile", f"Attractive in-sample, but fails: {names}."
    return "no-edge", f"No convincing edge; fails: {names}."


def build_report(stats: dict[str, Any], *, findings: Sequence[dict[str, Any]], preregistered: bool | None,
                 reproduction: str | None = None, holdout_touch: int | None = None, holdout_max: int | None = None,
                 config: ValidationConfig | None = None) -> dict[str, Any]:
    gates = scorecard(stats, findings=findings, preregistered=preregistered, reproduction=reproduction,
                      holdout_touch=holdout_touch, holdout_max=holdout_max, config=config)
    v, reason = verdict(stats, gates, findings, config)
    report = dict(stats)
    report["scorecard"] = [asdict(g) for g in gates]
    report["verdict"] = v
    report["verdict_reason"] = reason
    report["findings"] = [{"code": f["code"], "severity": f["severity"], "message": f["message"]} for f in findings]
    report["n_trials"] = stats.get("significance", {}).get("n_trials")
    return report
