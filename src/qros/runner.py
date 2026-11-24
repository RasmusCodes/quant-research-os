"""Managed runs: the research OS executes the strategy itself, so it can reproduce it exactly.

A managed experiment is ``module:function`` + parameters + seed + dataset versions + windows. The
function receives a :class:`RunContext` and returns a :class:`~qros.backtest.BacktestResult`. Because
the OS hands out the data, it can (a) hide a sealed holdout from the strategy, (b) audit point-in-time
access, and (c) re-run the identical inputs later and compare result hashes.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd

from qros.backtest.interface import as_result
from qros.data.pit import PITAudit, PointInTime
from qros.registry.models import Holdout, Periods
from qros.util.env import capture_environment, code_fingerprint, resolve_entry_point
from qros.util.hashing import fingerprint_series

if TYPE_CHECKING:  # pragma: no cover
    from qros.registry.core import Registry
    from qros.registry.models import Experiment


@dataclass
class RunContext:
    registry: "Registry"
    experiment_id: str | None
    params: dict[str, Any]
    seed: int | None
    periods: Periods
    holdout: Holdout | None = None
    holdout_opened: bool = False
    data_versions: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    _audits: list[PITAudit] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    @property
    def cutoff(self) -> pd.Timestamp | None:
        """Data from this date on is hidden while the holdout is sealed."""
        if self.holdout is not None and not self.holdout_opened:
            return pd.Timestamp(self.holdout.start)
        return None

    def param(self, name: str, default: Any = None) -> Any:
        return self.params.get(name, default)

    def data(self, role: str) -> pd.DataFrame:
        """The dataset version bound to ``role``, verified against its hash and cut at a sealed holdout."""
        if role not in self.data_versions:
            raise KeyError(f"No dataset bound to role {role!r}; bound roles: {sorted(self.data_versions)}")
        vid = self.data_versions[role]
        frame = self.registry.load(vid, verify=True)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{vid} is a raw file; read it with registry.load()")
        cut = self.cutoff
        if cut is None:
            return frame
        meta = self.registry.get_dataset(self.registry.get_version(vid).dataset_id)
        if isinstance(frame.index, pd.DatetimeIndex):
            frame = frame[frame.index < cut]
        if meta.get("time_col") and meta["time_col"] in frame.columns:
            frame = frame[pd.to_datetime(frame[meta["time_col"]]) < cut]
        if meta.get("available_col") and meta["available_col"] in frame.columns:
            frame = frame[pd.to_datetime(frame[meta["available_col"]]) < cut]
        return frame

    def pit(self, role: str, *, strict: bool = False) -> PointInTime:
        """A point-in-time view of a dataset whose metadata names entity, time and availability columns."""
        vid = self.data_versions[role]
        meta = self.registry.get_dataset(self.registry.get_version(vid).dataset_id)
        for key in ("entity_col", "time_col", "available_col"):
            if not meta.get(key):
                raise ValueError(f"{meta['name']} needs {key} for point-in-time access")
        view = PointInTime(self.data(role), entity=meta["entity_col"], time=meta["time_col"],
                           available=meta["available_col"], name=meta["name"], strict=strict)
        self._audits.append(view.audit)
        return view

    @property
    def audits(self) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self._audits]


def _context(reg: "Registry", exp_id: str | None, params: dict[str, Any], seed: int | None, periods: Periods,
             holdout: str | None, open_holdout: bool, data: dict[str, str]) -> RunContext:
    ho = reg.get_holdout(holdout) if holdout else None
    return RunContext(reg, exp_id, dict(params), seed, periods, ho, bool(open_holdout), dict(data))


def run_experiment(reg: "Registry", entry_point: str, *, params: dict[str, Any] | None = None,
                   seed: int | None = None, data: dict[str, str] | None = None, periods: Periods | None = None,
                   hypothesis: str | None = None, family: str | None = None, name: str | None = None,
                   holdout: str | None = None, open_holdout: bool = False, holdout_purpose: str | None = None,
                   tags: Iterable[str] = (), notes: str | None = None, model: str | None = None,
                   model_version: str | None = None, parent: str | None = None, project: str | None = None,
                   periods_per_year: float | None = None, validate: bool = True) -> "Experiment":
    fn = resolve_entry_point(entry_point)
    params = dict(params or {})
    periods = periods or Periods()
    exp = reg.create_experiment(
        name or entry_point.split(":")[-1], hypothesis=hypothesis, family=family, project=project, model=model,
        model_version=model_version, parent=parent, params=params, seed=seed, entry_point=entry_point,
        code=code_fingerprint(fn), periods=periods, holdout=holdout, open_holdout=open_holdout,
        holdout_purpose=holdout_purpose, data=data, tags=tags, notes=notes, source="managed",
        periods_per_year=periods_per_year)
    ctx = _context(reg, exp.id, params, seed, periods, holdout, open_holdout, data or {})
    try:
        result = as_result(fn(ctx))
    except Exception:
        reg.fail_experiment(exp.id, traceback.format_exc())
        raise
    audits = ctx.audits
    if audits:
        result.meta["pit_audit"] = list(result.meta.get("pit_audit", [])) + audits
    result.meta.update({k: v for k, v in ctx.meta.items() if k not in result.meta})
    return reg.complete_experiment(exp.id, result, validate=validate)


def reproduce(reg: "Registry", experiment_id: str, *, tolerance: float = 1e-10,
              revalidate: bool = True) -> dict[str, Any]:
    """Re-run a managed experiment from its recorded inputs and compare the result hash.

    Statuses: ``exact`` (identical hash), ``numerical`` (same dates, max |diff| <= tolerance),
    ``mismatch`` or ``error``. Re-runs read the holdout exactly as the original did and do not count as
    new holdout looks: they cannot reveal anything the original run did not.
    """
    exp = reg.get_experiment(experiment_id)
    if exp.source != "managed" or not exp.entry_point:
        raise ValueError(f"{experiment_id} was {exp.source}; only managed runs can be re-executed by qros")
    note = None
    try:
        fn = resolve_entry_point(exp.entry_point)
        current = code_fingerprint(fn)
        if exp.code and current.get("file_sha256") != exp.code.get("file_sha256"):
            note = "the strategy's source file has changed since the original run"
        ctx = _context(reg, None, exp.params, exp.seed, exp.periods, exp.holdout_id, exp.holdout_opened, exp.data)
        result = as_result(fn(ctx)).validate()
    except Exception as exc:  # noqa: BLE001 - recorded, then surfaced
        rec = reg.record_reproduction(experiment_id, "error", result_hash=None, max_abs_diff=None,
                                      environment=capture_environment(), note=f"{type(exc).__name__}: {exc}")
        if revalidate:
            from qros.evaluate import validate_experiment
            validate_experiment(reg, experiment_id)
        return rec
    new = result.returns.rename("net")
    new_hash = fingerprint_series(new)
    old = reg.returns(experiment_id)
    if new_hash == exp.result_hash:
        status, diff = "exact", 0.0
    elif new.index.equals(old.index):
        diff = float(np.max(np.abs(new.to_numpy() - old.to_numpy()))) if len(new) else 0.0
        status = "numerical" if diff <= tolerance else "mismatch"
    else:
        status, diff = "mismatch", float("nan")
        note = (note + "; " if note else "") + "the result covers different dates"
    rec = reg.record_reproduction(experiment_id, status, result_hash=new_hash,
                                  max_abs_diff=None if diff != diff else diff,
                                  environment=capture_environment(), note=note)
    if revalidate:
        from qros.evaluate import validate_experiment
        validate_experiment(reg, experiment_id)
    return rec
