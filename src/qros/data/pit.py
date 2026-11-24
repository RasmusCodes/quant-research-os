"""Point-in-time data access with an audit trail.

Fundamental data, estimates and many alternative datasets describe a period (a fiscal quarter) but
become known later (the filing date). Joining them on the period date lets a backtest trade on
information it could not have had. ``PointInTime`` keeps both timestamps and answers "what was known
at decision time d" by joining on the availability timestamp. Every query is audited: the audit counts
rows whose availability time is after the decision time, so a backtest that keys on the wrong column
(e.g. the period end) is detected from the rows it actually used, and ``strict=True`` refuses them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd


class PITViolation(RuntimeError):
    pass


@dataclass
class PITAudit:
    dataset: str
    queries: int = 0
    rows_used: int = 0
    violations: int = 0
    max_lead_days: float = 0.0
    keys_used: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PointInTime:
    def __init__(self, frame: pd.DataFrame, *, entity: str, time: str, available: str, name: str = "dataset",
                 strict: bool = False):
        missing = {entity, time, available} - set(frame.columns)
        if missing:
            raise KeyError(f"{name}: missing columns {sorted(missing)}")
        self.frame = frame.copy()
        for col in (time, available):
            self.frame[col] = pd.to_datetime(self.frame[col]).astype("datetime64[ns]")
        bad = self.frame[available] < self.frame[time]
        if bad.any():
            raise ValueError(f"{name}: {int(bad.sum())} rows become available before the period they describe")
        self.entity, self.time, self.available, self.name = entity, time, available, name
        self.strict = strict
        self.audit = PITAudit(dataset=name)

    def _record(self, decision: pd.Series, available: pd.Series, entity: pd.Series, key: str) -> None:
        used = available.notna()
        lead = (available - decision).dt.total_seconds() / 86400.0
        viol = used & (lead > 0)
        self.audit.queries += 1
        self.audit.rows_used += int(used.sum())
        self.audit.violations += int(viol.sum())
        if key not in self.audit.keys_used:
            self.audit.keys_used.append(key)
        if viol.any():
            self.audit.max_lead_days = max(self.audit.max_lead_days, float(lead[viol].max()))
            if len(self.audit.examples) < 5:
                i = viol[viol].index[0]
                self.audit.examples.append({"entity": str(entity.loc[i]), "decision": str(decision.loc[i].date()),
                                            "available": str(available.loc[i].date()), "key": key})
            if self.strict:
                raise PITViolation(f"{self.name}: {int(viol.sum())} rows were not available at decision time "
                                   f"(joined on {key!r})")

    def as_of(self, when: Any, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Latest known row per entity at ``when``."""
        t = pd.Timestamp(when)
        known = self.frame[self.frame[self.available] <= t]
        latest = known.sort_values([self.time, self.available]).groupby(self.entity).tail(1)
        self.audit.queries += 1
        self.audit.rows_used += len(latest)
        cols = [self.entity, self.time, self.available] + list(columns or [c for c in self.frame.columns
                                                                           if c not in (self.entity, self.time, self.available)])
        return latest[cols].set_index(self.entity)

    def panel(self, dates: Sequence[Any], value: str, *, key: str | None = None,
              entities: Sequence[Any] | None = None) -> pd.DataFrame:
        """Dates x entities panel of ``value`` as known at each decision date.

        ``key`` is the timestamp the join uses, by default the availability time. Passing the period
        column reproduces the classic look-ahead bug; the audit records it.
        """
        key = key or self.available
        dec = pd.DatetimeIndex(pd.to_datetime(dates)).astype("datetime64[ns]")
        ents = list(entities) if entities is not None else sorted(self.frame[self.entity].unique())
        left = pd.MultiIndex.from_product([dec, ents], names=["_decision", self.entity]).to_frame(index=False)
        left = left.sort_values("_decision", kind="mergesort")
        right = self.frame[[self.entity, key, self.available, value]].rename(columns={key: "_key"}) \
            if key != self.available else self.frame[[self.entity, self.available, value]].assign(
                _key=self.frame[self.available])
        right = right.dropna(subset=["_key"]).sort_values("_key", kind="mergesort")
        merged = pd.merge_asof(left, right, left_on="_decision", right_on="_key", by=self.entity,
                               direction="backward")
        self._record(merged["_decision"], merged[self.available], merged[self.entity], key)
        out = merged.pivot(index="_decision", columns=self.entity, values=value)
        out.index.name = None
        out.columns.name = None
        return out.reindex(index=dec, columns=ents)


def with_reporting_lag(frame: pd.DataFrame, time: str, lag_days: int | Sequence[int] | pd.Series,
                       column: str = "available_at") -> pd.DataFrame:
    """Add an availability column = period time + reporting lag (days)."""
    out = frame.copy()
    lag = pd.to_timedelta(np.asarray(lag_days, dtype="int64") if not np.isscalar(lag_days) else int(lag_days),
                          unit="D")
    out[column] = pd.to_datetime(out[time]) + lag
    return out


def check_availability(frame: pd.DataFrame, decision: str, available: str) -> dict[str, Any]:
    """Audit any feature table that carries both a decision time and the input's availability time."""
    d = pd.to_datetime(frame[decision])
    a = pd.to_datetime(frame[available])
    lead = (a - d).dt.total_seconds() / 86400.0
    viol = lead > 0
    return {"rows": int(len(frame)), "violations": int(viol.sum()),
            "max_lead_days": float(lead[viol].max()) if viol.any() else 0.0}
