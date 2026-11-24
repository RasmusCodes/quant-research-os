"""Record types returned by the registry."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from qros.util.timeutil import date_str


def _loads(text: str | None, default: Any = None) -> Any:
    if text is None or text == "":
        return default
    return json.loads(text)


def parse_window(value: Any) -> tuple[str, str] | None:
    """Accept ("2005-01-01", "2014-12-31"), "2005-01-01:2014-12-31" or "2005:2014"."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        if ":" not in value:
            raise ValueError(f"A window looks like 'start:end', got {value!r}")
        start, end = value.split(":", 1)
    else:
        start, end = value
    start, end = str(start).strip(), str(end).strip()
    if len(start) == 4 and start.isdigit():
        start = f"{start}-01-01"
    if len(end) == 4 and end.isdigit():
        end = f"{end}-12-31"
    s, e = date_str(start), date_str(end)
    if s is None or e is None or s > e:
        raise ValueError(f"Invalid window {value!r}")
    return (s, e)


@dataclass(frozen=True)
class Periods:
    """In-sample (train) and out-of-sample (test) windows of an experiment."""

    train: tuple[str, str] | None = None
    test: tuple[str, str] | None = None

    @classmethod
    def of(cls, train: Any = None, test: Any = None) -> "Periods":
        periods = cls(parse_window(train), parse_window(test))
        if periods.train and periods.test and periods.test[0] <= periods.train[1]:
            raise ValueError("The test window must start after the train window ends")
        return periods

    def to_dict(self) -> dict[str, Any]:
        return {"train": list(self.train) if self.train else None, "test": list(self.test) if self.test else None}


@dataclass
class HypothesisVersion:
    hypothesis_id: str
    version: int
    statement: str
    rationale: str | None
    prediction: dict[str, Any] | None
    criteria: list[dict[str, Any]] | None
    reason: str | None
    created_at: str
    text_sha256: str

    @classmethod
    def from_row(cls, row: Any) -> "HypothesisVersion":
        return cls(row["hypothesis_id"], row["version"], row["statement"], row["rationale"],
                   _loads(row["prediction_json"]), _loads(row["criteria_json"]), row["reason"],
                   row["created_at"], row["text_sha256"])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Hypothesis:
    id: str
    title: str
    family: str | None
    status: str
    researcher: str | None
    created_at: str
    registered_at: str | None
    current_version: int
    status_note: str | None
    status_updated_at: str | None
    versions: list[HypothesisVersion] = field(default_factory=list)

    @property
    def current(self) -> HypothesisVersion:
        return next(v for v in self.versions if v.version == self.current_version)

    @property
    def statement(self) -> str:
        return self.current.statement

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["statement"] = self.statement if self.versions else None
        return d


@dataclass
class DatasetVersion:
    id: str
    dataset_id: str
    label: str | None
    content_hash: str
    fingerprint_version: str
    storage: str | None
    object_sha256: str | None
    n_rows: int | None
    n_cols: int | None
    columns: list[str] | None
    time_start: str | None
    time_end: str | None
    knowledge_time: str | None
    transform_id: str | None
    quality: dict[str, Any] | None
    notes: str | None
    created_at: str

    @classmethod
    def from_row(cls, row: Any) -> "DatasetVersion":
        return cls(row["id"], row["dataset_id"], row["label"], row["content_hash"], row["fingerprint_version"],
                   row["storage"], row["object_sha256"], row["n_rows"], row["n_cols"], _loads(row["columns_json"]),
                   row["time_start"], row["time_end"], row["knowledge_time"], row["transform_id"],
                   _loads(row["quality_json"]), row["notes"], row["created_at"])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Holdout:
    id: str
    name: str
    start: str
    end: str
    description: str | None
    max_touches: int
    created_at: str

    @classmethod
    def from_row(cls, row: Any) -> "Holdout":
        return cls(row["id"], row["name"], row["start"], row["end"], row["description"], row["max_touches"],
                   row["created_at"])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EXPERIMENT_JSON_COLUMNS = {
    "params_json": "params", "code_json": "code", "environment_json": "environment", "summary_json": "summary",
}


@dataclass
class Experiment:
    id: str
    name: str
    status: str
    project: str | None = None
    hypothesis_id: str | None = None
    hypothesis_version: int | None = None
    family: str | None = None
    model: str | None = None
    model_version: str | None = None
    parent_id: str | None = None
    source: str = "managed"
    researcher: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    entry_point: str | None = None
    command: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    params_hash: str | None = None
    seed: int | None = None
    code: dict[str, Any] | None = None
    code_hash: str | None = None
    environment: dict[str, Any] | None = None
    train_start: str | None = None
    train_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None
    holdout_id: str | None = None
    holdout_opened: bool = False
    periods_per_year: float | None = None
    spec_hash: str | None = None
    data_fingerprint: str | None = None
    result_hash: str | None = None
    verdict: str | None = None
    summary: dict[str, Any] | None = None
    error: str | None = None
    data: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: Any) -> "Experiment":
        kwargs: dict[str, Any] = {}
        for key in row.keys():
            if key in EXPERIMENT_JSON_COLUMNS:
                kwargs[EXPERIMENT_JSON_COLUMNS[key]] = _loads(row[key], {} if key == "params_json" else None)
            elif key == "holdout_opened":
                kwargs[key] = bool(row[key])
            else:
                kwargs[key] = row[key]
        return cls(**kwargs)

    @property
    def periods(self) -> Periods:
        train = (self.train_start, self.train_end) if self.train_start else None
        test = (self.test_start, self.test_end) if self.test_start else None
        return Periods(train, test)

    def metric(self, name: str, scope: str = "oos", default: float | None = None) -> float | None:
        return self.metrics.get(scope, {}).get(name, default)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
