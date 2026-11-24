"""Tracking API for code that runs outside qros (notebooks, existing backtesters).

    with qros.track("momentum 12-1", hypothesis="H-001", params={"lookback": 252}, seed=7,
                    data={"prices": "dv:3f2a..."}, train="2005:2014", test="2015:2019") as run:
        result = my_backtest(...)
        run.log_result(result)          # a BacktestResult, a return Series or a result frame

The experiment is created (with code hash, git state and environment) when the block starts and frozen
when the result is logged; an exception or a missing result marks it failed.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Iterable

from qros.registry.core import Registry
from qros.registry.models import Periods
from qros.util.env import git_info
from qros.util.hashing import sha256_file


def caller_code(depth: int = 2) -> dict[str, Any] | None:
    frame = inspect.stack()[depth]
    path = Path(frame.filename)
    if not path.is_file():
        return None
    info: dict[str, Any] = {"file": path.name, "file_sha256": sha256_file(path), "module": frame.frame.f_globals.get("__name__"),
                            "qualname": frame.function}
    git = git_info(path)
    if git:
        info["git"] = git
    return info


class Run:
    def __init__(self, registry: Registry, experiment_id: str, owns_registry: bool = False):
        self.registry = registry
        self.id = experiment_id
        self._done = False
        self._owns = owns_registry

    def log_metric(self, name: str, value: float) -> None:
        self.registry.log_metrics(self.id, {name: value}, scope="custom")

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.registry.log_metrics(self.id, metrics, scope="custom")

    def log_artifact(self, name: str, obj: Any, kind: str = "file") -> str:
        return self.registry.log_artifact(self.id, name, obj, kind=kind)

    def note(self, text: str) -> None:
        self.registry.add_note(text, experiment=self.id)

    def log_result(self, result: Any, *, metrics: dict[str, float] | None = None, validate: bool = True) -> Any:
        exp = self.registry.complete_experiment(self.id, result, metrics=metrics, validate=validate)
        self._done = True
        return exp

    def fail(self, error: str) -> None:
        if not self._done:
            self.registry.fail_experiment(self.id, error)
            self._done = True

    @property
    def experiment(self) -> Any:
        return self.registry.get_experiment(self.id)

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        try:
            if exc_type is not None:
                import traceback
                self.fail("".join(traceback.format_exception(exc_type, exc, tb)))
            elif not self._done:
                self.fail("the tracked block ended without log_result()")
        finally:
            if self._owns:
                self.registry.close()
        return False


def track(name: str, *, registry: Registry | str | None = None, hypothesis: str | None = None,
          family: str | None = None, params: dict[str, Any] | None = None, seed: int | None = None,
          data: dict[str, str] | None = None, train: Any = None, test: Any = None, holdout: str | None = None,
          open_holdout: bool = False, holdout_purpose: str | None = None, tags: Iterable[str] = (),
          notes: str | None = None, model: str | None = None, model_version: str | None = None,
          parent: str | None = None, project: str | None = None, code: dict[str, Any] | None = None,
          periods_per_year: float | None = None) -> Run:
    owns = not isinstance(registry, Registry)
    reg = registry if isinstance(registry, Registry) else Registry.open(registry)
    exp = reg.create_experiment(
        name, hypothesis=hypothesis, family=family, project=project, model=model, model_version=model_version,
        parent=parent, params=params, seed=seed, code=code or caller_code(), periods=Periods.of(train, test),
        holdout=holdout, open_holdout=open_holdout, holdout_purpose=holdout_purpose, data=data, tags=tags,
        notes=notes, source="tracked", periods_per_year=periods_per_year)
    return Run(reg, exp.id, owns_registry=owns)
