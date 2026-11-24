"""qros_emit: write quant-research-os run manifests from any Python project, standard library only.

Copy this single file into a project that should not depend on quant-research-os (for example a
stdlib-only research layer), then:

    from qros_emit import RunManifest, write_returns_csv

    write_returns_csv("out/returns.csv", dates, net, gross=gross, cost=cost)
    m = RunManifest("my-project", "momentum 12-1", family="momentum", seed=7,
                    params={"lookback": 12}, train=("2005-01-01", "2014-12-31"),
                    test=("2015-01-01", "2019-12-31"))
    m.add_data("prices", "equity-prices", "data/prices.csv", source="vendor X", license="internal")
    m.set_returns("out/returns.csv")
    m.write("out/qros-run.json")          # then: qros ingest out/qros-run.json

The manifest schema is documented in schemas/run-manifest.schema.json of quant-research-os.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA = "qros.run-manifest/1"


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_state(path: str | os.PathLike = ".") -> dict[str, Any]:
    """Commit and dirty flag of the repository holding ``path`` (empty dict outside git)."""
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=str(path), capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None
    commit = run("rev-parse", "HEAD")
    if not commit:
        return {}
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status)}


def write_returns_csv(path: str | os.PathLike, dates: Sequence[Any], net: Sequence[float], *,
                      gross: Sequence[float] | None = None, cost: Sequence[float] | None = None) -> str:
    """Write ``date,net[,gross,cost]`` with full float precision; returns the file's SHA-256."""
    if len(dates) != len(net):
        raise ValueError("dates and net returns differ in length")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = ["date", "net"] + (["gross"] if gross is not None else []) + (["cost"] if cost is not None else [])
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(",".join(cols) + "\n")
        for i, d in enumerate(dates):
            row = [str(d)[:10], repr(float(net[i]))]
            if gross is not None:
                row.append(repr(float(gross[i])))
            if cost is not None:
                row.append(repr(float(cost[i])))
            fh.write(",".join(row) + "\n")
    return sha256_file(p)


class RunManifest:
    def __init__(self, project: str, name: str, *, family: str | None = None, hypothesis: str | None = None,
                 params: dict[str, Any] | None = None, seed: int | None = None,
                 train: tuple[str, str] | None = None, test: tuple[str, str] | None = None,
                 periods_per_year: float | None = None, tags: Iterable[str] = (), notes: str | None = None,
                 code: dict[str, Any] | None = None):
        self.started_at = _now()
        self.doc: dict[str, Any] = {
            "schema": SCHEMA, "project": project,
            "experiment": {"name": name, **({"family": family} if family else {}),
                           **({"hypothesis": {"id": hypothesis}} if hypothesis else {}),
                           **({"tags": list(tags)} if tags else {}), **({"notes": notes} if notes else {})},
            "code": {"language": "python", **git_state(), **(code or {})},
            "params": dict(params or {}),
            "data": [],
            "results": {},
        }
        if seed is not None:
            self.doc["seed"] = int(seed)
        periods = {}
        if train:
            periods["train"] = list(train)
        if test:
            periods["test"] = list(test)
        if periods:
            self.doc["periods"] = periods
        if periods_per_year:
            self.doc["periods_per_year"] = float(periods_per_year)

    def add_data(self, role: str, dataset: str, path: str | os.PathLike | None = None, *, sha256: str | None = None,
                 source: str | None = None, license: str | None = None, licensed: bool = False,
                 knowledge_time: str | None = None) -> "RunManifest":
        entry: dict[str, Any] = {"role": role, "dataset": dataset}
        if path is not None:
            entry["path"] = str(path).replace("\\", "/")
            entry["sha256"] = sha256_file(path)
        elif sha256:
            entry["sha256"] = sha256
        else:
            raise ValueError("add_data needs a path or a sha256")
        for key, value in (("source", source), ("license", license), ("knowledge_time", knowledge_time)):
            if value:
                entry[key] = value
        if licensed:
            entry["licensed"] = True
        self.doc["data"].append(entry)
        return self

    def set_returns(self, path: str | os.PathLike, *, date_column: str = "date",
                    columns: dict[str, str] | None = None, kind: str = "returns",
                    capital: float | None = None) -> "RunManifest":
        spec: dict[str, Any] = {"path": str(path).replace("\\", "/"), "sha256": sha256_file(path),
                                "date_column": date_column}
        if columns:
            spec["columns"] = columns
        if kind == "pnl":
            spec["kind"], spec["capital"] = "pnl", float(capital or 0)
        self.doc["results"]["returns"] = spec
        return self

    def add_metrics(self, **metrics: float) -> "RunManifest":
        self.doc["results"].setdefault("metrics", {}).update({k: float(v) for k, v in metrics.items()})
        return self

    def set_digest(self, value: str, json_path: str, command: Sequence[str], cwd: str = ".") -> "RunManifest":
        self.doc["results"]["digest"] = {"value": value, "from": "stdout-json", "json_path": json_path}
        self.doc["code"]["command"] = [str(c) for c in command]
        self.doc["code"]["cwd"] = cwd
        return self

    def write(self, path: str | os.PathLike) -> str:
        """Write the manifest (paths relative to its folder when possible); returns its SHA-256."""
        p = Path(path)
        base = p.parent.resolve()
        for entry in self.doc["data"] + ([self.doc["results"]["returns"]] if "returns" in self.doc["results"] else []):
            if "path" in entry:
                try:
                    entry["path"] = Path(entry["path"]).resolve().relative_to(base).as_posix()
                except ValueError:
                    pass
        if not self.doc["results"]:
            raise ValueError("a manifest needs a returns file or metrics")
        self.doc["started_at"] = self.started_at
        self.doc["finished_at"] = _now()
        p.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.doc, indent=2, sort_keys=False)
        p.write_text(text, encoding="utf-8")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
