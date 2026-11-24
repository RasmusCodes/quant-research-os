"""The experiment registry: a SQLite database plus a content-addressed object store.

A registry is a folder (by convention ``.qros/`` at a project root) holding ``registry.db`` and
``objects/``. Every write goes through this class, which records an event in the hash-chained audit
log; append-only history (hypothesis versions, the holdout ledger, dataset versions, finished
experiments) is additionally enforced by SQLite triggers.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from qros._version import __version__
from qros.artifacts import ObjectStore, frame_from_bytes, frame_to_bytes
from qros.registry import db
from qros.registry.models import (
    DatasetVersion, Experiment, Holdout, Hypothesis, HypothesisVersion, Periods,
)
from qros.util.hashing import (
    FRAME_FINGERPRINT_VERSION, canonical_json, fingerprint_frame, sha256_bytes, sha256_file, sha256_json,
    sha256_text,
)
from qros.util.timeutil import Clock, SystemClock, date_str, parse_iso, to_iso

DB_NAME = "registry.db"
HYPOTHESIS_STATUSES = ("draft", "registered", "supported", "rejected", "inconclusive", "abandoned")
EXPERIMENT_STATUSES = ("planned", "running", "completed", "failed", "invalidated")
SEVERITIES = ("info", "warning", "critical")
# What identifies "the same code" for trial de-duplication: the defining file and function for managed
# runs, or the commit / declared source hash for ingested runs. Full git state is still recorded.
CODE_HASH_KEYS = ("module", "qualname", "file_sha256", "source_sha256", "commit", "dirty", "code_hash")


class RegistryError(Exception):
    pass


class DataIntegrityError(RegistryError):
    pass


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    if not slug:
        raise ValueError(f"Cannot make an identifier from {text!r}")
    return slug


def _dumps(obj: Any) -> str | None:
    return None if obj is None else canonical_json(obj)


def _loads(text: str | None, default: Any = None) -> Any:
    return default if text in (None, "") else json.loads(text)


class Registry:
    """Open or create a registry folder."""

    def __init__(self, path: str | Path, *, create: bool = False, clock: Clock | None = None,
                 researcher: str | None = None):
        self.path = Path(path)
        db_path = self.path / DB_NAME
        if create:
            self.path.mkdir(parents=True, exist_ok=True)
        elif not db_path.is_file():
            raise FileNotFoundError(f"No qros registry at {self.path}. Create one with `qros init`.")
        self.conn = db.connect(db_path)
        db.migrate(self.conn)
        self.objects = ObjectStore(self.path / "objects")
        self.clock: Clock = clock or SystemClock()
        self._returns_cache: dict[tuple[str, str], pd.Series] = {}
        self.researcher = (researcher or os.environ.get("QROS_RESEARCHER")
                           or db.get_meta(self.conn, "researcher") or "researcher")
        if db.get_meta(self.conn, "registry_id") is None:
            with db.transaction(self.conn):
                db.set_meta(self.conn, "registry_id", secrets.token_hex(8))
                db.set_meta(self.conn, "created_at", self.now())
                db.set_meta(self.conn, "qros_version", __version__)
                if researcher:
                    db.set_meta(self.conn, "researcher", researcher)
                self._event("registry.created", "registry", None, {"qros": __version__})

    # ------------------------------------------------------------------ lifecycle

    @classmethod
    def init(cls, path: str | Path = ".qros", **kwargs: Any) -> "Registry":
        return cls(path, create=True, **kwargs)

    @classmethod
    def open(cls, path: str | Path | None = None, **kwargs: Any) -> "Registry":
        return cls(cls.locate(path), **kwargs)

    @staticmethod
    def locate(path: str | Path | None = None) -> Path:
        """Explicit path, else $QROS_REGISTRY, else the nearest ``.qros`` folder upwards from cwd."""
        if path is not None:
            p = Path(path)
            if (p / DB_NAME).is_file():
                return p
            if (p / ".qros" / DB_NAME).is_file():
                return p / ".qros"
            return p
        env = os.environ.get("QROS_REGISTRY")
        if env:
            return Path(env)
        here = Path.cwd().resolve()
        for folder in (here, *here.parents):
            if (folder / ".qros" / DB_NAME).is_file():
                return folder / ".qros"
        raise FileNotFoundError("No qros registry found here or above. Create one with `qros init`.")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Registry":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def registry_id(self) -> str:
        return db.get_meta(self.conn, "registry_id") or ""

    def now(self) -> str:
        return to_iso(self.clock.now())

    def _event(self, action: str, entity: str, entity_id: str | None, payload: Any = None,
               at: str | None = None) -> str:
        return db.append_event(self.conn, at or self.now(), self.researcher, action, entity, entity_id, payload)

    def _next_id(self, prefix: str, width: int) -> str:
        return f"{prefix}-{db.next_counter(self.conn, prefix):0{width}d}"

    def _one(self, sql: str, args: Sequence[Any] = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, args).fetchone()

    def _all(self, sql: str, args: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, args).fetchall()

    # ------------------------------------------------------------------ sources and datasets

    def add_source(self, name: str, *, vendor: str | None = None, url: str | None = None,
                   license: str | None = None, description: str | None = None) -> str:
        sid = "src:" + slugify(name)
        with db.transaction(self.conn):
            if self._one("SELECT 1 FROM sources WHERE id = ?", (sid,)):
                return sid
            self.conn.execute(
                "INSERT INTO sources(id, name, vendor, url, license, description, created_at) VALUES(?,?,?,?,?,?,?)",
                (sid, name, vendor, url, license, description, self.now()))
            self._event("source.added", "source", sid, {"name": name, "vendor": vendor, "license": license})
        return sid

    def add_dataset(self, name: str, *, source: str | None = None, kind: str = "table",
                    description: str | None = None, entity_col: str | None = None, time_col: str | None = None,
                    available_col: str | None = None, frequency: str | None = None,
                    survivorship_free: bool | None = None, adjusted: bool | None = None, licensed: bool = False,
                    schema: dict[str, Any] | None = None) -> str:
        did = "ds:" + slugify(name)
        if source and not source.startswith("src:"):
            source = "src:" + slugify(source)
        with db.transaction(self.conn):
            if self._one("SELECT 1 FROM datasets WHERE id = ?", (did,)):
                return did
            self.conn.execute(
                "INSERT INTO datasets(id, name, source_id, kind, description, entity_col, time_col, available_col,"
                " frequency, survivorship_free, adjusted, licensed, schema_json, created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (did, name, source, kind, description, entity_col, time_col, available_col, frequency,
                 None if survivorship_free is None else int(survivorship_free),
                 None if adjusted is None else int(adjusted), int(bool(licensed)), _dumps(schema), self.now()))
            self._event("dataset.added", "dataset", did, {
                "name": name, "source": source, "kind": kind, "available_col": available_col,
                "survivorship_free": survivorship_free, "adjusted": adjusted, "licensed": licensed})
        return did

    def get_dataset(self, dataset: str) -> dict[str, Any]:
        did = dataset if dataset.startswith("ds:") else "ds:" + slugify(dataset)
        row = self._one("SELECT * FROM datasets WHERE id = ?", (did,))
        if row is None:
            raise RegistryError(f"Unknown dataset {dataset}")
        out = dict(row)
        out["schema"] = _loads(out.pop("schema_json"))
        for key in ("survivorship_free", "adjusted", "licensed"):
            out[key] = None if out[key] is None else bool(out[key])
        return out

    def list_datasets(self) -> list[dict[str, Any]]:
        return [self.get_dataset(r["id"]) for r in self._all("SELECT id FROM datasets ORDER BY id")]

    def list_sources(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._all("SELECT * FROM sources ORDER BY id")]

    def record_transform(self, name: str, kind: str, *, func: Any = None, code_ref: str | None = None,
                         params: dict[str, Any] | None = None, description: str | None = None) -> str:
        code_sha = None
        if func is not None:
            from qros.util.env import code_fingerprint
            fp = code_fingerprint(func)
            code_ref = code_ref or f"{fp.get('module')}:{fp.get('qualname')}"
            code_sha = fp.get("file_sha256")
        definition = {"name": name, "kind": kind, "code_ref": code_ref, "code_sha256": code_sha,
                      "params": params or {}}
        tid = "tf:" + sha256_json(definition)[:12]
        with db.transaction(self.conn):
            if not self._one("SELECT 1 FROM transforms WHERE id = ?", (tid,)):
                self.conn.execute(
                    "INSERT INTO transforms(id, name, kind, code_ref, code_sha256, params_json, description,"
                    " created_at) VALUES(?,?,?,?,?,?,?,?)",
                    (tid, name, kind, code_ref, code_sha, canonical_json(params or {}), description, self.now()))
                self._event("transform.recorded", "transform", tid, definition)
        return tid

    def snapshot(self, dataset: str, data: pd.DataFrame | str | Path, *, label: str | None = None,
                 knowledge_time: Any = None, parents: Iterable[str] = (), transform: str | None = None,
                 notes: str | None = None, storage: str | None = None) -> DatasetVersion:
        """Freeze a dataset's content as an immutable, content-addressed version.

        ``knowledge_time`` is the snapshot's as-of: the moment up to which its contents were knowable
        (e.g. the vendor delivery date). Snapshotting identical content again returns the existing
        version. ``parents`` and ``transform`` record lineage for derived data.
        """
        meta = self.get_dataset(dataset)
        did = meta["id"]
        parents = list(parents)
        for parent in parents:
            self.get_version(parent)
        if isinstance(data, pd.DataFrame):
            content_hash = fingerprint_frame(data)
            payload, storage = frame_to_bytes(data, storage)
            columns = [str(c) for c in data.columns]
            n_rows, n_cols = int(data.shape[0]), int(data.shape[1])
            t0, t1 = self._time_range(data, meta.get("time_col"))
            from qros.data.quality import quality_report
            quality = quality_report(data, entity_col=meta.get("entity_col"), time_col=meta.get("time_col"))
            fp_version = FRAME_FINGERPRINT_VERSION
        else:
            src = Path(data)
            payload = src.read_bytes()
            content_hash = sha256_bytes(payload)
            storage, columns, n_rows, n_cols, t0, t1, quality = "file", None, None, None, None, None, None
            fp_version = "sha256-file"
        existing = self._one("SELECT id FROM dataset_versions WHERE dataset_id = ? AND content_hash = ?",
                             (did, content_hash))
        if existing:
            return self.get_version(existing["id"])
        object_sha = self.objects.put_bytes(payload)
        with db.transaction(self.conn):
            vid = "dv:" + content_hash[:12]
            n = 2
            while self._one("SELECT 1 FROM dataset_versions WHERE id = ?", (vid,)):
                vid, n = f"dv:{content_hash[:12]}-{n}", n + 1
            kt = None
            if knowledge_time is not None:
                kt = to_iso(parse_iso(str(knowledge_time))) if "T" in str(knowledge_time) else date_str(knowledge_time)
            self.conn.execute(
                "INSERT INTO dataset_versions(id, dataset_id, label, content_hash, fingerprint_version, storage,"
                " object_sha256, n_rows, n_cols, columns_json, time_start, time_end, knowledge_time, transform_id,"
                " quality_json, notes, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (vid, did, label, content_hash, fp_version, storage, object_sha, n_rows, n_cols, _dumps(columns),
                 t0, t1, kt, transform, _dumps(quality), notes, self.now()))
            for parent in parents:
                self.conn.execute("INSERT INTO lineage(output_id, input_id, transform_id) VALUES(?,?,?)",
                                  (vid, parent, transform))
            self._event("data.snapshot", "dataset_version", vid, {
                "dataset": did, "label": label, "content_hash": content_hash, "object_sha256": object_sha,
                "parents": parents, "transform": transform, "knowledge_time": kt, "rows": n_rows})
        return self.get_version(vid)

    def register_external_version(self, dataset: str, content_hash: str, *, label: str | None = None,
                                  knowledge_time: Any = None, notes: str | None = None,
                                  size: int | None = None) -> DatasetVersion:
        """Record a dataset version by hash only (data held elsewhere, e.g. licensed vendor files).

        Its content cannot be re-verified by this registry, which the version's storage ('external')
        makes explicit.
        """
        did = self.get_dataset(dataset)["id"]
        if not re.fullmatch(r"[0-9a-f]{64}", content_hash or ""):
            raise ValueError("content_hash must be a SHA-256 hex digest")
        existing = self._one("SELECT id FROM dataset_versions WHERE dataset_id = ? AND content_hash = ?",
                             (did, content_hash))
        if existing:
            return self.get_version(existing["id"])
        with db.transaction(self.conn):
            vid = "dv:" + content_hash[:12]
            n = 2
            while self._one("SELECT 1 FROM dataset_versions WHERE id = ?", (vid,)):
                vid, n = f"dv:{content_hash[:12]}-{n}", n + 1
            kt = date_str(knowledge_time) if knowledge_time else None
            self.conn.execute(
                "INSERT INTO dataset_versions(id, dataset_id, label, content_hash, fingerprint_version, storage,"
                " object_sha256, n_rows, n_cols, columns_json, time_start, time_end, knowledge_time, transform_id,"
                " quality_json, notes, created_at) VALUES(?,?,?,?,?,?,NULL,NULL,NULL,NULL,NULL,NULL,?,NULL,?,?,?)",
                (vid, did, label, content_hash, "sha256-file", "external", kt,
                 _dumps({"size": size} if size else None), notes, self.now()))
            self._event("data.external", "dataset_version", vid, {"dataset": did, "content_hash": content_hash,
                                                                 "label": label})
        return self.get_version(vid)

    def derive(self, dataset: str, data: pd.DataFrame, *, inputs: Sequence[str], transform: str,
               label: str | None = None, knowledge_time: Any = None, notes: str | None = None) -> DatasetVersion:
        """Snapshot derived data with lineage to its inputs (``transform`` from :meth:`record_transform`)."""
        if knowledge_time is None:
            times = [self.get_version(v).knowledge_time for v in inputs]
            times = [t for t in times if t]
            knowledge_time = max(times) if times else None
        return self.snapshot(dataset, data, label=label, knowledge_time=knowledge_time, parents=inputs,
                             transform=transform, notes=notes)

    @staticmethod
    def _time_range(df: pd.DataFrame, time_col: str | None) -> tuple[str | None, str | None]:
        values = None
        if time_col and time_col in df.columns:
            values = df[time_col]
        elif isinstance(df.index, pd.DatetimeIndex):
            values = df.index.to_series()
        if values is None or len(values) == 0:
            return None, None
        try:
            ts = pd.to_datetime(values)
        except (TypeError, ValueError):
            return None, None
        return date_str(ts.min()), date_str(ts.max())

    def get_version(self, version_id: str) -> DatasetVersion:
        row = self._one("SELECT * FROM dataset_versions WHERE id = ?", (version_id,))
        if row is None:
            raise RegistryError(f"Unknown dataset version {version_id}")
        return DatasetVersion.from_row(row)

    def list_versions(self, dataset: str | None = None) -> list[DatasetVersion]:
        if dataset:
            did = self.get_dataset(dataset)["id"]
            rows = self._all("SELECT * FROM dataset_versions WHERE dataset_id = ? ORDER BY created_at, id", (did,))
        else:
            rows = self._all("SELECT * FROM dataset_versions ORDER BY created_at, id")
        return [DatasetVersion.from_row(r) for r in rows]

    def latest_version(self, dataset: str) -> DatasetVersion:
        versions = self.list_versions(dataset)
        if not versions:
            raise RegistryError(f"Dataset {dataset} has no versions")
        return versions[-1]

    def load(self, version_id: str, verify: bool = True) -> pd.DataFrame | bytes:
        """Load a dataset version and check it still hashes to its recorded content hash."""
        version = self.get_version(version_id)
        if version.storage == "external":
            raise DataIntegrityError(f"{version_id} is an external reference: only its hash is recorded here")
        payload = self.objects.get_bytes(version.object_sha256, verify=verify)
        if version.storage == "file":
            if verify and sha256_bytes(payload) != version.content_hash:
                raise DataIntegrityError(f"{version_id}: stored bytes do not match the content hash")
            return payload
        frame = frame_from_bytes(payload, version.storage)
        if verify and fingerprint_frame(frame) != version.content_hash:
            raise DataIntegrityError(f"{version_id}: content no longer matches its fingerprint")
        return frame

    def verify_version(self, version_id: str) -> dict[str, Any]:
        if self.get_version(version_id).storage == "external":
            return {"version": version_id, "ok": None, "reason": "external reference: content not stored"}
        try:
            self.load(version_id, verify=True)
        except (DataIntegrityError, IOError, OSError) as exc:
            return {"version": version_id, "ok": False, "reason": str(exc)}
        return {"version": version_id, "ok": True, "reason": None}

    def lineage(self, version_id: str) -> dict[str, Any]:
        """Upstream graph of a dataset version: nodes (versions, transforms) and edges."""
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, str]] = []
        stack = [version_id]
        while stack:
            vid = stack.pop()
            if vid in nodes:
                continue
            version = self.get_version(vid)
            ds = self.get_dataset(version.dataset_id)
            nodes[vid] = {"id": vid, "type": "version", "dataset": version.dataset_id, "dataset_name": ds["name"],
                          "label": version.label, "content_hash": version.content_hash,
                          "knowledge_time": version.knowledge_time, "source": ds.get("source_id")}
            for row in self._all("SELECT input_id, transform_id FROM lineage WHERE output_id = ?", (vid,)):
                edges.append({"from": row["input_id"], "to": vid, "transform": row["transform_id"]})
                stack.append(row["input_id"])
        wanted = {e["transform"] for e in edges if e["transform"]}
        transforms = [t for t in self.list_transforms() if t["id"] in wanted]
        return {"root": version_id, "nodes": list(nodes.values()), "edges": edges, "transforms": transforms}

    def list_transforms(self) -> list[dict[str, Any]]:
        out = []
        for row in self._all("SELECT * FROM transforms ORDER BY created_at, id"):
            item = dict(row)
            item["params"] = _loads(item.pop("params_json"), {})
            out.append(item)
        return out

    def list_lineage_edges(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._all("SELECT * FROM lineage")]

    # ------------------------------------------------------------------ hypotheses

    def add_hypothesis(self, title: str, statement: str, *, rationale: str | None = None,
                       prediction: dict[str, Any] | None = None, criteria: list[dict[str, Any]] | None = None,
                       family: str | None = None, register: bool = False) -> Hypothesis:
        with db.transaction(self.conn):
            hid = self._next_id("H", 3)
            now = self.now()
            self.conn.execute(
                "INSERT INTO hypotheses(id, title, family, status, researcher, created_at, current_version)"
                " VALUES(?,?,?,?,?,?,1)", (hid, title, family, "draft", self.researcher, now))
            self._insert_hypothesis_version(hid, 1, statement, rationale, prediction, criteria, "initial", now)
            self._event("hypothesis.created", "hypothesis", hid, {
                "title": title, "statement": statement, "prediction": prediction, "criteria": criteria,
                "family": family}, at=now)
            if family:
                self.ensure_family(family)
            if register:
                self._register(hid)
        return self.get_hypothesis(hid)

    def _insert_hypothesis_version(self, hid: str, version: int, statement: str, rationale: str | None,
                                   prediction: Any, criteria: Any, reason: str | None, at: str) -> None:
        text_hash = sha256_json({"statement": statement, "rationale": rationale, "prediction": prediction,
                                 "criteria": criteria})
        self.conn.execute(
            "INSERT INTO hypothesis_versions(hypothesis_id, version, statement, rationale, prediction_json,"
            " criteria_json, reason, created_at, text_sha256) VALUES(?,?,?,?,?,?,?,?,?)",
            (hid, version, statement, rationale, _dumps(prediction), _dumps(criteria), reason, at, text_hash))

    def _register(self, hid: str) -> None:
        row = self._one("SELECT registered_at FROM hypotheses WHERE id = ?", (hid,))
        if row is None:
            raise RegistryError(f"Unknown hypothesis {hid}")
        if row["registered_at"]:
            raise RegistryError(f"{hid} is already registered (at {row['registered_at']})")
        now = self.now()
        version = self._one("SELECT current_version FROM hypotheses WHERE id = ?", (hid,))["current_version"]
        text_hash = self._one("SELECT text_sha256 FROM hypothesis_versions WHERE hypothesis_id = ? AND version = ?",
                              (hid, version))["text_sha256"]
        self.conn.execute("UPDATE hypotheses SET registered_at = ?, status = 'registered', status_updated_at = ?"
                          " WHERE id = ?", (now, now, hid))
        self._event("hypothesis.registered", "hypothesis", hid, {"version": version, "text_sha256": text_hash},
                    at=now)

    def register_hypothesis(self, hid: str) -> Hypothesis:
        """Pre-register (lock) a hypothesis before any experiment tests it."""
        with db.transaction(self.conn):
            self._register(hid)
        return self.get_hypothesis(hid)

    def amend_hypothesis(self, hid: str, *, reason: str, statement: str | None = None,
                         rationale: str | None = None, prediction: dict[str, Any] | None = None,
                         criteria: list[dict[str, Any]] | None = None) -> Hypothesis:
        """Add a new version. Earlier versions stay readable and amendments after results are flagged."""
        current = self.get_hypothesis(hid)
        cur = current.current
        with db.transaction(self.conn):
            version = current.current_version + 1
            now = self.now()
            new = {
                "statement": statement if statement is not None else cur.statement,
                "rationale": rationale if rationale is not None else cur.rationale,
                "prediction": prediction if prediction is not None else cur.prediction,
                "criteria": criteria if criteria is not None else cur.criteria,
            }
            self._insert_hypothesis_version(hid, version, new["statement"], new["rationale"], new["prediction"],
                                            new["criteria"], reason, now)
            self.conn.execute("UPDATE hypotheses SET current_version = ? WHERE id = ?", (version, hid))
            self._event("hypothesis.amended", "hypothesis", hid, {"version": version, "reason": reason, **new},
                        at=now)
        from qros.hygiene.rules import lint_hypothesis
        lint_hypothesis(self, hid)
        return self.get_hypothesis(hid)

    def set_hypothesis_status(self, hid: str, status: str, note: str | None = None,
                              evidence: Sequence[str] = ()) -> Hypothesis:
        if status not in HYPOTHESIS_STATUSES:
            raise ValueError(f"status must be one of {HYPOTHESIS_STATUSES}")
        with db.transaction(self.conn):
            now = self.now()
            cur = self.conn.execute("UPDATE hypotheses SET status = ?, status_note = ?, status_updated_at = ?"
                                    " WHERE id = ?", (status, note, now, hid))
            if cur.rowcount == 0:
                raise RegistryError(f"Unknown hypothesis {hid}")
            self._event("hypothesis.status", "hypothesis", hid, {"status": status, "note": note,
                                                                 "evidence": list(evidence)}, at=now)
        return self.get_hypothesis(hid)

    def get_hypothesis(self, hid: str) -> Hypothesis:
        row = self._one("SELECT * FROM hypotheses WHERE id = ?", (hid,))
        if row is None:
            raise RegistryError(f"Unknown hypothesis {hid}")
        versions = [HypothesisVersion.from_row(r) for r in self._all(
            "SELECT * FROM hypothesis_versions WHERE hypothesis_id = ? ORDER BY version", (hid,))]
        return Hypothesis(row["id"], row["title"], row["family"], row["status"], row["researcher"], row["created_at"],
                          row["registered_at"], row["current_version"], row["status_note"],
                          row["status_updated_at"], versions)

    def list_hypotheses(self) -> list[Hypothesis]:
        return [self.get_hypothesis(r["id"]) for r in self._all("SELECT id FROM hypotheses ORDER BY id")]

    # ------------------------------------------------------------------ holdouts and families

    def add_holdout(self, name: str, start: Any, end: Any, *, description: str | None = None,
                    max_touches: int = 1) -> Holdout:
        s, e = date_str(start), date_str(end)
        if not s or not e or s > e:
            raise ValueError("Invalid holdout window")
        with db.transaction(self.conn):
            hoid = self._next_id("HO", 2)
            self.conn.execute(
                "INSERT INTO holdouts(id, name, start, end, description, max_touches, created_at)"
                " VALUES(?,?,?,?,?,?,?)", (hoid, name, s, e, description, int(max_touches), self.now()))
            self._event("holdout.created", "holdout", hoid, {"name": name, "start": s, "end": e,
                                                             "max_touches": max_touches})
        return self.get_holdout(hoid)

    def get_holdout(self, holdout: str) -> Holdout:
        row = self._one("SELECT * FROM holdouts WHERE id = ? OR name = ?", (holdout, holdout))
        if row is None:
            raise RegistryError(f"Unknown holdout {holdout}")
        return Holdout.from_row(row)

    def list_holdouts(self) -> list[Holdout]:
        return [Holdout.from_row(r) for r in self._all("SELECT * FROM holdouts ORDER BY id")]

    def holdout_accesses(self, holdout: str | None = None) -> list[dict[str, Any]]:
        if holdout:
            hoid = self.get_holdout(holdout).id
            rows = self._all("SELECT * FROM holdout_access WHERE holdout_id = ? ORDER BY id", (hoid,))
        else:
            rows = self._all("SELECT * FROM holdout_access ORDER BY id")
        return [dict(r) for r in rows]

    def touch_holdout(self, holdout: str, experiment_id: str | None, purpose: str) -> None:
        hoid = self.get_holdout(holdout).id
        with db.transaction(self.conn):
            now = self.now()
            self.conn.execute("INSERT INTO holdout_access(holdout_id, experiment_id, purpose, accessed_at)"
                              " VALUES(?,?,?,?)", (hoid, experiment_id, purpose, now))
            n = self._one("SELECT COUNT(*) FROM holdout_access WHERE holdout_id = ?", (hoid,))[0]
            self._event("holdout.accessed", "holdout", hoid, {"experiment": experiment_id, "purpose": purpose,
                                                              "touch": n}, at=now)

    def ensure_family(self, name: str, description: str | None = None) -> str:
        with db.transaction(self.conn):
            if not self._one("SELECT 1 FROM families WHERE name = ?", (name,)):
                self.conn.execute("INSERT INTO families(name, description, external_trials, created_at)"
                                  " VALUES(?,?,0,?)", (name, description, self.now()))
                self._event("family.created", "family", name, {"description": description})
        return name

    def declare_external_trials(self, family: str, n: int, note: str) -> int:
        """Record configurations tried outside qros so they count in multiple-testing corrections."""
        if n < 0:
            raise ValueError("n must be non-negative")
        self.ensure_family(family)
        with db.transaction(self.conn):
            self.conn.execute("UPDATE families SET external_trials = external_trials + ? WHERE name = ?",
                              (int(n), family))
            total = self._one("SELECT external_trials FROM families WHERE name = ?", (family,))[0]
            self._event("family.external_trials", "family", family, {"added": n, "total": total, "note": note})
        return int(total)

    def get_family(self, name: str) -> dict[str, Any]:
        row = self._one("SELECT * FROM families WHERE name = ?", (name,))
        if row is None:
            raise RegistryError(f"Unknown family {name}")
        return dict(row)

    def list_families(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._all("SELECT * FROM families ORDER BY name")]

    def family_members(self, family: str, status: str | None = "completed") -> list[Experiment]:
        return self.list_experiments(family=family, status=status)

    def family_trial_count(self, family: str) -> dict[str, int]:
        members = self.family_members(family)
        distinct = len({m.spec_hash or m.id for m in members})
        fam = self._one("SELECT external_trials FROM families WHERE name = ?", (family,))
        external = int(fam[0]) if fam else 0
        return {"registered": len(members), "distinct": distinct, "external": external,
                "total": distinct + external}

    # ------------------------------------------------------------------ experiments

    def create_experiment(self, name: str, *, hypothesis: str | None = None, family: str | None = None,
                          project: str | None = None, model: str | None = None, model_version: str | None = None,
                          parent: str | None = None, params: dict[str, Any] | None = None, seed: int | None = None,
                          entry_point: str | None = None, command: str | None = None,
                          code: dict[str, Any] | None = None, environment: dict[str, Any] | None = None,
                          periods: Periods | None = None, holdout: str | None = None, open_holdout: bool = False,
                          holdout_purpose: str | None = None, data: dict[str, str] | None = None,
                          tags: Iterable[str] = (), notes: str | None = None, source: str = "tracked",
                          periods_per_year: float | None = None, started_at: str | None = None) -> Experiment:
        """Open a new experiment in status ``running``. Finish it with :meth:`complete_experiment`."""
        params = dict(params or {})
        periods = periods or Periods()
        hyp_version = None
        if hypothesis:
            hyp = self.get_hypothesis(hypothesis)
            hyp_version = hyp.current_version
            family = family or hyp.family or hyp.id
        family = family or slugify(name)
        holdout_obj = self.get_holdout(holdout) if holdout else None
        if parent:
            self.get_experiment(parent)
        data = dict(data or {})
        content = {}
        for role, vid in sorted(data.items()):
            content[role] = self.get_version(vid).content_hash
        if environment is None:
            from qros.util.env import capture_environment
            environment = capture_environment()
        code_hash = sha256_json({k: code.get(k) for k in CODE_HASH_KEYS}) if code else None
        data_fp = sha256_json(content) if content else None
        spec = {"entry_point": entry_point, "command": command, "code_hash": code_hash, "params": params,
                "seed": seed, "data": content, "periods": periods.to_dict(),
                "holdout": holdout_obj.id if holdout_obj else None, "holdout_opened": bool(open_holdout)}
        spec_hash = sha256_json(spec)
        self.ensure_family(family)
        with db.transaction(self.conn):
            eid = self._next_id("EXP", 4)
            now = started_at or self.now()
            self.conn.execute(
                "INSERT INTO experiments(id, name, project, hypothesis_id, hypothesis_version, family, model,"
                " model_version, parent_id, status, source, researcher, created_at, started_at, entry_point,"
                " command, params_json, params_hash, seed, code_json, code_hash, environment_json, train_start,"
                " train_end, test_start, test_end, holdout_id, holdout_opened, periods_per_year, spec_hash,"
                " data_fingerprint) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (eid, name, project, hypothesis, hyp_version, family, model, model_version, parent, "running",
                 source, self.researcher, now, now, entry_point, command, canonical_json(params),
                 sha256_json(params), seed, _dumps(code), code_hash, _dumps(environment),
                 periods.train[0] if periods.train else None, periods.train[1] if periods.train else None,
                 periods.test[0] if periods.test else None, periods.test[1] if periods.test else None,
                 holdout_obj.id if holdout_obj else None, int(bool(open_holdout)), periods_per_year, spec_hash,
                 data_fp))
            for role, vid in data.items():
                self.conn.execute("INSERT INTO experiment_data(experiment_id, role, version_id) VALUES(?,?,?)",
                                  (eid, role, vid))
            for tag in tags:
                self.conn.execute("INSERT OR IGNORE INTO tags(experiment_id, tag, created_at) VALUES(?,?,?)",
                                  (eid, tag, now))
            self._event("experiment.started", "experiment", eid, {
                "name": name, "hypothesis": hypothesis, "hypothesis_version": hyp_version, "family": family,
                "spec_hash": spec_hash, "params": params, "seed": seed, "data": data, "periods": periods.to_dict(),
                "holdout": holdout_obj.id if holdout_obj else None, "open_holdout": bool(open_holdout),
                "parent": parent, "model": model, "model_version": model_version, "source": source}, at=now)
            if notes:
                self.conn.execute("INSERT INTO notes(experiment_id, author, text, created_at) VALUES(?,?,?,?)",
                                  (eid, self.researcher, notes, now))
        if holdout_obj and open_holdout:
            self.touch_holdout(holdout_obj.id, eid, holdout_purpose or "declared holdout evaluation")
        return self.get_experiment(eid)

    def log_metrics(self, experiment_id: str, metrics: dict[str, float], scope: str = "custom") -> None:
        with db.transaction(self.conn):
            for name, value in metrics.items():
                v = None if value is None else float(value)
                if v is not None and v != v:  # NaN
                    v = None
                self.conn.execute(
                    "INSERT INTO metrics(experiment_id, scope, name, value) VALUES(?,?,?,?)"
                    " ON CONFLICT(experiment_id, scope, name) DO UPDATE SET value = excluded.value",
                    (experiment_id, scope, name, v))

    def log_artifact(self, experiment_id: str, name: str, obj: Any, *, kind: str = "file",
                     media_type: str | None = None, meta: dict[str, Any] | None = None) -> str:
        """Store bytes, a file path, a DataFrame/Series or a JSON-able object; returns its SHA-256."""
        if isinstance(obj, pd.Series):
            obj = obj.to_frame(name=obj.name if obj.name is not None else "value")
        if isinstance(obj, pd.DataFrame):
            payload, storage = frame_to_bytes(obj)
            media_type = media_type or ("application/vnd.apache.parquet" if storage == "parquet"
                                        else "application/x-qros-qcsv")
            meta = {**(meta or {}), "storage": storage, "content_hash": fingerprint_frame(obj)}
        elif isinstance(obj, (bytes, bytearray)):
            payload = bytes(obj)
        elif isinstance(obj, (str, Path)) and Path(obj).is_file():
            payload = Path(obj).read_bytes()
            meta = {**(meta or {}), "filename": Path(obj).name}
        else:
            payload = canonical_json(obj).encode("utf-8")
            media_type = media_type or "application/json"
        sha = self.objects.put_bytes(payload)
        with db.transaction(self.conn):
            now = self.now()
            self.conn.execute(
                "INSERT INTO artifacts(experiment_id, name, kind, sha256, size, media_type, meta_json, created_at)"
                " VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(experiment_id, name) DO UPDATE SET kind = excluded.kind,"
                " sha256 = excluded.sha256, size = excluded.size, media_type = excluded.media_type,"
                " meta_json = excluded.meta_json, created_at = excluded.created_at",
                (experiment_id, name, kind, sha, len(payload), media_type, _dumps(meta), now))
            self._event("artifact.logged", "experiment", experiment_id, {"name": name, "kind": kind,
                                                                        "sha256": sha, "size": len(payload)}, at=now)
        return sha

    def artifacts(self, experiment_id: str) -> list[dict[str, Any]]:
        out = []
        for row in self._all("SELECT * FROM artifacts WHERE experiment_id = ? ORDER BY id", (experiment_id,)):
            item = dict(row)
            item["meta"] = _loads(item.pop("meta_json"), {})
            out.append(item)
        return out

    def load_artifact(self, experiment_id: str, name: str) -> Any:
        row = self._one("SELECT * FROM artifacts WHERE experiment_id = ? AND name = ?", (experiment_id, name))
        if row is None:
            raise RegistryError(f"{experiment_id} has no artifact {name!r}")
        payload = self.objects.get_bytes(row["sha256"])
        meta = _loads(row["meta_json"], {})
        if meta.get("storage"):
            return frame_from_bytes(payload, meta["storage"])
        if row["media_type"] == "application/json":
            return json.loads(payload.decode("utf-8"))
        return payload

    def result_frame(self, experiment_id: str) -> pd.DataFrame:
        """The stored per-period result table (net, gross, turnover, costs, exposure columns)."""
        return self.load_artifact(experiment_id, "result")

    def returns(self, experiment_id: str, column: str = "net") -> pd.Series:
        """A completed experiment's per-period returns (cached: finished results are immutable)."""
        key = (experiment_id, column)
        if key in self._returns_cache:
            return self._returns_cache[key].copy()
        frame = self.result_frame(experiment_id)
        series = frame[column].astype(float)
        series.name = experiment_id
        self._returns_cache[key] = series
        return series.copy()

    def complete_experiment(self, experiment_id: str, result: Any, *, metrics: dict[str, float] | None = None,
                            validate: bool = True, finished_at: str | None = None) -> Experiment:
        """Record the result, compute standard metrics, validation and hygiene, and freeze the record."""
        from qros.evaluate import finalize
        return finalize(self, experiment_id, result, extra_metrics=metrics, validate=validate,
                        finished_at=finished_at)

    def fail_experiment(self, experiment_id: str, error: str) -> Experiment:
        with db.transaction(self.conn):
            now = self.now()
            self.conn.execute("UPDATE experiments SET status = 'failed', error = ?, finished_at = ? WHERE id = ?"
                              " AND status IN ('planned','running')", (error[-4000:], now, experiment_id))
            self._event("experiment.failed", "experiment", experiment_id, {"error": error[-1000:]}, at=now)
        return self.get_experiment(experiment_id)

    def invalidate_experiment(self, experiment_id: str, reason: str) -> Experiment:
        with db.transaction(self.conn):
            self.conn.execute("UPDATE experiments SET status = 'invalidated', verdict = 'invalid' WHERE id = ?",
                              (experiment_id,))
            self._event("experiment.invalidated", "experiment", experiment_id, {"reason": reason})
            self.conn.execute("INSERT INTO notes(experiment_id, author, text, created_at) VALUES(?,?,?,?)",
                              (experiment_id, self.researcher, f"Invalidated: {reason}", self.now()))
        return self.get_experiment(experiment_id)

    def get_experiment(self, experiment_id: str) -> Experiment:
        row = self._one("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        if row is None:
            raise RegistryError(f"Unknown experiment {experiment_id}")
        exp = Experiment.from_row(row)
        exp.data = {r["role"]: r["version_id"] for r in self._all(
            "SELECT role, version_id FROM experiment_data WHERE experiment_id = ? ORDER BY role", (experiment_id,))}
        exp.tags = [r["tag"] for r in self._all("SELECT tag FROM tags WHERE experiment_id = ? ORDER BY tag",
                                                (experiment_id,))]
        metrics: dict[str, dict[str, float]] = {}
        for r in self._all("SELECT scope, name, value FROM metrics WHERE experiment_id = ?", (experiment_id,)):
            metrics.setdefault(r["scope"], {})[r["name"]] = r["value"]
        exp.metrics = metrics
        return exp

    def list_experiments(self, *, family: str | None = None, hypothesis: str | None = None, tag: str | None = None,
                         status: str | None = None, project: str | None = None) -> list[Experiment]:
        sql = "SELECT e.id FROM experiments e"
        where, args = [], []
        if tag:
            sql += " JOIN tags t ON t.experiment_id = e.id"
            where.append("t.tag = ?")
            args.append(tag)
        for col, value in (("family", family), ("hypothesis_id", hypothesis), ("status", status),
                           ("project", project)):
            if value is not None:
                where.append(f"e.{col} = ?")
                args.append(value)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY e.id"
        return [self.get_experiment(r["id"]) for r in self._all(sql, args)]

    def tag(self, experiment_id: str, *tags: str) -> None:
        self.get_experiment(experiment_id)
        with db.transaction(self.conn):
            now = self.now()
            for t in tags:
                self.conn.execute("INSERT OR IGNORE INTO tags(experiment_id, tag, created_at) VALUES(?,?,?)",
                                  (experiment_id, t, now))
            self._event("experiment.tagged", "experiment", experiment_id, {"tags": list(tags)}, at=now)

    def untag(self, experiment_id: str, *tags: str) -> None:
        with db.transaction(self.conn):
            for t in tags:
                self.conn.execute("DELETE FROM tags WHERE experiment_id = ? AND tag = ?", (experiment_id, t))
            self._event("experiment.untagged", "experiment", experiment_id, {"tags": list(tags)})

    def add_note(self, text: str, *, experiment: str | None = None, hypothesis: str | None = None) -> int:
        if not experiment and not hypothesis:
            raise ValueError("A note belongs to an experiment or a hypothesis")
        with db.transaction(self.conn):
            now = self.now()
            cur = self.conn.execute(
                "INSERT INTO notes(experiment_id, hypothesis_id, author, text, created_at) VALUES(?,?,?,?,?)",
                (experiment, hypothesis, self.researcher, text, now))
            self._event("note.added", "experiment" if experiment else "hypothesis", experiment or hypothesis,
                        {"text": text}, at=now)
        return int(cur.lastrowid)

    def notes(self, *, experiment: str | None = None, hypothesis: str | None = None) -> list[dict[str, Any]]:
        if experiment:
            rows = self._all("SELECT * FROM notes WHERE experiment_id = ? ORDER BY id", (experiment,))
        elif hypothesis:
            rows = self._all("SELECT * FROM notes WHERE hypothesis_id = ? ORDER BY id", (hypothesis,))
        else:
            rows = self._all("SELECT * FROM notes ORDER BY id")
        return [dict(r) for r in rows]

    def set_verdict(self, experiment_id: str, verdict: str | None, summary: dict[str, Any] | None = None) -> None:
        with db.transaction(self.conn):
            if summary is None:
                self.conn.execute("UPDATE experiments SET verdict = ? WHERE id = ?", (verdict, experiment_id))
            else:
                self.conn.execute("UPDATE experiments SET verdict = ?, summary_json = ? WHERE id = ?",
                                  (verdict, canonical_json(summary), experiment_id))

    # ------------------------------------------------------------------ validation, warnings, reproductions

    def store_validation(self, experiment_id: str, report: dict[str, Any]) -> int:
        text = canonical_json(report)
        with db.transaction(self.conn):
            now = self.now()
            cur = self.conn.execute(
                "INSERT INTO validations(experiment_id, computed_at, n_trials, verdict, report_json, report_sha256)"
                " VALUES(?,?,?,?,?,?)", (experiment_id, now, report.get("n_trials"), report.get("verdict"), text,
                                         sha256_text(text)))
            self.conn.execute("UPDATE experiments SET verdict = ? WHERE id = ?", (report.get("verdict"),
                                                                                 experiment_id))
            self._event("validation.stored", "experiment", experiment_id, {
                "verdict": report.get("verdict"), "n_trials": report.get("n_trials"),
                "report_sha256": sha256_text(text)}, at=now)
        return int(cur.lastrowid)

    def latest_validation(self, experiment_id: str) -> dict[str, Any] | None:
        row = self._one("SELECT * FROM validations WHERE experiment_id = ? ORDER BY id DESC LIMIT 1",
                        (experiment_id,))
        if row is None:
            return None
        report = json.loads(row["report_json"])
        report["computed_at"] = row["computed_at"]
        return report

    def validation_history(self, experiment_id: str) -> list[dict[str, Any]]:
        out = []
        for row in self._all("SELECT id, computed_at, n_trials, verdict, report_json FROM validations"
                             " WHERE experiment_id = ? ORDER BY id", (experiment_id,)):
            report = json.loads(row["report_json"])
            sig = report.get("significance", {})
            out.append({"computed_at": row["computed_at"], "n_trials": row["n_trials"], "verdict": row["verdict"],
                        "dsr": sig.get("dsr"), "n_eff": sig.get("n_eff")})
        return out

    def upsert_warning(self, scope_key: str, code: str, severity: str, message: str, *,
                       details: dict[str, Any] | None = None, experiment_id: str | None = None,
                       hypothesis_id: str | None = None, family: str | None = None) -> str:
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")
        with db.transaction(self.conn):
            now = self.now()
            row = self._one("SELECT id, active, severity, message FROM warnings WHERE scope_key = ? AND code = ?",
                            (scope_key, code))
            if row:
                changed = (not row["active"]) or row["severity"] != severity or row["message"] != message
                self.conn.execute(
                    "UPDATE warnings SET severity = ?, message = ?, details_json = ?, active = 1, updated_at = ?"
                    " WHERE id = ?", (severity, message, _dumps(details), now, row["id"]))
                if changed:
                    self._event("warning.updated", "warning", row["id"], {"code": code, "severity": severity,
                                                                         "scope": scope_key}, at=now)
                return row["id"]
            wid = self._next_id("W", 5)
            self.conn.execute(
                "INSERT INTO warnings(id, scope_key, experiment_id, hypothesis_id, family, code, severity, message,"
                " details_json, active, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,1,?,?)",
                (wid, scope_key, experiment_id, hypothesis_id, family, code, severity, message, _dumps(details),
                 now, now))
            self._event("warning.raised", "warning", wid, {"code": code, "severity": severity, "scope": scope_key,
                                                          "message": message}, at=now)
            return wid

    def deactivate_warnings(self, scope_key: str, keep_codes: Iterable[str]) -> int:
        keep = set(keep_codes)
        rows = self._all("SELECT id, code FROM warnings WHERE scope_key = ? AND active = 1", (scope_key,))
        stale = [r["id"] for r in rows if r["code"] not in keep]
        if stale:
            with db.transaction(self.conn):
                now = self.now()
                for wid in stale:
                    self.conn.execute("UPDATE warnings SET active = 0, updated_at = ? WHERE id = ?", (now, wid))
                    self._event("warning.cleared", "warning", wid, {"scope": scope_key}, at=now)
        return len(stale)

    def warnings(self, *, experiment: str | None = None, family: str | None = None, hypothesis: str | None = None,
                 severity: str | None = None, active: bool | None = True) -> list[dict[str, Any]]:
        where, args = [], []
        if experiment:
            where.append("experiment_id = ?")
            args.append(experiment)
        if family:
            where.append("family = ?")
            args.append(family)
        if hypothesis:
            where.append("hypothesis_id = ?")
            args.append(hypothesis)
        if severity:
            where.append("severity = ?")
            args.append(severity)
        if active is not None:
            where.append("active = ?")
            args.append(int(active))
        sql = "SELECT * FROM warnings" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id"
        out = []
        for row in self._all(sql, args):
            item = dict(row)
            item["details"] = _loads(item.pop("details_json"), {})
            item["active"] = bool(item["active"])
            out.append(item)
        return out

    def acknowledge_warning(self, warning_id: str, note: str) -> None:
        with db.transaction(self.conn):
            now = self.now()
            cur = self.conn.execute("UPDATE warnings SET acknowledged_at = ?, acknowledged_note = ? WHERE id = ?",
                                    (now, note, warning_id))
            if cur.rowcount == 0:
                raise RegistryError(f"Unknown warning {warning_id}")
            self._event("warning.acknowledged", "warning", warning_id, {"note": note}, at=now)

    def record_reproduction(self, experiment_id: str, status: str, *, result_hash: str | None,
                            max_abs_diff: float | None, environment: dict[str, Any] | None = None,
                            note: str | None = None) -> dict[str, Any]:
        with db.transaction(self.conn):
            now = self.now()
            self.conn.execute(
                "INSERT INTO reproductions(experiment_id, run_at, status, result_hash, max_abs_diff,"
                " environment_json, note) VALUES(?,?,?,?,?,?,?)",
                (experiment_id, now, status, result_hash, max_abs_diff, _dumps(environment), note))
            self._event("experiment.reproduced", "experiment", experiment_id, {
                "status": status, "result_hash": result_hash, "max_abs_diff": max_abs_diff}, at=now)
        return {"experiment": experiment_id, "status": status, "result_hash": result_hash,
                "max_abs_diff": max_abs_diff, "run_at": now, "note": note}

    def reproductions(self, experiment_id: str | None = None) -> list[dict[str, Any]]:
        if experiment_id:
            rows = self._all("SELECT * FROM reproductions WHERE experiment_id = ? ORDER BY id", (experiment_id,))
        else:
            rows = self._all("SELECT * FROM reproductions ORDER BY id")
        out = []
        for row in rows:
            item = dict(row)
            item["environment"] = _loads(item.pop("environment_json"))
            out.append(item)
        return out

    # ------------------------------------------------------------------ audit

    def events(self, *, entity: str | None = None, entity_id: str | None = None,
               limit: int | None = None) -> list[dict[str, Any]]:
        where, args = [], []
        if entity:
            where.append("entity = ?")
            args.append(entity)
        if entity_id:
            where.append("entity_id = ?")
            args.append(entity_id)
        sql = "SELECT * FROM events" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY seq"
        if limit:
            sql += f" LIMIT {int(limit)}"
        out = []
        for row in self._all(sql, args):
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            out.append(item)
        return out

    def verify(self, deep: bool = True) -> dict[str, Any]:
        """Check the audit hash chain and (deep) every stored object and dataset version."""
        chain = db.verify_events(self.conn)
        bad_objects, bad_versions = [], []
        if deep:
            for row in self._all("SELECT DISTINCT sha256 FROM artifacts"):
                if not self.objects.verify(row["sha256"]):
                    bad_objects.append(row["sha256"])
            for row in self._all("SELECT id FROM dataset_versions"):
                res = self.verify_version(row["id"])
                if res["ok"] is False:
                    bad_versions.append(res)
        return {"audit_chain": chain, "bad_objects": bad_objects, "bad_versions": bad_versions,
                "ok": chain["ok"] and not bad_objects and not bad_versions}

    def stats(self) -> dict[str, Any]:
        q = lambda sql: self._one(sql)[0]  # noqa: E731
        return {
            "registry_id": self.registry_id,
            "experiments": q("SELECT COUNT(*) FROM experiments"),
            "completed": q("SELECT COUNT(*) FROM experiments WHERE status = 'completed'"),
            "hypotheses": q("SELECT COUNT(*) FROM hypotheses"),
            "families": q("SELECT COUNT(*) FROM families"),
            "datasets": q("SELECT COUNT(*) FROM datasets"),
            "dataset_versions": q("SELECT COUNT(*) FROM dataset_versions"),
            "warnings_active": q("SELECT COUNT(*) FROM warnings WHERE active = 1"),
            "critical_active": q("SELECT COUNT(*) FROM warnings WHERE active = 1 AND severity = 'critical'"),
            "holdout_touches": q("SELECT COUNT(*) FROM holdout_access"),
            "events": q("SELECT COUNT(*) FROM events"),
        }
