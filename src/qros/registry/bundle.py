"""Experiment serialisation: self-contained bundles that move between registries.

A bundle is a zip file with ``bundle.json`` (the experiment record, its hypothesis, data versions,
metrics, validation reports, reproductions and audit events) and ``objects/<sha256>`` (artifacts and,
optionally, the dataset versions themselves). Importing re-computes the result from the stored return
series and refuses the bundle if the recomputed result hash differs from the exported one.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qros._version import __version__
from qros.artifacts import frame_from_bytes
from qros.backtest.interface import BacktestResult
from qros.registry import db
from qros.registry.models import Periods
from qros.util.hashing import canonical_json, sha256_bytes

if TYPE_CHECKING:  # pragma: no cover
    from qros.registry.core import Registry

BUNDLE_FORMAT = "qros.experiment-bundle/1"


def _row(reg: "Registry", sql: str, args: tuple) -> dict[str, Any] | None:
    r = reg.conn.execute(sql, args).fetchone()
    return dict(r) if r else None


def export_experiment(reg: "Registry", experiment_id: str, path: str | Path, *, include_data: bool = False) -> Path:
    exp = reg.get_experiment(experiment_id)
    if exp.status != "completed":
        raise ValueError("only completed experiments can be exported")
    row = _row(reg, "SELECT * FROM experiments WHERE id = ?", (experiment_id,))
    objects: dict[str, bytes] = {}
    data = []
    for role, vid in exp.data.items():
        version = reg.get_version(vid)
        ds = reg.get_dataset(version.dataset_id)
        src = _row(reg, "SELECT * FROM sources WHERE id = ?", (ds["source_id"],)) if ds.get("source_id") else None
        data.append({"role": role, "version": version.to_dict(), "dataset": ds, "source": src})
        if include_data and version.object_sha256:
            objects[version.object_sha256] = reg.objects.get_bytes(version.object_sha256)
    hyp = reg.get_hypothesis(exp.hypothesis_id).to_dict() if exp.hypothesis_id else None
    holdout = reg.get_holdout(exp.holdout_id).to_dict() if exp.holdout_id else None
    arts = reg.artifacts(experiment_id)
    for a in arts:
        objects[a["sha256"]] = reg.objects.get_bytes(a["sha256"])
    validations = [dict(r) for r in reg.conn.execute(
        "SELECT computed_at, n_trials, verdict, report_json, report_sha256 FROM validations WHERE experiment_id = ?"
        " ORDER BY id", (experiment_id,))]
    bundle = {
        "format": BUNDLE_FORMAT, "qros": __version__, "exported_at": reg.now(), "registry_id": reg.registry_id,
        "experiment": row, "metrics": [dict(r) for r in reg.conn.execute(
            "SELECT scope, name, value FROM metrics WHERE experiment_id = ?", (experiment_id,))],
        "tags": exp.tags, "notes": reg.notes(experiment=experiment_id), "data": data, "hypothesis": hyp,
        "holdout": holdout, "holdout_accesses": [a for a in reg.holdout_accesses(exp.holdout_id) if a["experiment_id"] == experiment_id]
        if exp.holdout_id else [], "validations": validations, "reproductions": reg.reproductions(experiment_id),
        "events": reg.events(entity_id=experiment_id), "artifacts": arts, "include_data": include_data,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo("bundle.json", date_time=(2020, 1, 1, 0, 0, 0))
        zf.writestr(info, canonical_json(bundle))
        for sha, payload in sorted(objects.items()):
            zf.writestr(zipfile.ZipInfo(f"objects/{sha}", date_time=(2020, 1, 1, 0, 0, 0)), payload)
    with db.transaction(reg.conn):
        reg._event("experiment.exported", "experiment", experiment_id,
                   {"bundle_sha256": sha256_bytes(path.read_bytes()), "include_data": include_data})
    return path


def import_bundle(reg: "Registry", path: str | Path, *, validate: bool = True) -> str:
    """Import a bundle; returns the new experiment id in this registry."""
    with zipfile.ZipFile(path) as zf:
        bundle = json.loads(zf.read("bundle.json").decode("utf-8"))
        if bundle.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"not a qros experiment bundle: {bundle.get('format')!r}")
        objects = {}
        for name in zf.namelist():
            if name.startswith("objects/"):
                payload = zf.read(name)
                sha = name.split("/", 1)[1]
                if sha256_bytes(payload) != sha:
                    raise ValueError(f"bundle object {sha[:12]} is corrupt")
                objects[sha] = payload
    for payload in objects.values():
        reg.objects.put_bytes(payload)
    src_exp = bundle["experiment"]

    # hypothesis: reuse an identical one, otherwise import it under a new id
    hyp_id = None
    if bundle.get("hypothesis"):
        h = bundle["hypothesis"]
        current = next(v for v in h["versions"] if v["version"] == h["current_version"])
        for existing in reg.list_hypotheses():
            if existing.current.text_sha256 == current["text_sha256"]:
                hyp_id = existing.id
                break
        if hyp_id is None:
            created = reg.add_hypothesis(h["title"], current["statement"], rationale=current.get("rationale"),
                                         prediction=current.get("prediction"), criteria=current.get("criteria"),
                                         family=h.get("family"))
            hyp_id = created.id
            reg.add_note(f"Imported from registry {bundle['registry_id']} ({h['id']}); originally registered at "
                         f"{h.get('registered_at') or 'never'}. qros records its own registration time.", hypothesis=hyp_id)

    # data versions
    data_ids = {}
    for d in bundle["data"]:
        ds_meta, version = d["dataset"], d["version"]
        source = None
        if d.get("source"):
            s = d["source"]
            source = reg.add_source(s["name"], vendor=s.get("vendor"), url=s.get("url"), license=s.get("license"),
                                    description=s.get("description"))
        ds = reg.add_dataset(ds_meta["name"], source=source, kind=ds_meta.get("kind", "table"),
                             description=ds_meta.get("description"), entity_col=ds_meta.get("entity_col"),
                             time_col=ds_meta.get("time_col"), available_col=ds_meta.get("available_col"),
                             frequency=ds_meta.get("frequency"), survivorship_free=ds_meta.get("survivorship_free"),
                             adjusted=ds_meta.get("adjusted"), licensed=bool(ds_meta.get("licensed")))
        if version.get("object_sha256") and version["object_sha256"] in objects and version["storage"] != "file":
            frame = frame_from_bytes(objects[version["object_sha256"]], version["storage"])
            new = reg.snapshot(ds, frame, label=version.get("label"), knowledge_time=version.get("knowledge_time"),
                               notes=version.get("notes"))
            if new.content_hash != version["content_hash"]:
                raise ValueError(f"dataset version {version['id']} does not match its content hash")
        else:
            new = reg.register_external_version(ds, version["content_hash"], label=version.get("label"),
                                                knowledge_time=version.get("knowledge_time"),
                                                notes=f"imported reference to {version['id']}")
        data_ids[d["role"]] = new.id

    holdout_id = None
    if bundle.get("holdout"):
        ho = bundle["holdout"]
        try:
            holdout_id = reg.get_holdout(ho["name"]).id
        except Exception:
            holdout_id = reg.add_holdout(ho["name"], ho["start"], ho["end"], description=ho.get("description"),
                                         max_touches=ho.get("max_touches", 1)).id
    exp = reg.create_experiment(
        src_exp["name"], hypothesis=hyp_id, family=src_exp.get("family"), project=src_exp.get("project"),
        model=src_exp.get("model"), model_version=src_exp.get("model_version"),
        params=json.loads(src_exp["params_json"] or "{}"), seed=src_exp.get("seed"), entry_point=src_exp.get("entry_point"),
        command=src_exp.get("command"), code=json.loads(src_exp["code_json"]) if src_exp.get("code_json") else None,
        environment=json.loads(src_exp["environment_json"]) if src_exp.get("environment_json") else None,
        periods=Periods.of((src_exp["train_start"], src_exp["train_end"]) if src_exp.get("train_start") else None,
                           (src_exp["test_start"], src_exp["test_end"]) if src_exp.get("test_start") else None),
        holdout=holdout_id, open_holdout=bool(src_exp.get("holdout_opened")),
        holdout_purpose=f"imported: {src_exp['id']} from registry {bundle['registry_id']}", data=data_ids,
        tags=bundle.get("tags") or [], source="ingested", periods_per_year=src_exp.get("periods_per_year"),
        notes=f"Imported from registry {bundle['registry_id']} ({src_exp['id']}, finished {src_exp.get('finished_at')}).")
    result_art = next(a for a in bundle["artifacts"] if a["name"] == "result")
    frame = frame_from_bytes(objects[result_art["sha256"]], result_art["meta"]["storage"])
    for a in bundle["artifacts"]:
        if a["name"] not in ("result",):
            reg.log_artifact(exp.id, a["name"], objects[a["sha256"]], kind=a["kind"], media_type=a.get("media_type"),
                             meta=a.get("meta"))
    custom = {m["name"]: m["value"] for m in bundle["metrics"] if m["scope"] == "custom" and m["value"] is not None}
    done = reg.complete_experiment(exp.id, BacktestResult.from_frame(frame, periods_per_year=src_exp.get("periods_per_year")),
                                   metrics=custom or None, validate=validate)
    if done.result_hash != src_exp["result_hash"]:
        reg.invalidate_experiment(exp.id, "the imported result does not reproduce the exported result hash")
        raise ValueError("result hash mismatch after import; the experiment was marked invalid")
    return exp.id
