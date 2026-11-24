"""The language-agnostic run manifest: how any repository (Python, Rust, anything) reports a run.

A project writes ``qros-run.json`` next to its outputs (schema: ``schemas/run-manifest.schema.json``)::

    {
      "schema": "qros.run-manifest/1",
      "project": "hft-sim",
      "experiment": {"name": "mm ladder step 7", "family": "rq1", "hypothesis": {"id": "H-003"}},
      "code": {"repo": "hft-sim", "commit": "c9ae4d0", "dirty": false, "language": "rust",
               "command": ["target/release/hftsim", "run", "--seed", "7"], "cwd": "."},
      "params": {"mm_gamma": 0.5}, "seed": 7,
      "periods": {"train": ["2019-01-01", "2019-06-30"], "test": ["2019-07-01", "2019-12-31"]},
      "data": [{"role": "book", "dataset": "bitstamp-btcusd", "path": "data/l3.jsonl", "sha256": "..."}],
      "results": {"returns": {"path": "returns.csv", "sha256": "...", "date_column": "date",
                              "columns": {"net": "net"}}, "metrics": {"pnl": 12.5},
                  "digest": {"value": "0xd130e7689894d0af", "from": "stdout-json", "json_path": "manifest.digest"}},
      "started_at": "2026-11-20T10:00:00Z", "finished_at": "2026-11-20T10:05:00Z"
    }

``qros ingest qros-run.json`` verifies every declared hash, registers the data versions, creates the
experiment, recomputes all metrics with the same code as native runs, and validates it. Nothing from
the other repository is imported.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from qros.backtest.interface import BacktestResult
from qros.registry.models import Periods
from qros.util.hashing import canonical_json, sha256_bytes, sha256_file, sha256_text
from qros.util.timeutil import parse_iso, to_iso

if TYPE_CHECKING:  # pragma: no cover
    from qros.registry.core import Registry
    from qros.registry.models import Experiment

SCHEMA_ID = "qros.run-manifest/1"


class ManifestError(ValueError):
    pass


def load_manifest(path: str | Path) -> tuple[dict[str, Any], Path]:
    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        manifest = json.load(fh)
    return manifest, p.parent


def validate_manifest(m: dict[str, Any]) -> list[str]:
    errors = []
    if m.get("schema") != SCHEMA_ID:
        errors.append(f"schema must be {SCHEMA_ID!r}")
    exp = m.get("experiment")
    if not isinstance(exp, dict) or not exp.get("name"):
        errors.append("experiment.name is required")
    results = m.get("results")
    if not isinstance(results, dict):
        errors.append("results is required")
    else:
        ret = results.get("returns")
        if ret is not None and not isinstance(ret, dict):
            errors.append("results.returns must be an object")
        if ret is None and not results.get("metrics"):
            errors.append("results needs returns or metrics")
        if isinstance(ret, dict) and not (ret.get("path") or ret.get("frame")):
            errors.append("results.returns needs a path")
    for i, d in enumerate(m.get("data") or []):
        if not isinstance(d, dict) or not d.get("dataset"):
            errors.append(f"data[{i}].dataset is required")
        elif not (d.get("path") or d.get("sha256")):
            errors.append(f"data[{i}] needs a path or a sha256")
    if "params" in m and not isinstance(m["params"], dict):
        errors.append("params must be an object")
    if m.get("seed") is not None and not isinstance(m["seed"], int):
        errors.append("seed must be an integer")
    periods = m.get("periods") or {}
    for key in ("train", "test"):
        w = periods.get(key)
        if w is not None and (not isinstance(w, (list, tuple)) or len(w) != 2):
            errors.append(f"periods.{key} must be [start, end]")
    return errors


def _check_hash(path: Path, declared: str | None, what: str) -> str:
    actual = sha256_file(path)
    if declared and declared.lower() != actual:
        raise ManifestError(f"{what}: declared sha256 {declared[:12]} does not match the file ({actual[:12]})")
    return actual


def read_returns(spec: dict[str, Any], base: Path, frames: dict[str, pd.DataFrame] | None = None,
                 verify: bool = True) -> tuple[BacktestResult, str | None]:
    if spec.get("frame"):
        frame = (frames or {})[spec["frame"]].copy()
        file_hash = None
    else:
        path = (base / spec["path"]).resolve()
        file_hash = _check_hash(path, spec.get("sha256") if verify else None, "results.returns")
        frame = pd.read_csv(path) if path.suffix.lower() != ".parquet" else pd.read_parquet(path)
    date_col = spec.get("date_column", "date")
    if date_col in frame.columns:
        frame[date_col] = pd.to_datetime(frame[date_col])
        frame = frame.set_index(date_col)
    frame.index = pd.DatetimeIndex(frame.index).astype("datetime64[ns]")
    frame = frame.sort_index()
    standard = ("net", "gross", "cost", "turnover", "benchmark", "gross_exposure", "net_exposure")
    cols = spec.get("columns") or {c: c for c in standard if c in frame.columns} or {"net": "net"}
    out = pd.DataFrame(index=frame.index)
    for target, source in cols.items():
        if source in frame.columns:
            out[target] = pd.to_numeric(frame[source], errors="coerce")
    if "net" not in out.columns:
        raise ManifestError("results.returns: no 'net' column (map it with results.returns.columns)")
    if spec.get("kind") == "pnl":
        capital = float(spec.get("capital") or 0)
        if capital <= 0:
            raise ManifestError("results.returns.kind = 'pnl' needs a positive capital")
        out = out.apply(lambda s: s / capital if s.name in ("net", "gross", "cost") else s)
    out = out.dropna(subset=["net"])
    if "gross" in out.columns and "cost" not in out.columns:
        out["cost"] = out["gross"] - out["net"]
    if "cost" in out.columns and "gross" not in out.columns:
        out["gross"] = out["net"] + out["cost"]
    result = BacktestResult.from_frame(out, periods_per_year=spec.get("periods_per_year"))
    return result, file_hash


def ingest_manifest(reg: "Registry", manifest: dict[str, Any] | str | Path, *, base_dir: str | Path | None = None,
                    frames: dict[str, pd.DataFrame] | None = None, verify_hashes: bool = True,
                    validate: bool = True) -> "Experiment":
    if not isinstance(manifest, dict):
        manifest, base = load_manifest(manifest)
    else:
        base = Path(base_dir or ".")
    errors = validate_manifest(manifest)
    if errors:
        raise ManifestError("; ".join(errors))
    m = manifest
    project = m.get("project")
    exp_spec = m["experiment"]

    # data versions ------------------------------------------------------------------------------
    data_ids: dict[str, str] = {}
    for i, d in enumerate(m.get("data") or []):
        source = None
        if d.get("source"):
            source = reg.add_source(d["source"], license=d.get("license"), url=d.get("url"))
        ds = reg.add_dataset(d["dataset"], source=source, kind=d.get("kind", "file"),
                             survivorship_free=d.get("survivorship_free"), adjusted=d.get("adjusted"),
                             licensed=bool(d.get("licensed", False)), description=d.get("description"))
        path = (base / d["path"]).resolve() if d.get("path") else None
        if path is not None and path.is_file():
            actual = _check_hash(path, d.get("sha256") if verify_hashes else None, f"data[{i}]")
            if d.get("store", True) and path.stat().st_size <= int(d.get("max_store_bytes", 200_000_000)):
                version = reg.snapshot(ds, path, label=d.get("label"), knowledge_time=d.get("knowledge_time"),
                                       notes=d.get("notes"))
            else:
                version = reg.register_external_version(ds, actual, label=d.get("label"),
                                                        knowledge_time=d.get("knowledge_time"), notes=d.get("notes"),
                                                        size=path.stat().st_size)
        elif d.get("sha256"):
            version = reg.register_external_version(ds, d["sha256"].lower(), label=d.get("label"),
                                                    knowledge_time=d.get("knowledge_time"), notes=d.get("notes"))
        else:
            raise ManifestError(f"data[{i}]: {d.get('path')} not found and no sha256 declared")
        data_ids[d.get("role") or f"input{i}"] = version.id

    # hypothesis ---------------------------------------------------------------------------------
    hyp_id = None
    hyp = exp_spec.get("hypothesis")
    if isinstance(hyp, dict) and hyp.get("id"):
        hyp_id = reg.get_hypothesis(hyp["id"]).id
    elif isinstance(hyp, dict) and hyp.get("statement"):
        existing = [h for h in reg.list_hypotheses()
                    if h.title == hyp.get("title", hyp["statement"][:80]) and h.statement == hyp["statement"]]
        if existing:
            hyp_id = existing[0].id
        else:
            created = reg.add_hypothesis(hyp.get("title", hyp["statement"][:80]), hyp["statement"],
                                         prediction=hyp.get("prediction"), criteria=hyp.get("criteria"),
                                         family=exp_spec.get("family"), rationale=hyp.get("rationale"))
            hyp_id = created.id
            claimed = hyp.get("registered_at_claimed")
            if claimed:
                reg.add_note(f"Imported from {project or 'an external project'}; the source claims this hypothesis "
                             f"was pre-registered on {claimed}. qros cannot verify that date.", hypothesis=hyp_id)
            if hyp.get("register", True):
                reg.register_hypothesis(hyp_id)

    # holdout ------------------------------------------------------------------------------------
    holdout_id, open_holdout, purpose = None, False, None
    ho = m.get("holdout")
    if isinstance(ho, dict) and ho.get("name"):
        try:
            holdout_id = reg.get_holdout(ho["name"]).id
        except Exception:
            holdout_id = reg.add_holdout(ho["name"], ho["start"], ho["end"], description=ho.get("description"),
                                         max_touches=int(ho.get("max_touches", 1))).id
        open_holdout = bool(ho.get("opened", False))
        purpose = ho.get("purpose")

    # experiment ---------------------------------------------------------------------------------
    code = dict(m.get("code") or {})
    if isinstance(code.get("command"), list):
        code["command"] = [str(c) for c in code["command"]]
    started = m.get("started_at")
    started_iso = to_iso(parse_iso(started)) if started else None
    periods = m.get("periods") or {}
    env = m.get("environment") or {"declared": False}
    exp = reg.create_experiment(
        exp_spec["name"], hypothesis=hyp_id, family=exp_spec.get("family"), project=project,
        model=exp_spec.get("model"), model_version=exp_spec.get("model_version"), parent=exp_spec.get("parent"),
        params=m.get("params") or {}, seed=m.get("seed"), entry_point=code.get("entry_point"),
        command=" ".join(shlex.quote(c) for c in code["command"]) if isinstance(code.get("command"), list)
        else code.get("command"), code=code, environment=env,
        periods=Periods.of(periods.get("train") and tuple(periods["train"]), periods.get("test") and tuple(periods["test"])),
        holdout=holdout_id, open_holdout=open_holdout, holdout_purpose=purpose, data=data_ids,
        tags=exp_spec.get("tags") or [], notes=exp_spec.get("notes"), source="ingested",
        periods_per_year=m.get("periods_per_year"), started_at=started_iso)
    reg.log_artifact(exp.id, "manifest", m, kind="report")
    for art in m.get("artifacts") or []:
        path = (base / art["path"]).resolve()
        if path.is_file():
            if verify_hashes:
                _check_hash(path, art.get("sha256"), f"artifact {art.get('name') or art['path']}")
            reg.log_artifact(exp.id, art.get("name") or path.name, path, kind=art.get("kind", "file"))

    results = m["results"]
    metrics = {k: float(v) for k, v in (results.get("metrics") or {}).items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
    try:
        if results.get("returns"):
            result, _ = read_returns(results["returns"], base, frames, verify_hashes)
            result.meta.update(results.get("meta") or {})
            return reg.complete_experiment(exp.id, result, metrics=metrics or None, validate=validate)
        digest = (results.get("digest") or {}).get("value")
        from qros.evaluate import finalize_metrics_only
        return finalize_metrics_only(reg, exp.id, metrics, digest=sha256_text(digest) if digest else None)
    except Exception as exc:
        reg.fail_experiment(exp.id, f"{type(exc).__name__}: {exc}")
        raise


def _json_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def reproduce_command(reg: "Registry", experiment_id: str, *, cwd: str | Path, timeout: float = 900.0) -> dict[str, Any]:
    """Re-run an ingested experiment's recorded command and compare its digest or returns file."""
    exp = reg.get_experiment(experiment_id)
    manifest = reg.load_artifact(experiment_id, "manifest")
    code = manifest.get("code") or {}
    cmd = code.get("command")
    if not cmd:
        raise ValueError(f"{experiment_id} has no recorded command")
    args = list(cmd) if isinstance(cmd, list) else shlex.split(cmd)
    workdir = (Path(cwd) / (code.get("cwd") or ".")).resolve()
    exe = Path(args[0])
    if not exe.is_absolute():
        for candidate in (workdir / exe, workdir / exe.with_suffix(".exe")):
            if candidate.is_file():
                args[0] = str(candidate)
                break
    proc = subprocess.run(args, cwd=str(workdir), capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        return reg.record_reproduction(experiment_id, "error", result_hash=None, max_abs_diff=None,
                                       note=f"exit code {proc.returncode}: {proc.stderr[-300:]}")
    results = manifest.get("results") or {}
    digest_spec = results.get("digest") or {}
    if digest_spec.get("value"):
        if digest_spec.get("from", "stdout-json") == "stdout-json":
            value = _json_path(json.loads(proc.stdout), digest_spec.get("json_path", "digest"))
        else:
            value = proc.stdout.strip()
        status = "exact" if str(value) == str(digest_spec["value"]) else "mismatch"
        return reg.record_reproduction(experiment_id, status, result_hash=sha256_text(str(value)), max_abs_diff=None,
                                       note=f"digest {value} vs recorded {digest_spec['value']}")
    ret = results.get("returns") or {}
    if ret.get("path"):
        path = (workdir / ret["path"]).resolve() if not Path(ret["path"]).is_absolute() else Path(ret["path"])
        new_hash = sha256_file(path)
        status = "exact" if new_hash == (ret.get("sha256") or "").lower() else "mismatch"
        return reg.record_reproduction(experiment_id, status, result_hash=new_hash, max_abs_diff=None,
                                       note="returns file hash compared")
    raise ValueError("the manifest records neither a digest nor a returns file to compare")
