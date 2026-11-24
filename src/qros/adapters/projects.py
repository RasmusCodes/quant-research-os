"""Adapters for existing research repositories.

Each adapter reads a project's own output files and turns every run into a run manifest; nothing is
imported from the project, so its code base stays separate. The formats are those the projects write
today (documented in docs/INTEGRATION.md); point an adapter at the project's root folder.

* ``rl-trading-lab`` - one folder per run (config.json, eval.json, ledger_test.csv); holdout runs open
  the project's 2019+ holdout.
* ``statarb`` - the dashboard export (books.json + wide daily parquet files) and the notes'
  pre-registrations (reports/NN_preregistration.md).
* ``sports-quant`` - outputs/<league>/backtest_paths.parquet (daily bankroll paths per strategy).
* ``hft-sim`` - the Rust simulator's run manifests (reports/rqN.manifests.json + python/.cache/*.json):
  scalar results, reproducible by re-running the recorded command and comparing the event digest.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

import numpy as np
import pandas as pd

from qros.adapters.manifest import SCHEMA_ID, ingest_manifest
from qros.util.hashing import sha256_file, sha256_json

if TYPE_CHECKING:  # pragma: no cover
    from qros.registry.core import Registry

Record = tuple[dict[str, Any], Path, dict[str, pd.DataFrame]]


class ProjectAdapter:
    name = "project"

    def discover(self, root: str | Path, **options: Any) -> Iterator[Record]:  # pragma: no cover - interface
        raise NotImplementedError

    def ingest(self, reg: "Registry", root: str | Path, *, limit: int | None = None,
               select: Callable[[dict[str, Any]], bool] | None = None, validate: bool = True,
               **options: Any) -> list[str]:
        ids = []
        for manifest, base, frames in self.discover(root, **options):
            if select is not None and not select(manifest):
                continue
            if limit is not None and len(ids) >= limit:
                break
            ids.append(ingest_manifest(reg, manifest, base_dir=base, frames=frames, validate=validate).id)
        return ids


# ----------------------------------------------------------------------------- rl-trading-lab


class RLTradingLabAdapter(ProjectAdapter):
    name = "rl-trading-lab"
    holdout_start = "2019-01-01"

    def discover(self, root: str | Path, experiments: list[str] | None = None, **options: Any) -> Iterator[Record]:
        root = Path(root)
        runs = root / "runs"
        holdout_runs = sorted(p for p in (runs / "holdout_spy").glob("*") if (p / "eval.json").is_file()) \
            if (runs / "holdout_spy").is_dir() else []
        for run_dir in sorted(runs.glob("*/*")):
            group = run_dir.parent.name
            if group.startswith("_") or group == "baselines" or (experiments and group not in experiments):
                continue
            if not (run_dir / "eval.json").is_file() or not (run_dir / "ledger_test.csv").is_file():
                continue
            cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            ev = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
            spec = cfg.get("spec") or {}
            dates = (cfg.get("fold") or {}).get("dates") or {}
            holdout = bool(spec.get("holdout"))
            params = {"algo": spec.get("algo"), "universe": spec.get("universe"), "fold": spec.get("fold"),
                      "features": spec.get("features"), "total_steps": spec.get("total_steps"),
                      "tag": spec.get("tag") or None, "leak": spec.get("leak") or None,
                      **{f"algo.{k}": v for k, v in (cfg.get("algo_params") or {}).items()},
                      **{f"env.{k}": v for k, v in (spec.get("env") or {}).items()}}
            manifest: dict[str, Any] = {
                "schema": SCHEMA_ID, "project": self.name,
                "experiment": {
                    "name": f"{group}/{ev.get('run_id') or run_dir.name}", "family": f"{self.name}/{group}",
                    "model": spec.get("algo"), "model_version": f"env-v{ev.get('env_version', '?')}",
                    "tags": [t for t in (spec.get("universe"), spec.get("algo"), spec.get("fold"), spec.get("tag")) if t],
                },
                "code": {"repo": self.name, "language": "python",
                         "source_sha256": sha256_json(cfg.get("code")) if cfg.get("code") else None,
                         "versions": cfg.get("versions")},
                "params": {k: v for k, v in params.items() if v is not None},
                "seed": spec.get("seed"),
                "periods": {"train": dates.get("train"), "test": None if holdout else dates.get("test")},
                "periods_per_year": 252,
                "results": {
                    "returns": {"path": "ledger_test.csv", "date_column": "date",
                                "columns": {"net": "ret", "cost": "cost", "turnover": "turnover"},
                                "periods_per_year": 252},
                    "metrics": {k: v for k, v in ev.items() if k.startswith("test_") and isinstance(v, (int, float))},
                },
            }
            if holdout:
                manifest["holdout"] = {"name": f"{self.name} holdout", "start": self.holdout_start, "end": "2099-12-31",
                                       "opened": True, "purpose": f"holdout run {run_dir.name}",
                                       "max_touches": max(1, len(holdout_runs)),
                                       "description": "The project's holdout (2019 on); its own ledger is "
                                                      "runs/holdout_ledger.jsonl."}
            yield manifest, run_dir, {}


# ----------------------------------------------------------------------------- statarb


def _statarb_preregistrations(root: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted((root / "reports").glob("*_preregistration.md")):
        note = path.name.split("_")[0]
        text = path.read_text(encoding="utf-8")
        title_line = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
        written = re.search(r"written (\d{4}-\d{2}-\d{2})", text)
        m = re.search(r"\*\*Primary hypothes[ie]s?\s*([^.*]*)\.\*\*\s*(.+?)(?:\n\n|\Z)", text, re.S)
        if not m:
            m = re.search(r"##\s*(H[0-9]+[a-z]?)[^\n]*\n+(.+?)(?:\n\n|\Z)", text, re.S)
        if not m:
            continue
        label, statement = m.group(1).strip(), " ".join(m.group(2).split())
        out[note] = {"title": f"statarb note {note} {label}".strip(), "statement": statement,
                     "registered_at_claimed": written.group(1) if written else None, "source": path.name,
                     "rationale": title_line}
    return out


class StatarbAdapter(ProjectAdapter):
    name = "statarb"

    def discover(self, root: str | Path, **options: Any) -> Iterator[Record]:
        root = Path(root)
        dash = root / "results" / "dashboard"
        books = json.loads((dash / "books.json").read_text(encoding="utf-8"))
        tables = {}
        for key, file in (("net", "daily_net_base.parquet"), ("gross", "daily_gross.parquet"),
                          ("turnover", "daily_turnover.parquet"), ("gross_exposure", "daily_gross_exposure.parquet")):
            if (dash / file).is_file():
                tables[key] = pd.read_parquet(dash / file)
        net_hash = sha256_file(dash / "daily_net_base.parquet")
        prereg = _statarb_preregistrations(root)
        for book in books:
            bid = book["id"]
            if bid not in tables["net"].columns:
                continue
            frame = pd.DataFrame({k: t[bid] for k, t in tables.items() if bid in t.columns})
            frame = frame.loc[book.get("first") or frame.index.min(): book.get("last") or frame.index.max()]
            frame = frame.dropna(subset=["net"])
            if "gross" in frame.columns:
                frame["gross"] = frame["gross"].fillna(frame["net"])
            hyp = prereg.get(book.get("note", ""))
            manifest = {
                "schema": SCHEMA_ID, "project": self.name,
                "experiment": {"name": f"{bid}: {book.get('label', bid)}", "family": f"{self.name}/note-{book.get('note')}",
                               "model": bid, "tags": [f"note-{book.get('note')}", book.get("family", "")],
                               "notes": book.get("description"),
                               "hypothesis": {**hyp, "register": True} if hyp else None},
                "code": {"repo": self.name, "language": "python"},
                "params": {"book": bid, "costs_modeled": bool(book.get("costs_modeled"))},
                "periods_per_year": 252,
                "data": [{"role": "returns", "dataset": "statarb/dashboard-daily",
                          "path": "results/dashboard/daily_net_base.parquet",
                          "sha256": net_hash, "source": "statarb dashboard export", "store": False}],
                "results": {"returns": {"frame": bid, "columns": {c: c for c in frame.columns}, "periods_per_year": 252}},
            }
            yield manifest, root, {bid: frame.rename_axis("date").reset_index()}


# ----------------------------------------------------------------------------- sports-quant


class SportsQuantAdapter(ProjectAdapter):
    name = "sports-quant"

    def discover(self, root: str | Path, leagues: tuple[str, ...] = ("nba", "nfl"), **options: Any) -> Iterator[Record]:
        root = Path(root)
        for league in leagues:
            path = root / "outputs" / league / "backtest_paths.parquet"
            if not path.is_file():
                continue
            paths = pd.read_parquet(path)
            digest = sha256_file(path)
            for strategy, grp in paths.groupby("strategy"):
                g = grp.sort_values("date")
                bankroll_before = (g["bankroll"] - g["pnl"]).to_numpy()
                ret = np.where(bankroll_before > 0, g["pnl"].to_numpy() / bankroll_before, 0.0)
                frame = pd.DataFrame({"date": pd.to_datetime(g["date"]).to_numpy(), "net": ret})
                frame = frame.groupby("date", as_index=False)["net"].apply(lambda s: float(np.prod(1 + s) - 1))
                venue, _, rest = str(strategy).partition("|")
                manifest = {
                    "schema": SCHEMA_ID, "project": self.name,
                    "experiment": {"name": f"{league}: {strategy}", "family": f"{self.name}/{league}",
                                   "model": rest or str(strategy), "tags": [league, venue]},
                    "code": {"repo": self.name, "language": "python"},
                    "params": {"strategy": str(strategy), "league": league},
                    "data": [{"role": "paths", "dataset": f"sports-quant/{league}-paths",
                              "path": f"outputs/{league}/backtest_paths.parquet", "sha256": digest, "store": False}],
                    "results": {"returns": {"frame": str(strategy), "columns": {"net": "net"}}},
                }
                yield manifest, root, {str(strategy): frame}


# ----------------------------------------------------------------------------- hft-sim (Rust)


def _args_to_params(args: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    i = 0
    while i < len(args):
        key = args[i]
        if key.startswith("--"):
            value: Any = True
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                i += 1
            try:
                value = int(value) if isinstance(value, str) and re.fullmatch(r"-?\d+", value) else (
                    float(value) if isinstance(value, str) and re.fullmatch(r"-?\d*\.\d+(e-?\d+)?", value) else value)
            except ValueError:
                pass
            params[key[2:]] = value
        i += 1
    return params


class HftSimAdapter(ProjectAdapter):
    name = "hft-sim"

    def discover(self, root: str | Path, **options: Any) -> Iterator[Record]:
        root = Path(root)
        cache = {}
        for f in glob.glob(str(root / "python" / ".cache" / "*.json")):
            try:
                doc = json.loads(Path(f).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            man = doc.get("manifest") or {}
            if man.get("digest"):
                cache[(man.get("config_hash"), man.get("digest"))] = doc
        for mf in sorted((root / "reports").glob("rq*.manifests.json")):
            rq = mf.name.split(".")[0]
            for man in json.loads(mf.read_text(encoding="utf-8")):
                doc = cache.get((man.get("config_hash"), man.get("digest")))
                metrics: dict[str, float] = {"trades": man.get("trades"), "simulated_secs": man.get("simulated_secs")}
                if doc:
                    by_kind: dict[str, float] = {}
                    for agent in doc.get("agents") or []:
                        kind = str(agent.get("kind"))
                        by_kind[kind] = by_kind.get(kind, 0.0) + float(agent.get("pnl_dollars") or 0.0)
                    metrics.update({f"pnl.{k}": v for k, v in by_kind.items()})
                    market = doc.get("market") or {}
                    for key in ("mean_spread_ticks", "return_acf1", "cancel_share"):
                        if isinstance(market.get(key), (int, float)):
                            metrics[key] = market[key]
                params = _args_to_params(man.get("args") or [])
                manifest = {
                    "schema": SCHEMA_ID, "project": self.name,
                    "experiment": {"name": f"{rq} seed {man.get('seed')} {man.get('config_hash')}",
                                   "family": f"{self.name}/{rq}", "model": "hftsim", "tags": [rq, "simulation"],
                                   "notes": "Simulated market: results are properties of the simulator, not of a real market."},
                    "code": {"repo": self.name, "language": "rust", "version": man.get("version"),
                             "commit": None if man.get("git_commit") in (None, "unknown") else man.get("git_commit"),
                             "config_hash": man.get("config_hash"),
                             "command": ["target/release/hftsim", "run", *[str(a) for a in man.get("args") or []]],
                             "cwd": "."},
                    "params": params, "seed": man.get("seed"),
                    "results": {"metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                                "digest": {"value": man.get("digest"), "from": "stdout-json",
                                           "json_path": "manifest.digest"}},
                }
                yield manifest, root, {}


ADAPTERS: dict[str, type[ProjectAdapter]] = {
    RLTradingLabAdapter.name: RLTradingLabAdapter,
    StatarbAdapter.name: StatarbAdapter,
    SportsQuantAdapter.name: SportsQuantAdapter,
    HftSimAdapter.name: HftSimAdapter,
}


def get_adapter(name: str) -> ProjectAdapter:
    try:
        return ADAPTERS[name]()
    except KeyError:
        raise KeyError(f"Unknown adapter {name!r}; available: {sorted(ADAPTERS)}") from None
