"""Command-line interface: ``qros <command>``. Run ``qros --help`` or ``qros <command> --help``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from qros._version import __version__


def _registry(args: argparse.Namespace):
    from qros.registry.core import Registry
    return Registry.open(args.registry)


def _kv(pairs: Sequence[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"expected key=value, got {item!r}")
        key, value = item.split("=", 1)
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def _json_arg(text: str | None) -> Any:
    return json.loads(text) if text else None


def _yesno(text: str | None) -> bool | None:
    if text is None:
        return None
    return text.lower() in ("y", "yes", "true", "1")


def _fmt(v: Any, nd: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _table(rows: list[list[Any]], header: list[str]) -> str:
    cells = [header] + [[_fmt(c) for c in r] for r in rows]
    widths = [max(len(str(r[i])) for r in cells) for i in range(len(header))]
    lines = ["  ".join(str(c).ljust(w) for c, w in zip(r, widths)) for r in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _print(obj: Any, as_json: bool) -> None:
    if as_json:
        from qros.util.hashing import normalize
        print(json.dumps(normalize(obj), indent=2, default=str))
    else:
        print(obj)


# ----------------------------------------------------------------------------- commands


def cmd_init(args: argparse.Namespace) -> int:
    from qros.registry.core import Registry
    path = Path(args.path or ".qros")
    reg = Registry.init(path, researcher=args.researcher)
    print(f"Created registry {reg.registry_id} at {path}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    reg = _registry(args)
    stats = reg.stats()
    if args.json:
        _print(stats, True)
        return 0
    for k, v in stats.items():
        print(f"{k:>18}: {v}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    reg = _registry(args)
    res = reg.verify(deep=not args.quick)
    chain = res["audit_chain"]
    print(f"audit chain: {'OK' if chain['ok'] else 'BROKEN at event ' + str(chain['first_bad_seq'])} "
          f"({chain['events']} events, head {chain['head'][:16]})")
    if not args.quick:
        print(f"objects: {len(res['bad_objects'])} corrupt; dataset versions: {len(res['bad_versions'])} failing")
    return 0 if res["ok"] else 1


def cmd_hypothesis(args: argparse.Namespace) -> int:
    reg = _registry(args)
    if args.action == "add":
        h = reg.add_hypothesis(args.title, args.statement, rationale=args.rationale,
                               prediction=_json_arg(args.prediction), criteria=_json_arg(args.criteria),
                               family=args.family, register=args.register)
        print(f"{h.id} {h.status}: {h.title}")
    elif args.action == "register":
        h = reg.register_hypothesis(args.id)
        print(f"{h.id} registered at {h.registered_at}")
    elif args.action == "amend":
        h = reg.amend_hypothesis(args.id, reason=args.reason, statement=args.statement,
                                 prediction=_json_arg(args.prediction), criteria=_json_arg(args.criteria))
        print(f"{h.id} now at version {h.current_version}")
        for w in reg.warnings(hypothesis=h.id):
            print(f"  [{w['severity']}] {w['code']}: {w['message']}")
    elif args.action == "status":
        h = reg.set_hypothesis_status(args.id, args.status, note=args.note)
        print(f"{h.id} -> {h.status}")
    elif args.action == "list":
        rows = [[h.id, h.status, h.registered_at or "-", f"v{h.current_version}", h.title] for h in reg.list_hypotheses()]
        print(_table(rows, ["id", "status", "registered", "ver", "title"]))
    elif args.action == "show":
        h = reg.get_hypothesis(args.id)
        if args.json:
            _print(h.to_dict(), True)
        else:
            print(f"{h.id}  {h.title}  [{h.status}]  registered {h.registered_at or 'never'}")
            for v in h.versions:
                print(f"  v{v.version} {v.created_at} ({v.reason}): {v.statement}")
            for w in reg.warnings(hypothesis=h.id):
                print(f"  [{w['severity']}] {w['code']}: {w['message']}")
    return 0


def cmd_data(args: argparse.Namespace) -> int:
    reg = _registry(args)
    if args.action == "source":
        print(reg.add_source(args.name, vendor=args.vendor, url=args.url, license=args.license,
                             description=args.description))
    elif args.action == "dataset":
        print(reg.add_dataset(args.name, source=args.source, kind=args.kind, description=args.description,
                              entity_col=args.entity_col, time_col=args.time_col, available_col=args.available_col,
                              frequency=args.frequency, survivorship_free=_yesno(args.survivorship_free),
                              adjusted=_yesno(args.adjusted), licensed=args.licensed))
    elif args.action == "snapshot":
        import pandas as pd
        path = Path(args.file)
        data: Any = path
        if args.as_table or path.suffix.lower() in (".csv", ".parquet"):
            if path.suffix.lower() == ".parquet":
                data = pd.read_parquet(path)
            else:
                data = pd.read_csv(path, parse_dates=args.parse_dates or None)
                if args.index_col:
                    data = data.set_index(args.index_col)
        v = reg.snapshot(args.dataset, data, label=args.label, knowledge_time=args.knowledge_time, notes=args.notes)
        print(f"{v.id}  {v.dataset_id}  {v.label or ''}  content {v.content_hash[:16]}  rows {v.n_rows}")
    elif args.action == "list":
        rows = [[v.id, v.dataset_id, v.label or "", v.storage, v.n_rows, v.time_start, v.time_end, v.knowledge_time]
                for v in reg.list_versions(args.dataset)]
        print(_table(rows, ["version", "dataset", "label", "storage", "rows", "from", "to", "as of"]))
    elif args.action == "show":
        _print(reg.get_version(args.version).to_dict(), True)
    elif args.action == "lineage":
        lin = reg.lineage(args.version)
        if args.json:
            _print(lin, True)
        else:
            for n in lin["nodes"]:
                print(f"{n['id']}  {n['dataset_name']}  {n['label'] or ''}  as of {n['knowledge_time']}")
            for e in lin["edges"]:
                print(f"  {e['from']} -> {e['to']} via {e['transform']}")
    elif args.action == "verify":
        ids = [args.version] if args.version else [v.id for v in reg.list_versions()]
        bad = 0
        for vid in ids:
            res = reg.verify_version(vid)
            state = "ok" if res["ok"] else ("external" if res["ok"] is None else "FAILED")
            bad += res["ok"] is False
            print(f"{vid}: {state}{'' if not res['reason'] else ' - ' + res['reason']}")
        return 1 if bad else 0
    return 0


def cmd_holdout(args: argparse.Namespace) -> int:
    reg = _registry(args)
    if args.action == "add":
        ho = reg.add_holdout(args.name, args.start, args.end, description=args.description, max_touches=args.max_touches)
        print(f"{ho.id} {ho.name} {ho.start}..{ho.end} (max {ho.max_touches} looks)")
    else:
        for ho in reg.list_holdouts():
            acc = reg.holdout_accesses(ho.id)
            print(f"{ho.id} {ho.name} {ho.start}..{ho.end}: {len(acc)}/{ho.max_touches} looks")
            for a in acc:
                print(f"   {a['accessed_at']}  {a['experiment_id']}  {a['purpose']}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from qros.registry.models import Periods
    from qros.runner import run_experiment
    reg = _registry(args)
    params = _json_arg(args.params_json) or {}
    params.update(_kv(args.param))
    exp = run_experiment(reg, args.entry_point, params=params, seed=args.seed, data=_kv(args.data),
                         periods=Periods.of(args.train, args.test), hypothesis=args.hypothesis, family=args.family,
                         name=args.name, holdout=args.holdout, open_holdout=args.open_holdout,
                         holdout_purpose=args.purpose, tags=args.tag or [], notes=args.note)
    _print_experiment(reg, exp.id, False)
    return 0


def _print_experiment(reg: Any, eid: str, as_json: bool) -> None:
    exp = reg.get_experiment(eid)
    report = reg.latest_validation(eid) or {}
    if as_json:
        _print({"experiment": exp.to_dict(), "validation": report, "warnings": reg.warnings(experiment=eid)}, True)
        return
    print(f"{exp.id}  {exp.name}  [{exp.status}]  verdict: {exp.verdict}")
    print(f"  hypothesis {exp.hypothesis_id} v{exp.hypothesis_version}  family {exp.family}  seed {exp.seed}")
    print(f"  params {json.dumps(exp.params)}")
    print(f"  data {exp.data}")
    print(f"  spec {exp.spec_hash[:16] if exp.spec_hash else '-'}  result {exp.result_hash[:16] if exp.result_hash else '-'}")
    for scope in ("is", "oos", "holdout", "full"):
        m = exp.metrics.get(scope)
        if m:
            n_obs = m.get("n_obs")
            print(f"  {scope:>7}: Sharpe {_fmt(m.get('sharpe'))}  ann.ret {_fmt(m.get('ann_return'), 3)}  "
                  f"vol {_fmt(m.get('ann_vol'), 3)}  maxDD {_fmt(m.get('max_drawdown'), 3)}  "
                  f"n {int(n_obs) if n_obs is not None else '-'}")
    sig = report.get("significance") or {}
    if sig:
        print(f"  DSR {_fmt(sig.get('dsr'), 3)} (trials {sig.get('n_trials')}, effective {_fmt(sig.get('n_eff'), 1)})  "
              f"OOS p {_fmt(sig.get('oos_p'), 3)} raw / {_fmt(sig.get('oos_p_holm'), 3)} Holm")
    for g in report.get("scorecard", []):
        print(f"   [{g['status']:>4}] {g['label']}: {g['note']}")
    if report.get("verdict_reason"):
        print(f"  => {report['verdict']}: {report['verdict_reason']}")
    for w in reg.warnings(experiment=eid):
        print(f"  [{w['severity']}] {w['code']}: {w['message']}")


def cmd_exp(args: argparse.Namespace) -> int:
    reg = _registry(args)
    if args.action == "list":
        exps = reg.list_experiments(family=args.family, hypothesis=args.hypothesis, tag=args.tag)
        if args.verdict:
            exps = [e for e in exps if e.verdict == args.verdict]
        if args.json:
            _print([{"id": e.id, "name": e.name, "verdict": e.verdict, "family": e.family,
                     "is_sharpe": e.metric("sharpe", "is"), "oos_sharpe": e.metric("sharpe", "oos")} for e in exps], True)
            return 0
        rows = [[e.id, e.verdict or e.status, e.family, e.metric("sharpe", "is"), e.metric("sharpe", "oos"),
                 e.metric("sharpe", "holdout"), e.name[:48]] for e in exps]
        print(_table(rows, ["id", "verdict", "family", "IS SR", "OOS SR", "HO SR", "name"]))
    elif args.action == "show":
        _print_experiment(reg, args.id, args.json)
    elif args.action == "compare":
        from qros.compare import compare_experiments
        res = compare_experiments(reg, args.ids)
        if args.json:
            _print({"table": res["table"].to_dict(), "pairs": res["pairs"]}, True)
            return 0
        import pandas as pd
        with pd.option_context("display.width", 200, "display.max_colwidth", 40):
            print(res["table"].map(lambda v: _fmt(v, 3) if isinstance(v, float) else ("-" if v is None else v)).to_string())
        for p in res["pairs"]:
            print(f"Sharpe difference {p['a']} - {p['b']} on {p['window']}: {_fmt(p['diff'])} "
                  f"[{_fmt(p['lo'])}, {_fmt(p['hi'])}], p = {_fmt(p['p_value'], 3)}, correlation {_fmt(p.get('correlation'))}")
    elif args.action == "tag":
        reg.tag(args.id, *args.tags)
    elif args.action == "note":
        reg.add_note(args.text, experiment=args.id)
    elif args.action == "export":
        from qros.registry.bundle import export_experiment
        print(export_experiment(reg, args.id, args.out, include_data=args.include_data))
    elif args.action == "import":
        from qros.registry.bundle import import_bundle
        print(import_bundle(reg, args.bundle))
    elif args.action == "invalidate":
        reg.invalidate_experiment(args.id, args.reason)
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    reg = _registry(args)
    exp = reg.get_experiment(args.id)
    if exp.source == "managed":
        from qros.runner import reproduce
        rec = reproduce(reg, args.id)
    else:
        if not args.cwd:
            raise SystemExit("ingested experiments are re-run from their recorded command: pass --cwd <project root>")
        from qros.adapters.manifest import reproduce_command
        rec = reproduce_command(reg, args.id, cwd=args.cwd)
    print(f"{args.id}: {rec['status']}" + (f" (max |diff| {rec['max_abs_diff']:.3g})" if rec.get("max_abs_diff") else "")
          + (f" - {rec['note']}" if rec.get("note") else ""))
    return 0 if rec["status"] in ("exact", "numerical") else 1


def cmd_validate(args: argparse.Namespace) -> int:
    from qros.evaluate import revalidate_all, revalidate_family, validate_experiment
    reg = _registry(args)
    if args.all:
        res = revalidate_all(reg)
    elif args.family:
        res = revalidate_family(reg, args.family)
    elif args.id:
        res = {args.id: validate_experiment(reg, args.id)["verdict"]}
    else:
        raise SystemExit("pass an experiment id, --family or --all")
    for k, v in res.items():
        print(f"{k}: {v}")
    return 0


def cmd_warnings(args: argparse.Namespace) -> int:
    reg = _registry(args)
    items = reg.warnings(severity=args.severity, active=None if args.all else True)
    if args.json:
        _print(items, True)
        return 0
    rows = [[w["id"], w["severity"], w["code"], w["scope_key"], w["message"][:90]] for w in items]
    print(_table(rows, ["id", "severity", "code", "scope", "message"]))
    return 0


def cmd_ack(args: argparse.Namespace) -> int:
    reg = _registry(args)
    reg.acknowledge_warning(args.id, args.note)
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from qros.adapters.manifest import ingest_manifest
    reg = _registry(args)
    for path in args.manifests:
        exp = ingest_manifest(reg, path, verify_hashes=not args.no_verify)
        print(f"{path}: {exp.id} {exp.verdict}")
    return 0


def cmd_adapter(args: argparse.Namespace) -> int:
    from qros.adapters.projects import ADAPTERS, get_adapter
    if args.list:
        for name in sorted(ADAPTERS):
            print(name)
        return 0
    reg = _registry(args)
    adapter = get_adapter(args.name)
    opts: dict[str, Any] = {}
    if args.experiments:
        opts["experiments"] = args.experiments.split(",")
    ids = adapter.ingest(reg, args.root, limit=args.limit, **opts)
    print(f"ingested {len(ids)} runs from {args.name}")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    from qros.dashboard.export import export_site, serve
    reg = _registry(args)
    out = Path(args.out or (Path(reg.path) / "dashboard"))
    export_site(reg, out, title=args.title)
    print(f"dashboard written to {out}")
    if args.serve:
        serve(out, port=args.port, open_browser=args.open)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from qros.examples.demo import DEMO_SEED, build_demo, render_report
    out = Path(args.out)
    if (out / "registry.db").exists():
        raise SystemExit(f"{out} already holds a registry; choose another --out")
    reg = build_demo(out, seed=args.seed or DEMO_SEED, quick=args.quick)
    if args.report:
        Path(args.report).write_text(render_report(reg), encoding="utf-8")
        print(f"results tables written to {args.report}")
    print(f"demo registry at {out}; explore it with: qros --registry {out} dashboard --serve")
    return 0


def cmd_toy(args: argparse.Namespace) -> int:
    from qros.examples.toy import run
    res = run(args.seed, args.lookback, args.cost_bps, args.n)
    print(json.dumps({k: res[k] for k in ("version", "seed", "lookback", "cost_bps", "n", "data_sha256",
                                          "returns_sha256", "stats")}, indent=2))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    reg = _registry(args)
    events = reg.events(entity_id=args.entity) if args.entity else reg.events()
    for e in events[-args.limit:]:
        print(f"{e['seq']:>6} {e['at']} {e['action']:<24} {e['entity_id'] or '':<10} {e['hash'][:12]}")
    return 0


# ----------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qros", description="Quant Research OS: experiment registry, data provenance "
                                "and statistical validation.")
    p.add_argument("--version", action="version", version=f"qros {__version__}")
    p.add_argument("--registry", help="registry folder (default: $QROS_REGISTRY or the nearest .qros upwards)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="create a registry")
    s.add_argument("path", nargs="?")
    s.add_argument("--researcher")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("status", help="registry statistics")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("verify", help="verify the audit hash chain and every stored object")
    s.add_argument("--quick", action="store_true", help="audit chain only")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("hypothesis", help="pre-register and manage hypotheses")
    hs = s.add_subparsers(dest="action", required=True)
    a = hs.add_parser("add")
    a.add_argument("title")
    a.add_argument("--statement", required=True)
    a.add_argument("--rationale")
    a.add_argument("--prediction", help='JSON, e.g. {"direction": "positive", "metric": "oos.sharpe"}')
    a.add_argument("--criteria", help='JSON list, e.g. [{"metric": "oos.sharpe", "op": ">=", "value": 0.3}]')
    a.add_argument("--family")
    a.add_argument("--register", action="store_true")
    for name in ("register",):
        b = hs.add_parser(name)
        b.add_argument("id")
    b = hs.add_parser("amend")
    b.add_argument("id")
    b.add_argument("--reason", required=True)
    b.add_argument("--statement")
    b.add_argument("--prediction")
    b.add_argument("--criteria")
    b = hs.add_parser("status")
    b.add_argument("id")
    b.add_argument("status")
    b.add_argument("--note")
    hs.add_parser("list")
    b = hs.add_parser("show")
    b.add_argument("id")
    b.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_hypothesis)

    s = sub.add_parser("data", help="data sources, datasets, versions and lineage")
    ds = s.add_subparsers(dest="action", required=True)
    b = ds.add_parser("source")
    b.add_argument("name")
    b.add_argument("--vendor")
    b.add_argument("--url")
    b.add_argument("--license")
    b.add_argument("--description")
    b = ds.add_parser("dataset")
    b.add_argument("name")
    b.add_argument("--source")
    b.add_argument("--kind", default="table")
    b.add_argument("--description")
    b.add_argument("--entity-col")
    b.add_argument("--time-col")
    b.add_argument("--available-col")
    b.add_argument("--frequency")
    b.add_argument("--survivorship-free")
    b.add_argument("--adjusted")
    b.add_argument("--licensed", action="store_true")
    b = ds.add_parser("snapshot")
    b.add_argument("dataset")
    b.add_argument("file")
    b.add_argument("--label")
    b.add_argument("--knowledge-time")
    b.add_argument("--notes")
    b.add_argument("--as-table", action="store_true", help="parse the file as a table even without .csv/.parquet")
    b.add_argument("--index-col")
    b.add_argument("--parse-dates", nargs="*")
    b = ds.add_parser("list")
    b.add_argument("dataset", nargs="?")
    for name in ("show", "lineage"):
        b = ds.add_parser(name)
        b.add_argument("version")
        b.add_argument("--json", action="store_true")
    b = ds.add_parser("verify")
    b.add_argument("version", nargs="?")
    s.set_defaults(func=cmd_data)

    s = sub.add_parser("holdout", help="sealed holdout windows and their ledger")
    hs = s.add_subparsers(dest="action", required=True)
    b = hs.add_parser("add")
    b.add_argument("name")
    b.add_argument("start")
    b.add_argument("end")
    b.add_argument("--max-touches", type=int, default=1)
    b.add_argument("--description")
    hs.add_parser("list")
    s.set_defaults(func=cmd_holdout)

    s = sub.add_parser("run", help="run a managed (reproducible) experiment: module:function")
    s.add_argument("entry_point")
    s.add_argument("--param", "-p", action="append", help="key=value (value parsed as JSON when possible)")
    s.add_argument("--params-json")
    s.add_argument("--seed", type=int)
    s.add_argument("--data", "-d", action="append", help="role=dataset_version_id")
    s.add_argument("--train", help="start:end, e.g. 2005:2014")
    s.add_argument("--test", help="start:end")
    s.add_argument("--hypothesis")
    s.add_argument("--family")
    s.add_argument("--name")
    s.add_argument("--holdout")
    s.add_argument("--open-holdout", action="store_true")
    s.add_argument("--purpose")
    s.add_argument("--tag", action="append")
    s.add_argument("--note")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("exp", help="list, inspect, compare, tag, export and import experiments")
    es = s.add_subparsers(dest="action", required=True)
    b = es.add_parser("list")
    b.add_argument("--family")
    b.add_argument("--hypothesis")
    b.add_argument("--tag")
    b.add_argument("--verdict")
    b.add_argument("--json", action="store_true")
    b = es.add_parser("show")
    b.add_argument("id")
    b.add_argument("--json", action="store_true")
    b = es.add_parser("compare")
    b.add_argument("ids", nargs="+")
    b.add_argument("--json", action="store_true")
    b = es.add_parser("tag")
    b.add_argument("id")
    b.add_argument("tags", nargs="+")
    b = es.add_parser("note")
    b.add_argument("id")
    b.add_argument("text")
    b = es.add_parser("export")
    b.add_argument("id")
    b.add_argument("out")
    b.add_argument("--include-data", action="store_true")
    b = es.add_parser("import")
    b.add_argument("bundle")
    b = es.add_parser("invalidate")
    b.add_argument("id")
    b.add_argument("reason")
    s.set_defaults(func=cmd_exp)

    s = sub.add_parser("reproduce", help="re-run an experiment from its recorded inputs and compare result hashes")
    s.add_argument("id")
    s.add_argument("--cwd", help="project root for ingested runs with a recorded command")
    s.set_defaults(func=cmd_reproduce)

    s = sub.add_parser("validate", help="recompute validation reports and hygiene warnings")
    s.add_argument("id", nargs="?")
    s.add_argument("--family")
    s.add_argument("--all", action="store_true")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("warnings", help="research-hygiene warnings")
    s.add_argument("--severity", choices=["info", "warning", "critical"])
    s.add_argument("--all", action="store_true", help="include cleared warnings")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_warnings)

    s = sub.add_parser("ack", help="acknowledge a warning with a note")
    s.add_argument("id")
    s.add_argument("note")
    s.set_defaults(func=cmd_ack)

    s = sub.add_parser("ingest", help="ingest run manifests written by other repositories (any language)")
    s.add_argument("manifests", nargs="+")
    s.add_argument("--no-verify", action="store_true", help="skip file hash checks")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("adapter", help="ingest runs from a known project layout")
    s.add_argument("name", nargs="?")
    s.add_argument("root", nargs="?")
    s.add_argument("--limit", type=int)
    s.add_argument("--experiments", help="comma-separated run groups (rl-trading-lab)")
    s.add_argument("--list", action="store_true")
    s.set_defaults(func=cmd_adapter)

    s = sub.add_parser("dashboard", help="export (and optionally serve) the dashboard")
    s.add_argument("--out")
    s.add_argument("--title")
    s.add_argument("--serve", action="store_true")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--open", action="store_true")
    s.set_defaults(func=cmd_dashboard)

    s = sub.add_parser("demo", help="build the synthetic demo research project")
    s.add_argument("--out", default="demo-registry")
    s.add_argument("--seed", type=int)
    s.add_argument("--quick", action="store_true", help="40 firms and smaller sweeps (about a minute)")
    s.add_argument("--report", help="also write the results tables (Markdown) to this file")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("toy", help="run the browser-reproducible toy experiment and print its hashes")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--lookback", type=int, default=20)
    s.add_argument("--cost-bps", type=float, default=5.0)
    s.add_argument("--n", type=int, default=1260)
    s.set_defaults(func=cmd_toy)

    s = sub.add_parser("audit", help="show the hash-chained audit log")
    s.add_argument("--entity")
    s.add_argument("--limit", type=int, default=40)
    s.set_defaults(func=cmd_audit)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
