"""Integration paths for other repositories: manifests, the stdlib emitter, adapters, bundles."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import qros
from qros.adapters import ManifestError, get_adapter, ingest_manifest, register_returns, validate_manifest
from qros.registry.bundle import export_experiment, import_bundle
from qros.registry.core import Registry
from helpers import make_returns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integrations" / "python"))
import qros_emit  # noqa: E402


def _write_run(tmp_path, n=1500, mu=0.0005):
    r = make_returns(seed=3, n=n, mu=mu)
    gross = r + 0.0001
    dates = [d.strftime("%Y-%m-%d") for d in r.index]
    data_file = tmp_path / "input.csv"
    data_file.write_text("a,b\n1,2\n", encoding="utf-8")
    qros_emit.write_returns_csv(tmp_path / "returns.csv", dates, r.tolist(), gross=gross.tolist(), cost=[0.0001] * n)
    m = qros_emit.RunManifest("other-repo", "emitted run", family="emit", seed=11, params={"k": 3},
                              train=("2010-01-01", "2013-12-31"), test=("2014-01-01", "2015-09-30"),
                              periods_per_year=252)
    m.add_data("input", "small-input", data_file, source="unit test", license="CC0")
    m.set_returns(tmp_path / "returns.csv")
    m.add_metrics(custom_score=1.5)
    m.write(tmp_path / "qros-run.json")
    return tmp_path / "qros-run.json", r


def test_emitter_manifest_validates_and_ingests(reg, tmp_path):
    path, r = _write_run(tmp_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert validate_manifest(doc) == [] and doc["data"][0]["path"] == "input.csv"
    exp = ingest_manifest(reg, path)
    assert exp.source == "ingested" and exp.status == "completed"
    assert exp.metric("custom_score", "custom") == 1.5
    frame = reg.result_frame(exp.id)
    assert {"net", "gross", "cost"} <= set(frame.columns)
    np.testing.assert_allclose(frame["net"].to_numpy(), r.to_numpy(), rtol=0, atol=0)
    version = reg.get_version(exp.data["input"])
    assert version.storage == "file" and reg.verify_version(version.id)["ok"]
    assert reg.load_artifact(exp.id, "manifest")["project"] == "other-repo"


def test_tampered_file_is_rejected(reg, tmp_path):
    path, _ = _write_run(tmp_path)
    (tmp_path / "returns.csv").write_text((tmp_path / "returns.csv").read_text() + "2099-01-01,0.5,0.5,0\n")
    with pytest.raises(ManifestError):
        ingest_manifest(reg, path)


def test_manifest_validation_messages():
    assert "experiment.name is required" in validate_manifest({"schema": "qros.run-manifest/1", "results": {"metrics": {"a": 1}}})
    assert validate_manifest({"schema": "x", "experiment": {"name": "n"}, "results": {}})


def test_register_returns_one_call(reg):
    exp = register_returns(reg, "quick", make_returns(n=1300), train="2010:2012", test="2013:2014")
    assert exp.status == "completed"


def test_bundle_round_trip(tmp_path, reg, prices):
    reg.add_source("vendor")
    reg.add_dataset("prices", source="vendor")
    v = reg.snapshot("prices", prices)
    h = reg.add_hypothesis("H", "statement", register=True)
    exp = qros.run_experiment(reg, "strategies_fixture:trend", params={"lookback": 10}, seed=1, data={"prices": v.id},
                              periods=qros.Periods.of("2010:2014", "2015:2019"), hypothesis=h.id)
    bundle = export_experiment(reg, exp.id, tmp_path / "exp.zip", include_data=True)
    other = Registry.init(tmp_path / "other")
    new_id = import_bundle(other, bundle)
    imported = other.get_experiment(new_id)
    assert imported.result_hash == exp.result_hash
    assert imported.metric("sharpe", "oos") == pytest.approx(exp.metric("sharpe", "oos"))
    assert other.get_version(imported.data["prices"]).content_hash == v.content_hash
    other.close()


# ----------------------------------------------------------------------------- project adapters (synthetic fixtures)

def test_rl_trading_lab_adapter(reg, tmp_path):
    run = tmp_path / "runs" / "main_spy" / "a2c__wf1__s0"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({
        "spec": {"experiment": "main_spy", "universe": "spy", "algo": "a2c", "fold": "wf1", "seed": 0, "holdout": False},
        "fold": {"dates": {"train": ["2000-01-03", "2006-12-29"], "test": ["2007-01-02", "2012-12-31"]}},
        "algo_params": {"lr": 0.001}, "code": {"rlt/env.py": "abc123"}}))
    (run / "eval.json").write_text(json.dumps({"run_id": "a2c__wf1__s0", "env_version": 2, "test_sharpe": 0.4}))
    r = make_returns(seed=1, n=1500, start="2007-01-01")
    pd.DataFrame({"date": r.index.strftime("%Y-%m-%d"), "equity": (1 + r).cumprod() * 1e6, "ret": r.values,
                  "turnover": 0.1, "cost": 0.00001, "gross": 0.5, "net": 0.5}).to_csv(run / "ledger_test.csv", index=False)
    ids = get_adapter("rl-trading-lab").ingest(reg, tmp_path)
    exp = reg.get_experiment(ids[0])
    assert exp.family == "rl-trading-lab/main_spy" and exp.metric("test_sharpe", "custom") == 0.4
    np.testing.assert_allclose(reg.returns(exp.id).to_numpy(), r.to_numpy())


def test_statarb_adapter(reg, tmp_path):
    dash = tmp_path / "results" / "dashboard"
    dash.mkdir(parents=True)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "03_preregistration.md").write_text(
        "# Research note 3: pre-registration (written 2026-11-15, before any result)\n\n"
        "**Primary hypothesis H3.** Pairs entries in volatile regimes earn more.\n\nMore text.\n", encoding="utf-8")
    idx = pd.bdate_range("2010-01-01", periods=1200)
    rng = np.random.default_rng(0)
    pd.DataFrame({"BOOK-A": rng.normal(0.0003, 0.01, 1200)}, index=idx).rename_axis("date").to_parquet(dash / "daily_net_base.parquet")
    pd.DataFrame({"BOOK-A": rng.normal(0.0004, 0.01, 1200)}, index=idx).rename_axis("date").to_parquet(dash / "daily_gross.parquet")
    (dash / "books.json").write_text(json.dumps([{"id": "BOOK-A", "note": "03", "family": "Pairs", "label": "Book A",
                                                  "description": "test book", "first": "2010-01-01", "last": "2014-12-31"}]))
    pytest.importorskip("pyarrow")
    ids = get_adapter("statarb").ingest(reg, tmp_path)
    exp = reg.get_experiment(ids[0])
    hyp = reg.get_hypothesis(exp.hypothesis_id)
    assert "volatile regimes" in hyp.statement and hyp.registered_at
    notes = reg.notes(hypothesis=hyp.id)
    assert "2026-11-15" in notes[0]["text"]
    assert "NOT_PREREGISTERED" in {w["code"] for w in reg.warnings(experiment=exp.id)}


def test_hft_sim_adapter_metrics_only(reg, tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "python" / ".cache").mkdir(parents=True)
    man = {"args": ["--seed", "7", "--mm-gamma", "0.5"], "config_hash": "fnv1a64:abc", "seed": 7, "version": "0.1.0",
           "git_commit": "unknown", "digest": "0xdeadbeef", "trades": 100, "simulated_secs": 180}
    (tmp_path / "reports" / "rq1.manifests.json").write_text(json.dumps([man]))
    (tmp_path / "python" / ".cache" / "x.json").write_text(json.dumps({
        "manifest": man, "market": {"mean_spread_ticks": 1.01}, "agents": [{"kind": "mm", "pnl_dollars": 12.5}]}))
    ids = get_adapter("hft-sim").ingest(reg, tmp_path)
    exp = reg.get_experiment(ids[0])
    assert exp.verdict == "insufficient" and exp.metric("pnl.mm", "custom") == 12.5
    assert exp.params["mm-gamma"] == 0.5 and exp.seed == 7
    assert reg.load_artifact(exp.id, "manifest")["code"]["command"][:2] == ["target/release/hftsim", "run"]


@pytest.mark.skipif(shutil.which("cargo") is None, reason="Rust toolchain not installed")
def test_rust_crate_example_ingests(reg, tmp_path):
    crate = ROOT / "integrations" / "rust" / "qros-manifest"
    out = tmp_path / "rust-out"
    proc = subprocess.run(["cargo", "run", "--quiet", "--example", "emit_run", "--", str(out)], cwd=crate,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    exp = ingest_manifest(reg, out / "qros-run.json")
    assert exp.status == "completed" and exp.project == "rust-example"
    assert {"gross", "cost"} <= set(reg.result_frame(exp.id).columns)
