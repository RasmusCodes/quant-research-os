import json

import numpy as np
import pandas as pd
import pytest

from qros.cli.main import main


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QROS_REGISTRY", raising=False)
    assert main(["init", "--researcher", "cli-test"]) == 0
    return tmp_path


def test_cli_end_to_end(workdir, capsys):
    idx = pd.bdate_range("2010-01-01", "2016-12-31")
    rng = np.random.default_rng(0)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (len(idx), 3)), axis=0)), index=idx,
                      columns=["A", "B", "C"]).rename_axis("date")
    px.to_csv(workdir / "prices.csv")
    assert main(["data", "dataset", "prices", "--adjusted", "yes", "--survivorship-free", "yes"]) == 0
    assert main(["data", "snapshot", "prices", "prices.csv", "--index-col", "date", "--parse-dates", "date", "--label", "v1"]) == 0
    out = capsys.readouterr().out
    version = [line.split()[0] for line in out.splitlines() if line.startswith("dv:")][0]
    assert main(["hypothesis", "add", "Trend", "--statement", "Trend pays", "--register",
                 "--prediction", '{"direction": "positive"}']) == 0
    assert main(["run", "strategies_fixture:trend", "-p", "lookback=20", "--seed", "1", "-d", f"prices={version}",
                 "--train", "2010:2013", "--test", "2014:2016", "--hypothesis", "H-001"]) == 0
    assert main(["run", "strategies_fixture:trend", "-p", "lookback=60", "--seed", "1", "-d", f"prices={version}",
                 "--train", "2010:2013", "--test", "2014:2016", "--hypothesis", "H-001"]) == 0
    capsys.readouterr()
    assert main(["exp", "list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["id"] for r in rows] == ["EXP-0001", "EXP-0002"]
    assert main(["exp", "show", "EXP-0001"]) == 0
    shown = capsys.readouterr().out
    assert "verdict:" in shown and "DSR" in shown
    assert main(["exp", "compare", "EXP-0001", "EXP-0002"]) == 0
    assert "Sharpe difference" in capsys.readouterr().out
    assert main(["reproduce", "EXP-0001"]) == 0
    assert "exact" in capsys.readouterr().out
    assert main(["data", "verify"]) == 0
    assert main(["verify"]) == 0
    assert main(["warnings"]) == 0
    assert main(["validate", "--all"]) == 0
    assert main(["exp", "export", "EXP-0001", "bundle.zip"]) == 0
    assert (workdir / "bundle.zip").exists()
    assert main(["dashboard", "--out", "site"]) == 0
    assert (workdir / "site" / "data" / "index.json").exists() and (workdir / "site" / "index.html").exists()
    assert main(["audit", "--limit", "5"]) == 0


def test_cli_toy_prints_hashes(capsys):
    assert main(["toy", "--seed", "7", "--lookback", "20"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["returns_sha256"] == "3f62ca7d9a98cc8d164068e56fab975bc7170fe3fea1cf7c7ff845cb8278b634"


def test_cli_errors_are_clean(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QROS_REGISTRY", raising=False)
    assert main(["status"]) == 2
    assert "qros init" in capsys.readouterr().err


def test_hypothesis_amend_reports_harking(workdir, capsys):
    main(["hypothesis", "add", "H", "--statement", "s", "--register"])
    main(["hypothesis", "amend", "H-001", "--reason", "typo", "--statement", "s2"])
    assert "now at version 2" in capsys.readouterr().out
    assert main(["hypothesis", "show", "H-001", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["current_version"] == 2
