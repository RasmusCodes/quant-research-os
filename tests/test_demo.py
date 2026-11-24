"""The scripted demo project, built in quick mode (40 firms, short sweeps), plus the dashboard export."""

import json

import pytest

from qros.dashboard.export import export_site
from qros.examples.demo import build_demo
from qros.examples.synthetic_market import MarketConfig, generate


def test_synthetic_market_is_internally_consistent():
    m = generate(MarketConfig(seed=3, n_assets=20, start="2015-01-01", end="2018-12-31"))
    from qros.data.corporate_actions import adjust_prices
    adj = adjust_prices(m.close_raw_v2, m.actions)
    diff = (adj.pct_change(fill_method=None) - m.total_returns).abs().max().max()
    assert diff < 1e-12
    assert (m.fundamentals["available_at"] > m.fundamentals["period_end"]).all()
    assert len(m.bad_ticks) == 6 and not m.close_raw_v1.equals(m.close_raw_v2)
    assert set(m.survivors["status"]) == {"listed"}


@pytest.mark.slow
def test_quick_demo_builds_and_catches_the_planted_mistakes(tmp_path):
    reg = build_demo(tmp_path / "demo", quick=True, log=lambda s: None)
    codes = {w["code"] for w in reg.warnings()}
    expected = {"PIT_VIOLATION", "SURVIVORSHIP_BIAS", "UNADJUSTED_PRICES", "HYPOTHESIS_AMENDED_AFTER_RESULTS",
                "AMENDED_TESTED_ON_SEEN_DATA", "NONREPRODUCIBLE", "NOT_PREREGISTERED"}
    assert expected <= codes, expected - codes
    verdicts = {e.name: e.verdict for e in reg.list_experiments()}
    assert verdicts["Profitability, joined on period end"] == "invalid"
    assert verdicts["Past losers, survivors-only universe"] == "invalid"
    assert verdicts["Weekly reversal (same windows)"] == "invalid"
    assert verdicts["Momentum 12-1, randomised tie-break"] == "invalid"
    assert reg.verify()["ok"]
    out = export_site(reg, tmp_path / "site")
    index = json.loads((out / "data" / "index.json").read_text(encoding="utf-8"))
    assert index["demo"]["synthetic"] is True and len(index["experiments"]) == reg.stats()["experiments"]
    assert all((out / "data" / "experiments" / f"{e['id']}.json").exists() for e in index["experiments"])
    assert index["toy"] and index["toy"][0]["returns_sha256"]
    audit = json.loads((out / "data" / "audit.json").read_text(encoding="utf-8"))
    assert audit[-1]["hash"] == index["audit"]["head"]
    reg.close()
