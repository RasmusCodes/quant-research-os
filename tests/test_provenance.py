import numpy as np
import pandas as pd
import pytest

from qros.artifacts import frame_from_bytes, frame_to_bytes
from qros.data.corporate_actions import adjust_prices, detect_unadjusted_splits, split_check_summary, total_returns
from qros.data.pit import PITViolation, PointInTime, check_availability, with_reporting_lag
from qros.data.quality import quality_report
from qros.registry.core import DataIntegrityError
from qros.util.hashing import fingerprint_frame


def test_snapshot_is_content_addressed_and_idempotent(reg, prices):
    reg.add_source("vendor", license="test")
    reg.add_dataset("prices", source="vendor", adjusted=True, survivorship_free=True)
    v1 = reg.snapshot("prices", prices, label="v1", knowledge_time="2026-01-02")
    again = reg.snapshot("prices", prices.copy(), label="dup")
    assert v1.id == again.id == "dv:" + v1.content_hash[:12]
    assert v1.n_rows == len(prices) and v1.time_start == "2010-01-01"
    loaded = reg.load(v1.id)
    assert fingerprint_frame(loaded) == v1.content_hash
    changed = prices.copy()
    changed.iloc[5, 2] *= 1.01
    v2 = reg.snapshot("prices", changed, label="v2")
    assert v2.id != v1.id
    assert [v.label for v in reg.list_versions("prices")] == ["v1", "v2"]


@pytest.mark.parametrize("storage", ["qcsv", "parquet"])
def test_frame_storage_round_trip_preserves_fingerprint(storage):
    pytest.importorskip("pyarrow") if storage == "parquet" else None
    df = pd.DataFrame({
        "f": [0.1, np.nan, 1e-300, -2.5], "i": [1, 2, 3, 4], "s": ["a", "é", None, "d,e"],
        "t": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-02 13:45:00", None, "2021-06-30 00:00:00"], format="ISO8601"),
        "b": [True, False, True, True],
    }, index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]))
    payload, fmt = frame_to_bytes(df, storage)
    back = frame_from_bytes(payload, fmt)
    assert fingerprint_frame(back) == fingerprint_frame(df)


def test_qcsv_bytes_are_deterministic(prices):
    a, _ = frame_to_bytes(prices.iloc[:50], "qcsv")
    b, _ = frame_to_bytes(prices.iloc[:50].copy(), "qcsv")
    assert a == b


def test_corrupted_object_is_detected(reg, prices):
    reg.add_dataset("prices")
    v = reg.snapshot("prices", prices)
    path = reg.objects.path(v.object_sha256)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises((DataIntegrityError, IOError)):
        reg.load(v.id)
    assert reg.verify_version(v.id)["ok"] is False
    assert not reg.verify()["ok"]


def test_lineage_records_transforms(reg, prices):
    reg.add_dataset("raw")
    reg.add_dataset("events")
    reg.add_dataset("adjusted")
    raw = reg.snapshot("raw", prices)
    ev = reg.snapshot("events", pd.DataFrame({"entity": ["A1"], "date": [prices.index[100]], "split": [2.0], "dividend": [0.0]}))
    tf = reg.record_transform("adjust", "adjustment", func=adjust_prices, params={"method": "total return"})
    adj = reg.derive("adjusted", prices * 0.5, inputs=[raw.id, ev.id], transform=tf)
    lin = reg.lineage(adj.id)
    assert {n["id"] for n in lin["nodes"]} == {raw.id, ev.id, adj.id}
    assert {(e["from"], e["to"]) for e in lin["edges"]} == {(raw.id, adj.id), (ev.id, adj.id)}
    assert lin["transforms"][0]["name"] == "adjust"


def test_external_versions_are_hash_only(reg):
    reg.add_dataset("vendor", licensed=True)
    v = reg.register_external_version("vendor", "a" * 64, label="licensed file")
    assert v.storage == "external"
    assert reg.verify_version(v.id)["ok"] is None
    with pytest.raises(DataIntegrityError):
        reg.load(v.id)


# ----------------------------------------------------------------------------- point in time

def _fundamentals():
    return pd.DataFrame({
        "ticker": ["A", "A", "B", "B"],
        "period_end": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-03-31", "2020-06-30"]),
        "available_at": pd.to_datetime(["2020-05-10", "2020-08-12", "2020-04-20", "2020-07-25"]),
        "value": [1.0, 2.0, 10.0, 20.0],
    })


def test_pit_panel_uses_only_published_rows():
    pit = PointInTime(_fundamentals(), entity="ticker", time="period_end", available="available_at")
    dates = pd.to_datetime(["2020-04-30", "2020-06-30", "2020-07-31", "2020-08-31"])
    panel = pit.panel(dates, "value")
    assert np.isnan(panel.loc["2020-04-30", "A"]) and panel.loc["2020-04-30", "B"] == 10.0
    assert panel.loc["2020-06-30", "A"] == 1.0
    assert panel.loc["2020-07-31", "B"] == 20.0 and panel.loc["2020-07-31", "A"] == 1.0
    assert panel.loc["2020-08-31", "A"] == 2.0
    assert pit.audit.violations == 0


def test_pit_audit_catches_period_end_join():
    pit = PointInTime(_fundamentals(), entity="ticker", time="period_end", available="available_at")
    dates = pd.to_datetime(["2020-04-30", "2020-06-30"])
    pit.panel(dates, "value", key="period_end")
    # A's Q1 report (published 10 May) used on 30 Apr; A's and B's Q2 reports used on 30 Jun.
    # B's Q1 report (published 20 Apr) was already public on 30 Apr, so it is not a violation.
    assert pit.audit.violations == 3
    assert pit.audit.keys_used == ["period_end"]
    assert pit.audit.max_lead_days == pytest.approx(43.0)
    strict = PointInTime(_fundamentals(), entity="ticker", time="period_end", available="available_at", strict=True)
    with pytest.raises(PITViolation):
        strict.panel(dates, "value", key="period_end")


def test_pit_rejects_rows_available_before_their_period():
    bad = _fundamentals()
    bad.loc[0, "available_at"] = pd.Timestamp("2020-01-01")
    with pytest.raises(ValueError):
        PointInTime(bad, entity="ticker", time="period_end", available="available_at")


def test_reporting_lag_and_availability_check():
    f = with_reporting_lag(_fundamentals().drop(columns="available_at"), "period_end", 45)
    assert (f["available_at"] - f["period_end"]).dt.days.eq(45).all()
    feats = pd.DataFrame({"decision": pd.to_datetime(["2020-05-01", "2020-05-20"]),
                          "available": pd.to_datetime(["2020-05-10", "2020-05-10"])})
    assert check_availability(feats, "decision", "available") == {"rows": 2, "violations": 1, "max_lead_days": 9.0}


# ----------------------------------------------------------------------------- corporate actions

def _actions_case():
    idx = pd.bdate_range("2021-01-04", periods=6)
    close = pd.DataFrame({"X": [100.0, 102.0, 51.5, 52.0, 50.0, 51.0], "Y": [20.0, 20.2, 20.4, 19.9, 20.1, 20.3]}, index=idx)
    actions = pd.DataFrame({"entity": ["X", "Y", "X"], "date": [idx[2], idx[3], idx[4]],
                            "split": [2.0, 1.0, 1.0], "dividend": [0.0, 0.5, 0.25]})
    return close, actions


def test_adjusted_returns_equal_total_returns_exactly():
    close, actions = _actions_case()
    adj = adjust_prices(close, actions)
    np.testing.assert_allclose(adj.pct_change().iloc[1:].to_numpy(), total_returns(close, actions).iloc[1:].to_numpy(), rtol=0, atol=1e-14)
    # split day: holder of one old share now has two shares worth 51.5 each
    assert total_returns(close, actions).iloc[2]["X"] == pytest.approx(2 * 51.5 / 102.0 - 1)
    # dividend day: (P + D) / P_prev - 1
    assert total_returns(close, actions).iloc[3]["Y"] == pytest.approx((19.9 + 0.5) / 20.4 - 1)


def test_unadjusted_split_detection():
    close, actions = _actions_case()
    jumps = detect_unadjusted_splits(close, actions)
    assert jumps == [{"entity": "X", "date": "2021-01-06", "price_ratio": pytest.approx(102 / 51.5), "implied_split": 2.0,
                      "recorded_split": True}]
    assert split_check_summary(close, actions)["looks_unadjusted"]
    assert not split_check_summary(adjust_prices(close, actions), actions)["looks_unadjusted"]


def test_quality_report_profiles_a_panel():
    idx = pd.bdate_range("2020-01-01", periods=10)
    df = pd.DataFrame({"A": np.arange(10.0) + 1, "B": [1.0] * 5 + [np.nan] * 5}, index=idx)
    q = quality_report(df)
    assert q["missing"] == {"B": 0.5}
    assert q["entities_ending_early"] == 1
