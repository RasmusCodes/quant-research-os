import numpy as np
import pandas as pd
import pytest

from qros.util.hashing import (
    canonical_json, fingerprint_array, fingerprint_frame, fingerprint_series, float_bytes, normalize, sha256_text,
)


def test_canonical_json_is_order_independent_and_compact():
    a = canonical_json({"b": 1, "a": [1.0, 2.5], "c": {"y": None, "x": "é"}})
    b = canonical_json({"c": {"x": "é", "y": None}, "a": [1.0, 2.5], "b": 1})
    assert a == b == '{"a":[1.0,2.5],"b":1,"c":{"x":"é","y":null}}'


def test_normalize_handles_numpy_nan_and_dates():
    out = normalize({"i": np.int64(3), "f": np.float32(0.5), "nan": float("nan"), "t": pd.Timestamp("2020-01-02"),
                     "neg0": -0.0, "s": {2, 1}})
    assert out == {"i": 3, "f": 0.5, "nan": None, "t": "2020-01-02T00:00:00", "neg0": 0.0, "s": [1, 2]}


def test_float_bytes_canonicalises_nan_and_negative_zero():
    a = float_bytes([0.0, float("nan")])
    b = float_bytes([-0.0, np.float64(np.nan) * -1])
    assert a == b


def test_frame_fingerprint_ignores_column_order_not_values():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    assert fingerprint_frame(df) == fingerprint_frame(df[["y", "x"]])
    changed = df.copy()
    changed.iloc[0, 0] = 1.0000000001
    assert fingerprint_frame(df) != fingerprint_frame(changed)
    assert fingerprint_frame(df) != fingerprint_frame(df.iloc[::-1])


def test_frame_fingerprint_is_stable_across_datetime_resolution_and_int_types():
    idx_ns = pd.DatetimeIndex(["2020-01-01", "2020-01-02"]).astype("datetime64[ns]")
    idx_s = idx_ns.astype("datetime64[s]")
    a = pd.DataFrame({"v": np.array([1, 2], dtype="int32")}, index=idx_ns)
    b = pd.DataFrame({"v": np.array([1, 2], dtype="int64")}, index=idx_s)
    assert fingerprint_frame(a) == fingerprint_frame(b)


def test_series_fingerprint_ignores_name():
    s = pd.Series([0.1, 0.2], index=pd.to_datetime(["2020-01-01", "2020-01-02"]), name="a")
    assert fingerprint_series(s) == fingerprint_series(s.rename("b"))
    assert fingerprint_series(s, include_name=True) != fingerprint_series(s.rename("b"), include_name=True)


def test_array_fingerprint_shape_matters():
    a = np.arange(6, dtype=float)
    assert fingerprint_array(a) != fingerprint_array(a.reshape(2, 3))


def test_duplicate_columns_rejected():
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError):
        fingerprint_frame(df)


def test_sha256_text_known_value():
    assert sha256_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
