"""Canonical serialisation and content hashing.

Every identity in qros is a SHA-256 over a canonical encoding: dataset versions, experiment specs,
results, artifacts and audit events. The encodings below are versioned and documented in
docs/DATA_PROVENANCE.md, so the same content hashes to the same value on any machine and with any
pandas/numpy version, and a changed value always changes the hash.
"""

from __future__ import annotations

import datetime as dt
import decimal
import enum
import hashlib
import json
import math
import struct
from pathlib import Path, PurePath
from typing import Any, Iterable

import numpy as np
import pandas as pd

FRAME_FINGERPRINT_VERSION = "qros-frame-v1"
SERIES_FINGERPRINT_VERSION = "qros-series-v1"
_CANONICAL_NAN_BITS = 0x7FF8000000000000
_NAT_INT = np.iinfo(np.int64).min


# --------------------------------------------------------------------------- JSON


def normalize(obj: Any) -> Any:
    """Convert ``obj`` into plain JSON types with deterministic forms.

    numpy scalars become Python numbers, NaN/inf become ``None``, tuples and sets become lists (sets
    sorted), datetimes become ISO-8601 strings, paths become POSIX strings and dataclass-like objects
    with ``to_dict`` are expanded.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return 0.0 if value == 0.0 else value
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, decimal.Decimal):
        return normalize(float(obj))
    if isinstance(obj, enum.Enum):
        return normalize(obj.value)
    if isinstance(obj, pd.Timestamp):
        if pd.isna(obj):
            return None
        return obj.isoformat()
    if isinstance(obj, (dt.datetime, dt.date, dt.time)):
        return obj.isoformat()
    if isinstance(obj, np.datetime64):
        if np.isnat(obj):
            return None
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, PurePath):
        return obj.as_posix()
    if isinstance(obj, dict):
        return {str(k): normalize(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return sorted((normalize(v) for v in obj), key=lambda v: canonical_json(v))
    if isinstance(obj, (list, tuple)):
        return [normalize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [normalize(v) for v in obj.tolist()]
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return normalize(obj.to_dict())
    raise TypeError(f"Cannot normalise object of type {type(obj).__name__} for hashing")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, shortest round-trip floats, NaN as null."""
    return json.dumps(normalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def short_hash(digest: str | None, n: int = 12) -> str:
    return (digest or "")[:n]


# --------------------------------------------------------------------------- arrays


def float_bytes(values: Iterable[float] | np.ndarray) -> bytes:
    """Little-endian float64 bytes with one canonical NaN and -0.0 folded into +0.0."""
    a = np.array(values, dtype="<f8", copy=True).ravel()
    bits = a.view("<u8")
    bits[np.isnan(a)] = _CANONICAL_NAN_BITS
    a[a == 0.0] = 0.0
    return a.tobytes()


def _datetime_ints(values: Any) -> np.ndarray:
    idx = pd.DatetimeIndex(values)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    arr = idx.values.astype("datetime64[ns]")
    out = arr.view("i8").copy()
    out[np.isnat(arr)] = _NAT_INT
    return out.astype("<i8")


def _column_kind(values: pd.Series | pd.Index) -> str:
    dtype = values.dtype
    if pd.api.types.is_bool_dtype(dtype) and not values.isna().any():
        return "b"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "M"
    if pd.api.types.is_timedelta64_dtype(dtype):
        return "m"
    if pd.api.types.is_integer_dtype(dtype):
        return "i" if not values.isna().any() else "f"
    if pd.api.types.is_float_dtype(dtype):
        return "f"
    return "s"


def _encode_values(values: pd.Series | pd.Index, kind: str) -> bytes:
    if kind == "b":
        return np.asarray(values, dtype=bool).astype("u1").tobytes()
    if kind == "M":
        return _datetime_ints(values).tobytes()
    if kind == "m":
        arr = pd.TimedeltaIndex(values).values.astype("timedelta64[ns]")
        out = arr.view("i8").copy()
        out[np.isnat(arr)] = _NAT_INT
        return out.astype("<i8").tobytes()
    if kind == "i":
        return np.asarray(values, dtype="<i8").tobytes()
    if kind == "f":
        return float_bytes(pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype="f8", na_value=np.nan))
    parts = []
    for v in values.tolist():
        if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA or v is pd.NaT:
            parts.append("null")
        else:
            parts.append(canonical_json(v if not isinstance(v, (np.generic,)) else v.item()))
    return "\x1f".join(parts).encode("utf-8")


def _update_block(h: "hashlib._Hash", label: str, payload: bytes) -> None:
    head = label.encode("utf-8")
    h.update(struct.pack("<Q", len(head)))
    h.update(head)
    h.update(struct.pack("<Q", len(payload)))
    h.update(payload)


def fingerprint_series(series: pd.Series, include_name: bool = False) -> str:
    """Hash of a series' index and values (name optional). Used as the result hash of a run."""
    h = hashlib.sha256()
    h.update(SERIES_FINGERPRINT_VERSION.encode())
    if include_name:
        _update_block(h, "name", canonical_json(series.name).encode())
    index = series.index
    if isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1:
        _update_block(h, "index:range", str(len(index)).encode())
    else:
        kind = _column_kind(index)
        _update_block(h, f"index:{kind}", _encode_values(index, kind))
    kind = _column_kind(series)
    _update_block(h, f"values:{kind}", _encode_values(series, kind))
    return h.hexdigest()


def fingerprint_frame(df: pd.DataFrame, sort_columns: bool = True) -> str:
    """Hash of a DataFrame's index, column names, column kinds and values.

    Column order is ignored by default (columns are hashed in name order); row order is part of the
    content. Integer columns that contain missing values are hashed as floats.
    """
    h = hashlib.sha256()
    h.update(FRAME_FINGERPRINT_VERSION.encode())
    h.update(struct.pack("<QQ", df.shape[0], df.shape[1]))
    index = df.index
    if isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1:
        _update_block(h, "index:range", str(len(index)).encode())
    elif isinstance(index, pd.MultiIndex):
        for level in range(index.nlevels):
            values = index.get_level_values(level)
            kind = _column_kind(values)
            _update_block(h, f"index{level}:{kind}:{canonical_json(index.names[level])}", _encode_values(values, kind))
    else:
        kind = _column_kind(index)
        _update_block(h, f"index:{kind}:{canonical_json(index.name)}", _encode_values(index, kind))
    names = [str(c) for c in df.columns]
    if len(set(names)) != len(names):
        raise ValueError("Cannot fingerprint a frame with duplicate column names")
    order = sorted(range(len(names)), key=lambda i: names[i]) if sort_columns else range(len(names))
    for i in order:
        col = df.iloc[:, i]
        kind = _column_kind(col)
        _update_block(h, f"col:{kind}:{names[i]}", _encode_values(col, kind))
    return h.hexdigest()


def fingerprint_array(values: np.ndarray) -> str:
    a = np.asarray(values)
    h = hashlib.sha256()
    h.update(b"qros-array-v1")
    _update_block(h, "shape", canonical_json(list(a.shape)).encode())
    if np.issubdtype(a.dtype, np.floating):
        _update_block(h, "f", float_bytes(a))
    elif np.issubdtype(a.dtype, np.integer):
        _update_block(h, "i", a.astype("<i8").tobytes())
    elif np.issubdtype(a.dtype, np.bool_):
        _update_block(h, "b", a.astype("u1").tobytes())
    else:
        _update_block(h, "s", canonical_json(a.tolist()).encode())
    return h.hexdigest()


def fingerprint(obj: Any) -> str:
    """Dispatch to the right fingerprint for frames, series, arrays and JSON-like objects."""
    if isinstance(obj, pd.DataFrame):
        return fingerprint_frame(obj)
    if isinstance(obj, pd.Series):
        return fingerprint_series(obj)
    if isinstance(obj, np.ndarray):
        return fingerprint_array(obj)
    if isinstance(obj, (bytes, bytearray)):
        return sha256_bytes(bytes(obj))
    return sha256_json(obj)
