"""Content-addressed object store and frame serialisation.

Objects live at ``<registry>/objects/<sha[:2]>/<sha[2:]>`` and are written atomically, so an
artifact's address is its SHA-256 and a stored object can always be re-verified.

Frames are stored as Parquet when pyarrow is installed, otherwise as "qcsv": gzip-compressed CSV
(fixed gzip header, 17 significant digits) behind a JSON schema line that restores dtypes. Either way
the dataset's identity is its canonical fingerprint (qros.util.hashing), not the file bytes.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qros.util.hashing import sha256_bytes

QCSV_MAGIC = "#qros-csv-v1 "


class ObjectStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, sha: str) -> Path:
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError(f"Not a SHA-256 hex digest: {sha!r}")
        return self.root / sha[:2] / sha[2:]

    def exists(self, sha: str) -> bool:
        return self.path(sha).is_file()

    def put_bytes(self, data: bytes) -> str:
        sha = sha256_bytes(data)
        target = self.path(sha)
        if target.is_file():
            return sha
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return sha

    def put_file(self, path: str | Path) -> str:
        return self.put_bytes(Path(path).read_bytes())

    def get_bytes(self, sha: str, verify: bool = True) -> bytes:
        data = self.path(sha).read_bytes()
        if verify and sha256_bytes(data) != sha:
            raise IOError(f"Object {sha[:12]} is corrupt: its bytes no longer match its address")
        return data

    def verify(self, sha: str) -> bool:
        try:
            return sha256_bytes(self.path(sha).read_bytes()) == sha
        except FileNotFoundError:
            return False

    def copy_to(self, sha: str, dest: str | Path) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.path(sha), dest)
        return dest


# --------------------------------------------------------------------------- frames


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def _kind(values: pd.Series | pd.Index) -> str:
    dtype = values.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_integer_dtype(dtype):
        return "int" if not values.isna().any() else "float"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    return "string"


def _restore(values: pd.Series, kind: str) -> pd.Series:
    if kind == "datetime":
        return pd.to_datetime(values, format="ISO8601").astype("datetime64[ns]")
    if kind == "int":
        return values.astype("int64")
    if kind == "float":
        return values.astype("float64")
    if kind == "bool":
        return values.map({"True": True, "False": False, True: True, False: False}).astype(bool)
    return values.astype(object).where(values.notna(), None)


def frame_to_qcsv(df: pd.DataFrame) -> bytes:
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names cannot be stored")
    index_kind = None if isinstance(df.index, pd.RangeIndex) else _kind(df.index)
    schema = {
        "columns": [[str(c), _kind(df[c])] for c in df.columns],
        "index": None if index_kind is None else [df.index.name, index_kind],
    }
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    for name, kind in schema["columns"]:
        if kind == "datetime":
            out[name] = pd.to_datetime(out[name]).dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    write_index = index_kind is not None
    if write_index and index_kind == "datetime":
        out.index = pd.DatetimeIndex(out.index).strftime("%Y-%m-%dT%H:%M:%S.%f")
    text = out.to_csv(index=write_index, float_format="%.17g", lineterminator="\n")
    payload = (QCSV_MAGIC + json.dumps(schema, sort_keys=True) + "\n" + text).encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=6, filename="") as gz:
        gz.write(payload)
    return buf.getvalue()


def frame_from_qcsv(data: bytes) -> pd.DataFrame:
    text = gzip.decompress(data).decode("utf-8")
    first, rest = text.split("\n", 1)
    if not first.startswith(QCSV_MAGIC):
        raise ValueError("Not a qcsv object")
    schema = json.loads(first[len(QCSV_MAGIC):])
    index_spec = schema.get("index")
    dtypes = {name: (str if kind in ("string", "datetime", "bool") else None) for name, kind in schema["columns"]}
    # pandas >= 3 parses floats round-trip by default and deprecates the option; older pandas needs it.
    precision = {} if int(pd.__version__.split(".")[0]) >= 3 else {"float_precision": "round_trip"}
    df = pd.read_csv(io.StringIO(rest), keep_default_na=True, index_col=0 if index_spec else None,
                     dtype={k: v for k, v in dtypes.items() if v is not None}, **precision)
    for name, kind in schema["columns"]:
        df[name] = _restore(df[name], kind)
    if index_spec:
        name, kind = index_spec
        idx = pd.Series(df.index)
        df.index = pd.Index(_restore(idx, kind), name=name)
    return df


def frame_to_bytes(df: pd.DataFrame, storage: str | None = None) -> tuple[bytes, str]:
    storage = storage or ("parquet" if _has_pyarrow() else "qcsv")
    if storage == "parquet":
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow", index=not isinstance(df.index, pd.RangeIndex))
        return buf.getvalue(), "parquet"
    if storage == "qcsv":
        return frame_to_qcsv(df), "qcsv"
    raise ValueError(f"Unknown storage format {storage!r}")


def frame_from_bytes(data: bytes, storage: str) -> pd.DataFrame:
    if storage == "parquet":
        df = pd.read_parquet(io.BytesIO(data), engine="pyarrow")
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col].dtype):
                df[col] = df[col].astype("datetime64[ns]")
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.astype("datetime64[ns]")
        return df
    if storage == "qcsv":
        return frame_from_qcsv(data)
    raise ValueError(f"Unknown storage format {storage!r}")


def series_to_frame(series: pd.Series, name: str = "value") -> pd.DataFrame:
    return series.rename(name).to_frame()


def to_jsonable_series(series: pd.Series, decimals: int | None = None) -> dict[str, Any]:
    values = series.to_numpy(dtype=float)
    if decimals is not None:
        values = np.round(values, decimals)
    if isinstance(series.index, pd.DatetimeIndex):
        index = series.index.strftime("%Y-%m-%d").tolist()
    else:
        index = [str(i) for i in series.index]
    return {"index": index, "values": [None if not np.isfinite(v) else float(v) for v in values]}
