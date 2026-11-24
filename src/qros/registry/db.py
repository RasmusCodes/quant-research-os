"""SQLite connection handling, schema migrations, counters and the hash-chained audit log."""

from __future__ import annotations

import contextlib
import sqlite3
from importlib import resources
from pathlib import Path
from typing import Any, Iterator

from qros.util.hashing import canonical_json, sha256_text

SCHEMA_VERSION = 1
GENESIS_HASH = "0" * 64


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=60.0, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 60000")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:  # pragma: no cover - e.g. network drives
        pass
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def schema_sql() -> str:
    return resources.files("qros.registry").joinpath("schema.sql").read_text(encoding="utf-8")


def migrate(conn: sqlite3.Connection) -> int:
    """Create or upgrade the schema. Returns the schema version after migrating."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == 0:
        conn.executescript("BEGIN;\n" + schema_sql() + f"\nPRAGMA user_version = {SCHEMA_VERSION};\nCOMMIT;")
        version = SCHEMA_VERSION
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"Registry schema v{version} is newer than this qros (v{SCHEMA_VERSION}); upgrade quant-research-os."
        )
    return version


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE ... COMMIT, re-entrant via savepoints."""
    if conn.in_transaction:
        name = f"sp_{id(object())}"
        conn.execute(f"SAVEPOINT {name}")
        try:
            yield conn
        except BaseException:
            conn.execute(f"ROLLBACK TO {name}")
            conn.execute(f"RELEASE {name}")
            raise
        conn.execute(f"RELEASE {name}")
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def next_counter(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute("SELECT value FROM counters WHERE name = ?", (name,)).fetchone()
    value = (row[0] if row else 0) + 1
    conn.execute(
        "INSERT INTO counters(name, value) VALUES(?, ?) ON CONFLICT(name) DO UPDATE SET value = excluded.value",
        (name, value),
    )
    return value


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


# --------------------------------------------------------------------------- audit log


def event_hash(prev_hash: str, seq: int, at: str, actor: str | None, action: str, entity: str,
               entity_id: str | None, payload_json: str) -> str:
    body = canonical_json({
        "seq": seq, "at": at, "actor": actor, "action": action, "entity": entity,
        "entity_id": entity_id, "payload": payload_json,
    })
    return sha256_text(prev_hash + "|" + body)


def append_event(conn: sqlite3.Connection, at: str, actor: str | None, action: str, entity: str,
                 entity_id: str | None, payload: Any) -> str:
    row = conn.execute("SELECT seq, hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
    seq, prev = (row[0] + 1, row[1]) if row else (1, GENESIS_HASH)
    payload_json = canonical_json(payload if payload is not None else {})
    digest = event_hash(prev, seq, at, actor, action, entity, entity_id, payload_json)
    conn.execute(
        "INSERT INTO events(seq, at, actor, action, entity, entity_id, payload_json, prev_hash, hash)"
        " VALUES(?,?,?,?,?,?,?,?,?)",
        (seq, at, actor, action, entity, entity_id, payload_json, prev, digest),
    )
    return digest


def verify_events(conn: sqlite3.Connection) -> dict[str, Any]:
    """Recompute the hash chain. Any edited, inserted or removed event breaks it."""
    prev, count, first_bad = GENESIS_HASH, 0, None
    expected_seq = 1
    for row in conn.execute("SELECT * FROM events ORDER BY seq"):
        count += 1
        ok = (
            row["seq"] == expected_seq
            and row["prev_hash"] == prev
            and row["hash"] == event_hash(prev, row["seq"], row["at"], row["actor"], row["action"],
                                          row["entity"], row["entity_id"], row["payload_json"])
        )
        if not ok and first_bad is None:
            first_bad = row["seq"]
        prev = row["hash"]
        expected_seq = row["seq"] + 1
    return {"ok": first_bad is None, "events": count, "head": prev, "first_bad_seq": first_bad}
