# Data provenance and versioning

Every result in a registry can be traced to the exact data behind it: which dataset version, what
it was derived from, by which transform, and what was knowable when.

## Sources, datasets and versions

```python
reg.add_source("vendor", vendor="Example Data Inc.", license="internal use", url="https://...")
reg.add_dataset("fundamentals", source="vendor", entity_col="ticker", time_col="period_end",
                available_col="available_at", survivorship_free=True, licensed=True)
v = reg.snapshot("fundamentals", frame, label="2026-01 delivery", knowledge_time="2026-01-05")
```

- A **dataset** is a logical table with flags the hygiene rules read: point-in-time columns
  (`entity_col`, `time_col`, `available_col`), `survivorship_free`, `adjusted` (corporate actions) and
  `licensed` (vendor terms restrict redistribution).
- A **dataset version** is an immutable snapshot. Its id is `dv:` plus the first 12 hex digits of its
  content hash; snapshotting identical content again returns the existing version. SQLite triggers
  refuse updates and deletes of versions.
- `knowledge_time` is the snapshot's as-of: the moment up to which its contents were knowable (a
  vendor delivery date). Derived versions default to the latest knowledge time of their inputs.
- Raw files (any format) can be snapshotted as bytes (`storage = file`, content hash = SHA-256 of the
  file). Data that must stay elsewhere, such as licensed vendor files, can be registered by hash only
  (`register_external_version`, `storage = external`): the result still names the exact version, but
  this registry cannot re-verify its content.

## The canonical fingerprint (`qros-frame-v1`)

A version's identity is a hash of its *content*, not of a file, so the same table hashes the same
whether it was stored as Parquet or CSV, and with any pandas or numpy version. SHA-256 over:

1. the tag `qros-frame-v1` and the shape (rows, columns, as little-endian uint64);
2. the index: a default `RangeIndex` as its length; otherwise its kind and values;
3. each column, **in column-name order** (column order is ignored; row order is content): a block
   `col:<kind>:<name>` followed by its values.

Each block is written as `len(label) | label | len(payload) | payload` (lengths as little-endian uint64).
Values are encoded by kind:

| Kind | Columns | Encoding |
|---|---|---|
| `f` | floats, and integers with missing values | float64 little-endian; every NaN as `0x7ff8000000000000`; `-0.0` as `+0.0` |
| `i` | integers without missing values | int64 little-endian |
| `b` | booleans without missing values | one byte per value |
| `M` | datetimes | int64 nanoseconds since the epoch, UTC; NaT as int64 minimum |
| `m` | timedeltas | int64 nanoseconds |
| `s` | everything else | canonical JSON of each value (null for missing), joined by U+001F |

Series (used for result hashes) use `qros-series-v1`: the index and the values, without the name.

## Storage

Objects live in `objects/<sha[:2]>/<sha[2:]>` under their SHA-256 and are written atomically. Frames
are stored as Parquet when pyarrow is installed, otherwise as **qcsv**: gzip (fixed header, mtime 0)
around a JSON schema line and a CSV body with 17 significant digits, so the bytes are deterministic and
dtypes (datetimes, integers, booleans, strings) survive the round trip. `Registry.load(id)` re-computes
the fingerprint and raises `DataIntegrityError` if the content changed; `qros verify` checks every
object and every version.

## Lineage and transforms

```python
tf = reg.record_transform("adjust-prices", "adjustment", func=adjust_prices, params={"method": "total return"})
adjusted = reg.derive("prices-adjusted", adjust_prices(raw, actions), inputs=[raw_v.id, actions_v.id], transform=tf)
reg.lineage(adjusted.id)   # nodes, edges and transforms back to the raw snapshots
```

A transform's id hashes its name, kind, code reference, the source-file hash of the function and its
parameters, so changing the code or the parameters creates a new transform. Cleaning steps, corporate
action adjustments, universe filters and feature generation are all recorded the same way; the demo
records a price adjustment, a survivors-only filter and two feature-store snapshots.

## Point-in-time access

```python
pit = ctx.pit("fundamentals")                    # in a managed run; or qros.data.PointInTime(frame, ...)
panel = pit.panel(month_ends, "profitability")   # value known at each decision date
```

`PointInTime` keeps both timestamps of each row (the period it describes and when it became known)
and joins on the availability time with a backward as-of merge per entity. Every query is audited:
the audit counts rows whose availability time is after the decision time. Joining on the wrong key,
for example `pit.panel(dates, "profitability", key="period_end")`, is therefore detected from the rows
the backtest actually used; managed runs store the audit with the result and the `PIT_VIOLATION` rule
makes the experiment invalid. `strict=True` raises instead. Rows that become available before the
period they describe are rejected when the view is built.

In managed runs the data a strategy receives is also cut at the start of a sealed holdout
(`time_col`, `available_col` and a `DatetimeIndex` are all respected), so a strategy cannot read
holdout data until an experiment opens the holdout, which the ledger records.

## Corporate actions

`qros.data.corporate_actions` takes an actions table (`entity, date, split, dividend`, dividends per
post-split share):

- `adjust_prices(close, actions)` applies backward factors `f = P_e / ((P_e + D) * k)` for an event on
  ex-date e with closing price P_e, dividend D and split ratio k. With this factor, adjusted-price
  returns equal holding-period total returns `(P_e + D) * k / P_{e-1} - 1` exactly (dividends reinvested
  at the ex-date close); the test suite checks equality to 1e-14.
- `total_returns(close, actions)` computes those returns directly.
- `detect_unadjusted_splits(close, actions)` finds one-day moves of at least 20% whose price ratio is
  within 4% of a split ratio (2, 3, 4, 5, 10, 3:2, 5:4, and the reverse 1:2, 1:3, 1:4 and 1:10). When the jumps coincide with recorded
  splits, the series is unadjusted; strategies can put this check in their result metadata, and the
  `UNADJUSTED_PRICES` rule then makes the experiment invalid.

## Survivorship and missing data

Datasets flagged `survivorship_free = False` raise `SURVIVORSHIP_BIAS` when used (critical, and
therefore invalid, when bound to the `universe` role). Every frame snapshot stores a quality profile:
missing share per column, infinite values, non-positive prices, duplicate keys, entities that start
late or end early (listings and delistings) and the share of stale (repeated) values.

## Licensed data

`licensed = True` on a dataset raises an informational `LICENSED_DATA` warning on every experiment
that uses it, as a reminder that row-level data and possibly derived series may not be published.
The dashboard export contains summaries and weekly equity curves of experiments, never dataset rows.
