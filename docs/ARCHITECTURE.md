# Architecture

quant-research-os is a Python package around one idea: every number a researcher reports should be
traceable to a locked question, exact data, exact code and the full list of things that were tried.

```
             +---------------- qros CLI / Python API -----------------+
             |                                                         |
  strategies -> RunContext (data by version, sealed holdout, PIT audit) |
  notebooks  -> track() ------------------------------+                 |
  other repos -> run manifest (JSON) -> ingest -------+                 |
                                                      v                 |
                         finalize: metrics per window, artifacts, result hash
                                                      v
                         validate: family context -> statistics -> hygiene rules -> scorecard -> verdict
                                                      v
   Registry = SQLite (schema.sql, triggers, audit chain) + content-addressed object store
                                                      v
                         dashboard export (static JSON) -> web app (local or the public demo)
```

## Modules

| Module | Responsibility |
|---|---|
| `qros.registry` | `Registry` (the API), `schema.sql`, migrations, the hash-chained audit log, bundles |
| `qros.artifacts` | Content-addressed object store; frame storage (Parquet, or the deterministic "qcsv" format) |
| `qros.data` | Point-in-time access and audit, corporate actions, data-quality profiles |
| `qros.backtest` | `BacktestResult`, the reference weight/signal engine, cost models, risk statistics |
| `qros.validation` | Metrics, Sharpe inference, bootstrap, multiple testing, PBO, splits, sensitivity, regimes, the report and verdict |
| `qros.hygiene` | Research-hygiene rules and how their findings become warnings |
| `qros.evaluate` | Freezing a finished experiment and (re)computing its validation report |
| `qros.runner` | Managed runs and reproduction |
| `qros.tracking` | The `track()` context manager for code that runs outside qros |
| `qros.adapters` | Run manifests (any language) and adapters for existing project layouts |
| `qros.compare` | Side-by-side comparison with paired bootstrap tests |
| `qros.dashboard` | Registry export and the static web app |
| `qros.examples` | Synthetic market, demo strategies, the scripted demo, the browser-reproducible toy |

## Data model

A registry is a folder (`.qros/` by convention) holding `registry.db` and `objects/`. The tables
(see [`schema.sql`](../src/qros/registry/schema.sql)):

| Table | One row per | Mutable? |
|---|---|---|
| `sources`, `datasets` | data vendor / logical dataset (with flags: point-in-time columns, survivorship-free, adjusted, licensed) | no (insert-only) |
| `dataset_versions` | immutable content-addressed snapshot | **trigger-enforced immutable** |
| `transforms`, `lineage` | recorded transformation / input-to-output edge | lineage immutable |
| `hypotheses` | research question (status and registration time) | status only |
| `hypothesis_versions` | wording, prediction, success criteria, reason for change | **append-only** |
| `families` | trial family (with honestly declared external trials) | external count only |
| `experiments` | one evaluated configuration | **frozen once finished** (inputs, windows, hashes) |
| `experiment_data` | role -> dataset version used | frozen once finished |
| `metrics` | (experiment, scope, name) value; scopes `is`, `oos`, `holdout`, `full`, `custom` | frozen once finished |
| `artifacts` | stored output (result table, positions, metadata, manifest) | - |
| `holdouts`, `holdout_access` | sealed window / every look at it | ledger **append-only** |
| `validations` | validation report (JSON) with the trial count at the time | **append-only** |
| `warnings` | finding of a hygiene rule on a scope | updated in place, never deleted |
| `reproductions` | re-run and its outcome | append-only |
| `events` | audit-log entry (hash chain) | **append-only** |

## Experiment lifecycle

1. **Create** (`create_experiment`, called by `run_experiment`, `track` and `ingest_manifest`): records
   name, hypothesis and the version being tested, family, model and version, parent, parameters, seed,
   entry point or command, code fingerprint, environment, windows, holdout (and whether it is opened),
   dataset versions. Computes the **spec hash** = SHA-256 over code hash, parameters, seed, data content
   hashes, windows and holdout use. Opening a holdout appends to its ledger.
2. **Run**: for managed runs, `RunContext.data(role)` loads the bound version, verifies its hash and
   cuts everything from a sealed holdout's start; `RunContext.pit(role)` returns an audited
   point-in-time view.
3. **Finalize** (`complete_experiment`): validates the result (sorted unique dates, finite returns,
   net = gross - costs), detects undeclared holdout use, computes the standard metrics per window,
   stores the result table and metadata as artifacts, computes the **result hash** (fingerprint of the
   net return series) and freezes the record.
4. **Validate**: builds the family context, computes the statistics, runs the hygiene rules, grades
   the scorecard and stores a new validation report. Re-validation (for example after the family has
   grown) appends a new report; earlier ones stay, so the history shows a deflated Sharpe ratio fall as
   trials accumulate.
5. **Reproduce** (managed runs): re-executes with the recorded inputs and compares result hashes
   (`exact`), values (`numerical`, max |diff| <= 1e-10) or reports `mismatch`. Ingested runs can be
   re-run from their recorded command (`qros reproduce <id> --cwd <project root>`).

## Identities

| Hash | Over | Used for |
|---|---|---|
| dataset content hash | canonical frame fingerprint (or file bytes) | version id `dv:<12 hex>`, verification |
| object sha256 | stored bytes | object-store address, corruption detection |
| code hash | defining module, function and source-file hash (managed/tracked) or commit + declared source hash (ingested) | spec hash |
| spec hash | code hash, parameters, seed, data content hashes, windows, holdout use | trial de-duplication, comparisons |
| result hash | net return series (dates and values) | reproduction |
| text hash | hypothesis statement, rationale, prediction, criteria | locking a hypothesis version |
| event hash | previous hash + canonical event body | audit chain |

All are SHA-256 over canonical encodings, documented in [DATA_PROVENANCE.md](DATA_PROVENANCE.md).

## Immutability and the audit chain

Every write goes through `Registry`, which appends an event `{seq, at, actor, action, entity,
entity_id, payload}` whose hash is `sha256(prev_hash + "|" + canonical_json(event))`. SQLite triggers
refuse updates and deletes of events, hypothesis versions, dataset versions, the holdout ledger,
validation reports and the inputs and metrics of finished experiments, so history cannot be rewritten
through the API or plain SQL. `qros verify` recomputes the chain and re-hashes every stored object.

The chain is tamper-evident, not tamper-proof: someone with write access to the file can drop the
triggers and rewrite every event consistently, which yields a different head hash. Committing or
publishing the head hash (the dashboard prints it) pins it.

## Concurrency and storage

SQLite runs in WAL mode with a long busy timeout; writes use short `BEGIN IMMEDIATE` transactions.
Objects are written atomically (temporary file + rename) and never modified. Frames are stored as
Parquet when pyarrow is installed, otherwise as qcsv (gzip with a fixed header, CSV with 17 significant
digits and a JSON schema line), so a registry works with numpy and pandas alone.

## Extending

- **New metrics**: add to `validation.metrics.performance_summary`; all experiments get them on
  re-validation.
- **New hygiene rules**: add an entry to `hygiene.rules.RULES` and a check in `experiment_findings`,
  `family_findings` or `hypothesis_findings`; mark integrity failures in `validation.report.INTEGRITY_CODES`.
- **New adapters**: subclass `adapters.projects.ProjectAdapter` and yield run manifests.
- **Another database**: the SQL is standard apart from SQLite triggers and `PRAGMA`s; the `Registry`
  class is the only code that touches the connection.
