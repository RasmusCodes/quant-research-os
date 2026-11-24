-- qros registry schema, version 1.
-- Timestamps are UTC ISO-8601 strings; JSON columns hold canonical JSON (sorted keys).
-- Append-only tables are enforced with triggers, so history cannot be edited through SQL either.

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE counters (
    name  TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);

-- ------------------------------------------------------------------ data provenance

CREATE TABLE sources (
    id          TEXT PRIMARY KEY,               -- src:<slug>
    name        TEXT NOT NULL UNIQUE,
    vendor      TEXT,
    url         TEXT,
    license     TEXT,
    description TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE datasets (
    id                TEXT PRIMARY KEY,         -- ds:<slug>
    name              TEXT NOT NULL UNIQUE,
    source_id         TEXT REFERENCES sources(id),
    kind              TEXT NOT NULL DEFAULT 'table',   -- raw | table | feature | universe | events | file
    description       TEXT,
    entity_col        TEXT,                     -- e.g. ticker / permno
    time_col          TEXT,                     -- event or period time
    available_col     TEXT,                     -- when each row became known (point-in-time)
    frequency         TEXT,
    survivorship_free INTEGER,                  -- 1 yes, 0 no, NULL unknown
    adjusted          INTEGER,                  -- prices adjusted for corporate actions: 1/0/NULL
    licensed          INTEGER NOT NULL DEFAULT 0,  -- 1 = vendor terms restrict redistribution
    schema_json       TEXT,
    created_at        TEXT NOT NULL
);

CREATE TABLE transforms (
    id          TEXT PRIMARY KEY,               -- tf:<sha12 of definition>
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,                  -- cleaning | adjustment | feature | filter | merge | snapshot
    code_ref    TEXT,                           -- module:function
    code_sha256 TEXT,
    params_json TEXT NOT NULL DEFAULT '{}',
    description TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE dataset_versions (
    id                  TEXT PRIMARY KEY,       -- dv:<first 12 hex of content hash>[-n]
    dataset_id          TEXT NOT NULL REFERENCES datasets(id),
    label               TEXT,
    content_hash        TEXT NOT NULL,          -- canonical fingerprint (format independent)
    fingerprint_version TEXT NOT NULL,
    storage             TEXT,                   -- parquet | qcsv | file | external
    object_sha256       TEXT,                   -- bytes in the object store
    n_rows              INTEGER,
    n_cols              INTEGER,
    columns_json        TEXT,
    time_start          TEXT,
    time_end            TEXT,
    knowledge_time      TEXT,                   -- the snapshot's as-of: what was knowable when taken
    transform_id        TEXT REFERENCES transforms(id),
    quality_json        TEXT,
    notes               TEXT,
    created_at          TEXT NOT NULL,
    UNIQUE (dataset_id, content_hash)
);

CREATE TABLE lineage (
    output_id    TEXT NOT NULL REFERENCES dataset_versions(id),
    input_id     TEXT NOT NULL REFERENCES dataset_versions(id),
    transform_id TEXT REFERENCES transforms(id),
    PRIMARY KEY (output_id, input_id)
);

-- ------------------------------------------------------------------ hypotheses

CREATE TABLE hypotheses (
    id                TEXT PRIMARY KEY,         -- H-001
    title             TEXT NOT NULL,
    family            TEXT,
    status            TEXT NOT NULL DEFAULT 'draft',  -- draft | registered | supported | rejected | inconclusive | abandoned
    researcher        TEXT,
    created_at        TEXT NOT NULL,
    registered_at     TEXT,                     -- pre-registration lock time
    current_version   INTEGER NOT NULL DEFAULT 1,
    status_note       TEXT,
    status_updated_at TEXT
);

CREATE TABLE hypothesis_versions (
    hypothesis_id   TEXT NOT NULL REFERENCES hypotheses(id),
    version         INTEGER NOT NULL,
    statement       TEXT NOT NULL,
    rationale       TEXT,
    prediction_json TEXT,                       -- {"metric": "oos_sharpe", "direction": "positive", ...}
    criteria_json   TEXT,                       -- success criteria, e.g. [{"metric": "dsr", "op": ">=", "value": 0.95}]
    reason          TEXT,                       -- why this version exists
    created_at      TEXT NOT NULL,
    text_sha256     TEXT NOT NULL,
    PRIMARY KEY (hypothesis_id, version)
);

-- ------------------------------------------------------------------ holdouts

CREATE TABLE holdouts (
    id          TEXT PRIMARY KEY,               -- HO-01
    name        TEXT NOT NULL UNIQUE,
    start       TEXT NOT NULL,
    end         TEXT NOT NULL,
    description TEXT,
    max_touches INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL
);

-- ------------------------------------------------------------------ trial families

CREATE TABLE families (
    name            TEXT PRIMARY KEY,
    description     TEXT,
    external_trials INTEGER NOT NULL DEFAULT 0,  -- configurations tried outside qros, declared honestly
    created_at      TEXT NOT NULL
);

-- ------------------------------------------------------------------ experiments

CREATE TABLE experiments (
    id                 TEXT PRIMARY KEY,        -- EXP-0001
    name               TEXT NOT NULL,
    project            TEXT,
    hypothesis_id      TEXT REFERENCES hypotheses(id),
    hypothesis_version INTEGER,
    family             TEXT,                    -- trial family used for multiple-testing accounting
    model              TEXT,
    model_version      TEXT,
    parent_id          TEXT REFERENCES experiments(id),
    status             TEXT NOT NULL,           -- planned | running | completed | failed | invalidated
    source             TEXT NOT NULL DEFAULT 'managed',  -- managed | tracked | ingested
    researcher         TEXT,
    created_at         TEXT NOT NULL,
    started_at         TEXT,
    finished_at        TEXT,
    entry_point        TEXT,
    command            TEXT,
    params_json        TEXT NOT NULL DEFAULT '{}',
    params_hash        TEXT,
    seed               INTEGER,
    code_json          TEXT,
    code_hash          TEXT,
    environment_json   TEXT,
    train_start        TEXT,
    train_end          TEXT,
    test_start         TEXT,
    test_end           TEXT,
    holdout_id         TEXT REFERENCES holdouts(id),
    holdout_opened     INTEGER NOT NULL DEFAULT 0,
    periods_per_year   REAL,
    spec_hash          TEXT,                    -- identity of the inputs: code, params, seed, data, periods
    data_fingerprint   TEXT,                    -- hash over the dataset versions used
    result_hash        TEXT,                    -- fingerprint of the net return series
    verdict            TEXT,
    summary_json       TEXT,
    error              TEXT
);
CREATE INDEX ix_experiments_family ON experiments(family);
CREATE INDEX ix_experiments_hypothesis ON experiments(hypothesis_id);
CREATE INDEX ix_experiments_spec ON experiments(spec_hash);

CREATE TABLE experiment_data (
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    role          TEXT NOT NULL,
    version_id    TEXT NOT NULL REFERENCES dataset_versions(id),
    PRIMARY KEY (experiment_id, role)
);

CREATE TABLE metrics (
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    scope         TEXT NOT NULL,                -- full | is | oos | holdout | fold:<k> | custom
    name          TEXT NOT NULL,
    value         REAL,
    PRIMARY KEY (experiment_id, scope, name)
);

CREATE TABLE notes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT REFERENCES experiments(id),
    hypothesis_id TEXT REFERENCES hypotheses(id),
    author        TEXT,
    text          TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE tags (
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    tag           TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (experiment_id, tag)
);

CREATE TABLE artifacts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    name          TEXT NOT NULL,
    kind          TEXT NOT NULL,                -- returns | positions | signals | turnover | costs | exposure | report | file
    sha256        TEXT NOT NULL,
    size          INTEGER,
    media_type    TEXT,
    meta_json     TEXT,
    created_at    TEXT NOT NULL,
    UNIQUE (experiment_id, name)
);

CREATE TABLE holdout_access (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    holdout_id    TEXT NOT NULL REFERENCES holdouts(id),
    experiment_id TEXT REFERENCES experiments(id),
    purpose       TEXT,
    accessed_at   TEXT NOT NULL
);

CREATE TABLE validations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    computed_at   TEXT NOT NULL,
    n_trials      INTEGER,
    verdict       TEXT,
    report_json   TEXT NOT NULL,
    report_sha256 TEXT NOT NULL
);

CREATE TABLE warnings (
    id               TEXT PRIMARY KEY,          -- W-00001
    scope_key        TEXT NOT NULL,             -- EXP-0001 | family:<name> | hypothesis:<id> | holdout:<id> | registry
    experiment_id    TEXT REFERENCES experiments(id),
    hypothesis_id    TEXT REFERENCES hypotheses(id),
    family           TEXT,
    code             TEXT NOT NULL,
    severity         TEXT NOT NULL,             -- info | warning | critical
    message          TEXT NOT NULL,
    details_json     TEXT,
    active           INTEGER NOT NULL DEFAULT 1,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    acknowledged_at  TEXT,
    acknowledged_note TEXT,
    UNIQUE (scope_key, code)
);

CREATE TABLE reproductions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id    TEXT NOT NULL REFERENCES experiments(id),
    run_at           TEXT NOT NULL,
    status           TEXT NOT NULL,             -- exact | numerical | mismatch | error
    result_hash      TEXT,
    max_abs_diff     REAL,
    environment_json TEXT,
    note             TEXT
);

-- ------------------------------------------------------------------ audit log (hash chain)

CREATE TABLE events (
    seq          INTEGER PRIMARY KEY,
    at           TEXT NOT NULL,
    actor        TEXT,
    action       TEXT NOT NULL,
    entity       TEXT NOT NULL,
    entity_id    TEXT,
    payload_json TEXT NOT NULL,
    prev_hash    TEXT NOT NULL,
    hash         TEXT NOT NULL
);

-- ------------------------------------------------------------------ immutability

CREATE TRIGGER events_append_only_u BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER events_append_only_d BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;

CREATE TRIGGER hypothesis_versions_append_only_u BEFORE UPDATE ON hypothesis_versions
BEGIN SELECT RAISE(ABORT, 'hypothesis versions are append-only'); END;
CREATE TRIGGER hypothesis_versions_append_only_d BEFORE DELETE ON hypothesis_versions
BEGIN SELECT RAISE(ABORT, 'hypothesis versions are append-only'); END;

CREATE TRIGGER holdout_access_append_only_u BEFORE UPDATE ON holdout_access
BEGIN SELECT RAISE(ABORT, 'the holdout ledger is append-only'); END;
CREATE TRIGGER holdout_access_append_only_d BEFORE DELETE ON holdout_access
BEGIN SELECT RAISE(ABORT, 'the holdout ledger is append-only'); END;

CREATE TRIGGER validations_append_only_u BEFORE UPDATE ON validations
BEGIN SELECT RAISE(ABORT, 'validation reports are append-only'); END;

CREATE TRIGGER reproductions_append_only_u BEFORE UPDATE ON reproductions
BEGIN SELECT RAISE(ABORT, 'reproduction records are append-only'); END;

CREATE TRIGGER notes_append_only_u BEFORE UPDATE ON notes
BEGIN SELECT RAISE(ABORT, 'notes are append-only'); END;

CREATE TRIGGER dataset_versions_immutable_u BEFORE UPDATE ON dataset_versions
BEGIN SELECT RAISE(ABORT, 'dataset versions are immutable'); END;
CREATE TRIGGER dataset_versions_immutable_d BEFORE DELETE ON dataset_versions
BEGIN SELECT RAISE(ABORT, 'dataset versions are immutable'); END;

CREATE TRIGGER lineage_immutable_u BEFORE UPDATE ON lineage
BEGIN SELECT RAISE(ABORT, 'lineage is immutable'); END;

CREATE TRIGGER experiments_frozen BEFORE UPDATE OF
    name, hypothesis_id, hypothesis_version, family, entry_point, command, params_json, params_hash, seed,
    code_json, code_hash, train_start, train_end, test_start, test_end, holdout_id, spec_hash,
    data_fingerprint, result_hash, started_at, finished_at, periods_per_year
ON experiments
WHEN OLD.status IN ('completed', 'failed', 'invalidated')
BEGIN SELECT RAISE(ABORT, 'finished experiments are immutable'); END;

CREATE TRIGGER experiments_no_delete BEFORE DELETE ON experiments
WHEN OLD.status IN ('completed', 'failed', 'invalidated')
BEGIN SELECT RAISE(ABORT, 'finished experiments cannot be deleted'); END;

CREATE TRIGGER metrics_frozen_u BEFORE UPDATE ON metrics
WHEN (SELECT status FROM experiments WHERE id = OLD.experiment_id) IN ('completed', 'failed', 'invalidated')
BEGIN SELECT RAISE(ABORT, 'metrics of finished experiments are immutable'); END;
CREATE TRIGGER metrics_frozen_d BEFORE DELETE ON metrics
WHEN (SELECT status FROM experiments WHERE id = OLD.experiment_id) IN ('completed', 'failed', 'invalidated')
BEGIN SELECT RAISE(ABORT, 'metrics of finished experiments are immutable'); END;
CREATE TRIGGER metrics_frozen_i BEFORE INSERT ON metrics
WHEN (SELECT status FROM experiments WHERE id = NEW.experiment_id) IN ('completed', 'failed', 'invalidated')
BEGIN SELECT RAISE(ABORT, 'metrics of finished experiments are immutable'); END;

CREATE TRIGGER experiment_data_frozen_i BEFORE INSERT ON experiment_data
WHEN (SELECT status FROM experiments WHERE id = NEW.experiment_id) IN ('completed', 'failed', 'invalidated')
BEGIN SELECT RAISE(ABORT, 'data links of finished experiments are immutable'); END;
CREATE TRIGGER experiment_data_frozen_d BEFORE DELETE ON experiment_data
WHEN (SELECT status FROM experiments WHERE id = OLD.experiment_id) IN ('completed', 'failed', 'invalidated')
BEGIN SELECT RAISE(ABORT, 'data links of finished experiments are immutable'); END;
