# Integrating another repository

A project can report its experiments to quant-research-os without merging code bases. Pick the path
that fits; all four end in the same registry records, metrics, validation and warnings.

| Path | Use when | Dependency on qros |
|---|---|---|
| 1. Managed runs | the strategy can be called as `module:function(ctx)` | the project imports qros |
| 2. Tracking API | the backtest runs in your own code or a notebook | the project imports qros |
| 3. Run manifests | any language (Rust, C++, Julia...) or no qros dependency wanted | none: write JSON + CSV |
| 4. Project adapters | the project already writes outputs in a known layout | none at all |

## 1. Managed runs (strongest guarantees)

```python
# my_project/strategies.py
from qros import backtest_weights

def momentum(ctx):
    prices = ctx.data("prices")                 # the bound dataset version, hash-verified, holdout cut off
    fund = ctx.pit("fundamentals")              # audited point-in-time view
    lookback = ctx.param("lookback_months", 12)
    ...
    return backtest_weights(weights, returns, cost=10.0, lag=1)
```

```python
qros.run_experiment(reg, "my_project.strategies:momentum", params={"lookback_months": 12}, seed=1,
                    data={"prices": "dv:...", "fundamentals": "dv:..."},
                    periods=qros.Periods.of("2005:2014", "2015:2019"), hypothesis="H-001")
```

qros records the source-file hash and git state of the module, hides a sealed holdout from the
strategy, audits point-in-time access, and can re-run the experiment (`qros reproduce`).

## 2. Tracking API

```python
with qros.track("momentum 12-1", hypothesis="H-001", params={"lookback": 12}, seed=7,
                data={"prices": "dv:..."}, train="2005:2014", test="2015:2019",
                holdout="HO-01") as run:
    result = my_backtester.run(...)
    run.log_result(result)       # BacktestResult, a return Series, or a frame with net/gross/cost columns
```

The calling file's hash and git state are recorded. qros cannot hide data from code it does not run,
but it still detects results that reach into a sealed holdout (`HOLDOUT_IMPLICIT_ACCESS`).
`qros.adapters.register_returns(reg, name, returns, **same_kwargs)` does the same in one call.

## 3. Run manifests (any language)

A run writes its per-period returns and a `qros-run.json` manifest
([JSON Schema](../schemas/run-manifest.schema.json)):

```json
{
  "schema": "qros.run-manifest/1",
  "project": "my-sim",
  "experiment": {"name": "maker ladder step 7", "family": "rq1", "hypothesis": {"id": "H-003"}},
  "code": {"repo": "my-sim", "commit": "c9ae4d0", "dirty": false, "language": "rust",
           "command": ["target/release/sim", "--seed", "7"]},
  "params": {"gamma": 0.5},
  "seed": 7,
  "periods": {"train": ["2024-01-02", "2024-06-28"], "test": ["2024-07-01", "2024-12-31"]},
  "data": [{"role": "book", "dataset": "venue-l3", "path": "data/l3.jsonl", "sha256": "…"}],
  "results": {"returns": {"path": "returns.csv", "sha256": "…"}},
  "started_at": "2026-11-20T10:00:00Z"
}
```

`qros ingest qros-run.json` then:

1. checks the manifest against the schema rules and **verifies every declared SHA-256**, refusing the
   run if a file changed;
2. registers each input as a dataset version (stored, or by hash only for large or licensed files);
3. links or imports the hypothesis (an imported pre-registration date is kept as a note; qros uses its
   own registration time, and a run without a declared `started_at` cannot be shown to be
   pre-registered);
4. recomputes every metric from the returns file with the same code as native runs, validates the
   run against its family, and raises warnings.

Writers that need no qros install:

- **Python, stdlib only**: copy [`integrations/python/qros_emit.py`](../integrations/python/qros_emit.py)
  into the project.

  ```python
  from qros_emit import RunManifest, write_returns_csv
  write_returns_csv("out/returns.csv", dates, net, gross=gross, cost=cost)
  m = RunManifest("my-project", "momentum 12-1", family="momentum", seed=7, params={"lookback": 12},
                  train=("2005-01-01", "2014-12-31"), test=("2015-01-01", "2019-12-31"))
  m.add_data("prices", "equity-prices", "data/prices.csv")
  m.set_returns("out/returns.csv")
  m.write("out/qros-run.json")
  ```

- **Rust, zero dependencies**: [`integrations/rust/qros-manifest`](../integrations/rust/qros-manifest)
  (its own SHA-256 and JSON writer; `cargo run --example emit_run -- out` writes a complete example).

Runs with **scalar results only** (one simulated P&L, a forecast score) declare `results.metrics` and
optionally a `results.digest` that the program prints as JSON. They count as trials and get full
provenance; `qros reproduce <id> --cwd <project root>` re-runs `code.command` and compares the digest.

## 4. Project adapters

Adapters read a project's existing output files and turn each run into a manifest, importing nothing
from the project:

```
qros adapter --list
qros adapter rl-trading-lab ../rl-trading-lab --experiments main_spy,holdout_spy
qros adapter statarb ../statarb
qros adapter sports-quant ../sports-quant
qros adapter hft-sim ../hft-sim
```

| Adapter | Reads | Notes |
|---|---|---|
| `rl-trading-lab` | `runs/<experiment>/<run>/{config.json, eval.json, ledger_test.csv}` | one family per run group; `ret` is the net return; holdout runs open the project's 2019+ holdout |
| `statarb` | `results/dashboard/{books.json, daily_*.parquet}`, `reports/NN_preregistration.md` | one family per note; pre-registrations imported as hypotheses (their dates kept as claims) |
| `sports-quant` | `outputs/<league>/backtest_paths.parquet` | daily returns from bankroll paths |
| `hft-sim` | `reports/rqN.manifests.json`, `python/.cache/*.json` | scalar per-agent P&L; reproducible by re-running the recorded `hftsim run` command and comparing the event digest |

Writing a new adapter: subclass `qros.adapters.projects.ProjectAdapter`, implement `discover(root)`
to yield `(manifest_dict, base_dir, frames)` tuples, and register it in `ADAPTERS`.

## Checklist for a new integration

1. Put each configuration of one idea in the same **family**, and declare configurations tried
   outside qros with `declare_external_trials`.
2. **Register the hypothesis first** (`qros hypothesis add ... --register`).
3. Record **windows** (train, test) and use a **sealed holdout** for the final test.
4. Declare **data** by path or hash, and flag datasets honestly (survivorship, adjustment, licence).
5. Record the **code version** (commit and dirty flag, or source hashes) and the **seed**.
6. Emit **gross and cost** columns when you have them: they enable the cost-sensitivity check.
