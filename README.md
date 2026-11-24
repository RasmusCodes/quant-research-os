# quant-research-os

An operating system for quantitative research experiments: a database-backed registry, data
provenance and versioning, a standardised statistical validation layer, research-hygiene warnings,
and a dashboard. It answers the questions a careful reviewer asks of any backtest:

| Question | Where qros answers it |
|---|---|
| What hypothesis was tested? | Hypotheses are registered (locked) before testing; every amendment is a new, append-only version |
| What data was available at the time? | Content-addressed dataset versions with a knowledge time; point-in-time joins with an audit of the rows actually used |
| Which model, code and parameters? | Entry point, source-file hash, git state, parameters, seed and environment, combined into a spec hash |
| What was the out-of-sample performance? | Metrics for in-sample, test and sealed-holdout windows, computed by the same code for every strategy |
| How many experiments were run? | Trial families count every distinct configuration (plus trials declared from outside) |
| Is it statistically significant? | Deflated Sharpe ratio, Holm/BH-adjusted p-values, stationary-bootstrap intervals, probability of backtest overfitting |
| Can it be reproduced exactly? | Managed runs re-execute from the record and compare result hashes; a browser can re-run the demo's toy experiment bit for bit |
| Was the hypothesis changed after seeing the results? | Amendments after the first result are flagged, and "confirming" an amended hypothesis on data already seen makes the result invalid |

**Live demo:** [rasmuscodes.github.io/projects/research-os](https://rasmuscodes.github.io/projects/research-os/),
a scripted research project on a synthetic market with planted effects, so you can check the
verdicts against the truth.

## Install

```
pip install "quant-research-os[parquet] @ git+https://github.com/RasmusCodes/quant-research-os"
```

Python 3.10+, with numpy and pandas as the only required dependencies (pyarrow optional, for Parquet
storage). For development: `pip install -e ".[dev,parquet]"` and `pytest`.

## Quickstart

```python
import qros

reg = qros.Registry.init(".qros")                          # a folder: registry.db + objects/
reg.add_source("vendor", license="internal")
reg.add_dataset("prices", source="vendor", adjusted=True, survivorship_free=True)
prices_v1 = reg.snapshot("prices", prices_df, label="v1", knowledge_time="2026-01-05")

h = reg.add_hypothesis("Momentum", "12-1 month winners beat losers over the next month",
                       prediction={"direction": "positive", "metric": "oos.sharpe"},
                       criteria=[{"metric": "oos.sharpe", "op": ">=", "value": 0.3}],
                       register=True)                     # locked before any test

# Managed run: qros calls my_pkg.strategies.momentum(ctx) and can re-run it later.
exp = qros.run_experiment(reg, "my_pkg.strategies:momentum", params={"lookback_months": 12}, seed=1,
                          data={"prices": prices_v1.id}, periods=qros.Periods.of("2005:2014", "2015:2019"),
                          hypothesis=h.id)
print(exp.verdict, reg.latest_validation(exp.id)["verdict_reason"])
qros.reproduce(reg, exp.id)                                # re-runs and compares the result hash
```

Code that runs elsewhere (a notebook, an existing backtester) uses the tracking API instead:

```python
with qros.track("momentum 12-1", hypothesis=h.id, params={"lookback": 12}, seed=7,
                data={"prices": prices_v1.id}, train="2005:2014", test="2015:2019") as run:
    run.log_result(my_backtest())      # a qros.BacktestResult, a return Series or a result frame
```

The same from a terminal:

```
qros init
qros data dataset prices --adjusted yes --survivorship-free yes
qros data snapshot prices prices.csv --index-col date --parse-dates date --label v1
qros hypothesis add "Momentum" --statement "12-1 winners beat losers" --register
qros run my_pkg.strategies:momentum -p lookback_months=12 --seed 1 -d prices=dv:... --train 2005:2014 --test 2015:2019 --hypothesis H-001
qros exp list
qros exp compare EXP-0001 EXP-0002
qros reproduce EXP-0001
qros dashboard --serve
```

## What it does

- **Experiment registry** (SQLite): hypotheses with append-only versions, experiments with
  parameters, seed, code and data versions, windows, metrics, artifacts, tags, notes, validation
  reports and reproductions. Finished experiments are immutable, enforced by SQLite triggers, and
  every write appends to a hash-chained audit log.
- **Data provenance**: datasets and immutable, content-addressed versions; lineage through recorded
  transforms (cleaning, corporate-action adjustment, feature generation); knowledge times;
  point-in-time access with an audit; corporate-action adjustment and unadjusted-split detection;
  quality profiles (missing data, delistings, stale prices).
- **Statistical validation**: walk-forward and purged k-fold cross-validation, out-of-sample
  significance adjusted for every trial evaluated on the test window, the deflated Sharpe ratio,
  probability of backtest overfitting (CSCV), stationary-bootstrap intervals, Holm, Hochberg, BH and BY
  corrections, the Sharpe haircut, parameter-plateau and cost sensitivity, regime analysis.
- **Research-hygiene warnings**: 26 rules, including look-ahead, survivorship, unadjusted prices,
  undeclared or excessive holdout use, hypotheses rewritten after results, excessive tuning,
  non-reproducible results. See [docs/HYGIENE.md](docs/HYGIENE.md).
- **A transparent verdict**: robust, fragile, no edge, invalid or insufficient, from a scorecard
  anyone can check. See [docs/VALIDATION.md](docs/VALIDATION.md).
- **Backtest interface**: one `BacktestResult` (returns, gross, costs, turnover, positions, exposure,
  benchmark, audit metadata) and a small reference engine with an enforced execution lag.
- **Integration without merging code bases**: a language-agnostic run manifest
  ([schema](schemas/run-manifest.schema.json)), a single-file stdlib emitter
  ([integrations/python/qros_emit.py](integrations/python/qros_emit.py)), a zero-dependency Rust crate
  ([integrations/rust/qros-manifest](integrations/rust/qros-manifest)), and adapters for existing
  project layouts. See [docs/INTEGRATION.md](docs/INTEGRATION.md).
- **Dashboard**: a static, dependency-free web app over a registry export (experiment history, model
  comparisons, equity curves, drawdowns, Sharpe distributions, parameter heatmaps, PBO, hypotheses,
  lineage, warnings, in-browser reproduction and audit-chain verification).

## The demo project

`qros demo --out demo-registry` builds the demo: about two months of research on a synthetic market
(seed 20261121) in which a profitability effect is real, momentum is real but fades, short-term
reversal is real only before costs, and futures trend rules have no edge. The script makes the
classic mistakes on purpose: a look-ahead join, a survivor-only universe, unadjusted prices, a
hypothesis rewritten after its results, holdout over-use and an unseeded random generator. Each
is caught. [docs/DEMO.md](docs/DEMO.md) lists every experiment, its verdict and the planted truth.

## Documentation

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Components, data model, experiment lifecycle, identities, immutability, audit chain |
| [DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md) | Dataset versions, fingerprints, lineage, point-in-time access, corporate actions |
| [VALIDATION.md](docs/VALIDATION.md) | Every statistic with its formula and source, the scorecard and the verdict rule |
| [HYGIENE.md](docs/HYGIENE.md) | The warning rules and what triggers each |
| [INTEGRATION.md](docs/INTEGRATION.md) | How another repository (Python, Rust, anything) integrates |
| [CLI.md](docs/CLI.md) | Command reference |
| [DEMO.md](docs/DEMO.md) | The demo project: design, ground truth and results |

## Repository layout

```
src/qros/registry/     SQLite schema, registry API, bundles (export/import)
src/qros/data/         point-in-time access, corporate actions, quality profiles
src/qros/validation/   metrics, Sharpe inference, bootstrap, multiple testing, PBO, splits, sensitivity, regimes, scorecard
src/qros/hygiene/      research-hygiene rules
src/qros/backtest/     BacktestResult, reference engine, cost models, risk statistics
src/qros/adapters/     run manifests and project adapters
src/qros/dashboard/    export + static web app
src/qros/examples/     synthetic market, demo strategies, demo project, browser-reproducible toy
integrations/          stdlib Python emitter, Rust crate
schemas/               run-manifest JSON Schema
tests/                 pytest suite (+ a Node parity test for the browser port)
```

## Testing

`pytest` runs the suite (about a minute without the `slow` quick-demo build: `pytest -m "not slow"`).
Reference values come from independent sources: the multiple-testing corrections and Newey-West
t-statistics are checked against statsmodels, the deflated Sharpe ratio against the worked example in
Bailey and López de Prado (2014), and SHA-256 against the standard test vectors. CI runs the suite on
Linux and Windows, the JavaScript port of the toy experiment under Node (it must reproduce the Python
hashes bit for bit), and the Rust crate's tests plus an end-to-end Rust-to-qros ingest.

## What is not in this repository

No market data. The demo's data are generated by `qros.examples.synthetic_market` at build time, and
local registries (`.qros/`, `demo-registry/`) are git-ignored.

## License

MIT. Copyright (c) 2026 RasmusCodes.
