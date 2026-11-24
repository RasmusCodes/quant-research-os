# Research-hygiene rules

The rules in [`qros.hygiene.rules`](../src/qros/hygiene/rules.py) flag practices that make backtests
look better than they are. Each finding is stored as a warning on one scope (an experiment, a trial
family, a hypothesis or a holdout); re-validation updates warnings in place and clears the ones that no
longer apply. `qros warnings` lists them, `qros ack <id> "<note>"` records an acknowledgement (it does
not hide the warning).

**Integrity** findings make the verdict *invalid* when critical. Any other critical finding blocks a
*robust* verdict through the scorecard's hygiene check.

| Code | Default severity | Scope | Fires when |
|---|---|---|---|
| `PIT_VIOLATION` | critical, integrity | experiment | the point-in-time audit saw rows used before their availability time |
| `SURVIVORSHIP_BIAS` | warning (critical, integrity, for the `universe` role) | experiment | a dataset flagged not survivorship-free is used |
| `UNADJUSTED_PRICES` | warning (critical, integrity, when confirmed) | experiment | an unadjusted dataset is bound as `prices`/`returns`, or the split check matches one-day jumps to recorded splits |
| `HOLDOUT_IMPLICIT_ACCESS` | critical, integrity | experiment | the result reaches into a sealed holdout that was not opened |
| `HOLDOUT_OVERUSE` | critical, integrity | experiment and holdout | this experiment's look at the holdout exceeds its limit (and, on the holdout, more looks than allowed) |
| `HYPOTHESIS_AMENDED_AFTER_RESULTS` | critical | hypothesis | a new version was written after the hypothesis's first result |
| `PREDICTION_REVERSED` | critical | hypothesis | the predicted direction changed between versions |
| `AMENDED_TESTED_ON_SEEN_DATA` | critical, integrity | experiment | a version written after results is tested on windows already examined under earlier versions (a fresh holdout does not count as seen) |
| `NONREPRODUCIBLE` | critical, integrity | experiment | the latest re-run produced a different result (or failed) |
| `DATA_INTEGRITY` | critical, integrity | experiment | a dataset version's stored bytes no longer match their hash |
| `NOT_PREREGISTERED` | warning | experiment | no hypothesis, or it was registered after the run started (imported runs without a declared start time cannot be shown to be pre-registered) |
| `EXCESSIVE_TUNING` | warning (critical when the best in-sample Sharpe is no higher than the noise hurdle) | family | 20 or more trials in the family |
| `SIGNIFICANCE_LOST_AFTER_DEFLATION` | warning | experiment | PSR >= 0.95 alone but DSR < 0.95 |
| `HIGH_PBO` | warning (critical above 0.5) | family | PBO > 0.25 |
| `TEST_WINDOW_REUSE` | info (warning above 10) | family | more than one trial was evaluated on its test window |
| `SHORT_OOS` | warning | experiment | the test window has less than a year of observations |
| `NO_OOS` | warning | experiment | no test window and no opened holdout |
| `IS_OOS_DEGRADATION` | warning | experiment | in-sample Sharpe > 0.5 and out-of-sample below half of it |
| `FRAGILE_PEAK` | warning | experiment | two or more neighbours and plateau ratio < 0.5 |
| `COST_SENSITIVE` | warning | experiment | positive Sharpe at modelled costs, not positive at double |
| `OUTLIER_DEPENDENCE` | warning | experiment | positive Sharpe that is gone after trimming the 1% best and worst periods |
| `IMPLAUSIBLE_SHARPE` | warning | experiment | in-sample Sharpe above 3 on daily data |
| `SINGLE_REGIME` | info | experiment | positive in fewer than half of the volatility regimes |
| `CODE_NOT_VERSIONED` | info | experiment | no code hash or commit, or uncommitted changes |
| `MISSING_SEED` | info | experiment | a managed run without a random seed |
| `LICENSED_DATA` | info | experiment | a dataset flagged licensed is used |

## Why these, and how the registry makes them checkable

- *Look-ahead* is detected from the rows a strategy actually used, because managed runs receive data
  through audited point-in-time views. Tracked or ingested runs can pass their own audit in the
  result metadata (`meta["pit_audit"]`).
- *Holdout discipline* is enforced, not just recorded: in managed runs the data are cut at a sealed
  holdout's start, and every opening is appended to a ledger SQLite will not let anyone edit.
- *HARKing* (hypothesising after the results are known; Kerr 1998) is visible because hypothesis
  versions are append-only and timestamped against the experiments that test them.
- *Multiple testing* is accounted for because every trial in a family is registered, duplicates are
  recognised by their spec hash, and trials run outside qros can be declared
  (`Registry.declare_external_trials`).

Kerr, N. L. (1998). HARKing: hypothesizing after the results are known. *Personality and Social
Psychology Review* 2(3).
