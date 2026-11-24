# Command reference

`qros [--registry PATH] <command>`. Without `--registry`, qros uses `$QROS_REGISTRY` or the nearest
`.qros` folder above the current directory. `qros <command> --help` shows every option.

| Command | What it does |
|---|---|
| `qros init [PATH] [--researcher NAME]` | create a registry (default `./.qros`) |
| `qros status [--json]` | counts of experiments, hypotheses, versions, warnings, events |
| `qros verify [--quick]` | recompute the audit hash chain; re-hash every object and dataset version |
| `qros hypothesis add TITLE --statement S [--prediction JSON] [--criteria JSON] [--family F] [--register]` | add (and optionally lock) a hypothesis |
| `qros hypothesis register H-001` | pre-register (lock) a draft |
| `qros hypothesis amend H-001 --reason R [--statement S] [--prediction JSON]` | add a version (flagged if after results) |
| `qros hypothesis status H-001 supported [--note N]` | set the status |
| `qros hypothesis list` / `show H-001 [--json]` | list, or show the version history and warnings |
| `qros data source NAME [--vendor] [--url] [--license]` | register a data source |
| `qros data dataset NAME [--source] [--kind] [--entity-col] [--time-col] [--available-col] [--survivorship-free yes/no] [--adjusted yes/no] [--licensed]` | register a dataset and its flags |
| `qros data snapshot DATASET FILE [--label] [--knowledge-time] [--index-col] [--parse-dates ...]` | snapshot a CSV/Parquet table (or any file as bytes) |
| `qros data list [DATASET]` / `show VERSION` / `lineage VERSION` / `verify [VERSION]` | inspect and verify versions |
| `qros holdout add NAME START END [--max-touches N]` / `list` | seal a holdout window; list its ledger |
| `qros run MODULE:FUNCTION [-p k=v ...] [--seed N] [-d role=VERSION ...] [--train A:B] [--test A:B] [--hypothesis H] [--family F] [--holdout HO --open-holdout --purpose P] [--tag T] [--note N]` | managed, reproducible run |
| `qros exp list [--family] [--hypothesis] [--tag] [--verdict] [--json]` | list experiments |
| `qros exp show EXP [--json]` | metrics by window, significance, scorecard, warnings |
| `qros exp compare EXP EXP [...]` | side-by-side table and paired Sharpe-difference bootstraps |
| `qros exp tag EXP TAG...` / `note EXP TEXT` / `invalidate EXP REASON` | annotate |
| `qros exp export EXP OUT.zip [--include-data]` / `import BUNDLE.zip` | move an experiment between registries (import re-checks the result hash) |
| `qros reproduce EXP [--cwd DIR]` | re-run from the record and compare (`--cwd` for ingested runs with a command) |
| `qros validate [EXP] [--family F] [--all]` | recompute validation reports and warnings |
| `qros warnings [--severity S] [--all] [--json]` / `qros ack W-00001 NOTE` | list / acknowledge warnings |
| `qros ingest MANIFEST... [--no-verify]` | ingest run manifests from other repositories |
| `qros adapter NAME ROOT [--limit N] [--experiments a,b]` / `--list` | ingest a known project layout |
| `qros dashboard [--out DIR] [--serve] [--port 8765] [--open]` | export (and serve) the dashboard |
| `qros demo [--out DIR] [--quick] [--seed N]` | build the synthetic demo project |
| `qros toy [--seed 7] [--lookback 20] [--cost-bps 5]` | run the browser-reproducible toy and print its hashes |
| `qros audit [--entity ID] [--limit N]` | show the audit log |

Exit codes: 0 success, 1 a failed check (`verify`, `data verify`, `reproduce` mismatch), 2 a usage or
lookup error.
