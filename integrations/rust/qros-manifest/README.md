# qros-manifest

Write [quant-research-os](https://github.com/RasmusCodes/quant-research-os) run manifests from Rust.
No dependencies: the crate carries its own SHA-256 and JSON writer, so it builds offline.

A Rust backtester or simulator records each run by writing its per-period returns (CSV) and a
`qros-run.json` manifest; `qros ingest` then verifies every declared file hash, registers the data
versions and the run, and validates it with exactly the same statistics as native Python runs.

```rust
use qros_manifest::{write_returns_csv, Manifest, Value};

write_returns_csv("out/returns.csv", &dates, &net, Some(&gross), Some(&cost))?;
let mut m = Manifest::new("hft-sim", "maker ladder step 7");
m.family("rq1")
    .hypothesis("H-003")
    .seed(7)
    .param("gamma", Value::Float(0.5))
    .train("2024-01-02", "2024-06-28")
    .test("2024-07-01", "2024-12-31")
    .commit("c9ae4d0", false)
    .command(&["target/release/hftsim", "run", "--seed", "7"])
    .data("book", "bitstamp-btcusd", "data/l3.jsonl")?
    .returns_csv("out/returns.csv")?;
m.write("out/qros-run.json")?;
```

Then, in any folder with a registry:

```
qros ingest out/qros-run.json
```

Run the example and the tests:

```
cargo run --example emit_run -- out
cargo test
```

Scalar-result runs (for example one simulated P&L per run) can skip the returns file and declare
`metric(...)` values plus a `digest(value, json_path)` that the run prints as JSON; `qros reproduce
<id> --cwd <project root>` re-runs the recorded command and compares the digest.
