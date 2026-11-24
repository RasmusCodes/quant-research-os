//! Example: a toy Rust "backtest" that writes its returns and a qros run manifest.
//!
//!     cargo run --example emit_run -- out
//!     qros init && qros ingest out/qros-run.json
//!
//! The strategy is a deterministic toy (a seeded LCG drives an AR(1) series and a trend rule),
//! so the files and their hashes are identical on every run.

use qros_manifest::{write_returns_csv, Manifest, Value};
use std::env;
use std::path::PathBuf;

fn main() -> std::io::Result<()> {
    let out = PathBuf::from(env::args().nth(1).unwrap_or_else(|| "out".to_string()));
    let seed: u64 = 7;
    let lookback = 20usize;

    // Business days from 2018-01-01 (Monday), weekends skipped.
    let mut dates = Vec::new();
    let mut day = 0u64;
    while dates.len() < 1500 {
        if day % 7 < 5 {
            dates.push(qros_manifest::iso8601_utc(1_514_764_800 + day * 86_400)[..10].to_string());
        }
        day += 1;
    }
    let mut state = seed;
    let mut uniform = move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut r = Vec::with_capacity(dates.len());
    let mut prev = 0.0;
    for _ in 0..dates.len() {
        let z: f64 = (0..12).map(|_| uniform()).sum::<f64>() - 6.0;
        let x = 0.0002 + 0.06 * prev + 0.01 * z;
        r.push(x);
        prev = x;
    }
    let mut net = vec![0.0; r.len()];
    let mut gross = vec![0.0; r.len()];
    let mut cost = vec![0.0; r.len()];
    let mut pos_prev = 0.0;
    for t in 1..r.len() {
        let pos = if t > lookback {
            let s: f64 = r[t - 1 - lookback..t - 1].iter().sum();
            s.signum()
        } else {
            0.0
        };
        gross[t] = pos * r[t];
        cost[t] = 0.0005 * (pos - pos_prev as f64).abs();
        net[t] = gross[t] - cost[t];
        pos_prev = pos;
    }
    let returns_path = out.join("returns.csv");
    write_returns_csv(&returns_path, &dates, &net, Some(&gross), Some(&cost))?;

    let mut m = Manifest::new("rust-example", "toy trend rule (Rust)");
    m.family("rust-example")
        .seed(seed as i64)
        .param("lookback", Value::Int(lookback as i64))
        .param("cost_bps", Value::Float(5.0))
        .train("2018-01-01", "2021-12-31")
        .test("2022-01-03", "2023-09-30")
        .periods_per_year(252.0)
        .command(&["cargo", "run", "--example", "emit_run", "--", "out"])
        .code_field("crate", Value::str(env!("CARGO_PKG_NAME")))
        .code_field("crate_version", Value::str(env!("CARGO_PKG_VERSION")))
        .notes("Written by integrations/rust/qros-manifest/examples/emit_run.rs")
        .tag("rust")
        .returns_csv(&returns_path)?
        .metric("final_position", pos_prev);
    let digest = m.write(out.join("qros-run.json"))?;
    println!("wrote {} (manifest sha256 {})", out.join("qros-run.json").display(), &digest[..16]);
    Ok(())
}
