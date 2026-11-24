//! Write [quant-research-os](https://github.com/RasmusCodes/quant-research-os) run manifests from Rust.
//!
//! A Rust backtester or simulator records each run by writing its return series (CSV) and a
//! `qros-run.json` manifest next to it; `qros ingest qros-run.json` then registers the run, verifies
//! every file hash, and validates it with the same statistics as native Python runs. The crate has
//! no dependencies (it carries its own SHA-256 and JSON writer), so it builds offline.
//!
//! ```no_run
//! use qros_manifest::{Manifest, Value};
//! let dates = vec!["2024-01-02".to_string(), "2024-01-03".to_string()];
//! let net = vec![0.001, -0.002];
//! qros_manifest::write_returns_csv("out/returns.csv", &dates, &net, None, None).unwrap();
//! let mut m = Manifest::new("my-sim", "maker ladder step 7");
//! m.family("rq1").seed(7).param("gamma", Value::Float(0.5))
//!     .train("2024-01-02", "2024-06-28").test("2024-07-01", "2024-12-31")
//!     .returns_csv("out/returns.csv").unwrap()
//!     .metric("pnl", 12.5);
//! m.write("out/qros-run.json").unwrap();
//! ```

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const SCHEMA: &str = "qros.run-manifest/1";

// ----------------------------------------------------------------------------- JSON values

/// A JSON value. Objects keep insertion order; non-finite floats are written as `null`.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

impl Value {
    pub fn str(s: impl Into<String>) -> Value {
        Value::Str(s.into())
    }

    pub fn to_json(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Int(i) => {
                let _ = write!(out, "{}", i);
            }
            Value::Float(f) => {
                if f.is_finite() {
                    // Rust's Display for f64 prints the shortest string that round-trips.
                    let _ = write!(out, "{}", f);
                } else {
                    out.push_str("null");
                }
            }
            Value::Str(s) => write_json_string(out, s),
            Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Value::Object(fields) => {
                out.push('{');
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_json_string(out, k);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }
}

/// Escape a string as JSON (RFC 8259): quotes, backslashes and control characters.
pub fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

// ----------------------------------------------------------------------------- SHA-256

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 of a byte slice, as lowercase hex (FIPS 180-4).
pub fn sha256_hex(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    let mut w = [0u32; 64];
    for block in msg.chunks(64) {
        for i in 0..16 {
            w[i] = u32::from_be_bytes([block[4 * i], block[4 * i + 1], block[4 * i + 2], block[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) = (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(v);
        }
    }
    h.iter().map(|v| format!("{:08x}", v)).collect()
}

/// SHA-256 of a file's bytes.
pub fn sha256_file(path: impl AsRef<Path>) -> io::Result<String> {
    Ok(sha256_hex(&fs::read(path)?))
}

// ----------------------------------------------------------------------------- time

/// Seconds since the Unix epoch as an ISO-8601 UTC timestamp, e.g. `2026-11-21T09:30:00Z`.
pub fn iso8601_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m, d, rem / 3_600, (rem % 3_600) / 60, rem % 60)
}

pub fn now_utc() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    iso8601_utc(secs)
}

// ----------------------------------------------------------------------------- returns file

/// Write a returns CSV (`date,net[,gross,cost]`) in the layout `qros ingest` reads by default.
pub fn write_returns_csv(
    path: impl AsRef<Path>,
    dates: &[String],
    net: &[f64],
    gross: Option<&[f64]>,
    cost: Option<&[f64]>,
) -> io::Result<()> {
    if dates.len() != net.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "dates and net returns differ in length"));
    }
    let mut out = String::from("date,net");
    if gross.is_some() {
        out.push_str(",gross");
    }
    if cost.is_some() {
        out.push_str(",cost");
    }
    out.push('\n');
    for i in 0..dates.len() {
        let _ = write!(out, "{},{:?}", dates[i], net[i]);
        if let Some(g) = gross {
            let _ = write!(out, ",{:?}", g[i]);
        }
        if let Some(c) = cost {
            let _ = write!(out, ",{:?}", c[i]);
        }
        out.push('\n');
    }
    if let Some(parent) = path.as_ref().parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(path, out)
}

// ----------------------------------------------------------------------------- manifest

#[derive(Clone, Debug, Default)]
struct DataRef {
    role: String,
    dataset: String,
    path: Option<String>,
    sha256: String,
    source: Option<String>,
    license: Option<String>,
    licensed: bool,
}

/// A run manifest under construction. Paths are written relative to the manifest's folder.
#[derive(Clone, Debug)]
pub struct Manifest {
    project: String,
    name: String,
    family: Option<String>,
    hypothesis_id: Option<String>,
    notes: Option<String>,
    tags: Vec<String>,
    code: Vec<(String, Value)>,
    params: Vec<(String, Value)>,
    seed: Option<i64>,
    train: Option<(String, String)>,
    test: Option<(String, String)>,
    periods_per_year: Option<f64>,
    data: Vec<DataRef>,
    returns: Option<(String, String)>,
    metrics: Vec<(String, f64)>,
    digest: Option<(String, String)>,
    started_at: String,
    base: Option<PathBuf>,
}

impl Manifest {
    pub fn new(project: &str, name: &str) -> Self {
        Manifest {
            project: project.to_string(),
            name: name.to_string(),
            family: None,
            hypothesis_id: None,
            notes: None,
            tags: Vec::new(),
            code: vec![("language".to_string(), Value::str("rust"))],
            params: Vec::new(),
            seed: None,
            train: None,
            test: None,
            periods_per_year: None,
            data: Vec::new(),
            returns: None,
            metrics: Vec::new(),
            digest: None,
            started_at: now_utc(),
            base: None,
        }
    }

    /// Folder that relative paths are resolved against (defaults to the manifest's folder at write time).
    pub fn base_dir(&mut self, dir: impl AsRef<Path>) -> &mut Self {
        self.base = Some(dir.as_ref().to_path_buf());
        self
    }
    pub fn family(&mut self, family: &str) -> &mut Self {
        self.family = Some(family.to_string());
        self
    }
    pub fn hypothesis(&mut self, id: &str) -> &mut Self {
        self.hypothesis_id = Some(id.to_string());
        self
    }
    pub fn notes(&mut self, text: &str) -> &mut Self {
        self.notes = Some(text.to_string());
        self
    }
    pub fn tag(&mut self, tag: &str) -> &mut Self {
        self.tags.push(tag.to_string());
        self
    }
    pub fn param(&mut self, key: &str, value: Value) -> &mut Self {
        self.params.push((key.to_string(), value));
        self
    }
    pub fn seed(&mut self, seed: i64) -> &mut Self {
        self.seed = Some(seed);
        self
    }
    pub fn train(&mut self, start: &str, end: &str) -> &mut Self {
        self.train = Some((start.to_string(), end.to_string()));
        self
    }
    pub fn test(&mut self, start: &str, end: &str) -> &mut Self {
        self.test = Some((start.to_string(), end.to_string()));
        self
    }
    pub fn periods_per_year(&mut self, n: f64) -> &mut Self {
        self.periods_per_year = Some(n);
        self
    }
    /// Record the code version: commit hash and whether the working tree had uncommitted changes.
    pub fn commit(&mut self, commit: &str, dirty: bool) -> &mut Self {
        self.code.push(("commit".to_string(), Value::str(commit)));
        self.code.push(("dirty".to_string(), Value::Bool(dirty)));
        self
    }
    /// The command that reproduces the run, e.g. `["target/release/sim", "--seed", "7"]`.
    pub fn command(&mut self, args: &[&str]) -> &mut Self {
        self.code.push(("command".to_string(), Value::Array(args.iter().map(|a| Value::str(*a)).collect())));
        self
    }
    pub fn code_field(&mut self, key: &str, value: Value) -> &mut Self {
        self.code.push((key.to_string(), value));
        self
    }
    /// Declare an input dataset; the file's SHA-256 is computed now and verified at ingest.
    pub fn data(&mut self, role: &str, dataset: &str, path: impl AsRef<Path>) -> io::Result<&mut Self> {
        let sha = sha256_file(&path)?;
        self.data.push(DataRef {
            role: role.to_string(),
            dataset: dataset.to_string(),
            path: Some(path.as_ref().to_string_lossy().replace('\\', "/")),
            sha256: sha,
            ..Default::default()
        });
        Ok(self)
    }
    /// Declare an input by hash only (e.g. licensed data that stays elsewhere).
    pub fn data_hash(&mut self, role: &str, dataset: &str, sha256: &str, licensed: bool) -> &mut Self {
        self.data.push(DataRef {
            role: role.to_string(),
            dataset: dataset.to_string(),
            sha256: sha256.to_string(),
            licensed,
            ..Default::default()
        });
        self
    }
    pub fn data_source(&mut self, source: &str, license: &str) -> &mut Self {
        if let Some(last) = self.data.last_mut() {
            last.source = Some(source.to_string());
            last.license = Some(license.to_string());
        }
        self
    }
    /// The run's per-period return series (see [`write_returns_csv`]).
    pub fn returns_csv(&mut self, path: impl AsRef<Path>) -> io::Result<&mut Self> {
        let sha = sha256_file(&path)?;
        self.returns = Some((path.as_ref().to_string_lossy().replace('\\', "/"), sha));
        Ok(self)
    }
    pub fn metric(&mut self, key: &str, value: f64) -> &mut Self {
        self.metrics.push((key.to_string(), value));
        self
    }
    /// A digest the run prints as JSON on stdout, so `qros reproduce --cwd` can re-run and compare it.
    pub fn digest(&mut self, value: &str, json_path: &str) -> &mut Self {
        self.digest = Some((value.to_string(), json_path.to_string()));
        self
    }

    fn relative(&self, path: &str, manifest_dir: &Path) -> String {
        let p = Path::new(path);
        let base = self.base.clone().unwrap_or_else(|| manifest_dir.to_path_buf());
        match p.strip_prefix(&base) {
            Ok(rel) => rel.to_string_lossy().replace('\\', "/"),
            Err(_) => path.to_string(),
        }
    }

    pub fn to_value(&self, manifest_dir: &Path) -> Value {
        let mut exp = vec![("name".to_string(), Value::str(self.name.clone()))];
        if let Some(f) = &self.family {
            exp.push(("family".to_string(), Value::str(f.clone())));
        }
        if let Some(h) = &self.hypothesis_id {
            exp.push(("hypothesis".to_string(), Value::Object(vec![("id".to_string(), Value::str(h.clone()))])));
        }
        if let Some(n) = &self.notes {
            exp.push(("notes".to_string(), Value::str(n.clone())));
        }
        if !self.tags.is_empty() {
            exp.push(("tags".to_string(), Value::Array(self.tags.iter().map(|t| Value::str(t.clone())).collect())));
        }
        let mut periods = Vec::new();
        if let Some((a, b)) = &self.train {
            periods.push(("train".to_string(), Value::Array(vec![Value::str(a.clone()), Value::str(b.clone())])));
        }
        if let Some((a, b)) = &self.test {
            periods.push(("test".to_string(), Value::Array(vec![Value::str(a.clone()), Value::str(b.clone())])));
        }
        let data = self
            .data
            .iter()
            .map(|d| {
                let mut f = vec![
                    ("role".to_string(), Value::str(d.role.clone())),
                    ("dataset".to_string(), Value::str(d.dataset.clone())),
                    ("sha256".to_string(), Value::str(d.sha256.clone())),
                ];
                if let Some(p) = &d.path {
                    f.push(("path".to_string(), Value::str(self.relative(p, manifest_dir))));
                }
                if let Some(s) = &d.source {
                    f.push(("source".to_string(), Value::str(s.clone())));
                }
                if let Some(l) = &d.license {
                    f.push(("license".to_string(), Value::str(l.clone())));
                }
                if d.licensed {
                    f.push(("licensed".to_string(), Value::Bool(true)));
                }
                Value::Object(f)
            })
            .collect();
        let mut results = Vec::new();
        if let Some((path, sha)) = &self.returns {
            let mut r = vec![
                ("path".to_string(), Value::str(self.relative(path, manifest_dir))),
                ("sha256".to_string(), Value::str(sha.clone())),
                ("date_column".to_string(), Value::str("date")),
            ];
            if let Some(ppy) = self.periods_per_year {
                r.push(("periods_per_year".to_string(), Value::Float(ppy)));
            }
            results.push(("returns".to_string(), Value::Object(r)));
        }
        if !self.metrics.is_empty() {
            results.push((
                "metrics".to_string(),
                Value::Object(self.metrics.iter().map(|(k, v)| (k.clone(), Value::Float(*v))).collect()),
            ));
        }
        if let Some((value, path)) = &self.digest {
            results.push((
                "digest".to_string(),
                Value::Object(vec![
                    ("value".to_string(), Value::str(value.clone())),
                    ("from".to_string(), Value::str("stdout-json")),
                    ("json_path".to_string(), Value::str(path.clone())),
                ]),
            ));
        }
        let mut root = vec![
            ("schema".to_string(), Value::str(SCHEMA)),
            ("project".to_string(), Value::str(self.project.clone())),
            ("experiment".to_string(), Value::Object(exp)),
            ("code".to_string(), Value::Object(self.code.clone())),
            ("params".to_string(), Value::Object(self.params.clone())),
        ];
        if let Some(seed) = self.seed {
            root.push(("seed".to_string(), Value::Int(seed)));
        }
        if !periods.is_empty() {
            root.push(("periods".to_string(), Value::Object(periods)));
        }
        if let Some(ppy) = self.periods_per_year {
            root.push(("periods_per_year".to_string(), Value::Float(ppy)));
        }
        root.push(("data".to_string(), Value::Array(data)));
        root.push(("results".to_string(), Value::Object(results)));
        root.push(("started_at".to_string(), Value::str(self.started_at.clone())));
        root.push(("finished_at".to_string(), Value::str(now_utc())));
        Value::Object(root)
    }

    /// Write the manifest as JSON; returns its SHA-256.
    pub fn write(&self, path: impl AsRef<Path>) -> io::Result<String> {
        let path = path.as_ref();
        let dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
        if self.returns.is_none() && self.metrics.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "a manifest needs a returns file or metrics"));
        }
        let json = self.to_value(&dir).to_json();
        if !dir.as_os_str().is_empty() {
            fs::create_dir_all(&dir)?;
        }
        fs::write(path, &json)?;
        Ok(sha256_hex(json.as_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        let million = vec![b'a'; 1_000_000];
        assert_eq!(sha256_hex(&million), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    }

    #[test]
    fn json_escaping_and_floats() {
        let v = Value::Object(vec![
            ("s".to_string(), Value::str("a\"b\\c\nd\u{01}é")),
            ("x".to_string(), Value::Float(0.1)),
            ("nan".to_string(), Value::Float(f64::NAN)),
            ("n".to_string(), Value::Int(-3)),
            ("a".to_string(), Value::Array(vec![Value::Bool(true), Value::Null])),
        ]);
        assert_eq!(v.to_json(), r#"{"s":"a\"b\\c\nd\u0001é","x":0.1,"nan":null,"n":-3,"a":[true,null]}"#);
    }

    #[test]
    fn iso_dates() {
        assert_eq!(iso8601_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso8601_utc(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(iso8601_utc(1_798_761_599), "2026-12-31T23:59:59Z");
    }

    #[test]
    fn manifest_round_trip() {
        let dir = std::env::temp_dir().join(format!("qros-manifest-test-{}", std::process::id()));
        let dates: Vec<String> = (1..=5).map(|d| format!("2024-01-0{}", d + 1)).collect();
        let net = vec![0.01, -0.005, 0.002, 0.0, 0.003];
        write_returns_csv(dir.join("returns.csv"), &dates, &net, None, None).unwrap();
        let mut m = Manifest::new("test", "unit test run");
        m.family("f").seed(3).param("k", Value::Int(2)).returns_csv(dir.join("returns.csv")).unwrap().metric("pnl", 1.5);
        m.write(dir.join("qros-run.json")).unwrap();
        let text = fs::read_to_string(dir.join("qros-run.json")).unwrap();
        assert!(text.starts_with(r#"{"schema":"qros.run-manifest/1","project":"test""#));
        assert!(text.contains(r#""path":"returns.csv""#));
        assert!(text.contains(&sha256_file(dir.join("returns.csv")).unwrap()));
        let csv = fs::read_to_string(dir.join("returns.csv")).unwrap();
        assert!(csv.starts_with("date,net\n2024-01-02,0.01\n"));
        let _ = fs::remove_dir_all(&dir);
    }
}
