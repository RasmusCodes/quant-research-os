// Runs the dashboard's JavaScript port of the toy experiment under Node and checks it reproduces the
// Python golden hashes bit for bit. Usage: node tests/js/toy_parity.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import vm from "node:vm";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..");
const source = readFileSync(path.join(root, "src", "qros", "dashboard", "static", "toy.js"), "utf8");
const sandbox = { window: {}, TextEncoder, crypto: globalThis.crypto, Math, Date, Promise, setTimeout, console, ArrayBuffer, DataView, Uint8Array, Uint32Array };
vm.createContext(sandbox);
vm.runInContext(source, sandbox);
const QToy = sandbox.window.QToy;
const golden = JSON.parse(readFileSync(path.join(root, "tests", "golden_toy.json"), "utf8"));

let failures = 0;
for (const g of golden) {
  const res = await QToy.run(g.seed, g.lookback, g.cost_bps, g.n);
  const checks = {
    data: res.data_sha256 === g.data_sha256,
    returns: res.returns_sha256 === g.returns_sha256,
    sharpe: res.stats.sharpe === g.sharpe,
    final_equity: res.stats.final_equity === g.final_equity,
  };
  const ok = Object.values(checks).every(Boolean);
  if (!ok) failures++;
  console.log(`seed ${g.seed} lookback ${g.lookback}: ${ok ? "match" : "MISMATCH " + JSON.stringify(checks)}`);
}
// SHA-256 self-test against Node's own implementation.
const { createHash } = await import("node:crypto");
for (const text of ["", "abc", "x".repeat(55), "x".repeat(64), "ä€😀"]) {
  const ours = QToy.sha256Sync(text);
  const node = createHash("sha256").update(text, "utf8").digest("hex");
  if (ours !== node) { failures++; console.log(`sha256 mismatch for ${JSON.stringify(text)}`); }
}
if (failures) { console.log(`${failures} mismatches`); process.exit(1); }
console.log("all match");
