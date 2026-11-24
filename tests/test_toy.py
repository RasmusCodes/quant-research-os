"""The browser-reproducible toy: golden hashes in Python and, when Node is available, in JavaScript."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from qros.examples.toy import Xoshiro128ss, generate_returns, run

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests" / "golden_toy.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", GOLDEN, ids=lambda c: f"seed{c['seed']}-lb{c['lookback']}")
def test_python_toy_matches_golden(case):
    res = run(case["seed"], case["lookback"], case["cost_bps"], case["n"])
    assert res["data_sha256"] == case["data_sha256"]
    assert res["returns_sha256"] == case["returns_sha256"]
    assert res["stats"]["sharpe"] == case["sharpe"]          # bit for bit, not approximately
    assert res["stats"]["final_equity"] == case["final_equity"]


def test_generator_first_outputs_are_stable():
    rng = Xoshiro128ss(7)
    first = [rng.next_u32() for _ in range(3)]
    again = Xoshiro128ss(7)
    assert first == [again.next_u32() for _ in range(3)]
    assert all(0 <= x < 2 ** 32 for x in first)
    assert len(set(generate_returns(1, 50))) == 50


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed (the CI job runs this)")
def test_javascript_port_matches_golden():
    proc = subprocess.run(["node", str(ROOT / "tests" / "js" / "toy_parity.mjs")], capture_output=True, text=True,
                          timeout=120, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all match" in proc.stdout
