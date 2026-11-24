"""A small experiment that reproduces bit for bit in Python and in a browser.

The data are synthetic daily returns from a seeded 32-bit generator (splitmix32 seeding a xoshiro128**
stream). Every operation is an IEEE-754 double addition, multiplication, division or square root,
evaluated in a fixed order, so the JavaScript port in the dashboard (``toy.js``) produces the same
bytes and therefore the same SHA-256. Python's built-in ``sum`` is avoided on purpose: since Python 3.12
it uses compensated summation, which JavaScript does not.

Model: r_t = MU + PHI * r_{t-1} + SIGMA * z_t, with z_t = (u_1 + ... + u_12) - 6 (Irwin-Hall, an
approximately normal draw built from uniforms only). Strategy: hold sign(sum of the last ``lookback``
returns) for the next day, paying ``cost_bps`` per unit of position change.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

MASK32 = 0xFFFFFFFF
MU, PHI, SIGMA = 0.0002, 0.08, 0.01
N_DAYS = 1260
TOY_VERSION = "qros-toy-v1"


def _splitmix32(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B9) & MASK32
    z = state
    z = ((z ^ (z >> 16)) * 0x85EBCA6B) & MASK32
    z = ((z ^ (z >> 13)) * 0xC2B2AE35) & MASK32
    z = (z ^ (z >> 16)) & MASK32
    return state, z


def _rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (32 - k))) & MASK32


class Xoshiro128ss:
    """xoshiro128** (Blackman & Vigna) on 32-bit words."""

    def __init__(self, seed: int):
        state = seed & MASK32
        words = []
        for _ in range(4):
            state, value = _splitmix32(state)
            words.append(value)
        if not any(words):
            words[0] = 1
        self.s = words

    def next_u32(self) -> int:
        s0, s1, s2, s3 = self.s
        result = (_rotl((s1 * 5) & MASK32, 7) * 9) & MASK32
        t = (s1 << 9) & MASK32
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = _rotl(s3, 11)
        self.s = [s0, s1, s2, s3]
        return result

    def uniform(self) -> float:
        return self.next_u32() / 4294967296.0


def generate_returns(seed: int, n: int = N_DAYS) -> list[float]:
    rng = Xoshiro128ss(seed)
    out: list[float] = []
    prev = 0.0
    for _ in range(n):
        acc = 0.0
        for _ in range(12):
            acc = acc + rng.uniform()
        z = acc - 6.0
        r = MU + PHI * prev + SIGMA * z
        out.append(r)
        prev = r
    return out


def run_strategy(returns: list[float], lookback: int, cost_bps: float) -> dict[str, list[float]]:
    n = len(returns)
    cost = cost_bps / 10000.0
    pos = [0.0] * n
    for t in range(n):
        if t + 1 >= lookback:
            s = 0.0
            for k in range(t - lookback + 1, t + 1):
                s = s + returns[k]
            pos[t] = 1.0 if s > 0.0 else (-1.0 if s < 0.0 else 0.0)
    strat = [0.0] * n
    gross = [0.0] * n
    costs = [0.0] * n
    for t in range(1, n):
        held = pos[t - 1]
        before = pos[t - 2] if t >= 2 else 0.0
        gross[t] = held * returns[t]
        costs[t] = cost * abs(held - before)
        strat[t] = gross[t] - costs[t]
    return {"position": pos, "gross": gross, "cost": costs, "net": strat}


def _bytes(values: list[float]) -> bytes:
    return b"".join(struct.pack("<d", 0.0 if v == 0.0 else v) for v in values)


def sha256_floats(values: list[float]) -> str:
    return hashlib.sha256(_bytes(values)).hexdigest()


def stats(net: list[float], periods_per_year: float = 252.0) -> dict[str, float]:
    n = len(net)
    total = 0.0
    for v in net:
        total = total + v
    mean = total / n
    ss = 0.0
    for v in net:
        d = v - mean
        ss = ss + d * d
    sd = math.sqrt(ss / (n - 1))
    equity, peak, max_dd = 1.0, 1.0, 0.0
    for v in net:
        equity = equity * (1.0 + v)
        if equity > peak:
            peak = equity
        dd = equity / peak - 1.0
        if dd < max_dd:
            max_dd = dd
    return {"mean": mean, "sd": sd, "sharpe": mean / sd * math.sqrt(periods_per_year), "final_equity": equity,
            "max_drawdown": max_dd}


def run(seed: int, lookback: int, cost_bps: float = 5.0, n: int = N_DAYS) -> dict[str, Any]:
    returns = generate_returns(seed, n)
    res = run_strategy(returns, lookback, cost_bps)
    return {"version": TOY_VERSION, "seed": seed, "lookback": lookback, "cost_bps": cost_bps, "n": n,
            "data_sha256": sha256_floats(returns), "returns_sha256": sha256_floats(res["net"]),
            "stats": stats(res["net"]), "returns": returns, "net": res["net"], "gross": res["gross"],
            "cost": res["cost"]}


def toy_tsmom(ctx: Any) -> Any:
    """qros entry point: the toy experiment as a managed run (params: lookback, cost_bps, n)."""
    import pandas as pd

    from qros.backtest.interface import BacktestResult

    seed = int(ctx.seed if ctx.seed is not None else 0)
    out = run(seed, int(ctx.param("lookback", 20)), float(ctx.param("cost_bps", 5.0)), int(ctx.param("n", N_DAYS)))
    index = pd.bdate_range("2015-01-01", periods=out["n"])
    return BacktestResult(
        returns=pd.Series(out["net"], index=index), gross_returns=pd.Series(out["gross"], index=index),
        costs=pd.Series(out["cost"], index=index), benchmark=pd.Series(out["returns"], index=index),
        periods_per_year=252.0,
        meta={"toy": {k: out[k] for k in ("version", "seed", "lookback", "cost_bps", "n", "data_sha256",
                                          "returns_sha256", "stats")},
              "synthetic": True})
