"""A synthetic equity market with KNOWN ground truth, for demonstrating and testing the research OS.

Everything here is simulated. Because the data-generating process is known, each demo experiment can be
scored against the truth, which is how we check that the validation layer separates real effects from
noise. Planted structure:

* **Market**: a two-state (calm / turbulent) Markov volatility regime, Student-t shocks, no
  predictable mean, so market-timing rules have no true edge.
* **Profitability ("quality")**: each firm's quarterly profitability follows an AR(1); its current
  level earns a small daily drift (a real, persistent effect). Reports are published 30-75 days after
  quarter end, and the price jumps on the publication day in proportion to the surprise. Using the
  numbers at quarter end instead of at publication is look-ahead and captures those jumps.
* **Momentum**: a slowly varying expected-return component that makes past winners keep winning,
  switched off gradually between 2013 and 2016 (a real effect that decays).
* **Short-term reversal**: a transient price component that mean-reverts within days (real before
  costs; high turnover).
* **Delistings**: firms whose price falls 85% below their listing price delist with a -30% final
  return; a few are acquired at a premium. New firms list in their place.
* **Corporate actions**: splits (more likely at high prices) and quarterly dividends, so unadjusted
  prices differ from total-return prices.
* **Vendor errors**: data version 1 contains a handful of bad prints that version 2 corrects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MarketConfig:
    seed: int = 20261121
    n_assets: int = 200
    start: str = "2005-01-03"
    end: str = "2024-12-31"
    mkt_mu: float = 0.0003
    calm_vol: float = 0.008
    turb_vol: float = 0.020
    p_calm_turb: float = 0.01
    p_turb_calm: float = 0.04
    df_market: float = 5.0
    idio_vol_lo: float = 0.010
    idio_vol_hi: float = 0.022
    turb_idio_mult: float = 1.5
    df_idio: float = 4.0
    beta_lo: float = 0.6
    beta_hi: float = 1.4
    q_rho: float = 0.8
    q_drift: float = 1.5e-4
    surprise_jump: float = 0.02
    lag_lo: int = 30
    lag_hi: int = 75
    mom_sd: float = 4e-4
    mom_halflife_days: float = 300.0
    mom_decay_start: str = "2013-01-01"
    mom_decay_end: str = "2016-01-01"
    rev_rho: float = 0.8
    rev_psi: float = 0.010
    delist_level: float = 0.20
    delist_return: float = -0.50
    fut_calm_vol: float = 0.009
    fut_turb_vol: float = 0.018
    acquisition_prob_daily: float = 0.00004
    acquisition_premium: float = 0.25
    split_threshold: float = 250.0
    split_prob_daily: float = 0.0001
    payer_share: float = 0.6
    div_yield_lo: float = 0.01
    div_yield_hi: float = 0.04
    n_bad_ticks: int = 6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SyntheticMarket:
    config: MarketConfig
    dates: pd.DatetimeIndex
    total_returns: pd.DataFrame          # dates x tickers, NaN outside listing
    close_raw_v2: pd.DataFrame           # unadjusted closes (correct)
    close_raw_v1: pd.DataFrame           # unadjusted closes with vendor errors
    actions: pd.DataFrame                # entity, date, split, dividend
    fundamentals: pd.DataFrame           # ticker, period_end, available_at, profitability
    universe: pd.DataFrame               # ticker, first_date, last_date, status, reason
    market_returns: pd.Series            # the latent market factor
    regime: pd.Series                    # 0 calm / 1 turbulent
    futures_returns: pd.Series           # a driftless futures contract with no predictable component
    bad_ticks: list[dict[str, Any]] = field(default_factory=list)
    truth: dict[str, Any] = field(default_factory=dict)

    @property
    def survivors(self) -> pd.DataFrame:
        return self.universe[self.universe["status"] == "listed"].reset_index(drop=True)


def _momentum_scale(dates: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    frac = ((dates - t0) / (t1 - t0)).to_numpy(dtype=float)
    return np.clip(1.0 - frac, 0.0, 1.0)


def generate(cfg: MarketConfig | None = None) -> SyntheticMarket:
    cfg = cfg or MarketConfig()
    rng = np.random.default_rng(cfg.seed)
    dates = pd.bdate_range(cfg.start, cfg.end)
    T, N = len(dates), cfg.n_assets

    regime = np.zeros(T, dtype=np.int8)
    switch = rng.random(T)
    for t in range(1, T):
        p = cfg.p_calm_turb if regime[t - 1] == 0 else cfg.p_turb_calm
        regime[t] = 1 - regime[t - 1] if switch[t] < p else regime[t - 1]
    t_scale_m = math.sqrt(cfg.df_market / (cfg.df_market - 2))
    market = cfg.mkt_mu + np.where(regime == 1, cfg.turb_vol, cfg.calm_vol) * rng.standard_t(cfg.df_market, T) / t_scale_m
    # A futures contract with its own volatility regimes and zero drift: nothing to time.
    fut_regime = np.zeros(T, dtype=np.int8)
    fut_switch = rng.random(T)
    for t in range(1, T):
        p = cfg.p_calm_turb if fut_regime[t - 1] == 0 else cfg.p_turb_calm
        fut_regime[t] = 1 - fut_regime[t - 1] if fut_switch[t] < p else fut_regime[t - 1]
    futures = np.where(fut_regime == 1, cfg.fut_turb_vol, cfg.fut_calm_vol) * rng.standard_t(cfg.df_market, T) / t_scale_m

    # Quarter ends: last business day of each calendar quarter.
    qe_calendar = pd.date_range(cfg.start, cfg.end, freq="QE")
    qe_pos = sorted({int(dates.searchsorted(d, side="right") - 1) for d in qe_calendar if d >= dates[0]})
    qe_set = set(qe_pos)
    mom_scale = _momentum_scale(dates, cfg.mom_decay_start, cfg.mom_decay_end)
    mom_rho = 0.5 ** (1.0 / cfg.mom_halflife_days)
    mom_innov = cfg.mom_sd * math.sqrt(1 - mom_rho ** 2)
    t_scale_i = math.sqrt(cfg.df_idio / (cfg.df_idio - 2))

    capacity = N * 6
    R = np.full((T, capacity), np.nan)
    beta = np.empty(capacity)
    ivol = np.empty(capacity)
    price0 = np.empty(capacity)
    payer = np.zeros(capacity, dtype=bool)
    yld = np.zeros(capacity)
    first_day = np.full(capacity, -1)
    last_day = np.full(capacity, -1)
    reason = [""] * capacity

    def new_stock(sid: int, day: int) -> None:
        beta[sid] = rng.uniform(cfg.beta_lo, cfg.beta_hi)
        ivol[sid] = rng.uniform(cfg.idio_vol_lo, cfg.idio_vol_hi)
        price0[sid] = float(np.exp(rng.uniform(np.log(10), np.log(150))))
        payer[sid] = rng.random() < cfg.payer_share
        yld[sid] = rng.uniform(cfg.div_yield_lo, cfg.div_yield_hi) if payer[sid] else 0.0
        first_day[sid] = day

    slot_stock = np.arange(N)
    for sid in range(N):
        new_stock(sid, 0)
    next_id = N
    q_state = rng.standard_normal(N)
    q_reported = q_state.copy()                # what the market last learned (seeded with the start level)
    mu = rng.standard_normal(N) * cfg.mom_sd
    zeta = rng.standard_normal(N)
    cum = np.ones(N)
    pending_listing: dict[int, list[int]] = {}
    announcements: dict[int, list[tuple[int, float, int]]] = {}
    fundamentals: list[tuple[int, int, int, float]] = []   # stock id, period-end pos, available pos, value
    delist_next: dict[int, str] = {}

    for t in range(T):
        # New listings scheduled for today fill their slots.
        for slot in pending_listing.pop(t, []):
            sid = next_id
            next_id += 1
            if sid >= capacity:
                raise RuntimeError("increase capacity")
            new_stock(sid, t)
            slot_stock[slot] = sid
            q_state[slot] = rng.standard_normal()
            q_reported[slot] = q_state[slot]
            mu[slot] = rng.standard_normal() * cfg.mom_sd
            zeta[slot] = rng.standard_normal()
            cum[slot] = 1.0
        alive = slot_stock >= 0
        # Quarter roll happens the day after a quarter end.
        if t - 1 in qe_set:
            eps = rng.standard_normal(N)
            q_state = cfg.q_rho * q_state + math.sqrt(1 - cfg.q_rho ** 2) * eps
        mu = mom_rho * mu + mom_innov * rng.standard_normal(N)
        zeta_new = cfg.rev_rho * zeta + math.sqrt(1 - cfg.rev_rho ** 2) * rng.standard_normal(N)
        transient = cfg.rev_psi * (zeta_new - zeta)
        zeta = zeta_new
        sids = np.where(alive, slot_stock, 0)
        shock = rng.standard_t(cfg.df_idio, N) / t_scale_i * ivol[sids] * (cfg.turb_idio_mult if regime[t] else 1.0)
        r = cfg.q_drift * q_state + mom_scale[t] * mu + beta[sids] * market[t] + shock + transient
        for slot, surprise, sid in announcements.pop(t, []):
            if slot_stock[slot] == sid:
                r[slot] += cfg.surprise_jump * surprise
        acquired = rng.random(N) < cfg.acquisition_prob_daily
        for slot in range(N):
            sid = slot_stock[slot]
            if sid < 0:
                continue
            if slot in delist_next:
                r[slot] = cfg.delist_return if delist_next[slot] == "delisted" else cfg.acquisition_premium
                reason[sid] = delist_next.pop(slot)
                last_day[sid] = t
                slot_stock[slot] = -1
                R[t, sid] = r[slot]
                pending_listing.setdefault(t + 2, []).append(slot)
                continue
            r[slot] = max(r[slot], -0.9)
            R[t, sid] = r[slot]
            cum[slot] *= 1.0 + r[slot]
            if cum[slot] < cfg.delist_level and t >= first_day[sid] + 20:
                delist_next[slot] = "delisted"
            elif acquired[slot]:
                delist_next[slot] = "acquired"
        # Quarter-end reports: value is this quarter's profitability, published after a lag.
        if t in qe_set:
            for slot in range(N):
                sid = slot_stock[slot]
                if sid < 0 or slot in delist_next:
                    continue
                lag_days = int(rng.integers(cfg.lag_lo, cfg.lag_hi + 1))
                avail = int(dates.searchsorted(dates[t] + pd.Timedelta(days=lag_days)))
                if avail >= T:
                    avail = T  # published after the sample ends: never visible
                surprise = q_state[slot] - cfg.q_rho * q_reported[slot]
                fundamentals.append((sid, t, avail, float(q_state[slot])))
                q_reported[slot] = q_state[slot]
                if avail < T:
                    announcements.setdefault(avail, []).append((slot, float(surprise), sid))

    n_stocks = next_id
    tickers = [f"SYN{i + 1:03d}" for i in range(n_stocks)]
    for sid in range(n_stocks):
        if last_day[sid] < 0:
            last_day[sid] = T - 1
            reason[sid] = reason[sid] or "listed"
    total = pd.DataFrame(R[:, :n_stocks], index=dates, columns=tickers)

    # Unadjusted closes and corporate actions consistent with the total returns.
    closes = np.full((T, n_stocks), np.nan)
    actions = []
    for sid in range(n_stocks):
        f, l = int(first_day[sid]), int(last_day[sid])
        price = price0[sid]
        closes[f, sid] = price
        months_paid = set()
        for t in range(f + 1, l + 1):
            r = R[t, sid]
            split, div = 1.0, 0.0
            if payer[sid] and t - 1 in qe_set and t not in months_paid:
                div = round(yld[sid] / 4 * price, 4)
                months_paid.add(t)
            gross_price = price * (1 + r) - div
            if gross_price <= 0:
                div, gross_price = 0.0, price * (1 + r)
            if (gross_price > cfg.split_threshold and rng.random() < 0.05) or (rng.random() < cfg.split_prob_daily and gross_price > 40):
                split = 3.0 if gross_price > 2 * cfg.split_threshold else 2.0
            price = gross_price / split
            closes[t, sid] = price
            if split != 1.0 or div != 0.0:
                # Dividends are recorded per post-split share, the convention of qros.data.corporate_actions.
                actions.append((tickers[sid], dates[t], split, div / split))
    close_v2 = pd.DataFrame(closes, index=dates, columns=tickers)
    actions_df = pd.DataFrame(actions, columns=["entity", "date", "split", "dividend"])

    # Vendor errors in version 1: a few one-day bad prints.
    close_v1 = close_v2.copy()
    bad = []
    tries = 0
    while len(bad) < cfg.n_bad_ticks and tries < 1000:
        tries += 1
        sid = int(rng.integers(0, n_stocks))
        f, l = int(first_day[sid]), int(last_day[sid])
        if l - f < 50:
            continue
        t = int(rng.integers(f + 10, l - 10))
        factor = float(rng.choice([0.55, 0.6, 1.5, 1.7]))
        close_v1.iat[t, sid] = close_v2.iat[t, sid] * factor
        bad.append({"ticker": tickers[sid], "date": dates[t].strftime("%Y-%m-%d"), "factor": factor})

    fund = pd.DataFrame(fundamentals, columns=["sid", "pe", "av", "profitability"])
    fund = fund[fund["av"] < T]
    fund_df = pd.DataFrame({
        "ticker": [tickers[s] for s in fund["sid"]],
        "period_end": dates[fund["pe"].to_numpy()],
        "available_at": dates[fund["av"].to_numpy()],
        "profitability": fund["profitability"].round(6).to_numpy(),
    }).sort_values(["available_at", "ticker"]).reset_index(drop=True)

    universe = pd.DataFrame({
        "ticker": tickers,
        "first_date": dates[first_day[:n_stocks]],
        "last_date": dates[last_day[:n_stocks]],
        "status": ["listed" if reason[s] == "listed" else reason[s] for s in range(n_stocks)],
    })
    truth = {
        "market_timing_edge": False,
        "futures_trend": False,
        "quality_effect": {"present": True, "daily_drift_per_sd": cfg.q_drift, "report_lag_days": [cfg.lag_lo, cfg.lag_hi],
                           "announcement_jump_per_sd": cfg.surprise_jump},
        "momentum_effect": {"present_until": cfg.mom_decay_start, "gone_from": cfg.mom_decay_end},
        "short_term_reversal": {"present": True, "note": "transient price component; gross of costs"},
        "long_term_reversal": False,
        "delisted": int((universe["status"] == "delisted").sum()),
        "acquired": int((universe["status"] == "acquired").sum()),
        "splits": int((actions_df["split"] != 1.0).sum()),
        "dividends": int((actions_df["dividend"] != 0.0).sum()),
        "bad_ticks_v1": len(bad),
    }
    return SyntheticMarket(cfg, dates, total, close_v2, close_v1, actions_df, fund_df, universe,
                           pd.Series(market, index=dates, name="market"), pd.Series(regime, index=dates, name="regime"),
                           pd.Series(futures, index=dates, name="futures"), bad, truth)
