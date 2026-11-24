"""Tiny strategies used by the pipeline tests as managed-run entry points."""

import numpy as np
import pandas as pd

from qros.backtest import backtest_signal, backtest_weights


def trend(ctx):
    px = ctx.data("prices")
    r = px.pct_change(fill_method=None).fillna(0.0)
    sig = np.sign(px.pct_change(int(ctx.param("lookback", 20)), fill_method=None)).fillna(0.0)
    w = sig / sig.shape[1]
    return backtest_weights(w, r, cost=5.0, lag=1, keep_positions=False)


def noisy(ctx):
    px = ctx.data("prices")
    r = px.pct_change(fill_method=None).fillna(0.0)
    noise = np.random.default_rng().normal(size=px.shape)   # deliberately unseeded
    w = pd.DataFrame(np.sign(noise), index=px.index, columns=px.columns) / px.shape[1]
    return backtest_weights(w, r, cost=5.0, lag=1, keep_positions=False)


def fundamental(ctx):
    px = ctx.data("prices")
    r = px.pct_change(fill_method=None).fillna(0.0)
    fund = ctx.pit("fund")
    month_ends = px.groupby([px.index.year, px.index.month]).apply(lambda g: g.index.max())
    dates = pd.DatetimeIndex(month_ends.to_numpy())
    key = None if ctx.param("point_in_time", True) else fund.time
    panel = fund.panel(dates, "profit", key=key, entities=list(px.columns))
    w = panel.rank(axis=1).sub(1.5).fillna(0.0).reindex(px.index).ffill().fillna(0.0)
    return backtest_weights(w, r, cost=5.0, lag=1, keep_positions=False)
