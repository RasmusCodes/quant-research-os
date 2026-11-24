import numpy as np
import pandas as pd


def make_returns(seed=0, n=2600, mu=0.0004, sd=0.01, start="2010-01-01"):
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, n), index=idx)
