import numpy as np
import pandas as pd
import pytest

from qros.registry.core import Registry
from qros.util.timeutil import SimulatedClock


@pytest.fixture
def reg(tmp_path):
    r = Registry.init(tmp_path / ".qros", clock=SimulatedClock("2026-01-05T09:00:00Z"), researcher="tester")
    yield r
    r.close()


@pytest.fixture
def prices():
    idx = pd.bdate_range("2010-01-01", "2019-12-31")
    rng = np.random.default_rng(1)
    data = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, (len(idx), 6)), axis=0))
    return pd.DataFrame(data, index=idx, columns=[f"A{i}" for i in range(6)])
