"""Transaction-cost models (cost per period in return units)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class CostModel:
    def cost(self, turnover: pd.Series, positions: pd.DataFrame | None = None) -> pd.Series:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


@dataclass
class LinearCost(CostModel):
    """A flat rate per unit of one-way turnover (commission + half-spread), in basis points."""

    bps: float = 10.0

    def cost(self, turnover: pd.Series, positions: pd.DataFrame | None = None) -> pd.Series:
        return turnover.astype(float) * self.bps / 1e4

    def describe(self) -> dict[str, Any]:
        return {"model": "linear", "bps": self.bps}


@dataclass
class SquareRootImpact(CostModel):
    """Half-spread plus square-root market impact: bps + k * sqrt(turnover / participation_scale)."""

    half_spread_bps: float = 5.0
    impact_bps: float = 10.0
    scale: float = 1.0

    def cost(self, turnover: pd.Series, positions: pd.DataFrame | None = None) -> pd.Series:
        t = turnover.astype(float).clip(lower=0)
        rate = self.half_spread_bps / 1e4 + self.impact_bps / 1e4 * np.sqrt(t / self.scale)
        return t * rate

    def describe(self) -> dict[str, Any]:
        return {"model": "sqrt_impact", "half_spread_bps": self.half_spread_bps, "impact_bps": self.impact_bps,
                "scale": self.scale}
