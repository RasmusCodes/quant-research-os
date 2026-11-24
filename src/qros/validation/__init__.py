"""Standardised statistical validation: metrics, Sharpe inference, resampling, overfitting tests."""

from qros.validation.metrics import drawdown_series, infer_periods_per_year, newey_west_tstat, performance_summary
from qros.validation.sharpe import (
    deflated_sharpe_ratio, effective_number_of_trials, expected_max_sharpe, min_track_record_length,
    probabilistic_sharpe_ratio,
)
from qros.validation.multiple_testing import adjust_pvalues, haircut_sharpe
from qros.validation.bootstrap import bootstrap_sharpe, paired_sharpe_difference, stationary_bootstrap_indices
from qros.validation.pbo import probability_of_backtest_overfitting
from qros.validation.splits import PurgedKFold, walk_forward_splits, walk_forward_optimize

__all__ = [
    "performance_summary", "drawdown_series", "infer_periods_per_year", "newey_west_tstat",
    "probabilistic_sharpe_ratio", "deflated_sharpe_ratio", "expected_max_sharpe", "min_track_record_length",
    "effective_number_of_trials", "adjust_pvalues", "haircut_sharpe", "bootstrap_sharpe",
    "paired_sharpe_difference", "stationary_bootstrap_indices", "probability_of_backtest_overfitting",
    "PurgedKFold", "walk_forward_splits", "walk_forward_optimize",
]
