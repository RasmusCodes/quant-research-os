"""quant-research-os (qros): an operating system for quantitative research experiments.

Main entry points::

    import qros
    reg = qros.Registry.init(".qros")                      # or qros.Registry.open()
    h = reg.add_hypothesis("Momentum", "12-1 momentum earns a premium", register=True)
    exp = qros.run_experiment(reg, "my_pkg.strategies:momentum", params={...}, seed=1,
                              data={"prices": version.id}, periods=qros.Periods.of("2005:2014", "2015:2019"),
                              hypothesis=h.id)
    qros.reproduce(reg, exp.id)

See docs/ for the data model, the validation formulas and the integration guide.
"""

from qros._version import __version__
from qros.backtest import BacktestResult, as_result, backtest_signal, backtest_weights
from qros.evaluate import revalidate_all, revalidate_family, validate_experiment
from qros.registry import DataIntegrityError, Periods, Registry, RegistryError
from qros.runner import RunContext, reproduce, run_experiment
from qros.tracking import Run, track

__all__ = [
    "__version__", "Registry", "RegistryError", "DataIntegrityError", "Periods", "BacktestResult", "as_result",
    "backtest_weights", "backtest_signal", "run_experiment", "reproduce", "RunContext", "track", "Run",
    "validate_experiment", "revalidate_family", "revalidate_all",
]
