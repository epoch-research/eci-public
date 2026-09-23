"""
ECI (Epoch Capabilities Index) Fitting

A package for fitting the Item Response Theory model used to compute
ECI scores (model capabilities), EDI scores (benchmark difficulties),
and benchmark discriminabilities (how sharply scores rise with capability).

Two fits of the same curve are available: the Bayesian one in
``eci.bayesian`` (``fit_bayesian_eci``, posterior credible intervals), which
is the default, and the least-squares one in ``eci.fitting``
(``fit_eci_model``, bootstrap confidence intervals). The least-squares fit
is deprecated/historical - kept for reference and reproducing old results,
not for new work.
"""

from .fitting import fit_eci_model
from .dataloader import (
    prepare_benchmark_data,
    download_benchmark_data,
    load_benchmark_metadata,
    load_model_metadata,
)

# Served lazily: importing PyMC costs seconds, and callers that only load data
# or run the least-squares fit should not pay it
_LAZY_EXPORTS = {
    "fit_bayesian_eci": "bayesian",
    "build_bayesian_model": "bayesian",
    "BayesianPriors": "bayesian",
    "evaluate_fit": "evaluation",
    "summarise_evaluation": "evaluation",
    "save_evaluation": "evaluation",
}

__all__ = [
    "fit_eci_model",
    "prepare_benchmark_data",
    "download_benchmark_data",
    "load_benchmark_metadata",
    "load_model_metadata",
    *sorted(_LAZY_EXPORTS),
]
__version__ = "0.2.0"


def __getattr__(name: str):
    """Import the Bayesian fit and its evaluation suite on first use, so
    ``import eci`` stays cheap for callers that never touch them."""
    if name in _LAZY_EXPORTS:
        from importlib import import_module

        return getattr(import_module(f".{_LAZY_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
