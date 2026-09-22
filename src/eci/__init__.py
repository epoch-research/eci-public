"""
ECI (Epoch Capabilities Index) Fitting

A package for fitting the Item Response Theory model used to compute
ECI scores (model capabilities), EDI scores (benchmark difficulties),
and benchmark discriminabilities (how sharply scores rise with capability).

Two fits of the same curve are available: the least-squares one in
``eci.fitting`` (bootstrap confidence intervals) and the Bayesian one in
``eci.bayesian`` (posterior credible intervals). The Bayesian fit needs the
extra dependencies installed with ``pip install -e ".[bayesian]"``, and is
imported lazily so the least-squares fit does not require them.
"""

from .fitting import fit_eci_model
from .dataloader import (
    prepare_benchmark_data,
    download_benchmark_data,
    load_benchmark_metadata,
    load_model_metadata,
)

_BAYESIAN_EXPORTS = {"fit_bayesian_eci", "build_bayesian_model", "BayesianPriors"}

__all__ = [
    "fit_eci_model",
    "prepare_benchmark_data",
    "download_benchmark_data",
    "load_benchmark_metadata",
    "load_model_metadata",
    *sorted(_BAYESIAN_EXPORTS),
]
__version__ = "0.2.0"


def __getattr__(name: str):
    """Import the Bayesian fit on first use, so ``import eci`` stays cheap
    and works without PyMC installed."""
    if name in _BAYESIAN_EXPORTS:
        from . import bayesian

        return getattr(bayesian, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
