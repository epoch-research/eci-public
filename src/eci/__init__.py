"""
ECI (Epoch Capabilities Index) Fitting

A package for fitting the Item Response Theory model used to compute
ECI scores (model capabilities), EDI scores (benchmark difficulties),
and benchmark discriminabilities (how sharply scores rise with capability).

Two fits of the same curve are available: the Bayesian one in
``eci.bayesian`` (``fit_bayesian_eci``, posterior credible intervals), which
is the default, and the least-squares one in ``eci.frequentist``
(``fit_eci_model``, bootstrap confidence intervals). The least-squares fit
is deprecated/historical - kept for reference and reproducing old results,
not for new work.
"""

from .bayesian import (
    DEFAULT_ANCHOR_ECI_HIGH,
    DEFAULT_ANCHOR_ECI_LOW,
    DEFAULT_ANCHOR_MODEL_HIGH,
    DEFAULT_ANCHOR_MODEL_LOW,
    BayesianPriors,
    build_bayesian_model,
    fit_bayesian_eci,
)
from .dataloader import (
    download_benchmark_data,
    load_benchmark_metadata,
    load_model_metadata,
    prepare_benchmark_data,
)
from .evaluation import evaluate_fit, save_evaluation, summarise_evaluation
from .frequentist import fit_eci_model

__all__ = [
    "DEFAULT_ANCHOR_ECI_HIGH",
    "DEFAULT_ANCHOR_ECI_LOW",
    "DEFAULT_ANCHOR_MODEL_HIGH",
    "DEFAULT_ANCHOR_MODEL_LOW",
    "BayesianPriors",
    "build_bayesian_model",
    "download_benchmark_data",
    "evaluate_fit",
    "fit_bayesian_eci",
    "fit_eci_model",
    "load_benchmark_metadata",
    "load_model_metadata",
    "prepare_benchmark_data",
    "save_evaluation",
    "summarise_evaluation",
]
__version__ = "0.2.0"
