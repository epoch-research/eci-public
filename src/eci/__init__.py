"""
ECI (Epoch Compute Index) Fitting

A package for fitting the Item Response Theory model used to compute
ECI scores (model capabilities) and EDI scores (benchmark difficulties).
"""

from .fitting import fit_eci_model, load_benchmark_data, compute_eci_scores, fit_capabilities_given_benchmarks
from .dataloader import prepare_benchmark_data, download_benchmark_data

__all__ = [
    "fit_eci_model",
    "load_benchmark_data",
    "compute_eci_scores",
    "prepare_benchmark_data",
    "download_benchmark_data",
    "fit_capabilities_given_benchmarks",
]
__version__ = "0.1.0"
