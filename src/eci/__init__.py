"""
ECI (Epoch Compute Index) Fitting

A package for fitting the Item Response Theory model used to compute
ECI scores (model capabilities) and EDI scores (benchmark difficulties).
"""

from .fitting import fit_eci_model, load_benchmark_data, compute_eci_scores

__all__ = ["fit_eci_model", "load_benchmark_data", "compute_eci_scores"]
__version__ = "0.1.0"
