"""
ECI (Epoch Capabilities Index) Fitting

A package for fitting the Item Response Theory model used to compute
ECI scores (model capabilities) and EDI scores (benchmark difficulties).

fitting.py owns the raw-scale IRT fit; scaling.py owns the conversion to
the public ECI/EDI scale, including all confidence-interval construction.
"""

from .fitting import fit_eci_model, load_benchmark_data, BOOTSTRAP_METHODS
from .scaling import compute_eci_scores, EciResults
from .dataloader import prepare_benchmark_data, download_benchmark_data

__all__ = [
    "fit_eci_model",
    "load_benchmark_data",
    "compute_eci_scores",
    "EciResults",
    "prepare_benchmark_data",
    "download_benchmark_data",
    "BOOTSTRAP_METHODS",
]
__version__ = "0.2.0"
