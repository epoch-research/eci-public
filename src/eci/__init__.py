"""
ECI (Epoch Capabilities Index) Fitting

A package for fitting the Item Response Theory model used to compute
ECI scores (model capabilities) and EDI scores (benchmark difficulties).
"""

from .fitting import fit_eci_model, load_benchmark_data
from .dataloader import prepare_benchmark_data, download_benchmark_data

__all__ = [
    "fit_eci_model",
    "load_benchmark_data",
    "prepare_benchmark_data",
    "download_benchmark_data",
]
__version__ = "0.2.0"
