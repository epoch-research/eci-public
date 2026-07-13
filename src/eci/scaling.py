"""
ECI/EDI scale construction.

The raw IRT fit (see fitting.py) is identified relative to an anchor
*benchmark* (Winogrande: difficulty = 0, discriminability = 1). The public
ECI scale is defined by two anchor *models*:

    Claude 3.5 Sonnet -> 130,   GPT-5 -> 150

connected by the affine map ``eci = a + b * capability``. The same map sends
benchmark difficulties to EDI, and dividing discriminabilities by ``b``
preserves the IRT predictions. Every bootstrap refit lands on its own raw
scale, so the map is recomputed per draw from that draw's anchor
capabilities before quantile CIs are taken; the anchor models therefore sit
at exactly 130/150 in every draw and their CI cells are NaN.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# The anchor names are matched against the "Model" column of the fit input,
# i.e. the Airtable "Model aggregation" names. Renaming an aggregation in
# Airtable requires a coordinated change here (compute_eci_scores raises if
# an anchor is missing).
DEFAULT_ANCHOR_MODEL_LOW = "Claude 3.5 Sonnet"
DEFAULT_ANCHOR_ECI_LOW = 130.0
DEFAULT_ANCHOR_MODEL_HIGH = "GPT-5"
DEFAULT_ANCHOR_ECI_HIGH = 150.0

_MIN_ANCHOR_SPREAD = 1e-12


@dataclass
class EciResults:
    """ECI-scale results.

    eci_df: model_id, Model, capability, eci, and eci_ci_low/eci_ci_high
        when bootstrap draws were provided. The anchor models' CI cells are
        NaN: their values are fixed by definition, not estimated.
    edi_df: benchmark_id, benchmark, difficulty, discriminability,
        is_anchor, edi, discriminability_scaled, and edi_ci_low/edi_ci_high
        when draws were provided.
    scaling: the central affine map and the anchor definitions.
    draws: the scaled bootstrap draws, or None. Keys: model_ids,
        model_names, benchmark_ids, benchmark_names, the (n_draws, n)
        arrays eci, edi, slope, and the per-draw map coefficients a, b.
    """

    eci_df: pd.DataFrame
    edi_df: pd.DataFrame
    scaling: dict
    draws: dict | None


def _affine_map(cap_low, cap_high, eci_low, eci_high):
    """(a, b) of the map ``eci = a + b * capability`` sending cap_low to
    eci_low and cap_high to eci_high. Vectorized over draws."""
    spread = np.asarray(cap_high) - np.asarray(cap_low)
    bad = np.flatnonzero(~(spread > _MIN_ANCHOR_SPREAD))  # includes NaN
    if bad.size:
        where = f"draw(s) {bad.tolist()}" if spread.ndim else "the central fit"
        raise ValueError(
            f"anchor capabilities in {where} do not define a scale: "
            "the high anchor must sit above the low anchor"
        )
    b = (eci_high - eci_low) / spread
    return eci_low - b * np.asarray(cap_low), b


def _anchor_capability(model_df: pd.DataFrame, anchor_model: str) -> float:
    rows = model_df.loc[model_df["Model"] == anchor_model, "capability"]
    if rows.empty:
        raise ValueError(f"Anchor model '{anchor_model}' not found")
    return float(rows.iloc[0])


def compute_eci_scores(
    model_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    bootstrap_data: dict | None = None,
    *,
    anchor_model_low: str = DEFAULT_ANCHOR_MODEL_LOW,
    anchor_eci_low: float = DEFAULT_ANCHOR_ECI_LOW,
    anchor_model_high: str = DEFAULT_ANCHOR_MODEL_HIGH,
    anchor_eci_high: float = DEFAULT_ANCHOR_ECI_HIGH,
    ci_level: float = 0.90,
) -> EciResults:
    """Rescale a raw fit (the outputs of fit_eci_model) to the ECI scale.

    Central estimates are scaled with the (a, b) derived from the central
    fit's anchor capabilities. When bootstrap_data is provided, each draw is
    scaled with the (a, b) derived from its own anchor capabilities, and the
    CI columns are the ci_level quantiles of the scaled draws. A draw whose
    anchors coincide or invert raises ValueError; without bootstrap_data no
    CI columns are produced.
    """
    a, b = _affine_map(
        _anchor_capability(model_df, anchor_model_low),
        _anchor_capability(model_df, anchor_model_high),
        anchor_eci_low,
        anchor_eci_high,
    )
    eci_df = model_df.copy()
    eci_df["eci"] = a + b * eci_df["capability"]
    edi_df = bench_df.copy()
    edi_df["edi"] = a + b * edi_df["difficulty"]
    edi_df["discriminability_scaled"] = edi_df["discriminability"] / b

    scaling = {
        "a": float(a),
        "b": float(b),
        "anchor_model_low": anchor_model_low,
        "anchor_eci_low": float(anchor_eci_low),
        "anchor_model_high": anchor_model_high,
        "anchor_eci_high": float(anchor_eci_high),
        "ci_level": float(ci_level),
    }

    if not bootstrap_data or not len(bootstrap_data["capability_samples"]):
        return EciResults(eci_df, edi_df, scaling, None)

    model_names = list(bootstrap_data["model_names"])
    try:
        idx_low = model_names.index(anchor_model_low)
        idx_high = model_names.index(anchor_model_high)
    except ValueError as e:
        raise ValueError(f"Anchor model missing from bootstrap samples: {e}") from None

    caps = np.asarray(bootstrap_data["capability_samples"], dtype=float)
    diffs = np.asarray(bootstrap_data["difficulty_samples"], dtype=float)
    discs = np.asarray(bootstrap_data["discriminability_samples"], dtype=float)
    a_s, b_s = _affine_map(
        caps[:, idx_low], caps[:, idx_high], anchor_eci_low, anchor_eci_high
    )
    draws = {
        "model_ids": list(bootstrap_data["model_ids"]),
        "model_names": model_names,
        "benchmark_ids": list(bootstrap_data["benchmark_ids"]),
        "benchmark_names": list(bootstrap_data["benchmark_names"]),
        "eci": a_s[:, None] + b_s[:, None] * caps,
        "edi": a_s[:, None] + b_s[:, None] * diffs,
        "slope": discs / b_s[:, None],
        "a": a_s,
        "b": b_s,
    }

    tail = (1.0 - ci_level) / 2.0
    # model_df is sorted by capability while the draw columns are in fit
    # insertion order, so align the quantiles by id.
    lo, hi = np.quantile(draws["eci"], [tail, 1.0 - tail], axis=0)
    eci_df["eci_ci_low"] = eci_df["model_id"].map(pd.Series(lo, index=draws["model_ids"]))
    eci_df["eci_ci_high"] = eci_df["model_id"].map(pd.Series(hi, index=draws["model_ids"]))
    anchor_rows = eci_df["Model"].isin([anchor_model_low, anchor_model_high])
    eci_df.loc[anchor_rows, ["eci_ci_low", "eci_ci_high"]] = np.nan

    lo, hi = np.quantile(draws["edi"], [tail, 1.0 - tail], axis=0)
    edi_df["edi_ci_low"] = edi_df["benchmark_id"].map(pd.Series(lo, index=draws["benchmark_ids"]))
    edi_df["edi_ci_high"] = edi_df["benchmark_id"].map(pd.Series(hi, index=draws["benchmark_ids"]))

    return EciResults(eci_df, edi_df, scaling, draws)
