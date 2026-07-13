"""
ECI/EDI scale construction.

The raw IRT fit (see fitting.py) is identified relative to an anchor
*benchmark* (Winogrande: difficulty = 0, discriminability = 1). The public
ECI scale is instead defined by two anchor *models*:

    Claude 3.5 Sonnet -> 130,   GPT-5 -> 150

connected by an affine map ``eci = a + b * capability``, with the same map
applied to benchmark difficulties (EDI) and its slope ``b`` dividing the
discriminabilities.

Every bootstrap refit lands on its own slightly different raw scale, so the
map must be recomputed PER DRAW from that draw's own anchor capabilities
before any cross-draw statistic (quantile CIs) is taken. By construction the
anchor models then sit at exactly 130/150 in every draw, and their CI cells
are exported as NaN. Applying a single global (a, b) to every draw instead
would let the anchors' own sampling noise leak into every model's CI.

This module is deliberately the only place confidence intervals are
constructed: the raw fit exposes bootstrap draws, not CIs, so there is no
raw-scale CI lying around for a caller to scale incorrectly.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Scaling anchors - these define the ECI scale.
# NB: the names are matched against the "Model" column of the fit input,
# i.e. the Airtable "Model aggregation" names. Renaming an aggregation in
# Airtable requires a coordinated change here (the fit fails loudly if an
# anchor is missing, it never silently rescales).
DEFAULT_ANCHOR_MODEL_LOW = "Claude 3.5 Sonnet"
DEFAULT_ANCHOR_ECI_LOW = 130.0
DEFAULT_ANCHOR_MODEL_HIGH = "GPT-5"
DEFAULT_ANCHOR_ECI_HIGH = 150.0

# A draw only defines a valid ECI scale if the high anchor sits above the
# low anchor by at least this much on the raw scale. Zero spread makes the
# map undefined; negative spread (inverted anchors) would flip the scale.
_MIN_ANCHOR_SPREAD = 1e-12


@dataclass
class EciSamples:
    """Bootstrap draws on the ECI scale, plus the per-draw transforms used.

    All ``*_samples`` lists have one entry per retained draw. Model arrays
    are aligned with ``model_ids``/``model_names``; benchmark arrays with
    ``benchmark_ids``/``benchmark_names``. ``a_samples``/``b_samples`` record
    the affine map (``eci = a + b * raw``) that scaled each draw, so raw
    values can be reconstructed exactly.
    """

    model_ids: list
    model_names: list
    benchmark_ids: list
    benchmark_names: list
    eci_samples: list = field(default_factory=list)
    edi_samples: list = field(default_factory=list)
    slope_samples: list = field(default_factory=list)
    a_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    b_samples: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def num_samples(self) -> int:
        return len(self.eci_samples)


@dataclass
class EciResults:
    """Everything downstream consumers need, on the ECI scale.

    eci_df: one row per model - model_id, Model, capability (raw, for
        reference), eci, and (when bootstrap draws were provided)
        eci_ci_low / eci_ci_high. The anchor models' CI cells are NaN:
        they are fixed by definition, not estimated.
    edi_df: one row per benchmark - benchmark_id, benchmark, is_anchor,
        difficulty and discriminability (raw, for reference), edi,
        discriminability_scaled, and edi_ci_low / edi_ci_high when draws
        were provided. (No NaN masking here: the anchor *benchmark* pins
        the raw scale, but its position on the ECI scale is estimated.)
    scaling: the central affine map and the anchor definitions.
    samples: EciSamples (None when no draws were provided).
    diagnostics: draw bookkeeping - n_draws_total, n_draws_used,
        n_draws_dropped, dropped_reasons (one string per dropped draw).
    """

    eci_df: pd.DataFrame
    edi_df: pd.DataFrame
    scaling: dict
    samples: "EciSamples | None"
    diagnostics: dict


def _scale_from_anchor_capabilities(
    cap_low: float,
    cap_high: float,
    anchor_eci_low: float,
    anchor_eci_high: float,
) -> "tuple[float, float] | None":
    """Affine map (a, b) sending cap_low -> anchor_eci_low, cap_high -> anchor_eci_high.

    Returns None when the anchor spread is non-positive, NaN, or too small
    to define a scale.
    """
    spread = cap_high - cap_low
    if not spread > _MIN_ANCHOR_SPREAD:  # also catches NaN
        return None
    b = (anchor_eci_high - anchor_eci_low) / spread
    a = anchor_eci_low - b * cap_low
    return a, b


def _anchor_capability(model_df: pd.DataFrame, anchor_model: str) -> float:
    rows = model_df.loc[model_df["Model"] == anchor_model, "capability"]
    if rows.empty:
        raise ValueError(f"Anchor model '{anchor_model}' not found")
    return float(rows.iloc[0])


def compute_eci_scores(
    model_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    bootstrap_data: "dict | None" = None,
    *,
    anchor_model_low: str = DEFAULT_ANCHOR_MODEL_LOW,
    anchor_eci_low: float = DEFAULT_ANCHOR_ECI_LOW,
    anchor_model_high: str = DEFAULT_ANCHOR_MODEL_HIGH,
    anchor_eci_high: float = DEFAULT_ANCHOR_ECI_HIGH,
    ci_level: float = 0.90,
) -> EciResults:
    """
    Convert a raw fit to the ECI/EDI scale, with per-draw CI construction.

    Central estimates are scaled with the (a, b) derived from the central
    fit's anchor capabilities. When ``bootstrap_data`` (the third return
    value of ``fit_eci_model``) is provided, every bootstrap draw is
    re-anchored with its OWN (a, b) derived from that draw's anchor
    capabilities, and confidence intervals are the ``ci_level`` quantiles
    of the scaled draws. Draws whose anchors coincide or invert cannot
    define a scale; they are dropped with a warning and recorded in
    ``diagnostics`` - they are never scaled with a fallback map.

    Without ``bootstrap_data`` the result contains point estimates only:
    no CI columns exist, rather than incorrectly-scaled ones.

    Args:
        model_df: 'model_id', 'Model', 'capability' from fit_eci_model.
        bench_df: 'benchmark_id', 'benchmark', 'difficulty',
            'discriminability' from fit_eci_model.
        bootstrap_data: dict with 'model_ids', 'model_names',
            'benchmark_ids', 'benchmark_names', 'capability_samples',
            'difficulty_samples', 'discriminability_samples'.
        anchor_model_low/high: names of the scale-defining models,
            matched against the 'Model' column.
        anchor_eci_low/high: their fixed ECI values.
        ci_level: central quantile mass for the CIs (0.90 -> 5th/95th).

    Returns:
        EciResults (see its docstring for the exact contents).
    """
    cap_low = _anchor_capability(model_df, anchor_model_low)
    cap_high = _anchor_capability(model_df, anchor_model_high)

    central = _scale_from_anchor_capabilities(
        cap_low, cap_high, anchor_eci_low, anchor_eci_high
    )
    if central is None:
        raise ValueError(
            f"Anchor capabilities do not define a scale: "
            f"'{anchor_model_low}'={cap_low}, '{anchor_model_high}'={cap_high} "
            f"(high anchor must exceed low anchor)"
        )
    a, b = central

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

    n_total = len(bootstrap_data["capability_samples"]) if bootstrap_data else 0
    diagnostics = {
        "n_draws_total": n_total,
        "n_draws_used": 0,
        "n_draws_dropped": 0,
        "dropped_reasons": [],
    }

    if not n_total:
        return EciResults(eci_df, edi_df, scaling, None, diagnostics)

    model_names = list(bootstrap_data["model_names"])
    try:
        idx_low = model_names.index(anchor_model_low)
        idx_high = model_names.index(anchor_model_high)
    except ValueError as e:
        raise ValueError(
            f"Anchor model missing from bootstrap samples: {e}"
        ) from None

    samples = EciSamples(
        model_ids=list(bootstrap_data["model_ids"]),
        model_names=model_names,
        benchmark_ids=list(bootstrap_data["benchmark_ids"]),
        benchmark_names=list(bootstrap_data["benchmark_names"]),
    )
    a_list: list[float] = []
    b_list: list[float] = []

    draws = zip(
        bootstrap_data["capability_samples"],
        bootstrap_data["difficulty_samples"],
        bootstrap_data["discriminability_samples"],
    )
    for s, (cap, diff, disc) in enumerate(draws):
        cap = np.asarray(cap, dtype=float)
        draw_scale = _scale_from_anchor_capabilities(
            cap[idx_low], cap[idx_high], anchor_eci_low, anchor_eci_high
        )
        if draw_scale is None:
            reason = (
                f"draw {s}: anchor capabilities do not define a scale "
                f"('{anchor_model_low}'={cap[idx_low]}, "
                f"'{anchor_model_high}'={cap[idx_high]}); draw dropped"
            )
            diagnostics["dropped_reasons"].append(reason)
            warnings.warn(reason)
            continue
        a_s, b_s = draw_scale
        samples.eci_samples.append(a_s + b_s * cap)
        samples.edi_samples.append(a_s + b_s * np.asarray(diff, dtype=float))
        samples.slope_samples.append(np.asarray(disc, dtype=float) / b_s)
        a_list.append(a_s)
        b_list.append(b_s)

    samples.a_samples = np.array(a_list)
    samples.b_samples = np.array(b_list)
    diagnostics["n_draws_used"] = samples.num_samples
    diagnostics["n_draws_dropped"] = n_total - samples.num_samples

    if samples.num_samples >= 2:
        tail = (1.0 - ci_level) / 2.0

        eci_arr = np.vstack(samples.eci_samples)
        ci_low = pd.Series(
            np.quantile(eci_arr, tail, axis=0), index=samples.model_ids
        )
        ci_high = pd.Series(
            np.quantile(eci_arr, 1.0 - tail, axis=0), index=samples.model_ids
        )
        # Align by model_id: model_df is sorted by capability, the sample
        # arrays are in fit insertion order.
        eci_df["eci_ci_low"] = eci_df["model_id"].map(ci_low)
        eci_df["eci_ci_high"] = eci_df["model_id"].map(ci_high)
        # The anchors are fixed by definition (exactly 130/150 in every
        # draw); a CI is not a meaningful statement about them.
        anchor_mask = eci_df["Model"].isin([anchor_model_low, anchor_model_high])
        eci_df.loc[anchor_mask, ["eci_ci_low", "eci_ci_high"]] = np.nan

        edi_arr = np.vstack(samples.edi_samples)
        edi_low = pd.Series(
            np.quantile(edi_arr, tail, axis=0), index=samples.benchmark_ids
        )
        edi_high = pd.Series(
            np.quantile(edi_arr, 1.0 - tail, axis=0), index=samples.benchmark_ids
        )
        edi_df["edi_ci_low"] = edi_df["benchmark_id"].map(edi_low)
        edi_df["edi_ci_high"] = edi_df["benchmark_id"].map(edi_high)
    else:
        warnings.warn(
            f"Only {samples.num_samples} usable bootstrap draw(s) "
            f"(of {n_total}); skipping confidence intervals"
        )

    return EciResults(eci_df, edi_df, scaling, samples, diagnostics)
