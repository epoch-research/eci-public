"""
Bayesian fitting for the Epoch Capabilities Index (ECI).

Same IRT curve as :mod:`eci.fitting`,

    performance = sigmoid(slope * (capability - difficulty))

but fitted by MCMC rather than least squares, so every quantity comes out as
a posterior distribution and the intervals are read straight off it.

The observed scores live in [0, 1], so the likelihood is a Beta with a
per-benchmark concentration ``kappa``:

    mu[m, b]  = sigmoid(slope[b] * (capability[m] - difficulty[b]))
    y[m, b]  ~ Beta(mu * kappa[b], (1 - mu) * kappa[b])

giving variance ``mu * (1 - mu) / (kappa[b] + 1)``. A low ``kappa`` means a
benchmark's scores scatter widely around the curve (something other than
capability and difficulty is driving them); a high ``kappa`` means the curve
explains them tightly. Learning one per benchmark lets the fit lean on the
benchmarks that are actually informative about capability.

Priors (all customisable through :class:`BayesianPriors`):

    capability C_m  ~ Normal(0, tau_cd)
    difficulty D_b  ~ ZeroSumNormal(tau_cd)   sum_b D_b = 0
    slope a_b       ~ exp(ZeroSumNormal(tau_alpha))   prod_b a_b = 1
    tau_cd          ~ LogNormal(ln 3, 0.5)
    tau_alpha       ~ LogNormal(ln 0.5, 0.5)
    kappa_b         ~ LogNormal(ln 20, 1)

The two sum-to-zero constraints are what identify the fit: pinning the
difficulties' mean fixes the location, and pinning the slopes' geometric mean
fixes the scale. That replaces the anchor benchmark and L2 penalty used by the
least-squares fit; there is no anchor benchmark here.

Results are reported on the ECI scale, defined by two anchor models
(Claude 3.5 Sonnet -> 130, GPT-5 -> 150) via the affine map
``eci = a + b * capability``, exactly as in :mod:`eci.fitting`. The map is
applied per posterior draw, so the anchor models sit at exactly 130/150 in
every draw; their interval cells are NaN.
"""

import warnings
from dataclasses import dataclass, field

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from .fitting import (
    DEFAULT_ANCHOR_ECI_HIGH,
    DEFAULT_ANCHOR_ECI_LOW,
    DEFAULT_ANCHOR_MODEL_HIGH,
    DEFAULT_ANCHOR_MODEL_LOW,
    _MIN_ANCHOR_SPREAD,
)

# Keeps the Beta parameters away from 0 when the curve saturates; well inside
# the clipping applied to the observed scores.
_MU_EPS = 1e-6

# Variables checked for convergence after sampling.
_DIAGNOSTIC_VARS = ["capability", "difficulty", "slope", "kappa", "tau_cd", "tau_alpha"]


@dataclass(frozen=True)
class BayesianPriors:
    """
    The model's three hyperpriors, each a LogNormal given as
    ``(mu, sigma)`` of the underlying normal.

    Defaults are the weakly informative ones described in the methodology.
    Override any of them, e.g. a tighter concentration prior::

        BayesianPriors(kappa=(np.log(50.0), 1.0))

    Attributes:
        tau_cd: shared scale of capabilities and difficulties.
        tau_alpha: scale of the log slopes.
        kappa: per-benchmark Beta concentration. The default spreads mass
            over roughly 3-150, so benchmarks are free to be much noisier
            than the typical one.
    """

    tau_cd: tuple[float, float] = field(default=(float(np.log(3.0)), 0.5))
    tau_alpha: tuple[float, float] = field(default=(float(np.log(0.5)), 0.5))
    kappa: tuple[float, float] = field(default=(float(np.log(20.0)), 1.0))


def _lognormal(name: str, prior: tuple[float, float], dims: str | None = None):
    mu, sigma = prior
    return pm.LogNormal(name, mu=mu, sigma=sigma, dims=dims)


def _index_data(df: pd.DataFrame):
    """Map the long-format input onto contiguous model/benchmark indices.

    Returns (model_ids, model_names, bench_ids, bench_names, model_idx,
    bench_idx), with the ids in first-appearance order, matching
    :func:`eci.fitting.fit_eci_model`.
    """
    model_ids = df["model_id"].unique()
    bench_ids = df["benchmark_id"].unique()
    model_to_idx = {m: i for i, m in enumerate(model_ids)}
    bench_to_idx = {b: i for i, b in enumerate(bench_ids)}

    id_to_model = df.drop_duplicates("model_id").set_index("model_id")["Model"]
    id_to_bench = df.drop_duplicates("benchmark_id").set_index("benchmark_id")["benchmark"]

    return (
        model_ids,
        [id_to_model[m] for m in model_ids],
        bench_ids,
        [id_to_bench[b] for b in bench_ids],
        np.array([model_to_idx[m] for m in df["model_id"]]),
        np.array([bench_to_idx[b] for b in df["benchmark_id"]]),
    )


def _prepare_performance(df: pd.DataFrame, performance_clip_eps: float) -> np.ndarray:
    """Validate the observed scores and clip them into the Beta's open support."""
    performance = df["performance"].to_numpy(dtype=float)
    if np.isnan(performance).any():
        raise ValueError("Performance data contains NaN values")
    if (performance < 0).any() or (performance > 1).any():
        raise ValueError("Performance scores must be in [0, 1] range")
    at_boundary = ((performance <= 0) | (performance >= 1)).sum()
    if performance_clip_eps <= 0:
        if at_boundary:
            raise ValueError(
                f"{at_boundary} scores sit at exactly 0 or 1, where the Beta "
                "likelihood has no density; performance_clip_eps must be > 0"
            )
        return performance
    return np.clip(performance, performance_clip_eps, 1.0 - performance_clip_eps)


def build_bayesian_model(
    df: pd.DataFrame,
    priors: BayesianPriors = BayesianPriors(),
    performance_clip_eps: float = 1e-3,
) -> pm.Model:
    """
    Build the PyMC model, ready to sample from or inspect.

    Args:
        df: DataFrame with columns model_id, benchmark_id, performance,
            benchmark, Model.
        priors: hyperpriors; see :class:`BayesianPriors`.
        performance_clip_eps: clip performance into [eps, 1 - eps]. The Beta
            likelihood has no density at exactly 0 or 1, so this must be
            positive whenever the data touch either bound.

    Returns:
        A ``pm.Model`` whose coordinates are the model and benchmark *names*
        (not the positional ids, which shift whenever the input data change).
    """
    _, model_names, _, bench_names, model_idx, bench_idx = _index_data(df)
    performance = _prepare_performance(df, performance_clip_eps)

    coords = {"model": model_names, "benchmark": bench_names}
    with pm.Model(coords=coords) as model:
        tau_cd = _lognormal("tau_cd", priors.tau_cd)
        tau_alpha = _lognormal("tau_alpha", priors.tau_alpha)
        kappa = _lognormal("kappa", priors.kappa, dims="benchmark")

        # Non-centred: each item parameter is a unit-scale offset times its
        # hyperscale. Same posterior as writing the scales in directly, but it
        # decouples the offsets from tau_cd / tau_alpha, which is what keeps
        # NUTS out of Neal's funnel on the sparsely measured models.
        capability_z = pm.Normal("capability_z", 0.0, 1.0, dims="model")
        capability = pm.Deterministic("capability", capability_z * tau_cd, dims="model")

        # sum_b D_b = 0 fixes the location of the capability scale.
        difficulty_z = pm.ZeroSumNormal("difficulty_z", sigma=1.0, dims="benchmark")
        difficulty = pm.Deterministic("difficulty", difficulty_z * tau_cd, dims="benchmark")

        # prod_b alpha_b = 1 (geometric mean 1) fixes its scale.
        log_slope_z = pm.ZeroSumNormal("log_slope_z", sigma=1.0, dims="benchmark")
        slope = pm.Deterministic(
            "slope", pm.math.exp(log_slope_z * tau_alpha), dims="benchmark"
        )

        eta = slope[bench_idx] * (capability[model_idx] - difficulty[bench_idx])
        mu = pm.math.clip(pm.math.sigmoid(eta), _MU_EPS, 1.0 - _MU_EPS)
        kappa_obs = kappa[bench_idx]
        pm.Beta(
            "obs",
            alpha=mu * kappa_obs,
            beta=(1.0 - mu) * kappa_obs,
            observed=performance,
        )
    return model


def _check_convergence(trace: az.InferenceData, max_rhat: float = 1.01) -> None:
    """Warn about divergences and poor mixing rather than failing the fit;
    the draws are still returned so the user can inspect them."""
    n_divergent = int(trace.sample_stats["diverging"].sum()) if "sample_stats" in trace else 0
    if n_divergent:
        warnings.warn(
            f"{n_divergent} divergent transitions; the posterior may be biased. "
            "Consider raising target_accept.",
            stacklevel=3,
        )
    rhat = az.rhat(trace, var_names=_DIAGNOSTIC_VARS)
    worst = max(float(np.nanmax(v.to_numpy())) for v in rhat.data_vars.values())
    if worst > max_rhat:
        warnings.warn(
            f"max r_hat = {worst:.3f} > {max_rhat}; chains have not mixed. "
            "Consider more tuning or draws.",
            stacklevel=3,
        )


def _stack_draws(trace: az.InferenceData, name: str, dim: str) -> np.ndarray:
    """Posterior draws for one variable as a (n_draws, n_items) array."""
    return (
        trace.posterior[name]
        .stack(sample=("chain", "draw"))
        .transpose("sample", dim)
        .to_numpy()
    )


def _anchor_transform(
    capability: np.ndarray,
    anchor_low_idx: int,
    anchor_high_idx: int,
    eci_low: float,
    eci_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-draw (a, b) of ``eci = a + b * capability``.

    The vectorised twin of :func:`eci.fitting._affine_map`: one map per draw,
    each sending that draw's own anchor capabilities to eci_low / eci_high.
    """
    low = capability[:, anchor_low_idx]
    high = capability[:, anchor_high_idx]
    spread = high - low
    degenerate = ~(spread > _MIN_ANCHOR_SPREAD)  # also catches NaN
    if degenerate.any():
        raise ValueError(
            f"{degenerate.sum()} of {spread.size} posterior draws place the high "
            "anchor at or below the low anchor, defining no scale"
        )
    b = (eci_high - eci_low) / spread
    return eci_low - b * low, b


def fit_bayesian_eci(
    df: pd.DataFrame,
    priors: BayesianPriors = BayesianPriors(),
    performance_clip_eps: float = 1e-3,
    draws: int = 4000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 12345,
    *,
    anchor_model_low: str = DEFAULT_ANCHOR_MODEL_LOW,
    anchor_eci_low: float = DEFAULT_ANCHOR_ECI_LOW,
    anchor_model_high: str = DEFAULT_ANCHOR_MODEL_HIGH,
    anchor_eci_high: float = DEFAULT_ANCHOR_ECI_HIGH,
    ci_level: float = 0.90,
    progressbar: bool = True,
    sampler_kwargs: "dict | None" = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Fit the Bayesian IRT model and return ECI-scale scores with credible
    intervals.

    Args:
        df: DataFrame with columns model_id, benchmark_id, performance,
            benchmark, Model.
        priors: hyperpriors; see :class:`BayesianPriors`.
        performance_clip_eps: clip performance into [eps, 1 - eps].
        draws: posterior draws kept per chain.
        tune: tuning (warm-up) steps per chain, discarded.
        chains: number of chains, run in parallel where cores allow.
        target_accept: NUTS target acceptance rate; raise it towards 0.99 if
            the fit reports divergences.
        random_seed: seed for the sampler.
        anchor_model_low / anchor_eci_low / anchor_model_high /
            anchor_eci_high: the models that define the ECI scale and their
            fixed values.
        ci_level: central posterior mass for the intervals (0.90 -> 5th/95th
            percentiles), matching the least-squares fit's convention.
        progressbar: show the sampling progress bar.
        sampler_kwargs: extra keyword arguments forwarded to ``pm.sample``,
            e.g. ``{"idata_kwargs": {"log_likelihood": True}}`` to keep the
            pointwise log likelihood for ``az.loo``.

    Returns:
        (eci_df, edi_df, draws), everything on the ECI scale:
        - eci_df: model_id, Model, eci (posterior mean), and
          eci_ci_low / eci_ci_high (NaN for the anchor models, whose values
          are fixed by definition rather than estimated). Sorted by eci
          descending.
        - edi_df: benchmark_id, benchmark, edi, discriminability_scaled and
          kappa, each with _ci_low / _ci_high columns. Together edi and
          discriminability_scaled trace the fitted curve: a model's predicted
          score is ``sigmoid(discriminability_scaled * (eci - edi))``.
          Sorted by edi.
        - draws: dict with model_ids, model_names, benchmark_ids,
          benchmark_names, the (n_draws, n) arrays eci, edi, slope and kappa,
          the per-draw map coefficients a and b, and trace, the full
          ``arviz.InferenceData``.

    Divergences and poor mixing are warned about, not raised; a fit whose
    anchor capabilities coincide or invert in any draw raises, since no scale
    is then defined.

    On the published data the defaults give a clean fit apart from the global
    scale ``tau_cd``, which mixes slowly and lands at r_hat 1.01-1.02 whatever
    the tuning length, so the convergence warning fires intermittently. It is
    a nuisance parameter: everything the fit reports - capability, difficulty,
    slope and kappa - sits at or below 1.01.
    """
    model_ids, model_names, bench_ids, bench_names, _, _ = _index_data(df)
    try:
        anchor_low_idx = model_names.index(anchor_model_low)
        anchor_high_idx = model_names.index(anchor_model_high)
    except ValueError as e:
        raise ValueError(f"Anchor model not found in data: {e}") from None

    pymc_model = build_bayesian_model(df, priors, performance_clip_eps)
    with pymc_model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            **(sampler_kwargs or {}),
        )
    _check_convergence(trace)

    capability = _stack_draws(trace, "capability", "model")
    difficulty = _stack_draws(trace, "difficulty", "benchmark")
    slope = _stack_draws(trace, "slope", "benchmark")
    kappa = _stack_draws(trace, "kappa", "benchmark")

    # Re-anchor every draw with its own transform, so the anchors sit at
    # exactly 130/150 throughout the posterior and the intervals describe
    # position on the ECI scale rather than the unidentified raw scale.
    a, b = _anchor_transform(
        capability, anchor_low_idx, anchor_high_idx, anchor_eci_low, anchor_eci_high
    )
    eci = a[:, None] + b[:, None] * capability
    edi = a[:, None] + b[:, None] * difficulty
    slope_scaled = slope / b[:, None]

    draws_out = {
        "model_ids": list(model_ids),
        "model_names": model_names,
        "benchmark_ids": list(bench_ids),
        "benchmark_names": bench_names,
        "eci": eci,
        "edi": edi,
        "slope": slope_scaled,
        "kappa": kappa,
        "a": a,
        "b": b,
        "trace": trace,
    }

    tail = (1.0 - ci_level) / 2.0

    def summarise(values: np.ndarray, name: str) -> dict[str, np.ndarray]:
        low, high = np.quantile(values, [tail, 1.0 - tail], axis=0)
        return {name: values.mean(axis=0), f"{name}_ci_low": low, f"{name}_ci_high": high}

    eci_df = pd.DataFrame(
        {"model_id": model_ids, "Model": model_names, **summarise(eci, "eci")}
    )
    # The anchor models' values are fixed by definition, not estimated
    anchor_rows = eci_df["Model"].isin([anchor_model_low, anchor_model_high])
    eci_df.loc[anchor_rows, ["eci_ci_low", "eci_ci_high"]] = np.nan

    edi_df = pd.DataFrame(
        {
            "benchmark_id": bench_ids,
            "benchmark": bench_names,
            **summarise(edi, "edi"),
            **summarise(slope_scaled, "discriminability_scaled"),
            **summarise(kappa, "kappa"),
        }
    )

    return (
        eci_df.sort_values("eci", ascending=False),
        edi_df.sort_values("edi"),
        draws_out,
    )
