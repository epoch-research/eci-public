"""
Bayesian fitting for the Epoch Capabilities Index (ECI).

Implements the same IRT curve as :mod:`eci.fitting`, but as the mean of a
per-observation Beta likelihood sampled by MCMC rather than fit directly, so
every quantity comes out as a posterior distribution. See the README's
"Bayesian fit" section for the model, the priors, the centred/non-centred
sampling choice, and how results are mapped onto the ECI scale.

Besides the priors dataclass, two public entry points do the work:
:func:`build_bayesian_model` assembles the plain ``pm.Model`` (coordinates,
priors, likelihood) without touching the sampler, so it can be inspected or
sampled by hand; :func:`fit_bayesian_eci` calls it, runs ``pm.sample``, and
does everything downstream of the raw posterior - convergence warnings,
rescaling each draw onto the ECI scale via the anchor models, and (with
``evaluate=True``) handing the trace to :mod:`eci.evaluation`. Everything
else here is a private helper behind one of those two: pulling HDI/ETI
intervals out of an array, indexing the input data, and the per-draw anchor
transform.
"""

import warnings
from dataclasses import dataclass, field

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from .evaluation import evaluate_fit
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

# How the reported intervals are cut out of the posterior.
INTERVAL_KINDS = ("hdi", "eti")


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


def _hdi(values: np.ndarray, prob: float) -> tuple[np.ndarray, np.ndarray]:
    """Highest-density interval of each column of ``values`` (draws x items).

    The narrowest window containing ``prob`` of the draws, found by sliding it
    over the sorted draws. For a skewed posterior - a slope or a kappa - this
    is a tighter and more honest interval than cutting equal tails, and it
    never places the point estimate outside its own interval.
    """
    n_draws = values.shape[0]
    n_inside = max(1, min(n_draws, int(np.floor(prob * n_draws))))
    ordered = np.sort(values, axis=0)
    # widths[i] is the span of the window starting at draw i
    widths = ordered[n_inside - 1:] - ordered[: n_draws - n_inside + 1]
    start = np.argmin(widths, axis=0)
    columns = np.arange(values.shape[1])
    return ordered[start, columns], ordered[start + n_inside - 1, columns]


def _interval(values: np.ndarray, prob: float, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper bounds of each column's posterior interval."""
    if kind == "hdi":
        return _hdi(values, prob)
    if kind == "eti":
        tail = (1.0 - prob) / 2.0
        return tuple(np.quantile(values, [tail, 1.0 - tail], axis=0))
    raise ValueError(f"Unknown interval={kind!r}. Choose one of {INTERVAL_KINDS}.")


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
    performance_clip_eps: float = 1e-4,
    centred: bool = True,
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

            It is not a free numerical tolerance. Most boundary rows fit a
            shape parameter mu*kappa < 1, where the Beta density *diverges*
            as the value goes to 0 - so shrinking eps hands the likelihood an
            unbounded reward for pushing those rows' mu down, and the model
            pays for it by lowering the capability of the models involved and
            the kappa of the benchmarks carrying the zeros. Going from 1e-4 to
            1e-6 on the published data buys 0.03 of squared error at the 82
            exact zeros and adds 0.24 across the 2457-row bulk.

            The default is one item's worth of resolution for a benchmark of a
            few thousand items: an observed 0 says the true rate is below
            about 1/N, not that it equals eps.
        centred: sample the item parameters directly (the default), or, when
            False, as unit-scale offsets rescaled by their hyperparameter.
            The two define the *same* posterior and differ only in the
            coordinates NUTS explores; see the module docstring.

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

        if centred:
            capability = pm.Normal("capability", 0.0, tau_cd, dims="model")
            difficulty = pm.ZeroSumNormal("difficulty", sigma=tau_cd, dims="benchmark")
            log_slope = pm.ZeroSumNormal("log_slope", sigma=tau_alpha, dims="benchmark")
        else:
            capability_z = pm.Normal("capability_z", 0.0, 1.0, dims="model")
            capability = pm.Deterministic("capability", capability_z * tau_cd, dims="model")
            difficulty_z = pm.ZeroSumNormal("difficulty_z", sigma=1.0, dims="benchmark")
            difficulty = pm.Deterministic("difficulty", difficulty_z * tau_cd, dims="benchmark")
            log_slope_z = pm.ZeroSumNormal("log_slope_z", sigma=1.0, dims="benchmark")
            log_slope = log_slope_z * tau_alpha
        slope = pm.Deterministic("slope", pm.math.exp(log_slope), dims="benchmark")

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
    performance_clip_eps: float = 1e-4,
    centred: bool = True,
    draws: int = 4000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 12345,
    *,
    anchor_model_low: str = DEFAULT_ANCHOR_MODEL_LOW,
    anchor_eci_low: float = DEFAULT_ANCHOR_ECI_LOW,
    anchor_model_high: str = DEFAULT_ANCHOR_MODEL_HIGH,
    anchor_eci_high: float = DEFAULT_ANCHOR_ECI_HIGH,
    ci_level: float = 0.90,
    interval: str = "hdi",
    evaluate: bool = False,
    evaluation_max_draws: int = 2000,
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
        centred: parameterisation; see :func:`build_bayesian_model`.
        draws: posterior draws kept per chain.
        tune: tuning (warm-up) steps per chain, discarded.
        chains: number of chains, run in parallel where cores allow.
        target_accept: NUTS target acceptance rate; raise it towards 0.99 if
            the fit reports divergences. The centred parameterisation needs a
            higher value than the non-centred one to keep its step size small
            enough for the narrow part of the funnel.
        random_seed: seed for the sampler.
        anchor_model_low / anchor_eci_low / anchor_model_high /
            anchor_eci_high: the models that define the ECI scale and their
            fixed values.
        ci_level: posterior mass the reported intervals carry (0.90 by
            default, matching the least-squares fit's convention).
        interval: how to cut that mass out of the posterior - "hdi" for the
            highest-density interval (the default: the narrowest window
            holding ci_level of the draws), or "eti" for equal tails, i.e.
            the 5th and 95th percentiles at ci_level=0.90.
        evaluate: also run the diagnostic suite in :mod:`eci.evaluation` and
            return it under draws["evaluation"]. This keeps the pointwise log
            likelihood and draws a replicated dataset per posterior draw,
            which the fit otherwise skips: both are (draws x observations)
            arrays, so the fit uses noticeably more memory and time.
        evaluation_max_draws: draws the evaluation thins down to for its
            per-observation metrics.
        progressbar: show the sampling progress bar.
        sampler_kwargs: extra keyword arguments forwarded to ``pm.sample``,
            e.g. ``{"idata_kwargs": {"log_likelihood": True}}`` to keep the
            pointwise log likelihood for ``az.loo``.

    Returns:
        (eci_df, edi_df, draws), everything on the ECI scale:
        - eci_df: model_id, Model, eci (posterior mean), and the interval
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
          the per-draw map coefficients a and b, trace, the full
          ``arviz.InferenceData``, and - with evaluate=True - evaluation,
          the dict of diagnostic tables.

    Divergences and poor mixing are warned about, not raised; a fit whose
    anchor capabilities coincide or invert in any draw raises, since no scale
    is then defined.

    The global scale ``tau_cd`` is the slowest-mixing parameter and the one
    that trips the convergence warning first; it is a nuisance parameter, and
    the quantities the fit reports converge well before it does.
    """
    model_ids, model_names, bench_ids, bench_names, _, _ = _index_data(df)
    try:
        anchor_low_idx = model_names.index(anchor_model_low)
        anchor_high_idx = model_names.index(anchor_model_high)
    except ValueError as e:
        raise ValueError(f"Anchor model not found in data: {e}") from None

    sample_kwargs = dict(sampler_kwargs or {})
    if evaluate:
        # LOO needs the pointwise log likelihood, which PyMC does not keep by
        # default because it is one number per draw per observation.
        sample_kwargs["idata_kwargs"] = {
            **sample_kwargs.get("idata_kwargs", {}), "log_likelihood": True
        }

    pymc_model = build_bayesian_model(df, priors, performance_clip_eps, centred)
    with pymc_model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            **sample_kwargs,
        )
        if evaluate:
            pm.sample_posterior_predictive(
                trace,
                extend_inferencedata=True,
                random_seed=random_seed,
                progressbar=progressbar,
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
    if evaluate:
        draws_out["evaluation"] = evaluate_fit(
            trace, df,
            max_draws=evaluation_max_draws,
            performance_clip_eps=performance_clip_eps,
        )

    def summarise(values: np.ndarray, name: str) -> dict[str, np.ndarray]:
        low, high = _interval(values, ci_level, interval)
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
