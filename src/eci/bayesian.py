"""
Bayesian fitting for the Epoch Capabilities Index (ECI).

Implements the IRT curve as the mean of a per-observation Beta 
likelihood sampled by MCMC. See the README's "Bayesian fit" section 
for the model, the priors, and how results are mapped onto the ECI scale.

Besides the priors dataclass, two public entry points build
and fit the model: :func:`build_bayesian_model` assembles the plain
``pm.Model`` (coordinates, priors, likelihood) without touching the sampler, so
it can be inspected or sampled by hand; :func:`fit_bayesian_eci` calls it, runs
``pm.sample``, and does everything downstream of the raw posterior - convergence
warnings, rescaling each draw onto the ECI scale via the anchor models, and
(with ``evaluate=True``) handing the trace to :mod:`eci.evaluation`.

The fit samples with a dense mass matrix: the covariance of every parameter.
Capabilities, difficulties and slopes are strongly correlated (a model can sit
further below a benchmark if that benchmark's slope is shallower), and the
default diagonal mass matrix cannot see that, which forces many small steps per
draw. The covariance comes from a short diagonal pilot run. If that fit
diverges or still misses the convergence checks at 16,000 draws, the fit tries
again with the covariance of that fit's own draws, and only then falls back to
the default diagonal mass matrix. Each stage starts small and adds draws until
the checks pass. Everything else here is a private helper behind the public
entry points.
"""

import warnings
from dataclasses import dataclass, field

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pymc.distributions.transforms import ZeroSumTransform
from pymc.step_methods.hmc.quadpotential import QuadPotentialFull

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

# The Beta's SD at mu = 0.5 is 1 / (2 sqrt(kappa + 1)), which reaches 0.5 only
# as kappa -> 0, so the per-benchmark SD must stay below it.
_MAX_SIGMA = 0.5

# Capabilities and difficulties are uniform on [-_CD_BOUND, _CD_BOUND], wide
# enough to be flat wherever the data put them.
_CD_BOUND = 50.0

# Variables checked for convergence after sampling.
_DIAGNOSTIC_VARS = [
    "capability", "difficulty", "slope", "sigma",
    "tau_alpha", "sigma_median", "tau_sigma",
]

# The free variables whose draws give the dense mass matrix and each chain's
# starting point.
_FREE_VARS = ("tau_alpha", "sigma_median", "tau_sigma", "sigma", "capability", "difficulty", "log_slope")
# Posterior draws kept for that: enough to estimate the covariance of the ~460
# unconstrained parameters.
_COV_DRAWS = 2000
# Weight on the diagonal when shrinking the estimated covariance, which keeps it
# well conditioned however few draws it came from.
_COV_SHRINKAGE = 0.1
# A fit passes the convergence checks with no divergences, every bulk ESS at
# least this, and every r_hat at most _MAX_RHAT.
_MIN_ESS = 400
_MAX_RHAT = 1.01
# The dense stage's tuning steps and draws per chain: its first run, and the
# most draws it goes up to before falling back to the cold stage.
_DENSE_TUNE = 500
_DENSE_DRAWS = (1000, 16000)
# The pilot run's tuning steps and draws per chain, diagonal mass matrix.
_PILOT_TUNE = 1000
_PILOT_DRAWS = 1000
# The same for the cold stage (diagonal mass matrix). Tuning past ~1000 buys
# no ESS on this model; draws do.
_COLD_TUNE = 1000
_COLD_DRAWS = (4000, 32000)


@dataclass(frozen=True)
class BayesianPriors:
    """
    The model's three hyperpriors, each a LogNormal given as
    ``(mu, sigma)`` of the underlying normal.

    Defaults are the weakly informative ones described in the methodology.
    Override any of them, e.g. a prior expecting less noisy benchmarks::

        BayesianPriors(sigma_median=(np.log(0.07), 0.5))

    The per-benchmark noise SD ``sigma`` is the Beta's standard deviation
    where the curve passes mu = 0.5 (and its largest anywhere; elsewhere it
    shrinks by 2 sqrt(mu (1 - mu))). Its prior is hierarchical,
    ``LogNormal(ln sigma_median, tau_sigma)`` truncated to (0, 0.5) since the
    Beta cannot be wider, so benchmarks with few scores are pulled towards
    the noise level the others share rather than towards a fixed guess.

    Capabilities and difficulties take flat priors (see
    :func:`build_bayesian_model`), so they have no hyperprior here.

    Attributes:
        tau_alpha: scale of the log slopes.
        sigma_median: the typical benchmark's noise SD. The default centres
            it on 0.1 with 90% of the mass on roughly 0.04-0.23.
        tau_sigma: spread of log sigma across benchmarks. The default,
            around 0.5, lets one benchmark be several times noisier than
            another.
    """

    tau_alpha: tuple[float, float] = field(default=(float(np.log(0.5)), 0.5))
    sigma_median: tuple[float, float] = field(default=(float(np.log(0.1)), 0.5))
    tau_sigma: tuple[float, float] = field(default=(float(np.log(0.5)), 0.5))


def _interval(values: np.ndarray, prob: float) -> tuple[np.ndarray, np.ndarray]:
    """Equal-tailed interval of each column of ``values`` (draws x items):
    the (1 - prob) / 2 and (1 + prob) / 2 quantiles."""
    tail = (1.0 - prob) / 2.0
    return tuple(np.quantile(values, [tail, 1.0 - tail], axis=0))


def _lognormal(name: str, prior: tuple[float, float], dims: str | None = None):
    mu, sigma = prior
    return pm.LogNormal(name, mu=mu, sigma=sigma, dims=dims)


def _index_data(df: pd.DataFrame):
    """Map the long-format input onto contiguous model/benchmark indices.

    Returns (model_ids, model_names, bench_ids, bench_names, model_idx,
    bench_idx), with the ids in first-appearance order.
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

            It is not a free numerical tolerance. Most boundary rows fit a
            shape parameter mu*kappa < 1, where the Beta density *diverges*
            as the value goes to 0 - so shrinking eps hands the likelihood an
            unbounded reward for pushing those rows' mu down, and the model
            pays for it by lowering the capability of the models involved and
            raising the sigma of the benchmarks carrying the zeros.

    Returns:
        A ``pm.Model`` whose coordinates are the model and benchmark *names*
        (not the positional ids, which shift whenever the input data change).
    """
    _, model_names, _, bench_names, model_idx, bench_idx = _index_data(df)
    if len(bench_names) < 2:
        raise ValueError("the fit needs at least two benchmarks")
    performance = _prepare_performance(df, performance_clip_eps)

    coords = {"model": model_names, "benchmark": bench_names}
    with pm.Model(coords=coords) as model:
        tau_alpha = _lognormal("tau_alpha", priors.tau_alpha)
        sigma_median = _lognormal("sigma_median", priors.sigma_median)
        tau_sigma = _lognormal("tau_sigma", priors.tau_sigma)
        # The prior is on the noise SD; the Beta takes the concentration it
        # implies, inverting sigma = 1 / (2 sqrt(kappa + 1)).
        sigma = pm.Truncated(
            "sigma",
            pm.LogNormal.dist(mu=pm.math.log(sigma_median), sigma=tau_sigma),
            lower=0.0,
            upper=_MAX_SIGMA,
            dims="benchmark",
        )
        kappa = pm.Deterministic("kappa", 1.0 / (4.0 * sigma**2) - 1.0, dims="benchmark")

        capability = pm.Uniform(
            "capability", -_CD_BOUND, _CD_BOUND, dims="model"
        )
        # Uniform on the sum-to-zero slice of the box. PyMC has no zero-sum
        # uniform, and Uniform with a ZeroSumTransform evaluates to nan, so the
        # transform goes on a Flat and the box is imposed separately. The
        # constraint narrows each difficulty's marginal SD slightly below the
        # capabilities' (~1% at K = 20).
        difficulty = pm.Flat(
            "difficulty", dims="benchmark", transform=ZeroSumTransform([-1])
        )
        pm.Potential(
            "difficulty_bound",
            pm.math.switch(
                pm.math.all(pm.math.abs(difficulty) <= _CD_BOUND), 0.0, -np.inf
            ),
        )
        # A sum-to-zero vector of length K has marginal SD sigma * sqrt(1 - 1/K);
        # scaling sigma up by sqrt(K / (K - 1)) makes tau_alpha the marginal SD.
        zero_sum_scale = np.sqrt(len(bench_names) / (len(bench_names) - 1))
        log_slope = pm.ZeroSumNormal(
            "log_slope", sigma=tau_alpha * zero_sum_scale, dims="benchmark"
        )
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


def _check_convergence(trace: az.InferenceData, max_rhat: float = _MAX_RHAT) -> None:
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

    One map per draw, each sending that draw's own anchor capabilities to
    eci_low / eci_high.
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


def _thinned_draws(trace: az.InferenceData) -> dict:
    """_COV_DRAWS evenly spaced draws of each of _FREE_VARS, chains pooled, as
    arrays of shape (n_draws,) or (n_draws, n_items)."""
    posterior = trace.posterior
    total = posterior.sizes["chain"] * posterior.sizes["draw"]
    keep = np.linspace(0, total - 1, min(_COV_DRAWS, total)).astype(int)
    return {
        v: posterior[v].stack(sample=("chain", "draw")).transpose("sample", ...).to_numpy()[keep]
        for v in _FREE_VARS
    }


def _dense_cov(model: pm.Model, draws: dict) -> np.ndarray:
    """The (shrunk) covariance of ``draws`` on the sampler's unconstrained
    scale, for a dense mass matrix."""
    columns = []
    for value in model.value_vars:
        rv = model.values_to_rvs[value]
        transform = model.rvs_to_transforms[rv]
        x = draws[rv.name]
        if transform is not None:
            x = transform.forward(pt.as_tensor(x), *rv.owner.inputs).eval()
        columns.append(np.asarray(x).reshape(len(x), -1))
    cov = np.cov(np.hstack(columns), rowvar=False)
    return (1 - _COV_SHRINKAGE) * cov + _COV_SHRINKAGE * np.diag(np.diag(cov))


def _chain_starts(draws: dict, chains: int, rng) -> list[dict]:
    """A different draw per chain, so r_hat still compares chains that started
    in different places."""
    picks = rng.choice(len(draws["capability"]), chains, replace=False)
    return [{v: draws[v][i] for v in _FREE_VARS} for i in picks]


def _diagnostics(trace: az.InferenceData) -> dict:
    """Divergences, worst r_hat and lowest bulk ESS over _DIAGNOSTIC_VARS."""
    rhat = az.rhat(trace, var_names=_DIAGNOSTIC_VARS)
    ess = az.ess(trace, var_names=_DIAGNOSTIC_VARS)
    return {
        "divergences": int(trace.sample_stats["diverging"].sum()),
        "max_rhat": max(float(np.nanmax(v.to_numpy())) for v in rhat.data_vars.values()),
        "min_ess": min(float(np.nanmin(v.to_numpy())) for v in ess.data_vars.values()),
    }


def _next_draws(draws: int, min_ess: float) -> int:
    """Draws for the next run: enough, if ESS grows in proportion to draws, to
    lift ``min_ess`` to _MIN_ESS, rounded up to a power-of-two multiple of
    ``draws``. Doubles when ESS already passes (r_hat failed) or is zero."""
    if min_ess >= _MIN_ESS or not min_ess > 0:
        return 2 * draws
    return draws * 2 ** max(1, int(np.ceil(np.log2(_MIN_ESS / min_ess))))


def _ladder(run, draws_range: tuple[int, int], label: dict, attempts: list):
    """Call ``run(draws, rung)`` with more draws each time until a run passes
    the convergence checks, diverges, or reaches the top of ``draws_range``.

    More draws raise ESS and settle r_hat but do not remove divergences, so a
    run with any ends the ladder. Each run's ``label``, draws and diagnostics
    are appended to ``attempts``. Returns (trace, passed) for the last run.
    """
    draws, max_draws = draws_range
    rung = 0
    while True:
        trace = run(draws, rung)
        stats = _diagnostics(trace)
        attempts.append({**label, "draws": draws, **stats})
        if stats["divergences"]:
            return trace, False
        if stats["max_rhat"] <= _MAX_RHAT and stats["min_ess"] >= _MIN_ESS:
            return trace, True
        if draws >= max_draws:
            return trace, False
        draws = min(_next_draws(draws, stats["min_ess"]), max_draws)
        rung += 1


def _sample(
    model: pm.Model,
    *,
    draws: "int | None",
    tune: "int | None",
    chains: int,
    target_accept: float,
    random_seed: int,
    progressbar: bool,
    dense_mass_matrix: bool,
    sample_kwargs: dict,
) -> tuple[az.InferenceData, str, list[dict]]:
    """Sample the model, trying each stage until one passes the checks.

    The dense stages, each with its own mass matrix: "pilot" (a short diagonal
    run's draws), then "refresh" (the draws of the pilot stage's last, failed
    run). Then "cold", with
    the diagonal mass matrix. Returns the trace, the stage it came from, and
    one dict per run (stage, tune, draws, divergences, max_rhat, min_ess).
    """
    common = dict(chains=chains, cores=chains, progressbar=progressbar,
                  compute_convergence_checks=False, **sample_kwargs)
    attempts = []

    def draws_range(default):
        return default if draws is None else (draws, draws)

    def dense_stage(source, label, seed):
        rng = np.random.default_rng(seed)
        cov_draws = _thinned_draws(source)
        cov = _dense_cov(model, cov_draws)
        dense_tune = _DENSE_TUNE if tune is None else tune

        def run(n_draws, rung):
            with model:
                step = pm.NUTS(potential=QuadPotentialFull(cov), target_accept=target_accept)
                return pm.sample(draws=n_draws, tune=dense_tune, step=step,
                                 initvals=_chain_starts(cov_draws, chains, rng),
                                 random_seed=seed + rung, **common)

        return _ladder(run, draws_range(_DENSE_DRAWS), {"stage": label, "tune": dense_tune}, attempts)

    if dense_mass_matrix:
        # Pilot: a short run with the default diagonal mass matrix, only to
        # estimate the covariance and starting points for a dense run.
        with model:
            pilot = pm.sample(draws=_PILOT_DRAWS, tune=_PILOT_TUNE, target_accept=target_accept,
                              random_seed=random_seed + 1, **common)
        trace, passed = dense_stage(pilot, "pilot", random_seed + 200)
        if passed:
            return trace, "pilot", attempts
        # Refresh: the failed run's own draws, divergent or not, come from the
        # real posterior under a better mass matrix than the one before, so
        # their covariance is a better estimate (e.g. of a benchmark every
        # model scores near the floor on, whose difficulty has a long tail).
        trace, passed = dense_stage(trace, "refresh", random_seed + 100)
        if passed:
            return trace, "refresh", attempts
        last = attempts[-1]
        warnings.warn(
            f"The dense fits did not converge (last: {last['divergences']} divergences, "
            f"max r_hat {last['max_rhat']:.3f}, min bulk ESS {last['min_ess']:.0f} at "
            f"{last['draws']} draws); falling back to a cold fit.",
            stacklevel=3,
        )

    cold_tune = _COLD_TUNE if tune is None else tune

    def cold_run(n_draws, rung):
        with model:
            return pm.sample(draws=n_draws, tune=cold_tune, target_accept=target_accept,
                             random_seed=random_seed + rung, **common)

    trace, _ = _ladder(cold_run, draws_range(_COLD_DRAWS),
                       {"stage": "cold", "tune": cold_tune}, attempts)
    return trace, "cold", attempts


def fit_bayesian_eci(
    df: pd.DataFrame,
    priors: BayesianPriors = BayesianPriors(),
    performance_clip_eps: float = 1e-3,
    draws: "int | None" = None,
    tune: "int | None" = None,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 12345,
    *,
    dense_mass_matrix: bool = True,
    anchor_model_low: str = DEFAULT_ANCHOR_MODEL_LOW,
    anchor_eci_low: float = DEFAULT_ANCHOR_ECI_LOW,
    anchor_model_high: str = DEFAULT_ANCHOR_MODEL_HIGH,
    anchor_eci_high: float = DEFAULT_ANCHOR_ECI_HIGH,
    ci_level: float = 0.90,
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
        draws: posterior draws kept per chain. By default each stage starts
            small and adds draws until the fit passes the convergence checks
            (no divergences, bulk ESS at least 400 and r_hat at most 1.01):
            the dense stage from 1,000 up to 16,000, the cold stage from
            4,000 up to 32,000. Each next run is the draws ESS should need,
            assuming it grows in proportion, rounded up to a power-of-two
            multiple (at least double). Giving ``draws`` fixes every run at
            that size instead.
        tune: tuning (warm-up) steps per chain, discarded. Defaults to 500
            in the dense stage and 1000 in the cold stage.
        chains: number of chains, each on its own core.
        target_accept: NUTS target acceptance rate.
        random_seed: seed for the sampler; later runs use seeds derived
            from it.
        dense_mass_matrix: sample with a dense mass matrix first, the
            covariance of a short diagonal pilot run (tune 1000 + draws
            1000), whose draws also give each chain's starting point. If that
            "pilot" stage diverges or still misses the checks at its most
            draws, a "refresh" stage tries again with the covariance of the
            failed stage's own draws. Only if both fail does the fit go cold
            (PyMC's default initialisation and diagonal adaptation), with a
            warning. False goes straight to the
            cold stage, which needs about 12,000 draws on the full data where
            a dense stage needs 1,000.
        anchor_model_low / anchor_eci_low / anchor_model_high /
            anchor_eci_high: the models that define the ECI scale and their
            fixed values.
        ci_level: posterior mass the reported equal-tailed intervals carry
            (0.90 by default, i.e. the 5th and 95th percentiles).
        evaluate: also run the diagnostic suite in :mod:`eci.evaluation` and
            return it under draws["evaluation"]. This keeps the pointwise log
            likelihood and draws a replicated dataset per posterior draw,
            which the fit otherwise skips: both are (draws x observations)
            arrays, so the fit uses noticeably more memory and time.
        evaluation_max_draws: draws the evaluation thins down to for its
            per-observation metrics.
        progressbar: show the sampling progress bar.
        sampler_kwargs: extra keyword arguments forwarded to ``pm.sample``.

    Returns:
        (eci_df, edi_df, draws), everything on the ECI scale:
        - eci_df: model_id, Model, eci (posterior mean), and the interval
          eci_ci_low / eci_ci_high (NaN for the anchor models, whose values
          are fixed by definition rather than estimated). Sorted by eci
          descending.
        - edi_df: benchmark_id, benchmark, edi, discriminability_scaled
          and sigma, each with _ci_low / _ci_high columns. Together edi and
          discriminability_scaled trace the fitted curve: a model's predicted
          score is ``sigmoid(discriminability_scaled * (eci - edi))``.
          Sorted by edi.
        - draws: dict with model_ids, model_names, benchmark_ids,
          benchmark_names, the (n_draws, n) arrays eci, edi, slope and
          sigma,
          the per-draw map coefficients a and b, trace, the full
          ``arviz.InferenceData``, started_from (the stage the draws came
          from: "pilot", "refresh" or "cold"), attempts
          (one dict per run: stage, tune, draws, divergences, max_rhat,
          min_ess), and - with evaluate=True - evaluation, the dict of
          diagnostic tables.

    Divergences and poor mixing are warned about, not raised; a fit whose
    anchor capabilities coincide or invert in any draw raises, since no scale
    is then defined.
    """
    model_ids, model_names, bench_ids, bench_names, _, _ = _index_data(df)
    try:
        anchor_low_idx = model_names.index(anchor_model_low)
        anchor_high_idx = model_names.index(anchor_model_high)
    except ValueError as e:
        raise ValueError(f"Anchor model not found in data: {e}") from None

    sample_kwargs = dict(sampler_kwargs or {})

    pymc_model = build_bayesian_model(df, priors, performance_clip_eps)
    trace, started_from, attempts = _sample(
        pymc_model,
        draws=draws, tune=tune, chains=chains, target_accept=target_accept,
        random_seed=random_seed, progressbar=progressbar,
        dense_mass_matrix=dense_mass_matrix, sample_kwargs=sample_kwargs,
    )
    with pymc_model:
        if evaluate:
            # LOO needs the pointwise log likelihood, which PyMC does not
            # keep by default because it is one number per draw per
            # observation.
            pm.compute_log_likelihood(
                trace, extend_inferencedata=True, progressbar=progressbar
            )
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
    sigma = _stack_draws(trace, "sigma", "benchmark")

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
        "sigma": sigma,
        "a": a,
        "b": b,
        "trace": trace,
        "started_from": started_from,
        "attempts": attempts,
    }
    if evaluate:
        draws_out["evaluation"] = evaluate_fit(
            trace, df,
            max_draws=evaluation_max_draws,
            performance_clip_eps=performance_clip_eps,
        )

    def summarise(values: np.ndarray, name: str) -> dict[str, np.ndarray]:
        low, high = _interval(values, ci_level)
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
            **summarise(sigma, "sigma"),
        }
    )

    return (
        eci_df.sort_values("eci", ascending=False),
        edi_df.sort_values("edi"),
        draws_out,
    )
