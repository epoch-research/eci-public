"""
Evaluation suite for the Bayesian ECI fit.

Convergence, predictive accuracy, leave-one-out and calibration checks each
get their own function - :func:`sampler_diagnostics`,
:func:`predictive_accuracy`, :func:`loo_diagnostics`,
:func:`posterior_predictive_checks`, :func:`calibration`, :func:`qq_data`,
:func:`benchmark_forest_data` - so each can be run and tested alone.
:func:`evaluate_fit` runs all of them against a fitted trace and flattens the
headline numbers into one metrics table; :func:`save_evaluation` renders the
result as the three-page PDF report. See the README's "Evaluating a fit"
section for what each metric means, the raw-vs-clipped scoring split, and
what's saved to disk versus kept in the returned dict only.

Typical use is through the fit itself::

    eci_df, edi_df, draws = fit_bayesian_eci(df, evaluate=True)
    save_evaluation(draws["evaluation"], "outputs/evaluation")
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

# Parameters the diagnostics summarise. kappa and log_slope are left out:
# each is a one-to-one transform of a parameter already listed (sigma, slope),
# so its r_hat and ESS say nothing extra.
REPORTED_VARS = (
    "capability", "difficulty", "slope", "sigma",
    "tau_cd", "tau_alpha", "sigma_median", "tau_sigma",
)

# Coverage levels reported by the calibration check.
DEFAULT_COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)

# Test statistics used for the posterior predictive p-values. Each maps a
# (draws, n_obs) array (or a (n_obs,) one) to a value per draw.
PPC_STATISTICS = {
    "mean": lambda y: y.mean(axis=-1),
    "sd": lambda y: y.std(axis=-1),
    "variance": lambda y: y.var(axis=-1),
    "iqr": lambda y: (np.percentile(y, 75, axis=-1) - np.percentile(y, 25, axis=-1)),
    "min": lambda y: y.min(axis=-1),
    "max": lambda y: y.max(axis=-1),
    "skew": lambda y: stats.skew(y, axis=-1),
    "frac_below_0.05": lambda y: (y < 0.05).mean(axis=-1),
    "frac_above_0.95": lambda y: (y > 0.95).mean(axis=-1),
}

# Statistics the Beta cannot match once any score is clipped to the bound:
# no replicate can equal an observed exact 0 or 1, so their p-value pins at 0
# or 1 and says nothing about fit. Excluded from the flagged-check tally.
DEGENERATE_AT_BOUNDARY = ("min", "max")

# Pareto k bands from the PSIS-LOO literature: below 0.7 the importance
# sampling estimate is reliable, above 1.0 it is not usable.
PARETO_K_BANDS = ((0.5, "good"), (0.7, "ok"), (1.0, "bad"))

_OVERALL = "OVERALL"

# How to read each metric: which direction is good, and what value would mean
# the model is right. Blank where the number is context rather than a verdict.
METRIC_GUIDANCE = {
    # data
    "n_obs_clipped": "rows the Beta cannot see",
    # sampling
    "divergences": "0 ideal; any is a warning",
    "divergence_rate": "0 ideal",
    "max_r_hat": "lower; <= 1.01",
    "n_r_hat_above_1.01": "0 ideal",
    "min_ess_bulk": "higher; >= 400",
    "min_ess_tail": "higher; >= 400",
    "min_bfmi": "higher; > 0.3",
    # accuracy
    "rmse": "lower is better",
    "mae": "lower is better",
    "bias": "ideal 0",
    "rmse_ratio_median": "ideal 1; < 1 = intervals over-wide",
    "rmse_ratio_max": "ideal <= 1; > 1 = over-confident",
    "rmse_loo": "lower; out-of-sample",
    "mae_loo": "lower; out-of-sample",
    # loo
    "elpd_loo": "higher is better",
    "elpd_loo_se": "SE of elpd_loo",
    "p_loo": "effective parameters",
    "max_pareto_k": "lower; < 0.7",
    "pareto_k_good": "k <= 0.5; higher is better",
    "pareto_k_ok": "0.5 < k <= 0.7; usable",
    "pareto_k_bad": "0.7 < k <= 1; lower is better",
    "pareto_k_very_bad": "k > 1; 0 ideal",
    # calibration
    "mean_pit": "ideal 0.5",
    "ks_statistic": "lower is better",
    "ks_pvalue": "low p = detectably non-uniform",
    "coverage_0.5": "ideal 0.50",
    "coverage_0.8": "ideal 0.80",
    "coverage_0.9": "ideal 0.90",
    "coverage_0.95": "ideal 0.95",
    # The same on the PSIS-LOO PIT: free of the in-sample PIT's pull towards
    # 0.5, but only as reliable as the Pareto k of the points behind it.
    "mean_pit_loo": "ideal 0.5; out-of-sample",
    "ks_statistic_loo": "lower is better; out-of-sample",
    "ks_pvalue_loo": "low p = detectably non-uniform",
    "coverage_0.5_loo": "ideal 0.50; out-of-sample",
    "coverage_0.8_loo": "ideal 0.80; out-of-sample",
    "coverage_0.9_loo": "ideal 0.90; out-of-sample",
    "coverage_0.95_loo": "ideal 0.95; out-of-sample",
    # posterior predictive
    # Pooled checks are sensitive but unspecific: they mix the spread of the
    # fitted curve (32% of total variance) with the scatter sigma governs
    # (68%), and over 2,780 rows they flag biases far too small for any one
    # benchmark to see. Read them as "something small is off", not as a verdict.
    "ppp_mean": "ideal 0.5; pooled",
    "ppp_sd": "pooled; sigma is judged per benchmark",
    "ppp_variance": "pooled; sigma is judged per benchmark",
    "ppp_iqr": "pooled; sigma is judged per benchmark",
    "ppp_min": "meaningless; the clip pins it",
    "ppp_max": "meaningless; the clip pins it",
    "ppp_skew": "ideal 0.5; the Beta cannot flex its skew",
    "ppp_frac_below_0.05": "ideal 0.5; pooled, so very sensitive",
    "ppp_frac_above_0.95": "ideal 0.5; pooled, so very sensitive",
    "ppp_variance_median": "ideal 0.5; this judges sigma",
    "ppp_iqr_median": "ideal 0.5; this judges sigma",
    "ppp_variance_n_outside": "lower is better",
    "ppp_iqr_n_outside": "lower is better",
    "n_checks_outside": "10% expected by chance",
}


def _thin(trace: az.InferenceData, max_draws: int) -> az.InferenceData:
    """Subsample the draw dimension so the per-observation arrays stay small,
    keeping all chains and spacing the draws evenly."""
    n_chains = trace.posterior.sizes["chain"]
    n_draws = trace.posterior.sizes["draw"]
    if n_chains * n_draws <= max_draws:
        return trace
    keep = max(1, max_draws // n_chains)
    return trace.isel(draw=np.linspace(0, n_draws - 1, keep).astype(int))


def _flatten(values) -> np.ndarray:
    """Collapse (chain, draw, item) into (sample, item)."""
    return values.stack(sample=("chain", "draw")).transpose("sample", ...).to_numpy()


def _positions(trace: az.InferenceData, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Row -> position in the trace's model and benchmark coordinates.

    Matched on names, which is what the model uses as coordinates, so this
    does not depend on the positional ids in the input data.
    """
    model_pos = {name: i for i, name in enumerate(trace.posterior.coords["model"].values)}
    bench_pos = {name: i for i, name in enumerate(trace.posterior.coords["benchmark"].values)}
    return (
        df["Model"].map(model_pos).to_numpy(),
        df["benchmark"].map(bench_pos).to_numpy(),
    )


def _posterior_mu(trace: az.InferenceData, df: pd.DataFrame) -> np.ndarray:
    """The fitted curve per draw and observation, shape (n_draws, n_obs)."""
    model_idx, bench_idx = _positions(trace, df)
    capability = _flatten(trace.posterior["capability"])[:, model_idx]
    difficulty = _flatten(trace.posterior["difficulty"])[:, bench_idx]
    slope = _flatten(trace.posterior["slope"])[:, bench_idx]
    return 1.0 / (1.0 + np.exp(-slope * (capability - difficulty)))


def _predictive(trace: az.InferenceData) -> np.ndarray:
    """Posterior predictive draws, shape (n_draws, n_obs)."""
    return _flatten(trace.posterior_predictive["obs"])


def _observed(df: pd.DataFrame, clip_eps: float = 0.0) -> np.ndarray:
    """The observed scores, optionally clipped the way the likelihood saw them."""
    y = df["performance"].to_numpy(dtype=float)
    return np.clip(y, clip_eps, 1.0 - clip_eps) if clip_eps > 0 else y


def _pit(y_rep: np.ndarray, y: np.ndarray, weights: "np.ndarray | None" = None) -> np.ndarray:
    """PIT_i = P(y_rep_i <= y_i), per observation.

    Unweighted it is the in-sample PIT. With the (n_obs, n_draws) PSIS-LOO
    ``weights`` it is the LOO-PIT: the same share, but over the posterior
    reweighted as if observation i had been left out of the fit.
    """
    below = y_rep <= y  # (n_draws, n_obs)
    if weights is None:
        return below.mean(axis=0)
    # Normalised weights sum to 1 per observation; the clip only absorbs
    # floating-point overshoot past the ends.
    return np.clip((weights * below.T).sum(axis=1), 0.0, 1.0)


def _groups(df: pd.DataFrame):
    """Yield (benchmark name, row mask) per benchmark, then the overall mask."""
    benchmark = df["benchmark"].to_numpy()
    for name in pd.unique(benchmark):
        yield name, benchmark == name
    yield _OVERALL, np.ones(len(df), dtype=bool)


def _require_group(trace: az.InferenceData, group: str) -> None:
    if group not in trace:
        raise ValueError(
            f"the trace has no {group} group; refit with evaluate=True "
            "(or sample it yourself with that group requested)"
        )


def sampler_diagnostics(
    trace: az.InferenceData, var_names: "tuple[str, ...] | None" = REPORTED_VARS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Did NUTS sample the posterior properly?

    Returns (summary, per_parameter). The summary is a single row: divergences,
    the worst r_hat, the smallest effective sample sizes, the average
    acceptance rate and the per-chain BFMI (below ~0.2 suggests the sampler is
    exploring the energy distribution poorly). per_parameter carries r_hat,
    ess_bulk, ess_tail and the Monte Carlo standard errors for every
    parameter, worst r_hat first.

    ``var_names`` defaults to the parameters the fit reports; pass None to
    include every variable in the trace.
    """
    per_parameter = az.summary(
        trace, var_names=list(var_names) if var_names else None, kind="diagnostics"
    ).rename_axis("parameter")
    per_parameter = per_parameter.sort_values("r_hat", ascending=False).reset_index()

    stats_group = trace.sample_stats
    n_draws = int(stats_group.sizes["chain"] * stats_group.sizes["draw"])
    divergences = int(stats_group["diverging"].sum()) if "diverging" in stats_group else 0
    # arviz>=1.0 returns bfmi as a one-group DataTree; the per-chain values
    # are its "energy" variable (the name is a holdover, not raw NUTS energy).
    bfmi = az.bfmi(trace)["energy"].to_numpy()

    summary = {
        "n_chains": int(stats_group.sizes["chain"]),
        "n_draws": n_draws,
        "divergences": divergences,
        "divergence_rate": divergences / n_draws,
        "max_r_hat": float(per_parameter["r_hat"].max()),
        "n_r_hat_above_1.01": int((per_parameter["r_hat"] > 1.01).sum()),
        "n_parameters": len(per_parameter),
        "min_ess_bulk": float(per_parameter["ess_bulk"].min()),
        "min_ess_tail": float(per_parameter["ess_tail"].min()),
        "min_bfmi": float(bfmi.min()),
    }
    if "acceptance_rate" in stats_group:
        summary["mean_acceptance_rate"] = float(stats_group["acceptance_rate"].mean())
    if "tree_depth" in stats_group:
        summary["max_tree_depth"] = int(stats_group["tree_depth"].max())
    return pd.DataFrame([summary]), per_parameter


def predictive_accuracy(
    trace: az.InferenceData, df: pd.DataFrame, loo_weights: "np.ndarray | None" = None
) -> tuple[pd.DataFrame, dict]:
    """
    How closely does the fitted curve track the observed scores?

    Returns (per_benchmark, totals).

    ``totals`` holds the headline errors: RMSE, MAE and bias of the posterior
    mean curve, and - when ``loo_weights`` (n_obs, n_draws) are supplied - the
    same scored on leave-one-out weighted predictions. Those are
    out-of-sample, so the gap between the two is the optimism of the in-sample
    fit.

    ``per_benchmark`` deliberately does **not** carry raw RMSE. Per benchmark
    that number is not a measure of fit: it is a restatement of ``sigma``,
    which the fit already reports. Measured on the published data, a
    benchmark's RMSE regresses on the predictive SD the model itself claims
    with R^2 = 0.93, and ranks against the fitted noise level at spearman
    0.92 (measured on ``kappa``, the concentration ``sigma`` sets) - so
    ranking benchmarks by RMSE just re-ranks them by the noise parameter the
    model already estimated. A high-RMSE benchmark is not one the fit handles
    badly; it is one the fit already knows is noisy.

    What the ratio ``rmse_ratio = rmse / predicted_sd`` asks instead is
    whether that noise estimate is *honest*: near 1 the benchmark scatters as
    much as the model claims, below 1 the model reserves more spread than it
    needs, above 1 it is over-confident there. It ranks benchmarks differently
    from raw RMSE - on the published data only 2 of the top 5 coincide.

    Scored against the raw observed scores, not the clipped ones the
    likelihood sees, so the boundary rows are judged honestly.
    """
    mu = _posterior_mu(trace, df)
    y = _observed(df)  # raw: the error a user of the index would see
    mu_mean = mu.mean(axis=0)
    residual = mu_mean - y
    # The spread the model claims per observation, averaged over draws. For
    # Beta(mu*kappa, (1-mu)*kappa) that is mu(1-mu)/(kappa+1).
    kappa = _flatten(trace.posterior["kappa"])[:, _positions(trace, df)[1]]
    predictive_var = (mu * (1.0 - mu) / (kappa + 1.0)).mean(axis=0)
    loo_mu = (loo_weights * mu.T).sum(axis=1) if loo_weights is not None else None

    def rmse(mask, prediction):
        return float(np.sqrt(((prediction[mask] - y[mask]) ** 2).mean()))

    rows = []
    for name, mask in _groups(df):
        if name == _OVERALL:
            continue
        predicted_sd = float(np.sqrt(predictive_var[mask].mean()))
        rows.append({
            "benchmark": name,
            "n_obs": int(mask.sum()),
            "observed_mean": float(y[mask].mean()),
            "predicted_mean": float(mu_mean[mask].mean()),
            "bias": float(residual[mask].mean()),
            "predicted_sd": predicted_sd,
            "rmse_ratio": rmse(mask, mu_mean) / predicted_sd if predicted_sd else np.nan,
        })
    per_benchmark = pd.DataFrame(rows).sort_values("rmse_ratio", ascending=False)

    everything = np.ones(len(df), dtype=bool)
    totals = {
        "rmse": rmse(everything, mu_mean),
        "mae": float(np.abs(residual).mean()),
        "bias": float(residual.mean()),
        "rmse_ratio_median": float(per_benchmark["rmse_ratio"].median()),
        "rmse_ratio_max": float(per_benchmark["rmse_ratio"].max()),
    }
    if loo_mu is not None:
        totals["rmse_loo"] = rmse(everything, loo_mu)
        totals["mae_loo"] = float(np.abs(loo_mu - y).mean())
    return per_benchmark, totals


def loo_diagnostics(
    trace: az.InferenceData, df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Leave-one-out cross-validation by Pareto-smoothed importance sampling.

    Returns (summary, per_observation, weights):
    - summary: elpd_loo with its standard error, p_loo (the effective number
      of parameters), and how many observations fall in each Pareto k band.
      A k above 0.7 means the importance sampling estimate for that
      observation is unreliable - usually a point the model finds surprising.
    - per_observation: pointwise elpd and k, worst k first, with the model,
      benchmark and score so a bad point can be looked up directly.
    - weights: the (n_obs, n_draws) normalised LOO importance weights, for
      scoring out-of-sample predictions.

    elpd_loo is comparable across models fitted to the *same* observations
    with the same likelihood family; it is not comparable to a fit whose
    likelihood puts mass on different points (see the clipping note in
    :mod:`eci.bayesian`).
    """
    _require_group(trace, "log_likelihood")
    loo = az.loo(trace, pointwise=True)
    pareto_k = np.asarray(loo.pareto_k)
    elpd_i = np.asarray(loo.elpd_i)

    # loo(pointwise=True) already carries the PSIS log importance weights
    # (n_obs, chain, draw); flatten chain/draw the same way _flatten does
    # elsewhere and transpose to match the (n_obs, n_draws) shape below.
    weights = np.exp(_flatten(loo.log_weights).T)

    counts = {}
    previous = -np.inf
    for threshold, label in PARETO_K_BANDS:
        counts[f"pareto_k_{label}"] = int(((pareto_k > previous) & (pareto_k <= threshold)).sum())
        previous = threshold
    counts["pareto_k_very_bad"] = int((pareto_k > PARETO_K_BANDS[-1][0]).sum())

    summary = pd.DataFrame([{
        "elpd_loo": float(loo.elpd),
        "se": float(loo.se),
        "p_loo": float(loo.p),
        "n_obs": int(len(elpd_i)),
        "max_pareto_k": float(pareto_k.max()),
        **counts,
    }])

    per_observation = df[["Model", "benchmark", "performance"]].reset_index(drop=True)
    per_observation["elpd_i"] = elpd_i
    per_observation["pareto_k"] = pareto_k
    per_observation = per_observation.sort_values("pareto_k", ascending=False)
    return summary, per_observation, weights


def posterior_predictive_checks(
    trace: az.InferenceData,
    df: pd.DataFrame,
    statistics: "dict | None" = None,
    clip_eps: float = 0.0,
) -> pd.DataFrame:
    """
    Posterior predictive p-values.

    For each test statistic T and benchmark, replicate the data from the
    posterior and compare: ppp = P(T(y_rep) >= T(y_obs)). A value near 0.5
    means the model reproduces that feature of the data; near 0 or 1 means it
    systematically under- or over-produces it. Spread statistics (sd,
    variance, iqr) are the ones the per-benchmark ``sigma`` is there to get
    right; the tail statistics say whether the Beta reaches the extreme
    scores the benchmarks actually show.

    Reported as mid-p values - ``P(T_rep > T_obs) + 0.5 * P(T_rep = T_obs)``.
    The uncorrected ``P(T_rep >= T_obs)`` counts a tie as a disagreement,
    which pins a statistic the model reproduces exactly at 1.0 instead of 0.5.
    On the published data that alone accounted for 78 of 115 apparently
    failing per-benchmark checks: every benchmark with no scores above 0.95
    has ``T_obs = 0``, and ``P(T_rep >= 0) = 1`` no matter how the model
    behaves.

    These p-values are conservative - the same data informed the posterior -
    so treat 0.05/0.95 as a flag to look closer, not as a test.

    Read the spread statistics (sd, variance, iqr) **per benchmark**, not on
    the OVERALL row. Pooled across benchmarks they measure the spread of the
    fitted curve itself, which partial pooling deliberately shrinks, so the
    pooled p-value runs low even for a model that is right: on a correctly
    specified synthetic fit the ten per-benchmark variance p-values ranged
    0.11-0.65 while the pooled one was 0.0005. The per-benchmark rows ask the
    question ``sigma`` is actually there to answer.

    ``min`` and ``max`` are degenerate wherever the data contain scores of
    exactly 0 or 1: the Beta has no mass there, so no replicate can match the
    observed extreme and the p-value pins at 1. Read ``frac_below_0.05`` and
    ``frac_above_0.95`` instead - they ask the same question about the tails
    without needing the model to hit a point it cannot represent.
    """
    _require_group(trace, "posterior_predictive")
    statistics = statistics or PPC_STATISTICS
    y_rep = _predictive(trace)
    y = _observed(df, clip_eps)

    rows = []
    for name, mask in _groups(df):
        y_b, rep_b = y[mask], y_rep[:, mask]
        for stat_name, statistic in statistics.items():
            observed = float(statistic(y_b))
            replicated = np.asarray(statistic(rep_b), dtype=float)
            # Mid-p: ties split evenly. Without it a statistic the model
            # reproduces exactly scores 1.0 rather than 0.5, because
            # P(T_rep >= T_obs) counts every tie as a disagreement. That
            # matters for the discrete statistics: a benchmark with no scores
            # above 0.95 has T_obs = 0, so P(T_rep >= 0) = 1 whatever the
            # model does. For the continuous statistics exact ties are
            # essentially impossible and this changes nothing.
            ties = float((replicated == observed).mean())
            rows.append({
                "benchmark": name,
                "statistic": stat_name,
                "n_obs": int(mask.sum()),
                "observed": observed,
                "replicated_mean": float(replicated.mean()),
                "replicated_ci_low": float(np.quantile(replicated, 0.05)),
                "replicated_ci_high": float(np.quantile(replicated, 0.95)),
                "ppp": float((replicated > observed).mean()) + 0.5 * ties,
            })
    return pd.DataFrame(rows)


def calibration(
    trace: az.InferenceData,
    df: pd.DataFrame,
    levels: tuple[float, ...] = DEFAULT_COVERAGE_LEVELS,
    clip_eps: float = 0.0,
    loo_weights: "np.ndarray | None" = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Are the predictive intervals the width they claim to be?

    The probability integral transform PIT_i = P(y_rep <= y_obs_i) is uniform
    when the predictive distribution is calibrated. Returns
    (per_observation, coverage): the PIT of every observation, and per
    benchmark the share of observations inside each central interval (which
    should match the level) plus a Kolmogorov-Smirnov test of uniformity.

    The in-sample PIT scores each observation against a posterior that was
    fitted to it, so the observation sits nearer the middle of its own
    predictive distribution than a new one would: the PIT bunches towards 0.5
    and the intervals look wider than they are. When ``loo_weights``
    (n_obs, n_draws) are supplied, the same checks are repeated on the
    PSIS-LOO PIT, with a ``_loo`` suffix (``pit_loo``, ``coverage_0.9_loo``,
    ...), which removes that double use of the data. The LOO-PIT of a point
    with Pareto k > 0.7 is as unreliable as its importance weights.

    Scored against the clipped scores (``clip_eps``), for the reason given in
    the module docstring: the likelihood never saw an exact 0 or 1. Clipping
    still puts a clump of identical values at each clipped bound, which makes
    the PIT lumpy there, so read the KS statistic alongside the number of
    clipped observations rather than on its own.
    """
    _require_group(trace, "posterior_predictive")
    y_rep = _predictive(trace)
    y = _observed(df, clip_eps)
    pits = {"": _pit(y_rep, y)}
    if loo_weights is not None:
        pits["_loo"] = _pit(y_rep, y, loo_weights)

    per_observation = df[["Model", "benchmark", "performance"]].reset_index(drop=True)
    for suffix, pit in pits.items():
        per_observation[f"pit{suffix}"] = pit

    rows = []
    for name, mask in _groups(df):
        row = {"benchmark": name, "n_obs": int(mask.sum())}
        for suffix, pit in pits.items():
            pit_b = pit[mask]
            # The asymptotic p-value: the exact one overflows past a few
            # thousand observations, and the PIT is tied anyway (it is a
            # share of a finite number of draws), which the exact test
            # assumes away.
            ks = stats.kstest(pit_b, "uniform", method="asymp")
            row[f"mean_pit{suffix}"] = float(pit_b.mean())
            row[f"ks_statistic{suffix}"] = float(ks.statistic)
            row[f"ks_pvalue{suffix}"] = float(ks.pvalue)
            for level in levels:
                tail = (1.0 - level) / 2.0
                row[f"coverage_{level:g}{suffix}"] = float(
                    ((pit_b >= tail) & (pit_b <= 1.0 - tail)).mean()
                )
        rows.append(row)
    return per_observation, pd.DataFrame(rows)


def qq_data(
    trace: az.InferenceData,
    df: pd.DataFrame,
    n_quantiles: int = 200,
    clip_eps: float = 0.0,
    loo_weights: "np.ndarray | None" = None,
) -> dict:
    """
    Arrays for the two QQ plots, pooled over all observations.

    - PIT against Uniform(0, 1): the calibration plot. Points on the diagonal
      mean the predictive distribution has the right shape; an S-curve means
      the intervals are systematically too wide or too narrow. With
      ``loo_weights`` the PSIS-LOO PIT is added as ``pit_loo_sorted``, so the
      plot can show how much of any apparent over-width is the in-sample PIT
      scoring points against a fit that has already seen them (see
      :func:`calibration`).
    - Observed score quantiles against posterior predictive quantiles, with a
      90% band from the replicated datasets, which shows *where* on the score
      scale the fit departs from the data.
    """
    _require_group(trace, "posterior_predictive")
    y_rep = _predictive(trace)
    observed = _observed(df, clip_eps)
    y = np.sort(observed)
    pit = np.sort(_pit(y_rep, observed))

    levels = np.linspace(0, 1, n_quantiles + 2)[1:-1]
    replicated = np.percentile(y_rep, levels * 100, axis=1)  # (n_quantiles, n_draws)
    qq = {
        "pit_sorted": pit,
        "pit_uniform": (np.arange(1, len(pit) + 1) - 0.5) / len(pit),
        "quantile_levels": levels,
        "observed_quantiles": np.percentile(y, levels * 100),
        "replicated_median": np.median(replicated, axis=1),
        "replicated_ci_low": np.quantile(replicated, 0.05, axis=1),
        "replicated_ci_high": np.quantile(replicated, 0.95, axis=1),
    }
    if loo_weights is not None:
        qq["pit_loo_sorted"] = np.sort(_pit(y_rep, observed, loo_weights))
    return qq


def benchmark_forest_data(trace: az.InferenceData, hdi_prob: float = 0.90) -> pd.DataFrame:
    """
    Per-benchmark posterior mean and HDI of ``sigma`` and ``difficulty``, one
    row per benchmark, sorted by difficulty.

    Both are model parameters, not per-observation quantities, so this runs
    on the full trace regardless of ``max_draws``. In native model units
    (difficulty sums to 0 by its zero-sum prior; sigma is the Beta's
    SD at mu = 0.5) rather than the EDI scale, since the transform to EDI
    lives in :mod:`eci.bayesian` and isn't available here - this is a
    diagnostic view of the fit's own parameters, not the published scores.
    """
    posterior = trace.posterior.dataset[["sigma", "difficulty"]]
    hdi = az.hdi(posterior, prob=hdi_prob)
    mean = posterior.mean(dim=("chain", "draw"))
    forest = pd.DataFrame({
        "benchmark": trace.posterior.coords["benchmark"].values,
        "sigma_mean": mean["sigma"].values,
        "sigma_lo": hdi["sigma"].sel(ci_bound="lower").values,
        "sigma_hi": hdi["sigma"].sel(ci_bound="upper").values,
        "difficulty_mean": mean["difficulty"].values,
        "difficulty_lo": hdi["difficulty"].sel(ci_bound="lower").values,
        "difficulty_hi": hdi["difficulty"].sel(ci_bound="upper").values,
    })
    return forest.sort_values("difficulty_mean").reset_index(drop=True)


def evaluate_fit(
    trace: az.InferenceData,
    df: pd.DataFrame,
    max_draws: int = 2000,
    performance_clip_eps: float = 1e-3,
) -> dict:
    """
    Run the whole suite and return it as a dict of tables.

    Args:
        trace: the fit's InferenceData, which must carry the
            posterior_predictive and log_likelihood groups (``evaluate=True``
            in :func:`eci.bayesian.fit_bayesian_eci` asks for both).
        df: the same input the model was fitted to.
        max_draws: draws to keep for the per-observation metrics; the sampler
            diagnostics always use the full trace.
        performance_clip_eps: the clip the fit applied. The distributional
            checks are scored against the clipped scores and accuracy against
            the raw ones - see the module docstring. Pass the same value the
            fit used, or 0 to score everything raw.

    Returns a dict with keys sampler_summary, sampler_parameters, accuracy,
    loo_summary, loo_observations, ppc, calibration, pit, forest and qq
    (arrays for plotting, not a table).
    """
    sampler_summary, sampler_parameters = sampler_diagnostics(trace)
    raw = _observed(df)
    sampler_summary["n_obs_clipped"] = int(
        ((raw < performance_clip_eps) | (raw > 1.0 - performance_clip_eps)).sum()
        if performance_clip_eps > 0 else 0
    )
    thinned = _thin(trace, max_draws)

    loo_summary, loo_observations, weights = loo_diagnostics(thinned, df)
    accuracy, totals = predictive_accuracy(thinned, df, loo_weights=weights)
    pit, coverage = calibration(
        thinned, df, clip_eps=performance_clip_eps, loo_weights=weights
    )
    ppc = posterior_predictive_checks(thinned, df, clip_eps=performance_clip_eps)
    results = {
        "sampler_summary": sampler_summary,
        "sampler_parameters": sampler_parameters,
        "accuracy": accuracy,
        "loo_summary": loo_summary,
        "loo_observations": loo_observations,
        "ppc": ppc,
        "calibration": coverage,
        "pit": pit,
        "forest": benchmark_forest_data(trace),
        "qq": qq_data(thinned, df, clip_eps=performance_clip_eps, loo_weights=weights),
    }
    results["metrics"] = _metrics_table(results, totals, df)
    return results


def _metrics_table(results: dict, totals: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Every headline number as one flat (section, metric, value) table.

    This is the table to read first; the others break each section down.
    """
    sampler = results["sampler_summary"].iloc[0]
    loo = results["loo_summary"].iloc[0]
    coverage = _overall(results["calibration"])
    ppc = results["ppc"]
    pooled = ppc[ppc["benchmark"] == _OVERALL].set_index("statistic")["ppp"]
    per_benchmark = ppc[
        (ppc["benchmark"] != _OVERALL) & ~ppc["statistic"].isin(DEGENERATE_AT_BOUNDARY)
    ]

    rows = [
        ("data", "n_obs", len(df)),
        ("data", "n_models", df["Model"].nunique()),
        ("data", "n_benchmarks", df["benchmark"].nunique()),
        ("data", "n_obs_clipped", int(sampler["n_obs_clipped"])),
    ]
    rows += [("sampling", name, sampler[name]) for name in (
        "n_chains", "n_draws", "divergences", "divergence_rate", "max_r_hat",
        "n_r_hat_above_1.01", "n_parameters", "min_ess_bulk", "min_ess_tail", "min_bfmi",
    ) if name in sampler]
    rows += [("accuracy", name, value) for name, value in totals.items()]
    # n_obs would collide with the data section; se is ambiguous on its own
    rows += [("loo", "elpd_loo_se" if name == "se" else name, loo[name])
             for name in loo.index if name != "n_obs"]
    rows += [("calibration", name, coverage[name]) for name in coverage.index
             if name not in ("benchmark", "n_obs")]
    rows += [("ppc_pooled", f"ppp_{name}", value) for name, value in pooled.items()]
    for statistic in ("variance", "iqr"):
        values = per_benchmark.loc[per_benchmark["statistic"] == statistic, "ppp"]
        rows.append(("ppc_per_benchmark", f"ppp_{statistic}_median", values.median()))
        rows.append(("ppc_per_benchmark", f"ppp_{statistic}_n_outside",
                     int(((values < 0.05) | (values > 0.95)).sum())))
    rows.append(("ppc_per_benchmark", "n_checks", len(per_benchmark)))
    rows.append(("ppc_per_benchmark", "n_checks_outside", int(
        ((per_benchmark["ppp"] < 0.05) | (per_benchmark["ppp"] > 0.95)).sum())))

    table = pd.DataFrame(rows, columns=["section", "metric", "value"])
    table["guide"] = table["metric"].map(METRIC_GUIDANCE).fillna("")
    duplicated = table.loc[table["metric"].duplicated(), "metric"].tolist()
    if duplicated:  # names are the lookup key, so a collision is a bug
        raise ValueError(f"duplicate metric names in the metrics table: {duplicated}")
    return table


def _overall(table: pd.DataFrame) -> pd.Series:
    return table.loc[table["benchmark"] == _OVERALL].iloc[0]


def summarise_evaluation(results: dict) -> str:
    """One screenful of the numbers worth reading first."""
    sampler = results["sampler_summary"].iloc[0]
    metrics = results["metrics"].set_index("metric")["value"]
    loo = results["loo_summary"].iloc[0]
    coverage = _overall(results["calibration"])
    ppc = results["ppc"]
    ppc_overall = ppc[ppc["benchmark"] == _OVERALL].set_index("statistic")["ppp"]

    lines = [
        "Sampling",
        f"  divergences        {int(sampler['divergences'])} / {int(sampler['n_draws'])} draws",
        f"  max r_hat          {sampler['max_r_hat']:.4f} "
        f"({int(sampler['n_r_hat_above_1.01'])} of {int(sampler['n_parameters'])} above 1.01)",
        f"  min ESS            {sampler['min_ess_bulk']:.0f} bulk / {sampler['min_ess_tail']:.0f} tail",
        f"  min BFMI           {sampler['min_bfmi']:.2f}",
        "",
        f"{int(sampler['n_obs_clipped'])} observations sat at 0 or 1 and were clipped for the "
        "likelihood;\nthe distributional checks below are scored against the clipped values.",
        "",
        "Predictive accuracy (score units)",
        f"  RMSE               {metrics['rmse']:.4f}"
        + (f"   LOO {metrics['rmse_loo']:.4f}" if "rmse_loo" in metrics else ""),
        f"  MAE                {metrics['mae']:.4f}"
        + (f"   LOO {metrics['mae_loo']:.4f}" if "mae_loo" in metrics else ""),
        f"  bias               {metrics['bias']:+.4f}",
        f"  RMSE / predicted SD, per benchmark: median "
        f"{metrics['rmse_ratio_median']:.2f}, max {metrics['rmse_ratio_max']:.2f}",
        "  (near 1 = the benchmark scatters as much as the model claims;",
        "   below 1 = the model reserves more spread than it needs)",
        "",
        "Leave-one-out",
        f"  elpd_loo           {loo['elpd_loo']:.1f} +/- {loo['se']:.1f}",
        f"  p_loo              {loo['p_loo']:.1f}",
        f"  Pareto k           {int(loo['pareto_k_good'])} good, {int(loo['pareto_k_ok'])} ok, "
        f"{int(loo['pareto_k_bad'])} bad, {int(loo['pareto_k_very_bad'])} very bad "
        f"(max {loo['max_pareto_k']:.2f})",
        "",
        "Calibration (share of observations inside each central interval)",
    ]
    # One column per PIT: in-sample always, PSIS-LOO when the weights were there.
    suffixes = ["", "_loo"] if "mean_pit_loo" in coverage.index else [""]
    if len(suffixes) > 1:
        lines.append(f"  {'':<18}{'in-sample':>10}{'LOO':>10}")

    def row(label, metric, fmt=".3f"):
        values = "".join(f"{coverage[metric + s]:>10{fmt}}" for s in suffixes)
        return f"  {label:<18}{values}"

    for column in [c for c in coverage.index
                   if c.startswith("coverage_") and not c.endswith("_loo")]:
        lines.append(row(f"{column.split('_')[1]} interval", column))
    lines += [
        row("mean PIT", "mean_pit") + "   (0.5 when calibrated)",
        row("KS vs uniform", "ks_statistic"),
        row("  KS p-value", "ks_pvalue", ".3g"),
    ]
    lines += [
        "",
        "Posterior predictive p-values, pooled (0.5 is ideal, 0/1 is a red flag)",
    ]
    spread = {"sd", "variance", "iqr"}
    for name, value in ppc_overall.items():
        note = "  <- read per benchmark instead" if name in spread else ""
        lines.append(f"  {name:<18} {value:.3f}{note}")

    per_benchmark = ppc[
        (ppc["benchmark"] != _OVERALL)
        & ~ppc["statistic"].isin(DEGENERATE_AT_BOUNDARY)
    ]
    flagged = per_benchmark[(per_benchmark["ppp"] < 0.05) | (per_benchmark["ppp"] > 0.95)]
    lines += [
        "",
        "Per-benchmark spread checks (what the per-benchmark sigma is for)",
    ]
    for name in ("variance", "iqr"):
        values = per_benchmark.loc[per_benchmark["statistic"] == name, "ppp"]
        outside = ((values < 0.05) | (values > 0.95)).sum()
        lines.append(
            f"  {name:<18} median {values.median():.3f}, "
            f"{outside} of {len(values)} benchmarks outside [0.05, 0.95]"
        )
    lines += [
        "",
        f"{len(flagged)} of {len(per_benchmark)} per-benchmark checks "
        "fall outside [0.05, 0.95]",
        f"(excluding {'/'.join(DEGENERATE_AT_BOUNDARY)}, which the clipped "
        "boundary pins at 0 or 1)",
    ]
    return "\n".join(lines)


def _format_value(value) -> str:
    """Render a metric at a sensible precision for reading, not for parsing."""
    number = float(value)
    if number.is_integer() and abs(number) < 1e7:
        return f"{int(number):,}"
    magnitude = abs(number)
    if magnitude >= 100:
        return f"{number:,.1f}"
    if magnitude >= 1:
        return f"{number:.3f}"
    if magnitude < 1e-4:  # only genuinely tiny values are worth scientific form
        return f"{number:.2e}"
    return f"{number:.4f}"


# Headings for the metrics page, in the order the sections are laid out.
_SECTION_TITLES = {
    "data": "Data",
    "sampling": "Sampling",
    "accuracy": "Predictive accuracy (score units)",
    "loo": "Leave-one-out",
    "calibration": "Calibration",
    "ppc_pooled": "Posterior predictive p-values, pooled",
    "ppc_per_benchmark": "Posterior predictive, per benchmark",
}


def _draw_metrics_page(fig, metrics: pd.DataFrame) -> None:
    """Lay the metrics table out over two columns, split at a section break."""
    blocks = []
    for section, rows in metrics.groupby("section", sort=False):
        lines = [("header", _SECTION_TITLES.get(section, section), "", "")]
        lines += [("row", r.metric, _format_value(r.value), r.guide)
                  for r in rows.itertuples()]
        blocks.append(lines)

    # Balance the columns by line count, breaking only between sections
    total = sum(len(b) + 1 for b in blocks)
    left, right, running = [], [], 0
    for block in blocks:
        (left if running < total / 2 else right).extend(block + [("gap", "", "", "")])
        running += len(block) + 1

    for column, x in ((left, 0.06), (right, 0.55)):
        y = 0.90
        for kind, label, value, guide in column:
            if kind == "gap":
                y -= 0.018
                continue
            if kind == "header":
                fig.text(x, y, label, fontsize=9.5, weight="bold", color="#1A1D26")
                y -= 0.026
                continue
            fig.text(x + 0.012, y, label, fontsize=8, family="monospace", color="#5A6078")
            fig.text(x + 0.245, y, value, fontsize=8, family="monospace",
                     color="#1A1D26", ha="right", weight="medium")
            if guide:
                fig.text(x + 0.262, y, guide, fontsize=6.8, color="#868DA3", style="italic")
            y -= 0.0225


def _draw_forest(ax, forest: pd.DataFrame, column: str, title: str, log_scale: bool = False):
    """One horizontal forest plot: posterior mean and HDI per benchmark."""
    y = np.arange(len(forest))
    mean, lo, hi = forest[f"{column}_mean"], forest[f"{column}_lo"], forest[f"{column}_hi"]
    ax.errorbar(
        mean, y, xerr=[mean - lo, hi - mean], fmt="o", ms=3, lw=1,
        color="#2563A8", ecolor="#2563A8", capsize=0,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(forest["benchmark"], fontsize=6.5)
    ax.set_ylim(-1, len(forest))
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    if log_scale:
        ax.set_xscale("log")
        # Plain labels (0.04, 0.1, 0.2) rather than 4x10^-2: the range spans
        # well under two decades, so a few ticks per decade are needed.
        ax.xaxis.set_major_locator(ticker.LogLocator(subs=(1.0, 2.0, 4.0, 6.0)))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def plot_evaluation(results: dict, path: "str | Path") -> Path:
    """Write the evaluation as a three-page PDF (needs matplotlib).

    Page 1 is the metrics table — every headline number, i.e. ``results["metrics"]``.
    Page 2 is the two QQ plots, which are here because they
    show something a table cannot: *where* on the score scale the fit departs
    from the data. Page 3 is a forest plot of every benchmark's ``sigma`` and
    ``difficulty`` (posterior mean and 90% HDI, from ``results["forest"]``) -
    the per-benchmark item parameters, visually, in place of a CSV.
    """
    path = Path(path)
    qq = results["qq"]
    forest = results["forest"]

    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Evaluation metrics", fontsize=14, weight="bold", x=0.06, ha="left", y=0.96)
        _draw_metrics_page(fig, results["metrics"])
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

        # Calibration: are the predictive distributions the right shape?
        axes[0].plot(qq["pit_uniform"], qq["pit_sorted"], lw=1.6, color="#2563A8",
                     label="in-sample")
        if "pit_loo_sorted" in qq:
            axes[0].plot(qq["pit_uniform"], qq["pit_loo_sorted"], lw=1.6, color="#0F766E",
                         label="PSIS-LOO")
        axes[0].plot([0, 1], [0, 1], color="#B45309", ls="--", lw=1)
        axes[0].set(
            xlabel="uniform quantile", ylabel="PIT quantile",
            title="PIT QQ — calibration", xlim=(0, 1), ylim=(0, 1),
        )
        axes[0].text(
            0.03, 0.95, "above the line = intervals too wide\nbelow = too narrow",
            transform=axes[0].transAxes, fontsize=8, va="top", color="#5A6078",
        )
        if "pit_loo_sorted" in qq:
            axes[0].legend(loc="lower right", fontsize=9)

        # Fit: where on the score scale do observed and predicted diverge?
        axes[1].fill_betweenx(
            qq["observed_quantiles"], qq["replicated_ci_low"], qq["replicated_ci_high"],
            alpha=0.22, color="#2563A8", label="90% band",
        )
        axes[1].plot(qq["replicated_median"], qq["observed_quantiles"], lw=1.6, color="#2563A8")
        axes[1].plot([0, 1], [0, 1], color="#B45309", ls="--", lw=1)
        axes[1].set(
            xlabel="predicted quantile", ylabel="observed quantile",
            title="Score QQ — where the fit departs", xlim=(0, 1), ylim=(0, 1),
        )
        axes[1].legend(loc="lower right", fontsize=9)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, max(4.8, 0.16 * len(forest) + 1.2)))
        fig.suptitle(
            "Per-benchmark item parameters (posterior mean, 90% HDI)",
            fontsize=11, weight="bold", x=0.06, ha="left",
        )
        _draw_forest(axes[0], forest, "sigma", "sigma (noise SD at mu = 0.5, log scale)", log_scale=True)
        _draw_forest(axes[1], forest, "difficulty", "difficulty (model units)")
        axes[1].axvline(0, color="#B45309", ls="--", lw=1)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig)
        plt.close(fig)

    return path


def save_evaluation(results: dict, output_dir: "str | Path") -> list[Path]:
    """
    Write the evaluation as ``evaluation_report.pdf``: the metrics table, the
    two QQ plots, and a forest plot of every benchmark's ``sigma`` and
    ``difficulty`` (see :func:`plot_evaluation`).

    The tables behind it - sampler parameters, per-benchmark accuracy, LOO,
    posterior predictive checks, calibration, PIT - are not written to disk;
    they're in the ``results`` dict returned by :func:`evaluate_fit` (and
    ``summarise_evaluation(results)`` for a terminal-readable version) for
    when a headline number needs drilling into.

    Returns the paths written (currently just the one PDF).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return [plot_evaluation(results, output_dir / "evaluation_report.pdf")]
