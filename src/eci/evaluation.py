"""
Evaluation suite for the Bayesian ECI fit.

Answers three separate questions, which are easy to conflate:

1. *Did the sampler work?* r_hat, effective sample size, divergences, BFMI.
   A failure here invalidates everything else, so read it first.
2. *Does the fitted curve predict the scores?* RMSE and MAE, both in-sample
   and leave-one-out weighted, overall and per benchmark.
3. *Is the model's uncertainty honest?* Posterior predictive p-values on the
   mean, spread and tails; PIT calibration and coverage; QQ arrays.

Every metric is reported per benchmark with an ``OVERALL`` row, so a single
badly behaved benchmark cannot hide inside a pooled number.

Two different observed vectors are in play, and mixing them up quietly breaks
the calibration numbers. Accuracy is scored against the **raw** scores, since
that is the error a user of the index would actually see. The distributional
checks - posterior predictive p-values, PIT, coverage, QQ - are scored against
the **clipped** scores the likelihood was fitted to: the model was never shown
a score of exactly 0 or 1 and cannot be asked to have predicted one. Scoring
those checks against the raw scores makes the clipping masquerade as
miscalibration (measured on a correctly specified synthetic fit: 90% coverage
reads 0.78 against raw scores and 0.96 against clipped ones).

The per-observation work (posterior predictive, PIT, QQ, LOO) runs on a thinned
subsample of the draws - see ``max_draws`` - because those arrays are
(draws x observations) and the full trace makes them needlessly large. The
sampler diagnostics always use every draw.

Typical use is through the fit itself::

    eci_df, edi_df, draws = fit_bayesian_eci(df, evaluate=True)
    results = draws["evaluation"]
    save_evaluation(results, "outputs/evaluation")

``save_evaluation`` writes one CSV per table plus a multi-page PDF report.
"""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy import stats

# Parameters the diagnostics summarise. The model's unit-scale offsets
# (capability_z and friends) are excluded: they are sampling coordinates, not
# results, and their r_hat says nothing extra once the quantity they build is
# converged.
REPORTED_VARS = ("capability", "difficulty", "slope", "kappa", "tau_cd", "tau_alpha")

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
    include the sampler's own unit-scale offsets as well.
    """
    per_parameter = az.summary(
        trace, var_names=list(var_names) if var_names else None, kind="diagnostics"
    ).rename_axis("parameter")
    per_parameter = per_parameter.sort_values("r_hat", ascending=False).reset_index()

    stats_group = trace.sample_stats
    n_draws = int(stats_group.sizes["chain"] * stats_group.sizes["draw"])
    divergences = int(stats_group["diverging"].sum()) if "diverging" in stats_group else 0
    bfmi = np.asarray(az.bfmi(trace))

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
) -> pd.DataFrame:
    """
    How closely does the fitted curve track the observed scores?

    RMSE, MAE and bias of the posterior mean curve, per benchmark and overall.
    When ``loo_weights`` (n_obs, n_draws) are supplied, the leave-one-out
    weighted predictions are scored as well: those are out-of-sample, so the
    gap between the two columns is the optimism of the in-sample fit.

    ``rmse_per_draw_ci_low``/``_high`` bound each individual draw's RMSE, and
    are not an interval around ``rmse``: averaging the curve before scoring it
    beats scoring each draw and averaging, so ``rmse`` normally sits below
    that interval rather than inside it.

    Scored against the raw observed scores, not the clipped ones the
    likelihood sees, so the boundary rows are judged honestly.
    """
    mu = _posterior_mu(trace, df)
    y = _observed(df)  # raw: the error a user of the index would see
    mu_mean = mu.mean(axis=0)
    residual = mu_mean - y
    loo_mu = (loo_weights * mu.T).sum(axis=1) if loo_weights is not None else None

    rows = []
    for name, mask in _groups(df):
        draw_rmse = np.sqrt(((mu[:, mask] - y[mask]) ** 2).mean(axis=1))
        row = {
            "benchmark": name,
            "n_obs": int(mask.sum()),
            "rmse": float(np.sqrt((residual[mask] ** 2).mean())),
            "mae": float(np.abs(residual[mask]).mean()),
            "bias": float(residual[mask].mean()),
            "rmse_per_draw_ci_low": float(np.quantile(draw_rmse, 0.05)),
            "rmse_per_draw_ci_high": float(np.quantile(draw_rmse, 0.95)),
            "observed_mean": float(y[mask].mean()),
            "predicted_mean": float(mu_mean[mask].mean()),
        }
        if loo_mu is not None:
            row["rmse_loo"] = float(np.sqrt(((loo_mu[mask] - y[mask]) ** 2).mean()))
            row["mae_loo"] = float(np.abs(loo_mu[mask] - y[mask]).mean())
        rows.append(row)

    return pd.DataFrame(rows)


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
    elpd_i = np.asarray(loo.loo_i)

    log_likelihood = _flatten(trace.log_likelihood["obs"])  # (n_draws, n_obs)
    log_weights, _ = az.psislw(-log_likelihood.T)
    weights = np.exp(log_weights)

    counts = {}
    previous = -np.inf
    for threshold, label in PARETO_K_BANDS:
        counts[f"pareto_k_{label}"] = int(((pareto_k > previous) & (pareto_k <= threshold)).sum())
        previous = threshold
    counts["pareto_k_very_bad"] = int((pareto_k > PARETO_K_BANDS[-1][0]).sum())

    summary = pd.DataFrame([{
        "elpd_loo": float(loo.elpd_loo),
        "se": float(loo.se),
        "p_loo": float(loo.p_loo),
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
    variance, iqr) are the ones the per-benchmark ``kappa`` is there to get
    right; the tail statistics say whether the Beta reaches the extreme
    scores the benchmarks actually show.

    These p-values are conservative - the same data informed the posterior -
    so treat 0.05/0.95 as a flag to look closer, not as a test.

    Read the spread statistics (sd, variance, iqr) **per benchmark**, not on
    the OVERALL row. Pooled across benchmarks they measure the spread of the
    fitted curve itself, which partial pooling deliberately shrinks, so the
    pooled p-value runs low even for a model that is right: on a correctly
    specified synthetic fit the ten per-benchmark variance p-values ranged
    0.11-0.65 while the pooled one was 0.0005. The per-benchmark rows ask the
    question ``kappa`` is actually there to answer.

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
            rows.append({
                "benchmark": name,
                "statistic": stat_name,
                "n_obs": int(mask.sum()),
                "observed": observed,
                "replicated_mean": float(replicated.mean()),
                "replicated_ci_low": float(np.quantile(replicated, 0.05)),
                "replicated_ci_high": float(np.quantile(replicated, 0.95)),
                "ppp": float((replicated >= observed).mean()),
            })
    return pd.DataFrame(rows)


def calibration(
    trace: az.InferenceData,
    df: pd.DataFrame,
    levels: tuple[float, ...] = DEFAULT_COVERAGE_LEVELS,
    clip_eps: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Are the predictive intervals the width they claim to be?

    The probability integral transform PIT_i = P(y_rep <= y_obs_i) is uniform
    when the predictive distribution is calibrated. Returns
    (per_observation, coverage): the PIT of every observation, and per
    benchmark the share of observations inside each central interval (which
    should match the level) plus a Kolmogorov-Smirnov test of uniformity.

    Scored against the clipped scores (``clip_eps``), for the reason given in
    the module docstring: the likelihood never saw an exact 0 or 1. Clipping
    still puts a clump of identical values at each clipped bound, which makes
    the PIT lumpy there, so read the KS statistic alongside the number of
    clipped observations rather than on its own.
    """
    _require_group(trace, "posterior_predictive")
    y_rep = _predictive(trace)
    y = _observed(df, clip_eps)
    pit = (y_rep <= y).mean(axis=0)

    per_observation = df[["Model", "benchmark", "performance"]].reset_index(drop=True)
    per_observation["pit"] = pit

    rows = []
    for name, mask in _groups(df):
        pit_b = pit[mask]
        # The asymptotic p-value: the exact one overflows past a few
        # thousand observations, and the PIT is tied anyway (it is a
        # share of a finite number of draws), which the exact test
        # assumes away.
        ks = stats.kstest(pit_b, "uniform", method="asymp")
        row = {
            "benchmark": name,
            "n_obs": int(mask.sum()),
            "mean_pit": float(pit_b.mean()),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
        }
        for level in levels:
            tail = (1.0 - level) / 2.0
            row[f"coverage_{level:g}"] = float(
                ((pit_b >= tail) & (pit_b <= 1.0 - tail)).mean()
            )
        rows.append(row)
    return per_observation, pd.DataFrame(rows)


def qq_data(
    trace: az.InferenceData, df: pd.DataFrame, n_quantiles: int = 200, clip_eps: float = 0.0
) -> dict:
    """
    Arrays for the two QQ plots, pooled over all observations.

    - PIT against Uniform(0, 1): the calibration plot. Points on the diagonal
      mean the predictive distribution has the right shape; an S-curve means
      the intervals are systematically too wide or too narrow.
    - Observed score quantiles against posterior predictive quantiles, with a
      90% band from the replicated datasets, which shows *where* on the score
      scale the fit departs from the data.
    """
    _require_group(trace, "posterior_predictive")
    y_rep = _predictive(trace)
    observed = _observed(df, clip_eps)
    y = np.sort(observed)
    pit = np.sort((y_rep <= observed).mean(axis=0))

    levels = np.linspace(0, 1, n_quantiles + 2)[1:-1]
    replicated = np.percentile(y_rep, levels * 100, axis=1)  # (n_quantiles, n_draws)
    return {
        "pit_sorted": pit,
        "pit_uniform": (np.arange(1, len(pit) + 1) - 0.5) / len(pit),
        "quantile_levels": levels,
        "observed_quantiles": np.percentile(y, levels * 100),
        "replicated_median": np.median(replicated, axis=1),
        "replicated_ci_low": np.quantile(replicated, 0.05, axis=1),
        "replicated_ci_high": np.quantile(replicated, 0.95, axis=1),
    }


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
    loo_summary, loo_observations, ppc, calibration, pit and qq (arrays for
    plotting, not a table).
    """
    sampler_summary, sampler_parameters = sampler_diagnostics(trace)
    raw = _observed(df)
    sampler_summary["n_obs_clipped"] = int(
        ((raw < performance_clip_eps) | (raw > 1.0 - performance_clip_eps)).sum()
        if performance_clip_eps > 0 else 0
    )
    thinned = _thin(trace, max_draws)

    loo_summary, loo_observations, weights = loo_diagnostics(thinned, df)
    pit, coverage = calibration(thinned, df, clip_eps=performance_clip_eps)
    return {
        "sampler_summary": sampler_summary,
        "sampler_parameters": sampler_parameters,
        "accuracy": predictive_accuracy(thinned, df, loo_weights=weights),
        "loo_summary": loo_summary,
        "loo_observations": loo_observations,
        "ppc": posterior_predictive_checks(thinned, df, clip_eps=performance_clip_eps),
        "calibration": coverage,
        "pit": pit,
        "qq": qq_data(thinned, df, clip_eps=performance_clip_eps),
    }


def _overall(table: pd.DataFrame) -> pd.Series:
    return table.loc[table["benchmark"] == _OVERALL].iloc[0]


def summarise_evaluation(results: dict) -> str:
    """One screenful of the numbers worth reading first."""
    sampler = results["sampler_summary"].iloc[0]
    accuracy = _overall(results["accuracy"])
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
        f"  RMSE               {accuracy['rmse']:.4f}"
        + (f"   LOO {accuracy['rmse_loo']:.4f}" if "rmse_loo" in accuracy else ""),
        f"  MAE                {accuracy['mae']:.4f}"
        + (f"   LOO {accuracy['mae_loo']:.4f}" if "mae_loo" in accuracy else ""),
        f"  bias               {accuracy['bias']:+.4f}",
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
    for column in [c for c in coverage.index if c.startswith("coverage_")]:
        level = column.split("_")[1]
        lines.append(f"  {level:>18}     {coverage[column]:.3f}")
    lines += [
        f"  mean PIT           {coverage['mean_pit']:.3f} (0.5 when calibrated)",
        f"  KS vs uniform      {coverage['ks_statistic']:.3f} (p = {coverage['ks_pvalue']:.3g})",
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
        "Per-benchmark spread checks (what the per-benchmark kappa is for)",
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


def plot_evaluation(results: dict, path: "str | Path") -> Path:
    """Write the diagnostic plots as a multi-page PDF (needs matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(path)
    accuracy = results["accuracy"]
    per_benchmark = accuracy[accuracy["benchmark"] != _OVERALL]
    qq = results["qq"]

    with PdfPages(path) as pdf:
        # Page 1: did the sampler work?
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        parameters = results["sampler_parameters"]
        axes[0].hist(parameters["r_hat"], bins=40, color="#4477aa")
        axes[0].axvline(1.01, color="#cc3311", ls="--", label="1.01")
        axes[0].set(xlabel="r_hat", ylabel="parameters", title="Convergence")
        axes[0].legend()
        axes[1].hist(parameters["ess_bulk"], bins=40, color="#4477aa")
        axes[1].set(xlabel="ESS (bulk)", ylabel="parameters", title="Effective sample size")
        axes[2].scatter(parameters["ess_bulk"], parameters["r_hat"], s=6, alpha=0.5)
        axes[2].axhline(1.01, color="#cc3311", ls="--")
        axes[2].set(xlabel="ESS (bulk)", ylabel="r_hat", title="ESS vs r_hat")
        fig.suptitle("Sampler diagnostics")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: how well does it predict?
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].hist(per_benchmark["rmse"], bins=30, color="#4477aa")
        axes[0].axvline(_overall(accuracy)["rmse"], color="#cc3311", ls="--", label="overall")
        axes[0].set(xlabel="RMSE", ylabel="benchmarks", title="RMSE by benchmark")
        axes[0].legend()
        axes[1].scatter(per_benchmark["n_obs"], per_benchmark["rmse"], s=14, alpha=0.7)
        axes[1].set(xlabel="observations", ylabel="RMSE", title="RMSE vs benchmark size")
        axes[2].scatter(per_benchmark["observed_mean"], per_benchmark["predicted_mean"], s=14, alpha=0.7)
        lims = [0, 1]
        axes[2].plot(lims, lims, color="#cc3311", ls="--")
        axes[2].set(xlabel="observed mean score", ylabel="predicted mean score",
                    title="Benchmark means", xlim=lims, ylim=lims)
        fig.suptitle("Predictive accuracy")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: calibration, as a QQ plot and as coverage
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(qq["pit_uniform"], qq["pit_sorted"], lw=1.5)
        axes[0].plot([0, 1], [0, 1], color="#cc3311", ls="--")
        axes[0].set(xlabel="uniform quantile", ylabel="PIT quantile",
                    title="PIT QQ plot (calibration)")
        axes[1].hist(results["pit"]["pit"], bins=40, density=True, color="#4477aa")
        axes[1].axhline(1.0, color="#cc3311", ls="--")
        axes[1].set(xlabel="PIT", ylabel="density", title="PIT histogram")
        coverage = _overall(results["calibration"])
        levels = [c for c in coverage.index if c.startswith("coverage_")]
        nominal = [float(c.split("_")[1]) for c in levels]
        axes[2].plot(nominal, [coverage[c] for c in levels], "o-")
        axes[2].plot([0, 1], [0, 1], color="#cc3311", ls="--")
        axes[2].set(xlabel="nominal coverage", ylabel="observed coverage",
                    title="Interval coverage", xlim=(0, 1), ylim=(0, 1))
        fig.suptitle("Calibration")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: posterior predictive and LOO
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(qq["replicated_median"], qq["observed_quantiles"], lw=1.5)
        axes[0].fill_betweenx(qq["observed_quantiles"], qq["replicated_ci_low"],
                              qq["replicated_ci_high"], alpha=0.25, label="90% band")
        axes[0].plot([0, 1], [0, 1], color="#cc3311", ls="--")
        axes[0].set(xlabel="predicted quantile", ylabel="observed quantile",
                    title="Score QQ plot")
        axes[0].legend()
        ppc = results["ppc"]
        for statistic, colour in (("variance", "#4477aa"), ("iqr", "#ee7733")):
            values = ppc[(ppc["statistic"] == statistic) & (ppc["benchmark"] != _OVERALL)]["ppp"]
            axes[1].hist(values, bins=20, range=(0, 1), alpha=0.6, label=statistic, color=colour)
        axes[1].set(xlabel="posterior predictive p-value", ylabel="benchmarks",
                    title="Spread checks by benchmark")
        axes[1].legend()
        loo_observations = results["loo_observations"]
        axes[2].scatter(range(len(loo_observations)), loo_observations["pareto_k"].to_numpy(),
                        s=5, alpha=0.5)
        for threshold, label in PARETO_K_BANDS:
            axes[2].axhline(threshold, color="#cc3311", ls="--", lw=0.8)
        axes[2].set(xlabel="observation (ranked)", ylabel="Pareto k", title="PSIS-LOO reliability")
        fig.suptitle("Posterior predictive and leave-one-out")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return path


def save_evaluation(results: dict, output_dir: "str | Path", plots: bool = True) -> list[Path]:
    """
    Write every table as a CSV, the headline numbers as a text summary, and
    (unless ``plots=False``) the plots as ``evaluation_report.pdf``.

    Returns the paths written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name, table in results.items():
        if not isinstance(table, pd.DataFrame):
            continue  # qq holds plotting arrays, not a table
        destination = output_dir / f"{name}.csv"
        table.to_csv(destination, index=False)
        written.append(destination)

    summary_path = output_dir / "summary.txt"
    summary_path.write_text(summarise_evaluation(results) + "\n")
    written.append(summary_path)

    if plots:
        written.append(plot_evaluation(results, output_dir / "evaluation_report.pdf"))
    return written
