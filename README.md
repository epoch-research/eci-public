# ECI: Epoch Capabilities Index

This package fits the ECI model, whose core is the IRT curve

    performance = sigmoid(discriminability * (capability - difficulty))

to a table of benchmark scores, estimating:

- **ECI scores**: one capability score per model
- **EDI scores**: one difficulty score per benchmark
- **Discriminabilities**: one slope per benchmark, governing how sharply
  scores rise as capability increases

Two fits are available: 
1. A Bayesian one (the default) that treats the IRT curve as the *mean* 
of a per-observation Beta likelihood and infers a full posterior over its 
parameters by MCMC, with intervals read straight off that posterior.
2. A least-squares frequentist approach which is deprecated but available for 
historical reproduction of results. This fits the curve directly, by minimizing its 
distance to the observed scores, with bootstrap confidence intervals.


See [Bayesian fit](#bayesian-fit-the-default) and
[Least-squares fit](#least-squares-fit).

For details on the methodology, see:
- **Paper**: [A Rosetta Stone for AI Benchmarks](https://arxiv.org/abs/2512.00193)
- **Website**: [Epoch Capabilities Index](https://epoch.ai/benchmarks/eci#overview)

## Installation

```bash
git clone https://github.com/epoch-research/eci-public.git
cd eci-public
pip install -e .
```

The Bayesian fit is the default, so PyMC and ArviZ are ordinary dependencies
and install with the package and may take some seconds to import. 

## Usage

The fit runs from Epoch's raw benchmark data,
[benchmark_data.zip](https://epoch.ai/data/benchmark_data.zip), which the
loader downloads and processes into one row per model x benchmark (see
`src/eci/dataloader.py`).

### Command Line

```bash
# Download benchmark_data.zip, fit the Bayesian model and save results to
# outputs/ (about two minutes on the published data)
python scripts/fit_eci.py

# Fit from a local copy of the zip
python scripts/fit_eci.py --input path/to/benchmark_data.zip

# Longer chains
python scripts/fit_eci.py --draws 8000 --chains 4

# Also write the diagnostic suite to outputs/evaluation/
python scripts/fit_eci.py --evaluate

# Fit by frequentist approach (least squares error) instead
python scripts/fit_eci.py --method least-squares

# With fewer bootstrap samples for a quicker run
python scripts/fit_eci.py --method least-squares --bootstrap-samples 100

# Use numerical Jacobian (slower)
python scripts/fit_eci.py --method least-squares --numeric-jacobian
```

`--evaluate` scores a posterior, so it applies to the Bayesian fit only;
passing it with `--method least-squares` raises an error.

### Python API

```python
from eci import prepare_benchmark_data, fit_bayesian_eci

df = prepare_benchmark_data()  # downloads benchmark_data.zip
eci_df, edi_df, draws = fit_bayesian_eci(df)

print(eci_df[["Model", "eci", "eci_ci_low", "eci_ci_high"]].head(10))
```

`fit_eci_model` is the least-squares fit and takes the same first argument,
returning the same three objects:

```python
from eci import fit_eci_model

eci_df, edi_df, draws = fit_eci_model(df, bootstrap_samples=100)
```

Everything is returned on the ECI scale, defined by two anchor models
(Claude 3.5 Sonnet = 130, GPT-5 = 150):

- `eci_df`: each model's `eci`, with confidence intervals.
- `edi_df`: each benchmark's difficulty (`edi`) and its slope in ECI units
  (`discriminability_scaled`), both with confidence intervals. Together
  they trace a benchmark's fitted curve: a model's predicted score is
  `sigmoid(discriminability_scaled * (eci - edi))`.
- `draws`: the scaled draws (model ECIs, benchmark difficulties and slopes)
  plus the per-draw scale transforms — posterior draws from the Bayesian fit,
  bootstrap draws from the least-squares one.

Both fits carry the same interval columns, cut from their own draws: the
least-squares fit takes quantiles of the bootstrap draws, the Bayesian fit
quantiles of the posterior draws. Either way each draw
is re-anchored with its own scale transform, so the anchor models sit at
exactly 130/150 in every draw; the anchors themselves get NaN intervals, being
pinned by definition.

### Bayesian fit (the default)

`fit_bayesian_eci` infers the curve's parameters by MCMC: the IRT curve is the 
mean of a per-observation Beta likelihood (below), and what's sampled is 
the posterior over that likelihood's parameters. Every quantity comes out 
as a posterior distribution and the intervals are read straight off it:

```python
from eci import prepare_benchmark_data, fit_bayesian_eci

df = prepare_benchmark_data()
eci_df, edi_df, draws = fit_bayesian_eci(df)

print(eci_df[["Model", "eci", "eci_ci_low", "eci_ci_high"]].head(10))
print(edi_df[["benchmark", "edi", "edi_ci_low", "edi_ci_high"]].head(10))
print(draws["trace"])  # the full arviz.InferenceData
```

`eci` and `edi` are posterior means; the `_ci_low` / `_ci_high` columns are
90% **equal-tailed intervals** — the 5th and 95th percentiles of the
posterior. The same applies to `discriminability_scaled` and `sigma`. Pass
`ci_level=` for a different mass, e.g. `ci_level=0.95`.

Scores live in [0, 1], consistent with a Beta likelihood, with a per-benchmark
noise SD `sigma_b`:

    mu[m, b]  = sigmoid(slope[b] * (capability[m] - difficulty[b]))
    kappa[b]  = 1 / (4 * sigma[b]^2) - 1
    y[m, b]  ~ Beta(mu * kappa[b], (1 - mu) * kappa[b])

The Beta's variance is `mu * (1 - mu) / (kappa_b + 1)`, so its SD is
`2 * sqrt(mu * (1 - mu)) * sigma_b`: `sigma_b` is the SD of the scores where the
curve passes 50%, and the largest it gets anywhere on that benchmark, shrinking
towards 0 and 1. It has to stay below 0.5, where `kappa_b` reaches 0. A high
`sigma_b` means the benchmark's scores scatter widely around the curve —
something other than capability and difficulty is driving them; a low
`sigma_b` means the curve explains them tightly. One `sigma` per benchmark lets
the fit learn from the benchmarks that are most informative about capability.
`edi_df` reports `sigma` posteriors alongside difficulty and slope.

The priors are weakly informative:

| Parameter | Prior | Notes |
| --- | --- | --- |
| `C_m` (capability) | `Normal(0, tau_cd)` | |
| `D_b` (difficulty) | `ZeroSumNormal(tau_cd)` | `sum_b D_b = 0` fixes the location |
| `alpha_b` (slope) | `exp(ZeroSumNormal(tau_alpha))` | `prod_b alpha_b = 1` fixes the scale |
| `tau_cd` | `LogNormal(ln 3, 0.5)` | shared scale of capability and difficulty |
| `tau_alpha` | `LogNormal(ln 0.5, 0.5)` | |
| `sigma_b` | `LogNormal(ln sigma_median, tau_sigma)` truncated to `(0, 0.5)` | per-benchmark noise SD at `mu = 0.5` |
| `sigma_median` | `LogNormal(ln 0.1, 0.5)` | the typical benchmark's noise SD |
| `tau_sigma` | `LogNormal(ln 0.5, 0.5)` | spread of `ln sigma_b` across benchmarks |

Those two sum-to-zero constraints are what identify the fit. The noise prior
is hierarchical: the benchmarks share a typical noise level, learned from all
of them, so one with only a handful of scores is pulled towards it rather than
towards a fixed guess.
The four hyperpriors are customisable, each given as the
`(mu, sigma)` of the underlying normal:

```python
from eci import BayesianPriors, fit_bayesian_eci

priors = BayesianPriors(sigma_median=(np.log(0.07), 0.5))
eci_df, edi_df, draws = fit_bayesian_eci(df, priors=priors)
```

The sampler runs a default 4 chains of 2000 tuning steps and 4000 draws at
`target_accept=0.95`.


### Evaluating a fit

`evaluate=True` runs a diagnostic suite alongside the fit and returns it under
`draws["evaluation"]`:

```python
from eci import fit_bayesian_eci, save_evaluation, summarise_evaluation

eci_df, edi_df, draws = fit_bayesian_eci(df, evaluate=True)
print(summarise_evaluation(draws["evaluation"]))
save_evaluation(draws["evaluation"], "outputs/evaluation")
```

or from the command line, which writes the same thing to
`<output-dir>/evaluation/`:

```bash
python scripts/fit_eci.py --evaluate
```

`save_evaluation` writes a single file, `evaluation_report.pdf`, with three
pages: the metrics table; the two QQ plots, which show *where* on the score 
scale the fit departs from the data; and a forest plot of every benchmark's 
`sigma` and `difficulty` (posterior mean and 90% interval). 

The suite looks at sampling, accuracy and calibration metrics. 
`draws["evaluation"]` (passed to `save_evaluation`) stores every data table 
used for the summary statistics as a dataframe, but is not by default saved
to disk:

| Table | Metrics |
| --- | --- |
| **`metrics`** | **Every headline number, flat — read this one first** |
| `sampler_summary`, `sampler_parameters` | Divergences, r-hat, ESS (bulk and tail), MCSE, BFMI |
| `accuracy` | Per benchmark: bias, the predictive SD the model claims, and `rmse_ratio` |
| `loo_summary`, `loo_observations` | `elpd_loo`, `p_loo`, and the Pareto k of every observation |
| `ppc` | Posterior predictive p-values (mid-p) for mean, sd, variance, IQR, min, max, skew and tail shares |
| `calibration`, `pit` | PIT per observation, coverage against nominal, KS test of uniformity — each both in-sample and PSIS-LOO (`_loo` suffix) |
| `forest` | Per benchmark: posterior mean and 90% interval of `sigma` and `difficulty` — the PDF's page 3 |
| `qq` | Arrays for the PIT-uniform (in-sample and PSIS-LOO) and score QQ plots |

Notes on metrics:
- **Accuracy is scored against the raw scores, the distributional checks
  against the clipped ones.** The Beta likelihood is never shown a score of
  exactly 0 or 1, so it cannot be asked to have predicted one; scoring the PIT
  against raw scores makes clipping look like miscalibration.
- **Calibration is reported in-sample and leave-one-out.** The in-sample PIT
  scores each observation against a posterior fitted to it, which bunches the
  PIT towards 0.5 and makes the intervals look wider than they are. The
  PSIS-LOO PIT reweights the same draws as if each point had been left out, so
  it costs no refits; it is only as reliable as the point's Pareto k, so read
  it alongside the k counts.
- **The p-values are mid-p**, `P(T_rep > T_obs) + 0.5 * P(T_rep = T_obs)`.
  The uncorrected `P(T_rep >= T_obs)` counts a tie as a disagreement, pinning
  a statistic the model reproduces exactly at 1.0 instead of 0.5: a benchmark
  with no scores above 0.95 has `T_obs = 0` for `frac_above_0.95`, and
  `P(T_rep >= 0) = 1` however the model behaves.
- **Consider per-benchmark spread checks (sd, variance, IQR) alongside pooled.**
  Pooled across benchmarks they measure the spread of the fitted curve, which
  partial pooling deliberately shrinks, so the pooled p-value runs low even
  for a model that is right.

`evaluate=True` keeps the pointwise log likelihood and draws a replicated
dataset per posterior draw, both of which are (draws × observations) arrays,
so the fit uses more memory and time. The per-observation metrics thin to
`evaluation_max_draws` (2000 by default); the sampler diagnostics always use
every draw.

`build_bayesian_model(df)` returns the `pm.Model` itself if you want to sample
it differently or inspect it. Divergences and poor mixing are reported as
warnings, not errors, so the draws are still returned for inspection.


### Least-squares fit (deprecated)

Deprecated/historical: this predates the Bayesian fit above, which is now
the default. It's kept for reference and for reproducing old results.

`fit_eci_model` fits the same IRT curve by least squares. It is identified
differently: the anchor benchmark's discriminability is pinned to fix the
multiplicative scale and an L2 penalty resolves the remaining translation
freedom, where the Bayesian fit uses its two sum-to-zero constraints. Its
intervals come from a bootstrap rather than a posterior. It is the faster 
of the two and needs no MCMC diagnostics, but it reports no `sigma`, has 
no per-observation noise model, and there is no diagnostic suite for it. 
`edi_df` carries an `is_anchor` column marking the pinned benchmark.

Bootstrap resampling holds the set of models fixed and resamples each
model's benchmark results with replacement. 

The anchor model names are matched against the `Model` column of the input
data; renaming those models upstream requires passing the new names
explicitly (the fit fails if an anchor is missing).

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Citation

```bibtex
@article{ho2025rosetta,
  title={A Rosetta Stone for AI Benchmarks},
  author={Ho, Anson and Denain, Jean-Stanislas and Atanasov, David and Albanie, Samuel and Shah, Rohin},
  journal={arXiv preprint arXiv:2512.00193},
  year={2025}
}
```

## License

MIT
