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

For more detail on the methodology, see the
[Epoch Capabilities Index](https://epoch.ai/benchmarks/eci#overview) page.

The ECI was originally motivated by the paper
[A Rosetta Stone for AI Benchmarks](https://arxiv.org/abs/2512.00193). The
paper describes an earlier version of the methodology.

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

The model is fit on an arbitrary raw scale, then mapped linearly onto the ECI
scale by fixing two anchor models (Claude 3.5 Sonnet = 130, GPT-5 = 150).
Everything is returned on the ECI scale:

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

The Beta's standard deviation (SD) is `2 * sqrt(mu * (1 - mu)) * sigma_b`. Thus at 
`mu=0.5` the SD is maximised and takes value `sigma_b`, and it shrinks as `mu` 
approaches 0 or 1. `sigma_b` is bounded above by 0.5 as the results are scores in 
[0,1]. `sigma_b` varies per benchmark to reflect how noisy each benchmarks' results 
are; a high value could be either high run-to-run variance or a poor fit of the 
ECI modelling assumptions.

The priors are weakly informative:

| Parameter | Prior | Notes |
| --- | --- | --- |
| `C_m` (capability) | `Uniform(-50, 50)` | flat over any plausible range |
| `D_b` (difficulty) | uniform on `[-50, 50]^K` with `sum_b D_b = 0` | the sum fixes the location |
| `alpha_b` (slope) | `exp(ZeroSumNormal(tau_alpha))` | `prod_b alpha_b = 1` fixes the scale |
| `tau_alpha` | `LogNormal(ln 0.5, 0.5)` | |
| `sigma_b` | `LogNormal(ln sigma_median, tau_sigma)` truncated to `(0, 0.5)` | per-benchmark noise SD at `mu = 0.5` |
| `sigma_median` | `LogNormal(ln 0.1, 0.5)` | the typical benchmark's noise SD |
| `tau_sigma` | `LogNormal(ln 0.5, 0.5)` | spread of `ln sigma_b` across benchmarks |

Those two sum-to-zero constraints are what identify the fit on the raw scale.
We apply a `sqrt(K / (K - 1))` correction term to the log slopes' zero-sum prior scale, where
`K` is the number of benchmarks, so that `tau_alpha` is the marginal SD. The uniform difficulty
prior gets no such correction: the sum-to-zero constraint leaves each difficulty's marginal SD
about 1% narrower than a capability's for `K` around 20, which a prior this wide does not notice. The noise prior
is hierarchical: the benchmarks share a typical noise level, learned from all
of them, so one with only a handful of scores is pulled towards it rather than
towards a fixed guess.
The three hyperpriors are customisable, each given as the
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
pages: a metrics table; two QQ plots which show *where* on the score 
scale the fit departs from the data; and a forest plot of every benchmark's 
`sigma` and `difficulty` (posterior mean and 90% interval). 

The suite looks at sampling, accuracy and calibration metrics. 
`draws["evaluation"]` (passed to `save_evaluation`) stores every data table 
used for the summary statistics as a dataframe, but is not by default saved
to disk:

| Table | Metrics |
| --- | --- |
| `sampler_summary`, `sampler_parameters` | Divergences, r-hat, ESS (bulk and tail), MCSE, BFMI |
| `accuracy` | Per benchmark: bias, the predictive SD the model claims, and `rmse_ratio` |
| `loo_summary`, `loo_observations` | `elpd_loo`, `p_loo`, and the Pareto k of every observation |
| `ppc` | Posterior predictive p-values (mid-p) for mean, sd, variance, IQR, min, max, skew and tail shares |
| `calibration`, `pit` | PIT per observation, coverage against nominal, KS test of uniformity — each both in-sample and PSIS-LOO (`_loo` suffix) |
| `forest` | Per benchmark: posterior mean and 90% interval of `sigma` and `difficulty` — the PDF's page 3 |
| `qq` | Arrays for the PIT-uniform (in-sample and PSIS-LOO) and score QQ plots |


`evaluate=True` keeps the pointwise log likelihood and draws a replicated
dataset per posterior draw, both of which are (draws × observations) arrays,
so the fit uses more memory and time. The per-observation metrics thin to
`evaluation_max_draws` (2000 by default); the sampler diagnostics always use
every draw.

`build_bayesian_model(df)` returns the `pm.Model` itself if you want to sample
it differently or inspect it. Divergences and poor mixing are reported as
warnings, not errors, so the draws are still returned for inspection.


### Least-squares fit (deprecated)

Kept for reference and for reproducing old results.

`fit_eci_model` fits the capabilities, difficulties and discriminabilities
that minimise the squared error between the IRT curve and the observed scores, 
plus an L2 penalty on all parameters. Scores are clipped by 
`performance_clip_eps` (default `[0.001,0.999]), and the
penalty's weight is set by `regularization_strength` (default `0.1`). It's 
solved with SciPy's bounded `least_squares`, keeping capabilities and difficulties 
in `[-10, 10]` and discriminabilities in `[0.1, 10]`.

The curve only depends on `discriminability * (capability - difficulty)`, so
the parameters need to be anchored. The discriminability of the `anchor_benchmark`
(Winogrande by default) is fixed at `anchor_discriminability`
(default `1.0`), which sets the scale, and the L2 penalty sets the location.
`edi_df` marks the anchor benchmark in its `is_anchor` column. These choices
set the raw scale the model is fit on; the fitted values are then
mapped linearly onto the ECI scale so that the two anchor models sit at 130
and 150.

Intervals come from a bootstrap of `bootstrap_samples` resamples (default 500, 
seeded by `bootstrap_seed`). The intervals cover the central `ci_level` of 
those refits (default `0.90`, i.e. the 5th and 95th percentiles).


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
