# ECI: Epoch Capabilities Index

This package fits the ECI model, whose core is the IRT curve

    performance = sigmoid(discriminability * (capability - difficulty))

to a table of benchmark scores, estimating:

- **ECI scores**: one capability score per model
- **EDI scores**: one difficulty score per benchmark
- **Discriminabilities**: one slope per benchmark, governing how sharply
  scores rise as capability increases

Two fits are available: a Bayesian one (the default) that treats the curve as
the *mean* of a per-observation Beta likelihood and infers a full posterior
over its parameters by MCMC, with intervals read straight off that posterior;
and a least-squares one, now deprecated/historical, that fits the curve
directly by minimizing its distance to the observed scores, with bootstrap
confidence intervals. See [Bayesian fit](#bayesian-fit-the-default) and
[Least-squares fit (deprecated)](#least-squares-fit-deprecated).

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
and install with the package. Importing them costs a few seconds, so `eci`
defers that until you call the Bayesian fit: `import eci` and the least-squares
fit stay fast.

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

# Fit by least squares instead
python scripts/fit_eci.py --method least-squares

# With fewer bootstrap samples for a quicker run
python scripts/fit_eci.py --method least-squares --bootstrap-samples 100

# Use numerical Jacobian (slower)
python scripts/fit_eci.py --method least-squares --numeric-jacobian
```

`--evaluate` scores a posterior, so it applies to the Bayesian fit only;
passing it with `--method least-squares` is an error rather than a silent
no-op.

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
takes the 90% highest-density interval of the posterior. Either way each draw
is re-anchored with its own scale transform, so the anchor models sit at
exactly 130/150 in every draw; the anchors themselves get NaN intervals, being
pinned by definition.

### Bayesian fit (the default)

`fit_bayesian_eci` infers the curve's parameters by MCMC rather than fitting
the curve directly: the curve is the mean of a per-observation Beta
likelihood (below), and what's sampled is the posterior over that likelihood's
parameters. Every quantity comes out as a posterior distribution and the
intervals are read straight off it:

```python
from eci import prepare_benchmark_data, fit_bayesian_eci

df = prepare_benchmark_data()
eci_df, edi_df, draws = fit_bayesian_eci(df)

print(eci_df[["Model", "eci", "eci_ci_low", "eci_ci_high"]].head(10))
print(edi_df[["benchmark", "edi", "edi_ci_low", "edi_ci_high"]].head(10))
print(draws["trace"])  # the full arviz.InferenceData
```

`eci` and `edi` are posterior means; the `_ci_low` / `_ci_high` columns are
90% **highest-density intervals** — the narrowest window holding 90% of the
posterior. The same applies to `discriminability_scaled` and `kappa`, whose
posteriors are skewed and where an HDI is meaningfully tighter than equal
tails. Pass `interval="eti"` for equal-tailed intervals (5th/95th percentiles)
instead, or `ci_level=` for a different mass.

Scores live in [0, 1], so the likelihood is a Beta with a per-benchmark
concentration `kappa_b`:

    mu[m, b]  = sigmoid(slope[b] * (capability[m] - difficulty[b]))
    y[m, b]  ~ Beta(mu * kappa[b], (1 - mu) * kappa[b])

whose variance is `mu * (1 - mu) / (kappa_b + 1)`. A low `kappa_b` means the
benchmark's scores scatter widely around the curve — something other than
capability and difficulty is driving them; a high `kappa_b` means the curve
explains them tightly. One `kappa` per benchmark lets the fit lean on the
benchmarks that are actually informative about capability, and `edi_df`
reports it alongside difficulty and slope.

The priors are weakly informative:

| Parameter | Prior | Notes |
| --- | --- | --- |
| `C_m` (capability) | `Normal(0, tau_cd)` | |
| `D_b` (difficulty) | `ZeroSumNormal(tau_cd)` | `sum_b D_b = 0` fixes the location |
| `alpha_b` (slope) | `exp(ZeroSumNormal(tau_alpha))` | `prod_b alpha_b = 1` fixes the scale |
| `tau_cd` | `LogNormal(ln 3, 0.5)` | shared scale of capability and difficulty |
| `tau_alpha` | `LogNormal(ln 0.5, 0.5)` | |
| `kappa_b` | `LogNormal(ln 20, 1)` | per-benchmark Beta concentration |

Those two sum-to-zero constraints are what identify the fit, so it uses no
anchor benchmark and no L2 penalty. The three hyperpriors are customisable,
each given as the `(mu, sigma)` of the underlying normal:

```python
from eci import BayesianPriors, fit_bayesian_eci

priors = BayesianPriors(kappa=(np.log(50.0), 1.0))
eci_df, edi_df, draws = fit_bayesian_eci(df, priors=priors)
```

The sampler runs 4 chains of 2000 tuning steps and 4000 draws at
`target_accept=0.95`, and the item parameters are **centred** — drawn straight
from their priors rather than as unit-scale offsets rescaled by their
hyperparameter. `centred=False` (or `--non-centred`) switches to the other
form. The two define the *same* posterior and differ only in the coordinates
NUTS explores, so this is a sampling choice, not a modelling one.

Which is better depends on the data, and the two settings interact: the
centred geometry needs a smaller step size to get through the narrow part of
the funnel, so it needs the higher `target_accept` to work at all. Measured on
the published data, centred at 0.95 beats non-centred at 0.9 on every
diagnostic — including `tau_cd`, the global scale, which is the slowest mixer
and the first parameter to trip the convergence warning. **Retune
`target_accept` within a parameterisation before comparing them**; holding it
fixed makes centred look far worse than it is.

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
pages: the metrics table; the two QQ plots, which are there because they show
something a table cannot, namely *where* on the score scale the fit departs
from the data; and a forest plot of every benchmark's `kappa` and
`difficulty` (posterior mean and 90% HDI), in place of a CSV. The metrics
table carries a `guide` column saying which direction is good and what value
would mean the model is right.

The suite answers three separate questions, and `draws["evaluation"]` (what
`save_evaluation` is given) keeps every table behind those headline numbers as
a dataframe, for drilling into one that the PDF flags — none of these are
written to disk:

| Table | What it tells you |
| --- | --- |
| **`metrics`** | **Every headline number, flat — the PDF's page 1** |
| `sampler_summary`, `sampler_parameters` | Divergences, r-hat, ESS (bulk and tail), MCSE, BFMI |
| `accuracy` | Per benchmark: bias, the predictive SD the model claims, and `rmse_ratio` |
| `loo_summary`, `loo_observations` | `elpd_loo`, `p_loo`, and the Pareto k of every observation |
| `ppc` | Posterior predictive p-values (mid-p) for mean, sd, variance, IQR, min, max, skew and tail shares |
| `calibration`, `pit` | PIT per observation, coverage against nominal, KS test of uniformity |
| `forest` | Per benchmark: posterior mean and 90% HDI of `kappa` and `difficulty` — the PDF's page 3 |
| `qq` | Arrays for the PIT-uniform and score QQ plots |

Five things to know before reading the numbers:

- **Per-benchmark RMSE is not reported, and should not be.** It is a
  restatement of `kappa`, which the fit already gives you: on the published
  data a benchmark's RMSE regresses on the predictive SD the model itself
  claims with R² = 0.93, and ranks against `kappa` at spearman −0.92. A
  high-RMSE benchmark is not one the fit handles badly, it is one the fit
  already knows is noisy. `rmse_ratio` — observed RMSE over the SD the model
  claims — asks the question that isn't already answered: near 1 the
  benchmark scatters as much as the model says, below 1 the model reserves
  more spread than it needs. It ranks benchmarks differently; only 2 of the
  top 5 coincide with raw RMSE.

- **`performance_clip_eps` is a modelling choice, not a tolerance.** The Beta
  has no density at exactly 0 or 1, so boundary scores are clipped to
  `[eps, 1-eps]` — default `1e-4`. Most boundary rows fit a shape parameter
  `mu*kappa < 1`, where the density *diverges* as the value goes to 0, so a
  smaller eps hands the likelihood an unbounded reward for pushing those rows
  down; the model pays by lowering the capability of the models involved and
  the `kappa` of the benchmarks carrying the zeros. Going to `1e-6` buys 0.03
  of squared error at the 82 exact zeros and adds 0.24 across the 2457-row
  bulk. ELPD is not comparable across different eps.
- **Accuracy is scored against the raw scores, the distributional checks
  against the clipped ones.** The Beta likelihood is never shown a score of
  exactly 0 or 1, so it cannot be asked to have predicted one; scoring the PIT
  against raw scores makes clipping look like miscalibration. On a correctly
  specified synthetic fit, 90% coverage reads 0.78 against raw scores and 0.96
  against clipped ones.
- **The p-values are mid-p**, `P(T_rep > T_obs) + 0.5 * P(T_rep = T_obs)`.
  The uncorrected `P(T_rep >= T_obs)` counts a tie as a disagreement, pinning
  a statistic the model reproduces exactly at 1.0 instead of 0.5 — on the
  published data that alone accounted for 78 of 115 apparently failing
  per-benchmark checks, since a benchmark with no scores above 0.95 has
  `T_obs = 0` and `P(T_rep >= 0) = 1` however the model behaves.
- **Read the spread checks (sd, variance, IQR) per benchmark, not pooled.**
  Pooled across benchmarks they measure the spread of the fitted curve, which
  partial pooling deliberately shrinks, so the pooled p-value runs low even
  for a model that is right. On that same synthetic fit the ten per-benchmark
  variance p-values ranged 0.11–0.65 while the pooled one was 0.0005.

`evaluate=True` keeps the pointwise log likelihood and draws a replicated
dataset per posterior draw, both of which are (draws × observations) arrays,
so the fit uses more memory and time. The per-observation metrics thin to
`evaluation_max_draws` (2000 by default); the sampler diagnostics always use
every draw.

`build_bayesian_model(df)` returns the `pm.Model` itself if you want to sample
it differently or inspect it. Divergences and poor mixing are reported as
warnings, not errors, so the draws are still returned for inspection.

On the published data the defaults take about two minutes and mix cleanly: 0
divergences in 16,000 draws, 0 of 444 parameters above r-hat 1.01, minimum ESS
586 bulk / 924 tail, minimum BFMI 0.85.

### Least-squares fit (deprecated)

Deprecated/historical: this predates the Bayesian fit above, which is now
the default and the one to use for new work. It's kept for reference and
for reproducing old results.

`fit_eci_model` fits the same curve by least squares. It is identified
differently: the anchor benchmark's discriminability is pinned to fix the
multiplicative scale and an L2 penalty resolves the remaining translation
freedom, where the Bayesian fit uses its two sum-to-zero constraints. Its
intervals come from a bootstrap rather than a posterior. It is the faster of the two and needs no MCMC diagnostics, but it
reports no `kappa`, has no per-observation noise model, and there is no
diagnostic suite for it — `--evaluate` is Bayesian-only. `edi_df` carries an
`is_anchor` column marking the pinned benchmark.

Bootstrap resampling holds the set of models fixed and resamples each
model's benchmark results with replacement. Any fit — central or bootstrap —
that fails to converge, or whose anchor capabilities coincide or invert
(defining no scale), raises.

The anchor model names are matched against the `Model` column of the input
data; renaming those models upstream requires passing the new names
explicitly (the fit fails loudly if an anchor is missing).

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
