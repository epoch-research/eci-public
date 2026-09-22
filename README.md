# ECI: Epoch Capabilities Index

This package fits the ECI model,

    performance = sigmoid(discriminability * (capability - difficulty))

to a table of benchmark scores, estimating:

- **ECI scores**: one capability score per model
- **EDI scores**: one difficulty score per benchmark
- **Discriminabilities**: one slope per benchmark, governing how sharply
  scores rise as capability increases

Two fits of that curve are available: a least-squares one with bootstrap
confidence intervals (the default), and a Bayesian one fitted by MCMC, whose
intervals come straight from the posterior. See [Bayesian fit](#bayesian-fit).

For details on the methodology, see:
- **Paper**: [A Rosetta Stone for AI Benchmarks](https://arxiv.org/abs/2512.00193)
- **Website**: [Epoch Capabilities Index](https://epoch.ai/benchmarks/eci#overview)

## Installation

```bash
git clone https://github.com/epoch-research/eci-public.git
cd eci-public
pip install -e .

# For the Bayesian fit (installs PyMC and ArviZ)
pip install -e ".[bayesian]"
```

## Usage

The fit runs from Epoch's raw benchmark data,
[benchmark_data.zip](https://epoch.ai/data/benchmark_data.zip), which the
loader downloads and processes into one row per model x benchmark (see
`src/eci/dataloader.py`).

### Command Line

```bash
# Download benchmark_data.zip, fit model and save results to outputs/
python scripts/fit_eci.py

# Fit from a local copy of the zip
python scripts/fit_eci.py --input path/to/benchmark_data.zip

# With fewer bootstrap samples for a quicker run
python scripts/fit_eci.py --bootstrap-samples 100

# Use numerical Jacobian (slower)
python scripts/fit_eci.py --numeric-jacobian

# Fit the Bayesian model instead (about a minute on the published data)
python scripts/fit_eci.py --method bayesian
python scripts/fit_eci.py --method bayesian --draws 8000 --chains 4
```

### Python API

```python
from eci import prepare_benchmark_data, fit_eci_model

df = prepare_benchmark_data()  # downloads benchmark_data.zip
eci_df, edi_df, draws = fit_eci_model(df, bootstrap_samples=100)

print(eci_df[["Model", "eci", "eci_ci_low", "eci_ci_high"]].head(10))
```

Everything is returned on the ECI scale, defined by two anchor models
(Claude 3.5 Sonnet = 130, GPT-5 = 150):

- `eci_df`: each model's `eci`, with confidence intervals.
- `edi_df`: each benchmark's difficulty (`edi`) and its slope in ECI units
  (`discriminability_scaled`), both with confidence intervals. Together
  they trace a benchmark's fitted curve: a model's predicted score is
  `sigmoid(discriminability_scaled * (eci - edi))`.
- `draws`: the scaled bootstrap draws (model ECIs, benchmark difficulties
  and slopes) plus the per-draw scale transforms.

Both fits carry the same interval columns, cut from their own draws: the
least-squares fit takes quantiles of the bootstrap draws, the Bayesian fit
takes the 90% highest-density interval of the posterior. Either way each draw
is re-anchored with its own scale transform, so the anchor models sit at
exactly 130/150 in every draw; the anchors themselves get NaN intervals, being
pinned by definition.

### Bayesian fit

`fit_bayesian_eci` fits the same curve with MCMC, returning the same three
objects on the same ECI scale:

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
eci_df, edi_df, draws = fit_bayesian_eci(df, priors=priors, target_accept=0.95)
```

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
python scripts/fit_eci.py --method bayesian --evaluate
```

It answers three separate questions, and the tables are written one CSV each
alongside a `summary.txt` and a four-page `evaluation_report.pdf`:

| Table | What it tells you |
| --- | --- |
| **`metrics`** | **Every headline number, flat — read this one first** |
| `sampler_summary`, `sampler_parameters` | Did NUTS work? Divergences, r-hat, ESS (bulk and tail), MCSE, BFMI |
| `accuracy` | Per benchmark: bias, the predictive SD the model claims, and `rmse_ratio` |
| `loo_summary`, `loo_observations` | `elpd_loo`, `p_loo`, and the Pareto k of every observation |
| `ppc` | Posterior predictive p-values for mean, sd, variance, IQR, min, max, skew and tail shares |
| `calibration`, `pit` | PIT per observation, coverage against nominal, KS test of uniformity |
| `qq` | Arrays for the PIT-uniform and score QQ plots |

The PDF is only those two QQ plots — they show something a table cannot,
namely *where* on the score scale the fit departs from the data. Everything
else is a number and lives in `metrics.csv`.

Four things to know before reading the numbers:

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
warnings, not errors, so the draws are still returned for inspection. On the
published data the default 4 chains x (2000 tune + 4000 draws) take about 90
seconds and mix cleanly, except for the global scale `tau_cd`, which lands at
r-hat 1.01-1.02 however long it tunes and so trips that warning now and then;
it is a nuisance parameter, and every reported quantity sits at or below 1.01.

### Bootstrap

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
