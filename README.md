# ECI: Epoch Capabilities Index

This package fits the ECI model,

    performance = sigmoid(discriminability * (capability - difficulty))

to a table of benchmark scores, estimating:

- **ECI scores**: one capability score per model
- **EDI scores**: one difficulty score per benchmark
- **Discriminabilities**: one slope per benchmark, governing how sharply
  scores rise as capability increases

For details on the methodology, see:
- **Paper**: [A Rosetta Stone for AI Benchmarks](https://arxiv.org/abs/2512.00193)
- **Website**: [Epoch Capabilities Index](https://epoch.ai/benchmarks/eci#overview)

## Installation

```bash
git clone https://github.com/epoch-research/eci-public.git
cd eci-public
pip install -e .
```

## Usage

### Command Line

```bash
# Fit model and save results to outputs/
python scripts/fit_eci.py

# With fewer bootstrap samples for a quicker run
python scripts/fit_eci.py --bootstrap-samples 100

# Use numerical Jacobian (slower)
python scripts/fit_eci.py --numeric-jacobian
```

### Python API

```python
from eci import load_benchmark_data, fit_eci_model

df = load_benchmark_data("https://epoch.ai/data/eci_benchmarks.csv")
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

CIs are quantiles of the bootstrap draws, each draw re-anchored with its
own scale transform so the anchor models sit at exactly 130/150 in every
draw; the anchors themselves get NaN CIs, being pinned by definition.

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
