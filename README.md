# ECI: Epoch Capability Index

This package fits the ECI model to compute:
- **ECI scores**: Unified capability scores for LLMs
- **EDI scores**: Difficulty scores for benchmarks

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

- `eci_df` / `edi_df`: central estimates and confidence intervals. CIs are
  quantiles of the bootstrap draws, each draw re-anchored with its own scale
  transform so the anchor models sit at exactly 130/150 in every draw; the
  anchors themselves get NaN CIs, being pinned by definition.
- `draws`: the scaled bootstrap draws (model ECIs, benchmark difficulties
  and slopes) plus the per-draw scale transforms.

Bootstrap resampling holds the set of models fixed and resamples each
model's benchmark results with replacement. A draw whose anchor capabilities
coincide or invert cannot define a scale; the fit raises a `ValueError`
naming the offending draws.

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
@article{epoch2024aci,
  title={Artificial Capable Intelligence},
  author={Epoch AI},
  journal={arXiv preprint arXiv:2512.00193},
  year={2024}
}
```

## License

MIT
