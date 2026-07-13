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

# Use a different bootstrap scheme for confidence intervals
python scripts/fit_eci.py --bootstrap-method observation
```

Available bootstrap methods (`--bootstrap-method` / `bootstrap_method=`):

- `hierarchical` (default): hold the set of models fixed and resample each model's benchmark results with replacement, so no model ever loses all its observations.
- `observation`: resample all (model, benchmark) observations with replacement from the pooled data.

### Python API

```python
from eci import load_benchmark_data, fit_eci_model, compute_eci_scores

df = load_benchmark_data("https://epoch.ai/data/eci_benchmarks.csv")

# Raw-scale IRT fit: point estimates + bootstrap draws (no CIs here)
model_df, bench_df, bootstrap_data = fit_eci_model(df, bootstrap_samples=100)

# ECI/EDI scale conversion. Confidence intervals are constructed here, by
# re-anchoring every bootstrap draw with its own scale transform so the
# anchor models (Claude 3.5 Sonnet = 130, GPT-5 = 150) are fixed in every
# draw. The anchors themselves get NaN CIs: they are pinned by definition.
results = compute_eci_scores(model_df, bench_df, bootstrap_data)

print(results.eci_df[["Model", "eci", "eci_ci_low", "eci_ci_high"]].head(10))
```

`compute_eci_scores` returns an `EciResults` with:

- `eci_df` / `edi_df`: central estimates and CIs on the ECI scale
- `draws`: the bootstrap draws on the ECI scale (models, benchmark
  difficulties, and slopes), plus the per-draw scale transforms
- `scaling`: the central affine map and anchor definitions

A draw whose anchor capabilities coincide or invert cannot define a scale;
`compute_eci_scores` raises a `ValueError` naming the offending draws.

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
