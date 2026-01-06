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

# With more bootstrap samples for more precise confidence intervals
python scripts/fit_eci.py --bootstrap-samples 500

# Use numerical Jacobian (slower, matches website exactly)
python scripts/fit_eci.py --numeric-jacobian
```

### Python API

```python
from eci import load_benchmark_data, fit_eci_model, compute_eci_scores

df = load_benchmark_data("https://epoch.ai/data/eci_benchmarks.csv")
model_df, bench_df = fit_eci_model(df, bootstrap_samples=100)
eci_df, edi_df = compute_eci_scores(model_df, bench_df)

print(eci_df[["Model", "eci"]].head(10))
```

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
