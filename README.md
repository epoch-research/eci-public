# ECI: Epoch Capability Index

This package fits an Item Response Theory (IRT) model to LLM benchmark data to compute:

- **ECI (Epoch Capability Index)**: A unified score measuring model capabilities across benchmarks
- **EDI (Epoch Difficulty Index)**: A score measuring benchmark difficulty

## How It Works

The ECI model assumes that benchmark performance follows a logistic function:

```
P(correct) = sigmoid(discriminability × (capability - difficulty))
```

Where:
- **Capability** measures how capable a model is (higher = more capable)
- **Difficulty** measures how hard a benchmark is (higher = harder)
- **Discriminability** measures how sharply performance transitions around the difficulty threshold

The model jointly estimates all these parameters from observed benchmark scores using least squares optimization.

### Scale Anchoring

The raw fitted parameters are anchored to create the ECI/EDI scale:
- **Winogrande** benchmark is anchored at difficulty 0 with discriminability 1 (for model identification)
- **Claude 3.5 Sonnet** is anchored at ECI 130
- **GPT-5** is anchored at ECI 150

All other scores are determined relative to these anchor points.

## Installation

```bash
pip install eci
```

Or install from source:

```bash
git clone https://github.com/epoch-research/eci-public.git
cd eci-public
pip install -e .
```

## Quick Start

```python
from eci import load_benchmark_data, fit_eci_model, compute_eci_scores

# Load benchmark performance data
df = load_benchmark_data("https://epoch.ai/data/eci_benchmarks.csv")

# Fit the IRT model
model_df, bench_df = fit_eci_model(df, bootstrap_samples=100)

# Convert to ECI/EDI scale
eci_df, edi_df = compute_eci_scores(model_df, bench_df)

# View top models by ECI
print(eci_df[["Model", "eci", "eci_ci_low", "eci_ci_high"]].head(10))

# View benchmark difficulties
print(edi_df[["benchmark", "edi"]].head(10))
```

## API Reference

### `load_benchmark_data(url)`

Load benchmark performance data from a CSV file or URL.

**Arguments:**
- `url`: Path or URL to CSV with columns: `model_id`, `benchmark_id`, `performance`, `benchmark`, `Model`

**Returns:** pandas DataFrame

### `fit_eci_model(df, **kwargs)`

Fit the IRT model to estimate capabilities and difficulties.

**Arguments:**
- `df`: DataFrame with benchmark performance data
- `anchor_benchmark`: Benchmark to anchor (default: "Winogrande")
- `anchor_difficulty`: Difficulty value for anchor (default: 0.0)
- `anchor_discriminability`: Discriminability for anchor (default: 1.0)
- `regularization_strength`: L2 regularization strength (default: 0.1)
- `bootstrap_samples`: Number of bootstrap resamples for CIs (default: 100)
- `bootstrap_seed`: Random seed for reproducibility (default: 12345)
- `ci_level`: Confidence interval level (default: 0.90)

**Returns:** Tuple of (model_capabilities_df, benchmark_params_df)

### `compute_eci_scores(model_df, bench_df, **kwargs)`

Convert raw capabilities to ECI/EDI scale using anchor models.

**Arguments:**
- `model_df`: Model capabilities from `fit_eci_model`
- `bench_df`: Benchmark parameters from `fit_eci_model`
- `anchor_model_low`: Model for lower anchor (default: "Claude 3.5 Sonnet")
- `anchor_eci_low`: ECI value for lower anchor (default: 130)
- `anchor_model_high`: Model for upper anchor (default: "GPT-5")
- `anchor_eci_high`: ECI value for upper anchor (default: 150)

**Returns:** Tuple of (eci_df, edi_df) with scaled scores

## Data Sources

- **Input data**: https://epoch.ai/data/eci_benchmarks.csv
- **Expected ECI scores**: https://epoch.ai/data/eci_scores.csv
- **Expected EDI scores**: https://epoch.ai/data/edi_scores.csv

## Running Tests

```bash
pip install -e ".[dev]"
pytest -v
```

Tests verify that the fitted scores match the scores published on epoch.ai.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{epoch_eci,
  author = {Epoch AI},
  title = {ECI: Epoch Capability Index},
  url = {https://github.com/epoch-research/eci-public},
  year = {2025}
}
```

## Links

- [Epoch AI](https://epoch.ai)
- [ECI Methodology](https://epoch.ai/blog/eci)
