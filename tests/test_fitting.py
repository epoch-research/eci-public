"""
Tests for ECI fitting that verify outputs match the Epoch AI website.

These tests load benchmark data from https://epoch.ai/data/eci_benchmarks.csv
and verify that the fitted scores match:
- https://epoch.ai/data/eci_scores.csv (model capabilities)
- https://epoch.ai/data/edi_scores.csv (benchmark difficulties)
"""

import pandas as pd
import pytest
from scipy.stats import spearmanr

from eci import fit_eci_model, load_benchmark_data, compute_eci_scores


# URLs for test data
BENCHMARKS_URL = "https://epoch.ai/data/eci_benchmarks.csv"
EXPECTED_ECI_URL = "https://epoch.ai/data/eci_scores.csv"
EXPECTED_EDI_URL = "https://epoch.ai/data/edi_scores.csv"

# Tolerance for score comparisons
ECI_TOLERANCE = 0.1  # ECI points
EDI_TOLERANCE = 0.1  # EDI points


@pytest.fixture(scope="module")
def benchmark_data():
    """Load benchmark performance data."""
    return load_benchmark_data(BENCHMARKS_URL)


@pytest.fixture(scope="module")
def fitted_model(benchmark_data):
    """Fit the ECI model on benchmark data (no bootstrap for speed)."""
    model_df, bench_df = fit_eci_model(
        benchmark_data,
        bootstrap_samples=0,
    )
    eci_df, edi_df = compute_eci_scores(model_df, bench_df)
    return eci_df, edi_df


@pytest.fixture(scope="module")
def expected_eci():
    """Load expected ECI scores from website."""
    return pd.read_csv(EXPECTED_ECI_URL)


@pytest.fixture(scope="module")
def expected_edi():
    """Load expected EDI scores from website."""
    return pd.read_csv(EXPECTED_EDI_URL)


class TestDataLoading:
    """Tests for data loading functionality."""

    def test_load_benchmark_data(self, benchmark_data):
        """Test that benchmark data loads correctly."""
        assert len(benchmark_data) > 0
        assert "model_id" in benchmark_data.columns
        assert "benchmark_id" in benchmark_data.columns
        assert "performance" in benchmark_data.columns
        assert "benchmark" in benchmark_data.columns
        assert "Model" in benchmark_data.columns

    def test_performance_range(self, benchmark_data):
        """Test that performance values are in valid range."""
        assert benchmark_data["performance"].min() >= 0
        assert benchmark_data["performance"].max() <= 1

    def test_no_missing_values(self, benchmark_data):
        """Test that required columns have no missing values."""
        assert benchmark_data["model_id"].notna().all()
        assert benchmark_data["benchmark_id"].notna().all()
        assert benchmark_data["performance"].notna().all()


class TestModelFitting:
    """Tests for the model fitting process."""

    def test_model_fit_produces_output(self, fitted_model):
        """Test that fitting produces non-empty results."""
        eci_df, edi_df = fitted_model
        assert len(eci_df) > 0
        assert len(edi_df) > 0

    def test_eci_has_required_columns(self, fitted_model):
        """Test that ECI output has required columns."""
        eci_df, _ = fitted_model
        assert "Model" in eci_df.columns
        assert "eci" in eci_df.columns

    def test_edi_has_required_columns(self, fitted_model):
        """Test that EDI output has required columns."""
        _, edi_df = fitted_model
        assert "benchmark" in edi_df.columns
        assert "edi" in edi_df.columns

    def test_anchor_models_have_correct_eci(self, fitted_model):
        """Test that anchor models have approximately correct ECI values."""
        eci_df, _ = fitted_model

        claude_eci = eci_df.loc[eci_df["Model"] == "Claude 3.5 Sonnet", "eci"]
        gpt5_eci = eci_df.loc[eci_df["Model"] == "GPT-5", "eci"]

        if not claude_eci.empty:
            assert abs(claude_eci.iloc[0] - 130) < 0.01
        if not gpt5_eci.empty:
            assert abs(gpt5_eci.iloc[0] - 150) < 0.01


class TestECIScoreAccuracy:
    """Tests comparing ECI scores to expected website values."""

    def test_eci_scores_match_website(self, fitted_model, expected_eci):
        """Test that computed ECI scores match website within tolerance."""
        eci_df, _ = fitted_model

        comparison = eci_df.merge(
            expected_eci[["Model", "eci"]],
            on="Model",
            how="inner",
            suffixes=("_computed", "_expected")
        )

        # Ensure we matched most models
        n_matched = len(comparison)
        n_expected = min(len(eci_df), len(expected_eci))
        assert n_matched > 0.9 * n_expected, \
            f"Only matched {n_matched} models out of {n_expected}"

        # Compute differences
        comparison["diff"] = comparison["eci_computed"] - comparison["eci_expected"]
        comparison["abs_diff"] = comparison["diff"].abs()

        # Statistics
        mae = comparison["abs_diff"].mean()
        max_diff = comparison["abs_diff"].max()
        worst_model = comparison.loc[comparison["abs_diff"].idxmax()]

        # Print detailed stats
        print(f"\n{'='*60}")
        print("ECI Score Comparison")
        print(f"{'='*60}")
        print(f"Models compared: {n_matched}")
        print(f"Mean absolute error: {mae:.6f}")
        print(f"Max absolute error: {max_diff:.6f}")
        print(f"Largest deviation: {worst_model['Model']}")
        print(f"  Computed: {worst_model['eci_computed']:.4f}")
        print(f"  Expected: {worst_model['eci_expected']:.4f}")
        print(f"  Diff: {worst_model['diff']:+.6f}")

        assert mae < ECI_TOLERANCE, \
            f"Mean absolute ECI error {mae:.6f} exceeds tolerance {ECI_TOLERANCE}"
        assert max_diff < ECI_TOLERANCE * 10, \
            f"Max ECI error {max_diff:.6f} is too large"

    def test_eci_ranking_correlation(self, fitted_model, expected_eci):
        """Test that ECI rankings are highly correlated with expected."""
        eci_df, _ = fitted_model

        comparison = eci_df.merge(
            expected_eci[["Model", "eci"]],
            on="Model",
            how="inner",
            suffixes=("_computed", "_expected")
        )

        if len(comparison) < 10:
            pytest.skip("Not enough models to compute ranking correlation")

        corr, p_value = spearmanr(
            comparison["eci_computed"].rank(),
            comparison["eci_expected"].rank()
        )

        print(f"\nECI Spearman correlation: {corr:.6f} (p={p_value:.2e})")

        assert corr > 0.99, f"Ranking correlation {corr:.6f} is too low"


class TestEDIScoreAccuracy:
    """Tests comparing EDI scores to expected website values."""

    def test_edi_scores_match_website(self, fitted_model, expected_edi):
        """Test that computed EDI scores match website within tolerance."""
        _, edi_df = fitted_model

        comparison = edi_df.merge(
            expected_edi[["benchmark_name", "edi"]],
            left_on="benchmark",
            right_on="benchmark_name",
            how="inner",
            suffixes=("_computed", "_expected")
        )

        # Ensure we matched most benchmarks
        n_matched = len(comparison)
        n_expected = min(len(edi_df), len(expected_edi))
        assert n_matched > 0.9 * n_expected, \
            f"Only matched {n_matched} benchmarks out of {n_expected}"

        # Compute differences
        comparison["diff"] = comparison["edi_computed"] - comparison["edi_expected"]
        comparison["abs_diff"] = comparison["diff"].abs()

        # Statistics
        mae = comparison["abs_diff"].mean()
        max_diff = comparison["abs_diff"].max()
        worst_bench = comparison.loc[comparison["abs_diff"].idxmax()]

        # Print detailed stats
        print(f"\n{'='*60}")
        print("EDI Score Comparison")
        print(f"{'='*60}")
        print(f"Benchmarks compared: {n_matched}")
        print(f"Mean absolute error: {mae:.6f}")
        print(f"Max absolute error: {max_diff:.6f}")
        print(f"Largest deviation: {worst_bench['benchmark']}")
        print(f"  Computed: {worst_bench['edi_computed']:.4f}")
        print(f"  Expected: {worst_bench['edi_expected']:.4f}")
        print(f"  Diff: {worst_bench['diff']:+.6f}")

        assert mae < EDI_TOLERANCE, \
            f"Mean absolute EDI error {mae:.6f} exceeds tolerance {EDI_TOLERANCE}"

    def test_edi_ranking_correlation(self, fitted_model, expected_edi):
        """Test that EDI rankings are highly correlated with expected."""
        _, edi_df = fitted_model

        comparison = edi_df.merge(
            expected_edi[["benchmark_name", "edi"]],
            left_on="benchmark",
            right_on="benchmark_name",
            how="inner",
            suffixes=("_computed", "_expected")
        )

        if len(comparison) < 10:
            pytest.skip("Not enough benchmarks to compute ranking correlation")

        corr, p_value = spearmanr(
            comparison["edi_computed"].rank(),
            comparison["edi_expected"].rank()
        )

        print(f"\nEDI Spearman correlation: {corr:.6f} (p={p_value:.2e})")

        assert corr > 0.99, f"Ranking correlation {corr:.6f} is too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
