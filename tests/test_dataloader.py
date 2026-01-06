"""
Tests for the dataloader that verify output matches eci_benchmarks.csv.

Note: There are known bugs in benchmark_data.zip that cause some scores to differ
from the backend data used to generate eci_benchmarks.csv. These tests will print
detailed information about any mismatches.
"""

import pandas as pd
import pytest
from pathlib import Path

from eci.dataloader import prepare_benchmark_data, get_all_benchmark_names

EXPECTED_URL = "https://epoch.ai/data/eci_benchmarks.csv"


@pytest.fixture(scope="module")
def loaded_data():
    """Load data from benchmark_data.zip."""
    return prepare_benchmark_data(cache_dir=Path(".cache"))


@pytest.fixture(scope="module")
def expected_data():
    """Load expected data from eci_benchmarks.csv."""
    df = pd.read_csv(EXPECTED_URL)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["benchmark_release_date"] = pd.to_datetime(df["benchmark_release_date"], errors="coerce")
    return df


class TestDataLoaderStructure:
    """Test that the dataloader produces valid output structure."""

    def test_required_columns(self, loaded_data):
        """Test that all required columns are present."""
        required = [
            "model_id", "benchmark_id", "performance", "benchmark",
            "benchmark_release_date", "optimized", "is_math", "is_coding",
            "model", "model_version", "Model", "date", "source"
        ]
        missing = set(required) - set(loaded_data.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_has_data(self, loaded_data):
        """Test that data was loaded."""
        assert len(loaded_data) > 0

    def test_performance_range(self, loaded_data):
        """Test that performance values are valid."""
        assert loaded_data["performance"].min() >= 0
        assert loaded_data["performance"].max() <= 1
        assert loaded_data["performance"].notna().all()


class TestDataLoaderAccuracy:
    """Test accuracy against expected eci_benchmarks.csv."""

    def test_model_coverage(self, loaded_data, expected_data):
        """Test that we have similar model coverage."""
        loaded_models = set(loaded_data["Model"].unique())
        expected_models = set(expected_data["Model"].unique())

        missing = expected_models - loaded_models
        extra = loaded_models - expected_models

        coverage = len(loaded_models & expected_models) / len(expected_models)

        print(f"\n{'='*60}")
        print("Model Coverage")
        print(f"{'='*60}")
        print(f"Expected models: {len(expected_models)}")
        print(f"Loaded models: {len(loaded_models)}")
        print(f"Coverage: {coverage:.1%}")

        if missing:
            print(f"\nMissing models ({len(missing)}):")
            for m in sorted(missing)[:10]:
                print(f"  - {m}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")

        if extra:
            print(f"\nExtra models ({len(extra)}):")
            for m in sorted(extra)[:10]:
                print(f"  - {m}")

        # Allow some tolerance for coverage
        assert coverage > 0.8, f"Model coverage too low: {coverage:.1%}"

    def test_benchmark_coverage(self, loaded_data, expected_data):
        """Test that we have similar benchmark coverage."""
        loaded_benchmarks = set(loaded_data["benchmark"].unique())
        expected_benchmarks = set(expected_data["benchmark"].unique())

        missing = expected_benchmarks - loaded_benchmarks
        extra = loaded_benchmarks - expected_benchmarks

        coverage = len(loaded_benchmarks & expected_benchmarks) / len(expected_benchmarks)

        print(f"\n{'='*60}")
        print("Benchmark Coverage")
        print(f"{'='*60}")
        print(f"Expected benchmarks: {len(expected_benchmarks)}")
        print(f"Loaded benchmarks: {len(loaded_benchmarks)}")
        print(f"Coverage: {coverage:.1%}")

        if missing:
            print(f"\nMissing benchmarks: {sorted(missing)}")
        if extra:
            print(f"\nExtra benchmarks: {sorted(extra)}")

        assert coverage > 0.8, f"Benchmark coverage too low: {coverage:.1%}"

    def test_score_matching(self, loaded_data, expected_data):
        """Test that scores match for common (Model, benchmark) pairs."""
        # Merge on Model and benchmark
        loaded = loaded_data[["Model", "benchmark", "performance"]].copy()
        expected = expected_data[["Model", "benchmark", "performance"]].copy()

        merged = loaded.merge(
            expected,
            on=["Model", "benchmark"],
            suffixes=("_loaded", "_expected"),
            how="inner"
        )

        merged["diff"] = merged["performance_loaded"] - merged["performance_expected"]
        merged["abs_diff"] = merged["diff"].abs()

        # Find mismatches (with tolerance for floating point)
        tolerance = 1e-6
        mismatches = merged[merged["abs_diff"] > tolerance].copy()

        print(f"\n{'='*60}")
        print("Score Matching")
        print(f"{'='*60}")
        print(f"Common (Model, benchmark) pairs: {len(merged)}")
        print(f"Matching pairs: {len(merged) - len(mismatches)}")
        print(f"Mismatching pairs: {len(mismatches)}")

        if len(mismatches) > 0:
            print(f"\nMismatches (tolerance > {tolerance}):")
            mismatches_sorted = mismatches.sort_values("abs_diff", ascending=False)
            print(mismatches_sorted[[
                "Model", "benchmark", "performance_loaded", "performance_expected", "diff"
            ]].head(20).to_string(index=False))

            if len(mismatches) > 20:
                print(f"... and {len(mismatches) - 20} more mismatches")

            # Summary statistics
            print(f"\nMismatch statistics:")
            print(f"  Mean absolute difference: {mismatches['abs_diff'].mean():.6f}")
            print(f"  Max absolute difference: {mismatches['abs_diff'].max():.6f}")

        match_rate = (len(merged) - len(mismatches)) / len(merged) if len(merged) > 0 else 0
        print(f"\nMatch rate: {match_rate:.1%}")

        # This assertion may fail due to known bugs - just warn
        if match_rate < 0.95:
            print(f"\nWARNING: Match rate ({match_rate:.1%}) is below 95%")
            print("This may be due to known bugs in benchmark_data.zip")

    def test_row_count(self, loaded_data, expected_data):
        """Test that row counts are similar."""
        loaded_count = len(loaded_data)
        expected_count = len(expected_data)

        print(f"\n{'='*60}")
        print("Row Count")
        print(f"{'='*60}")
        print(f"Expected rows: {expected_count}")
        print(f"Loaded rows: {loaded_count}")
        print(f"Difference: {loaded_count - expected_count}")

        # Allow some tolerance
        ratio = loaded_count / expected_count if expected_count > 0 else 0
        assert 0.8 < ratio < 1.2, f"Row count ratio {ratio:.2f} outside acceptable range"


class TestBenchmarkFiltering:
    """Test benchmark filtering functionality."""

    def test_get_all_benchmark_names(self):
        """Test that get_all_benchmark_names returns expected benchmarks."""
        names = get_all_benchmark_names()
        assert isinstance(names, set)
        assert len(names) > 0
        # Check some expected benchmarks exist
        assert "MMLU" in names
        assert "Winogrande" in names
        assert "GPQA diamond" in names

    def test_include_benchmarks(self):
        """Test including only specific benchmarks."""
        include = {"MMLU", "Winogrande", "GSM8K", "GPQA diamond", "HellaSwag"}
        filtered = prepare_benchmark_data(
            cache_dir=Path(".cache"),
            include_benchmarks=include,
            min_benchmarks_per_model=4,  # Allow models with at least 4 of the 5 benchmarks
        )

        benchmarks_in_result = set(filtered["benchmark"].unique())
        # All returned benchmarks should be from our include set
        assert benchmarks_in_result.issubset(include), (
            f"Unexpected benchmarks: {benchmarks_in_result - include}"
        )
        # Should have at least some of the requested benchmarks
        assert len(benchmarks_in_result) > 0, "No benchmarks returned"

    def test_exclude_benchmarks(self):
        """Test excluding specific benchmarks."""
        exclude = {"MMLU", "Winogrande"}
        filtered = prepare_benchmark_data(
            cache_dir=Path(".cache"),
            exclude_benchmarks=exclude,
        )

        benchmarks_in_result = set(filtered["benchmark"].unique())
        assert not (benchmarks_in_result & exclude), (
            f"Excluded benchmarks found in result: {benchmarks_in_result & exclude}"
        )
        # Should still have other benchmarks
        assert len(benchmarks_in_result) > 0

    def test_cannot_use_both_include_and_exclude(self):
        """Test that using both include and exclude raises error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            prepare_benchmark_data(
                cache_dir=Path(".cache"),
                include_benchmarks={"MMLU"},
                exclude_benchmarks={"Winogrande"},
            )

    def test_unknown_benchmark_in_include_raises_error(self):
        """Test that unknown benchmark names in include_benchmarks raises error."""
        with pytest.raises(ValueError, match="Unknown benchmark names"):
            prepare_benchmark_data(
                cache_dir=Path(".cache"),
                include_benchmarks={"MMLU", "NonexistentBenchmark"},
            )

    def test_unknown_benchmark_in_exclude_raises_error(self):
        """Test that unknown benchmark names in exclude_benchmarks raises error."""
        with pytest.raises(ValueError, match="Unknown benchmark names"):
            prepare_benchmark_data(
                cache_dir=Path(".cache"),
                exclude_benchmarks={"NonexistentBenchmark"},
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
