"""
Tests for ECI fitting that verify outputs match the Epoch AI website.

These tests load benchmark data from https://epoch.ai/data/eci_benchmarks.csv
and verify that the fitted scores match:
- https://epoch.ai/data/eci_scores.csv (model capabilities)
- https://epoch.ai/data/edi_scores.csv (benchmark difficulties)
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from eci import fit_eci_model, load_benchmark_data, compute_eci_scores, BOOTSTRAP_METHODS
from eci.fitting import _bootstrap_indices, _irt_jacobian


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
    model_df, bench_df, _ = fit_eci_model(
        benchmark_data,
        bootstrap_samples=0,
    )
    results = compute_eci_scores(model_df, bench_df)
    return results.eci_df, results.edi_df


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


def _synthetic_benchmark_data(seed: int = 0) -> pd.DataFrame:
    """Small dataset generated from the IRT model itself (runs offline)."""
    rng = np.random.default_rng(seed)
    n_models, n_benchmarks = 12, 8
    capabilities = np.linspace(-1.5, 1.5, n_models)
    difficulties = np.linspace(-1.0, 1.0, n_benchmarks)
    discriminabilities = rng.uniform(0.8, 1.5, n_benchmarks)

    rows = []
    for m in range(n_models):
        # First few models get minimal coverage (4 benchmarks, like the
        # least-covered real models); all models include the anchor benchmark
        if m < 3:
            benches = np.concatenate(
                [[0], rng.choice(np.arange(1, n_benchmarks), size=3, replace=False)]
            )
        else:
            benches = np.arange(n_benchmarks)
        for b in benches:
            p = 1.0 / (1.0 + np.exp(-discriminabilities[b] * (capabilities[m] - difficulties[b])))
            p = float(np.clip(p + rng.normal(0, 0.03), 0.001, 0.999))
            rows.append({
                "model_id": f"model_{m}",
                "benchmark_id": f"bench_{b}",
                "performance": p,
                "benchmark": "Winogrande" if b == 0 else f"bench_{b}",
                "Model": f"Model {m}",
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_data():
    return _synthetic_benchmark_data()


# Anchor models used when scaling synthetic fits (well-separated capabilities)
SYNTH_ANCHORS = dict(anchor_model_low="Model 2", anchor_model_high="Model 9")


def _scaled_synthetic_results(synthetic_data, bootstrap_samples, method="hierarchical"):
    model_df, bench_df, bootstrap_data = fit_eci_model(
        synthetic_data, bootstrap_samples=bootstrap_samples, bootstrap_method=method
    )
    return compute_eci_scores(model_df, bench_df, bootstrap_data, **SYNTH_ANCHORS)


class TestBootstrapMethods:
    """Offline tests for the bootstrap resampling schemes."""

    def test_invalid_method_raises(self, synthetic_data):
        with pytest.raises(ValueError, match="bootstrap_method"):
            fit_eci_model(
                synthetic_data, bootstrap_samples=1, bootstrap_method="jackknife"
            )

    @pytest.mark.parametrize("method", BOOTSTRAP_METHODS)
    def test_method_produces_valid_cis(self, synthetic_data, method):
        results = _scaled_synthetic_results(synthetic_data, 8, method)
        eci_df = results.eci_df
        anchors = eci_df["Model"].isin(SYNTH_ANCHORS.values())
        non_anchor = eci_df[~anchors]
        assert np.isfinite(non_anchor["eci_ci_low"]).all()
        assert np.isfinite(non_anchor["eci_ci_high"]).all()
        assert (non_anchor["eci_ci_low"] <= non_anchor["eci_ci_high"]).all()
        assert eci_df.loc[anchors, ["eci_ci_low", "eci_ci_high"]].isna().all().all()

    @pytest.mark.parametrize("method", BOOTSTRAP_METHODS)
    def test_method_reproducible_with_seed(self, synthetic_data, method):
        def run():
            return fit_eci_model(
                synthetic_data, bootstrap_samples=4, bootstrap_method=method
            )[0]

        pd.testing.assert_frame_equal(run(), run())

    def test_hierarchical_resampling_covers_every_model(self):
        rng = np.random.default_rng(0)
        model_idx = np.repeat(np.arange(10), 4)
        rows_by_model = [np.flatnonzero(model_idx == m) for m in range(10)]
        for _ in range(200):
            idx = _bootstrap_indices(rng, "hierarchical", model_idx.size, rows_by_model)
            assert idx.size == model_idx.size
            assert set(model_idx[idx]) == set(range(10))
            # Each model's rows are resampled only from its own rows
            assert (model_idx[np.sort(idx)] == model_idx).all()

    def test_methods_give_different_cis(self, synthetic_data):
        cis = {}
        for method in ("observation", "hierarchical"):
            results = _scaled_synthetic_results(synthetic_data, 6, method)
            eci_df = results.eci_df[~results.eci_df["Model"].isin(SYNTH_ANCHORS.values())]
            cis[method] = eci_df.sort_values("model_id")["eci_ci_high"].to_numpy()
        assert not np.allclose(cis["observation"], cis["hierarchical"])

    def test_end_to_end_anchors_pinned_in_every_draw(self, synthetic_data):
        """Full fit -> scale integration: every scaled draw must place the
        anchor models exactly on their fixed ECI values."""
        results = _scaled_synthetic_results(synthetic_data, 5)
        names = results.draws["model_names"]
        eci_draws = results.draws["eci"]
        assert eci_draws.shape[0] > 1
        idx_low = names.index(SYNTH_ANCHORS["anchor_model_low"])
        idx_high = names.index(SYNTH_ANCHORS["anchor_model_high"])
        np.testing.assert_allclose(eci_draws[:, idx_low], 130.0, atol=1e-9)
        np.testing.assert_allclose(eci_draws[:, idx_high], 150.0, atol=1e-9)
        eci = results.eci_df.set_index("Model")["eci"]
        assert eci["Model 2"] == pytest.approx(130.0, abs=1e-9)
        assert eci["Model 9"] == pytest.approx(150.0, abs=1e-9)


def _irt_jacobian_loop_reference(
    params, capability, difficulty, discriminability, model_idx, bench_idx,
    anchor_idx, n_models, n_benchmarks, regularization_strength,
):
    """Loop-based Jacobian assembly used as the verification reference."""
    from scipy.sparse import lil_matrix

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    n_obs = len(model_idx)
    n_params = params.size
    z = discriminability[bench_idx] * (capability[model_idx] - difficulty[bench_idx])
    s = sigmoid(z)
    ds = s * (1 - s)

    n_rows = n_obs + 1 if regularization_strength > 0 else n_obs
    jac = lil_matrix((n_rows, n_params))

    cap_derivs = ds * discriminability[bench_idx]
    diff_derivs = -ds * discriminability[bench_idx]
    discrim_derivs = ds * (capability[model_idx] - difficulty[bench_idx])

    for i in range(n_obs):
        jac[i, model_idx[i]] = cap_derivs[i]
        jac[i, n_models + bench_idx[i]] = diff_derivs[i]
        b = bench_idx[i]
        if b != anchor_idx:
            param_idx = n_models + n_benchmarks + (b if b < anchor_idx else b - 1)
            jac[i, param_idx] = discrim_derivs[i]

    if regularization_strength > 0:
        reg = regularization_strength * np.sum(params**2) / n_params
        if reg > 0:
            scale = regularization_strength / (n_params * np.sqrt(reg))
            for j in range(n_params):
                jac[n_obs, j] = scale * params[j]

    return jac.tocsr()


class TestJacobians:
    """Verify the vectorized Jacobian assembly against loop/numerical references."""

    def _random_problem(self, seed=0, n_models=9, n_benchmarks=6, n_obs=60):
        rng = np.random.default_rng(seed)
        model_idx = rng.integers(0, n_models, size=n_obs)
        bench_idx = rng.integers(0, n_benchmarks, size=n_obs)
        # Ensure every model and benchmark appears at least once
        model_idx[:n_models] = np.arange(n_models)
        bench_idx[:n_benchmarks] = np.arange(n_benchmarks)
        capability = rng.normal(0, 1.5, n_models)
        difficulty = rng.normal(0, 1.0, n_benchmarks)
        discriminability = rng.uniform(0.5, 2.0, n_benchmarks)
        return model_idx, bench_idx, capability, difficulty, discriminability

    @pytest.mark.parametrize("regularization", [0.0, 0.1])
    @pytest.mark.parametrize("anchor_idx", [0, 3, 5])
    def test_irt_jacobian_matches_loop_reference(self, regularization, anchor_idx):
        model_idx, bench_idx, capability, difficulty, discriminability = self._random_problem()
        n_models, n_benchmarks = len(capability), len(difficulty)
        discriminability[anchor_idx] = 1.0
        free = np.arange(n_benchmarks) != anchor_idx
        params = np.concatenate([capability, difficulty, discriminability[free]])

        args = (
            params, capability, difficulty, discriminability, model_idx,
            bench_idx, anchor_idx, n_models, n_benchmarks, regularization,
        )
        vectorized = _irt_jacobian(*args).toarray()
        reference = _irt_jacobian_loop_reference(*args).toarray()

        assert vectorized.shape == reference.shape
        np.testing.assert_array_equal(vectorized, reference)

    def test_analytical_fit_matches_numerical_fit(self, synthetic_data):
        fast = fit_eci_model(
            synthetic_data, bootstrap_samples=0, use_analytical_jacobian=True
        )[0]
        slow = fit_eci_model(
            synthetic_data, bootstrap_samples=0, use_analytical_jacobian=False
        )[0]
        merged = fast.merge(slow, on="model_id", suffixes=("_fast", "_slow"))
        # Exact vs finite-difference Jacobians converge along slightly
        # different optimizer paths; agreement is to ~1e-5, not machine eps
        np.testing.assert_allclose(
            merged["capability_fast"], merged["capability_slow"], atol=1e-4
        )


class TestFitReturns:
    """Tests for the fit's output contract."""

    def test_bootstrap_data_shape(self, synthetic_data):
        model_df, bench_df, bootstrap_data = fit_eci_model(
            synthetic_data,
            bootstrap_samples=3,
        )

        expected_keys = {
            "model_ids", "model_names", "benchmark_ids", "benchmark_names",
            "capability_samples", "difficulty_samples", "discriminability_samples",
        }
        assert expected_keys.issubset(bootstrap_data.keys())

        assert len(bootstrap_data["model_names"]) == len(model_df)
        assert len(bootstrap_data["benchmark_names"]) == len(bench_df)

        for key in ("capability_samples", "difficulty_samples", "discriminability_samples"):
            samples = bootstrap_data[key]
            assert len(samples) > 0, f"No samples collected for {key}"
            assert len(samples) <= 3, f"Got more {key} than requested"

        cap_sample = bootstrap_data["capability_samples"][0]
        diff_sample = bootstrap_data["difficulty_samples"][0]
        disc_sample = bootstrap_data["discriminability_samples"][0]
        assert cap_sample.shape == (len(model_df),)
        assert diff_sample.shape == (len(bench_df),)
        assert disc_sample.shape == (len(bench_df),)

    def test_no_bootstrap_gives_empty_samples(self, synthetic_data):
        model_df, bench_df, bootstrap_data = fit_eci_model(
            synthetic_data, bootstrap_samples=0
        )
        assert bootstrap_data["capability_samples"] == []
        assert len(bootstrap_data["model_names"]) == len(model_df)

    def test_fit_outputs_are_bare(self, synthetic_data):
        """The fit returns statistical results only: no CI/SE columns (CIs
        are constructed on the ECI scale in scaling.py) and no metadata
        passthrough (callers join their own)."""
        model_df, bench_df, _ = fit_eci_model(synthetic_data, bootstrap_samples=2)
        assert list(model_df.columns) == ["model_id", "Model", "capability"]
        assert list(bench_df.columns) == [
            "benchmark_id", "benchmark", "difficulty", "discriminability", "is_anchor",
        ]

    def test_scaling_reproduces_eci_column(self, synthetic_data):
        model_df, bench_df, _ = fit_eci_model(synthetic_data, bootstrap_samples=0)
        results = compute_eci_scores(model_df, bench_df, **SYNTH_ANCHORS)

        scaling = results.scaling
        assert scaling["anchor_model_low"] == "Model 2"
        assert scaling["anchor_eci_low"] == 130.0
        assert scaling["anchor_model_high"] == "Model 9"
        assert scaling["anchor_eci_high"] == 150.0

        # eci = a + b * capability should reproduce the eci column
        recomputed = scaling["a"] + scaling["b"] * results.eci_df["capability"]
        assert (recomputed - results.eci_df["eci"]).abs().max() < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
