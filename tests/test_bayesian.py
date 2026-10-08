"""
Tests for the Bayesian ECI fit.

These run on synthetic data drawn from the model itself, so they are fast,
offline, and have a known ground truth to check recovery against.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

pytest.importorskip("pymc", reason="PyMC is required; reinstall with pip install -e .")

import pymc as pm  # noqa: E402

from eci.bayesian import BayesianPriors, build_bayesian_model, fit_bayesian_eci  # noqa: E402

ANCHOR_LOW, ANCHOR_HIGH = "Claude 3.5 Sonnet", "GPT-5"
N_MODELS, N_BENCHMARKS = 40, 30
# Spread of the true capabilities and difficulties. Wider spreads put many
# models far from a benchmark's difficulty, where the Beta draws land at 0 or
# 1 and get clipped: at 3, a quarter of the scores are, against 3.5% in the
# real data, and the clipping distorts the noise the fit can recover.
SPREAD = 1.5

# Small but enough to mix on data this size; the fit takes a few seconds.
SAMPLING = dict(draws=1000, tune=1000, chains=2, progressbar=False)


@pytest.fixture(scope="module")
def truth():
    """Draw model parameters and a sparse table of scores from the model."""
    rng = np.random.default_rng(0)
    capability = rng.normal(0, SPREAD, N_MODELS)
    difficulty = rng.normal(0, SPREAD, N_BENCHMARKS)
    difficulty -= difficulty.mean()
    log_slope = rng.normal(0, 0.5, N_BENCHMARKS)
    slope = np.exp(log_slope - log_slope.mean())
    kappa = np.exp(rng.normal(np.log(20), 0.7, N_BENCHMARKS))

    names = [f"Model {i}" for i in range(N_MODELS)]
    names[3], names[30] = ANCHOR_LOW, ANCHOR_HIGH
    capability[3], capability[30] = -1.0, 2.0  # anchors the right way round

    rows = []
    for m in range(N_MODELS):
        for b in range(N_BENCHMARKS):
            if rng.random() < 0.25:  # sparse coverage, as in the real data
                continue
            mu = 1.0 / (1.0 + np.exp(-slope[b] * (capability[m] - difficulty[b])))
            rows.append((m, b, rng.beta(mu * kappa[b], (1 - mu) * kappa[b]),
                         f"Bench {b}", names[m]))
    df = pd.DataFrame(
        rows, columns=["model_id", "benchmark_id", "performance", "benchmark", "Model"]
    )

    # The truth on the ECI scale: the same affine map the fit applies, built
    # from the true anchor capabilities.
    b_scale = (150.0 - 130.0) / (capability[30] - capability[3])
    a_shift = 130.0 - b_scale * capability[3]
    return {
        "df": df,
        "names": names,
        "bench_names": [f"Bench {b}" for b in range(N_BENCHMARKS)],
        "eci": pd.Series(a_shift + b_scale * capability, index=names),
        "edi": pd.Series(a_shift + b_scale * difficulty, index=[f"Bench {b}" for b in range(N_BENCHMARKS)]),
        "sigma": pd.Series(0.5 / np.sqrt(kappa + 1.0), index=[f"Bench {b}" for b in range(N_BENCHMARKS)]),
    }


@pytest.fixture(scope="module")
def fitted(truth):
    """Fit the Bayesian model once and reuse it across the tests."""
    return fit_bayesian_eci(truth["df"], **SAMPLING)


class TestModelConstruction:
    """Tests for the PyMC model itself."""

    def test_coords_are_names(self, truth):
        """Coordinates carry model and benchmark names, not positional ids,
        in first-appearance order as in the least-squares fit."""
        df = truth["df"]
        model = build_bayesian_model(df)
        assert list(model.coords["model"]) == df["Model"].drop_duplicates().tolist()
        assert list(model.coords["benchmark"]) == df["benchmark"].drop_duplicates().tolist()

    def test_expected_variables(self, truth):
        """The model exposes the parameters the methodology names."""
        model = build_bayesian_model(truth["df"])
        names = {v.name for v in model.named_vars.values()}
        assert {
            "capability", "difficulty", "slope", "sigma", "kappa",
            "tau_cd", "tau_alpha", "sigma_median", "tau_sigma",
        } <= names

    def test_rejects_nan_performance(self, truth):
        df = truth["df"].copy()
        df.loc[df.index[0], "performance"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            build_bayesian_model(df)

    def test_rejects_out_of_range_performance(self, truth):
        df = truth["df"].copy()
        df.loc[df.index[0], "performance"] = 1.5
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            build_bayesian_model(df)

    def test_rejects_boundary_scores_without_clipping(self, truth):
        """The Beta has no density at 0 or 1, so clipping cannot be switched
        off when the data touch either bound."""
        df = truth["df"].copy()
        df.loc[df.index[0], "performance"] = 0.0
        with pytest.raises(ValueError, match="performance_clip_eps"):
            build_bayesian_model(df, performance_clip_eps=0.0)

    def test_priors_are_customisable(self, truth):
        """A changed hyperprior reaches the model."""
        model = build_bayesian_model(
            truth["df"], BayesianPriors(sigma_median=(float(np.log(0.03)), 0.05))
        )
        with model:
            prior = pm.sample_prior_predictive(
                draws=200, var_names=["sigma_median"], random_seed=0
            )
        assert 0.025 < float(prior.prior["sigma_median"].median()) < 0.035

    def test_kappa_follows_sigma(self, truth):
        """kappa is the concentration whose Beta has SD sigma at mu = 0.5,
        and the sigma prior never reaches the 0.5 where kappa would hit 0."""
        model = build_bayesian_model(
            truth["df"],
            BayesianPriors(sigma_median=(float(np.log(0.4)), 0.5), tau_sigma=(0.0, 0.5)),
        )
        with model:
            prior = pm.sample_prior_predictive(
                draws=500, var_names=["sigma", "kappa"], random_seed=0
            ).prior
        sigma, kappa = prior["sigma"].to_numpy(), prior["kappa"].to_numpy()
        assert sigma.max() < 0.5
        assert (kappa > 0).all()
        np.testing.assert_allclose(0.5 / np.sqrt(kappa + 1.0), sigma)

    def test_tau_is_the_marginal_sd(self, truth):
        """tau_cd and tau_alpha are the marginal SDs of capability, difficulty
        and log slope alike. Without the zero-sum correction the two zero-sum
        priors would come out sqrt(1 - 1/K) narrower, 1.7% at K = 30."""
        model = pm.do(build_bayesian_model(truth["df"]), {"tau_cd": 2.0, "tau_alpha": 0.5})
        capability, difficulty, log_slope = pm.draw(
            [model["capability"], model["difficulty"], model["log_slope"]],
            draws=20000, random_seed=0,
        )
        assert np.sqrt((capability**2).mean()) == pytest.approx(2.0, rel=0.005)
        assert np.sqrt((difficulty**2).mean()) == pytest.approx(2.0, rel=0.005)
        assert np.sqrt((log_slope**2).mean()) == pytest.approx(0.5, rel=0.005)

    def test_rejects_a_single_benchmark(self, truth):
        df = truth["df"]
        with pytest.raises(ValueError, match="two benchmarks"):
            build_bayesian_model(df[df["benchmark_id"] == df["benchmark_id"].iloc[0]])


class TestIdentifyingConstraints:
    """The two constraints that pin down the otherwise free location and
    scale of the capability axis."""

    def test_difficulties_sum_to_zero(self, fitted):
        _, _, draws = fitted
        difficulty = draws["trace"].posterior["difficulty"].to_numpy()
        assert np.abs(difficulty.sum(axis=-1)).max() < 1e-8

    def test_slopes_have_geometric_mean_one(self, fitted):
        _, _, draws = fitted
        log_slope = np.log(draws["trace"].posterior["slope"].to_numpy())
        assert np.abs(log_slope.sum(axis=-1)).max() < 1e-8


class TestOutputFormat:
    """Tests for the returned frames, which mirror the least-squares fit."""

    def test_eci_columns_and_sorting(self, fitted):
        eci_df, _, _ = fitted
        assert list(eci_df.columns) == [
            "model_id", "Model", "eci", "eci_ci_low", "eci_ci_high"
        ]
        assert eci_df["eci"].is_monotonic_decreasing
        assert len(eci_df) == N_MODELS

    def test_edi_columns_and_sorting(self, fitted):
        _, edi_df, _ = fitted
        for name in ("edi", "discriminability_scaled", "sigma"):
            assert {name, f"{name}_ci_low", f"{name}_ci_high"} <= set(edi_df.columns)
        assert edi_df["edi"].is_monotonic_increasing
        assert len(edi_df) == N_BENCHMARKS

    def test_anchors_are_pinned(self, fitted):
        """Every draw is re-anchored, so the anchor models land exactly on
        their fixed values and get no interval."""
        eci_df, _, draws = fitted
        anchors = eci_df.set_index("Model").loc[[ANCHOR_LOW, ANCHOR_HIGH]]
        assert anchors["eci"].to_numpy() == pytest.approx([130.0, 150.0], abs=1e-9)
        assert anchors[["eci_ci_low", "eci_ci_high"]].isna().all().all()

        low = draws["model_names"].index(ANCHOR_LOW)
        high = draws["model_names"].index(ANCHOR_HIGH)
        assert draws["eci"][:, low] == pytest.approx(130.0)
        assert draws["eci"][:, high] == pytest.approx(150.0)

    def test_intervals_bracket_the_point_estimate(self, fitted):
        eci_df, edi_df, _ = fitted
        free = eci_df.dropna(subset=["eci_ci_low"])
        assert (free["eci_ci_low"] <= free["eci"]).all()
        assert (free["eci"] <= free["eci_ci_high"]).all()
        assert (edi_df["edi_ci_low"] <= edi_df["edi"]).all()
        assert (edi_df["edi"] <= edi_df["edi_ci_high"]).all()

    def test_slopes_and_noise_are_in_range(self, fitted):
        _, edi_df, _ = fitted
        assert (edi_df["discriminability_scaled"] > 0).all()
        assert ((edi_df["sigma"] > 0) & (edi_df["sigma"] < 0.5)).all()

    def test_draws_shapes(self, fitted):
        _, _, draws = fitted
        n_draws = SAMPLING["draws"] * SAMPLING["chains"]
        assert draws["eci"].shape == (n_draws, N_MODELS)
        assert draws["edi"].shape == (n_draws, N_BENCHMARKS)
        assert draws["slope"].shape == (n_draws, N_BENCHMARKS)
        assert draws["sigma"].shape == (n_draws, N_BENCHMARKS)
        assert draws["a"].shape == draws["b"].shape == (n_draws,)
        assert draws["model_names"][0] == "Model 0"


class TestParameterRecovery:
    """The fit should find the parameters the data were drawn from."""

    def test_recovers_capability(self, fitted, truth):
        eci_df, _, _ = fitted
        fit = eci_df.set_index("Model")["eci"]
        true = truth["eci"][fit.index]
        corr = spearmanr(fit, true)[0]
        mae = (fit - true).abs().mean()
        print(f"\nECI recovery: spearman {corr:.4f}, MAE {mae:.3f} points")
        assert corr > 0.95
        assert mae < 5.0

    def test_recovers_difficulty(self, fitted, truth):
        _, edi_df, _ = fitted
        fit = edi_df.set_index("benchmark")["edi"]
        true = truth["edi"][fit.index]
        assert spearmanr(fit, true)[0] > 0.9

    def test_recovers_benchmark_noise(self, fitted, truth):
        """sigma should at least order the benchmarks by how tightly their
        scores track the curve."""
        _, edi_df, _ = fitted
        fit = edi_df.set_index("benchmark")["sigma"]
        assert spearmanr(fit, truth["sigma"][fit.index])[0] > 0.6

    def test_interval_coverage(self, fitted, truth):
        """The 90% intervals should cover most of the truth.

        A sanity check, not a calibration test: every model's ECI goes through
        the same per-draw anchor transform, so their interval errors are
        correlated and one dataset's coverage swings widely (0.58-0.97 over
        five synthetic draws). Broken intervals would still fail it.
        """
        eci_df, _, _ = fitted
        free = eci_df.dropna(subset=["eci_ci_low"]).set_index("Model")
        true = truth["eci"][free.index]
        covered = ((true >= free["eci_ci_low"]) & (true <= free["eci_ci_high"])).mean()
        print(f"\n90% interval coverage: {covered:.2f} over {len(free)} models")
        assert covered > 0.5


class TestAnchorHandling:
    """Tests for the anchor models that define the ECI scale."""

    def test_missing_anchor_raises(self, truth):
        with pytest.raises(ValueError, match="Anchor model not found"):
            fit_bayesian_eci(truth["df"], anchor_model_high="No Such Model", **SAMPLING)

    def test_custom_anchor_values(self, truth):
        eci_df, _, _ = fit_bayesian_eci(
            truth["df"], anchor_eci_low=0.0, anchor_eci_high=100.0, **SAMPLING
        )
        anchors = eci_df.set_index("Model").loc[[ANCHOR_LOW, ANCHOR_HIGH], "eci"]
        assert anchors.to_numpy() == pytest.approx([0.0, 100.0], abs=1e-9)
