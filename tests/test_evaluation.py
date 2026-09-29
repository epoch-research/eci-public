"""
Tests for the Bayesian fit's evaluation suite.

Run on the same synthetic data as the fit tests, so the model is correctly
specified: the diagnostics should come back clean, which is what makes them
worth trusting when a real fit comes back dirty.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymc", reason="PyMC is required; reinstall with pip install -e .")

from eci.bayesian import fit_bayesian_eci  # noqa: E402
from eci.evaluation import (  # noqa: E402
    PPC_STATISTICS,
    evaluate_fit,
    save_evaluation,
    summarise_evaluation,
)

from .test_bayesian import ANCHOR_HIGH, ANCHOR_LOW, N_BENCHMARKS, SAMPLING, truth  # noqa: F401


@pytest.fixture(scope="module")
def evaluated(truth):
    """Fit with the evaluation suite switched on, once, for the whole module."""
    _, _, draws = fit_bayesian_eci(truth["df"], evaluate=True, **SAMPLING)
    return draws["evaluation"], truth["df"]


class TestEvaluationWiring:
    """The evaluate flag has to produce the groups the suite needs."""

    def test_evaluate_populates_the_trace(self, evaluated):
        results, _ = evaluated
        assert set(results) == {
            "sampler_summary", "sampler_parameters", "accuracy", "loo_summary",
            "loo_observations", "ppc", "calibration", "pit", "forest", "qq", "metrics",
        }

    def test_fit_without_evaluate_has_no_evaluation(self, truth):
        _, _, draws = fit_bayesian_eci(truth["df"], **SAMPLING)
        assert "evaluation" not in draws
        assert "log_likelihood" not in draws["trace"]

    def test_evaluate_fit_needs_the_extra_groups(self, truth):
        """Calling the suite on a plain trace says what is missing."""
        _, _, draws = fit_bayesian_eci(truth["df"], **SAMPLING)
        with pytest.raises(ValueError, match="log_likelihood"):
            evaluate_fit(draws["trace"], truth["df"])


class TestSamplerDiagnostics:
    def test_summary_is_one_row_with_the_headline_numbers(self, evaluated):
        results, _ = evaluated
        summary = results["sampler_summary"]
        assert len(summary) == 1
        assert {"divergences", "max_r_hat", "min_ess_bulk", "min_ess_tail",
                "min_bfmi"} <= set(summary.columns)

    def test_parameters_are_ranked_worst_r_hat_first(self, evaluated):
        results, _ = evaluated
        r_hat = results["sampler_parameters"]["r_hat"]
        assert r_hat.is_monotonic_decreasing
        assert (results["sampler_parameters"]["ess_bulk"] > 0).all()

    def test_a_correctly_specified_fit_converges(self, evaluated):
        """The data came from this very model, so nothing should be wrong."""
        results, _ = evaluated
        summary = results["sampler_summary"].iloc[0]
        assert summary["max_r_hat"] < 1.05
        assert summary["divergence_rate"] < 0.05
        assert summary["min_bfmi"] > 0.2



class TestMetricsTable:
    def test_every_section_is_present(self, evaluated):
        results, df = evaluated
        metrics = results["metrics"].set_index("metric")["value"]
        assert set(results["metrics"]["section"]) == {
            "data", "sampling", "accuracy", "loo", "calibration",
            "ppc_pooled", "ppc_per_benchmark",
        }
        assert metrics["n_obs"] == len(df)
        assert metrics["n_benchmarks"] == N_BENCHMARKS

    def test_metric_names_are_unique(self, evaluated):
        """The name is the lookup key, so a collision silently turns a scalar
        into a Series for every caller."""
        results, _ = evaluated
        metrics = results["metrics"]["metric"]
        assert not metrics.duplicated().any(), list(metrics[metrics.duplicated()])

    def test_headline_errors_are_plausible(self, evaluated):
        """MAE cannot exceed RMSE, and the in-sample fit cannot beat
        leave-one-out."""
        results, _ = evaluated
        metrics = results["metrics"].set_index("metric")["value"]
        assert metrics["mae"] <= metrics["rmse"] + 1e-12
        assert metrics["rmse"] <= metrics["rmse_loo"] + 1e-9
        assert 0 < metrics["rmse"] < 0.5
        assert abs(metrics["bias"]) < 0.05


class TestAccuracy:
    def test_one_row_per_benchmark_no_pooled_row(self, evaluated):
        """The per-benchmark table carries no OVERALL row: the pooled numbers
        live in the metrics table."""
        results, _ = evaluated
        accuracy = results["accuracy"]
        assert len(accuracy) == N_BENCHMARKS
        assert "OVERALL" not in set(accuracy["benchmark"])

    def test_reports_the_ratio_not_raw_rmse(self, evaluated):
        """Raw per-benchmark RMSE is a restatement of sigma, so it is not
        reported; the ratio against the model's own predicted SD is."""
        results, _ = evaluated
        accuracy = results["accuracy"]
        assert "rmse" not in accuracy.columns
        assert {"predicted_sd", "rmse_ratio", "bias"} <= set(accuracy.columns)
        assert (accuracy["predicted_sd"] > 0).all()
        assert (accuracy["rmse_ratio"] > 0).all()
        assert accuracy["rmse_ratio"].is_monotonic_decreasing

    def test_ratio_is_near_one_when_the_model_is_right(self, evaluated):
        """The data are Beta draws with the model's own noise, so each
        benchmark should scatter about as much as the model claims."""
        results, _ = evaluated
        ratio = results["accuracy"]["rmse_ratio"]
        assert 0.5 < ratio.median() < 1.5


class TestLoo:
    def test_summary_counts_every_observation_into_a_pareto_band(self, evaluated):
        results, df = evaluated
        summary = results["loo_summary"].iloc[0]
        bands = ["pareto_k_good", "pareto_k_ok", "pareto_k_bad", "pareto_k_very_bad"]
        assert sum(int(summary[b]) for b in bands) == len(df)
        assert summary["n_obs"] == len(df)

    def test_pointwise_table_is_ranked_by_pareto_k(self, evaluated):
        results, df = evaluated
        observations = results["loo_observations"]
        assert len(observations) == len(df)
        assert observations["pareto_k"].is_monotonic_decreasing
        assert {"Model", "benchmark", "performance", "elpd_i"} <= set(observations.columns)

    def test_well_specified_fit_has_reliable_importance_sampling(self, evaluated):
        results, df = evaluated
        summary = results["loo_summary"].iloc[0]
        assert summary["pareto_k_very_bad"] / len(df) < 0.05


class TestPosteriorPredictiveChecks:
    def test_every_statistic_is_checked_for_every_benchmark(self, evaluated):
        results, _ = evaluated
        ppc = results["ppc"]
        assert set(ppc["statistic"]) == set(PPC_STATISTICS)
        assert len(ppc) == (N_BENCHMARKS + 1) * len(PPC_STATISTICS)
        assert ppc["ppp"].between(0, 1).all()

    def test_spread_is_reproduced_when_the_model_is_right(self, evaluated):
        """The data are Beta draws around the curve, so each benchmark's
        spread should be reproduced. Checked per benchmark: pooled across
        benchmarks the statistic measures the spread of the fitted curve,
        which partial pooling shrinks, and runs low even for a correct fit.
        """
        results, _ = evaluated
        ppc = results["ppc"]
        per_benchmark = ppc[ppc["benchmark"] != "OVERALL"]
        for statistic in ("variance", "iqr"):
            values = per_benchmark.loc[per_benchmark["statistic"] == statistic, "ppp"]
            assert 0.15 < values.median() < 0.85
            assert ((values < 0.05) | (values > 0.95)).mean() < 0.35

    def test_pooled_mean_is_reproduced(self, evaluated):
        results, _ = evaluated
        ppc = results["ppc"]
        overall = ppc[ppc["benchmark"] == "OVERALL"].set_index("statistic")["ppp"]
        assert 0.02 < overall["mean"] < 0.98


class TestCalibration:
    def test_pit_is_a_probability_per_observation(self, evaluated):
        results, df = evaluated
        pit = results["pit"]
        assert len(pit) == len(df)
        assert pit["pit"].between(0, 1).all()

    def test_coverage_is_close_to_nominal(self, evaluated):
        """The intervals should be the width they claim; this is the check
        that would catch an over-confident posterior."""
        results, _ = evaluated
        overall = results["calibration"]
        overall = overall[overall["benchmark"] == "OVERALL"].iloc[0]
        assert abs(overall["coverage_0.9"] - 0.90) < 0.07
        assert abs(overall["coverage_0.5"] - 0.50) < 0.12
        assert abs(overall["mean_pit"] - 0.5) < 0.07

    def test_qq_arrays_line_up(self, evaluated):
        results, df = evaluated
        qq = results["qq"]
        assert len(qq["pit_sorted"]) == len(qq["pit_uniform"]) == len(df)
        assert np.all(np.diff(qq["pit_sorted"]) >= 0)
        assert (qq["replicated_ci_low"] <= qq["replicated_ci_high"]).all()
        assert len(qq["observed_quantiles"]) == len(qq["quantile_levels"])


class TestForest:
    def test_one_row_per_benchmark_sorted_by_difficulty(self, evaluated):
        results, _ = evaluated
        forest = results["forest"]
        assert len(forest) == N_BENCHMARKS
        assert forest["difficulty_mean"].is_monotonic_increasing

    def test_intervals_bracket_the_mean(self, evaluated):
        results, _ = evaluated
        forest = results["forest"]
        assert (forest["sigma_lo"] <= forest["sigma_mean"]).all()
        assert (forest["sigma_mean"] <= forest["sigma_hi"]).all()
        assert (forest["difficulty_lo"] <= forest["difficulty_mean"]).all()
        assert (forest["difficulty_mean"] <= forest["difficulty_hi"]).all()
        assert ((forest["sigma_lo"] > 0) & (forest["sigma_hi"] < 0.5)).all()  # sigma's support


class TestSaving:
    def test_writes_only_the_pdf_report(self, evaluated, tmp_path):
        results, _ = evaluated
        written = save_evaluation(results, tmp_path)
        names = {path.name for path in written}
        assert names == {"evaluation_report.pdf"}
        assert all(path.exists() and path.stat().st_size > 0 for path in written)

    def test_summary_mentions_each_section(self, evaluated):
        results, _ = evaluated
        text = summarise_evaluation(results)
        for heading in ("Sampling", "Predictive accuracy", "Leave-one-out",
                        "Calibration", "Posterior predictive p-values"):
            assert heading in text
