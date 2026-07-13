"""
Closed-form tests for ECI scale construction (eci.scaling).

The synthetic bootstrap draws are built so the correct per-draw-scaled
results are known exactly: the anchors' raw capabilities wobble across
draws (so a global-scaling implementation produces materially different,
wrong CIs - asserted directly below), while each auxiliary model sits at a
known fraction between the anchors, making its ECI-scale value per draw
exact by construction.
"""

import numpy as np
import pandas as pd
import pytest

from eci.scaling import compute_eci_scores

ANCHOR_LOW = "Anchor Low"
ANCHOR_HIGH = "Anchor High"
LOW_ECI = 130.0
HIGH_ECI = 150.0

# In fit order: anchors first, then a model pinned halfway between the
# anchors, then a model whose anchor-relative position varies by draw
MODEL_IDS = ["m0", "m1", "m2", "m3"]
MODEL_NAMES = [ANCHOR_LOW, ANCHOR_HIGH, "midpoint-model", "fraction-model"]
BENCH_IDS = ["b0", "b1"]
BENCH_NAMES = ["bench_a", "bench_b"]

N_DRAWS = 200

ANCHOR_KWARGS = dict(
    anchor_model_low=ANCHOR_LOW,
    anchor_eci_low=LOW_ECI,
    anchor_model_high=ANCHOR_HIGH,
    anchor_eci_high=HIGH_ECI,
)


def make_central_frames():
    """Central fit: anchor capabilities 1.0/3.0 -> a=120, b=10."""
    model_df = pd.DataFrame({
        "model_id": MODEL_IDS,
        "Model": MODEL_NAMES,
        "capability": [1.0, 3.0, 2.0, 2.5],
    }).sort_values("capability", ascending=False)
    bench_df = pd.DataFrame({
        "benchmark_id": BENCH_IDS,
        "benchmark": BENCH_NAMES,
        "difficulty": [0.0, 1.0],
        "discriminability": [1.0, 2.0],
        "is_anchor": [True, False],
    })
    return model_df, bench_df


def make_bootstrap_data(seed: int = 20260713):
    """Draws with known per-draw-scaled values.

    Per draw i:
      - the low anchor is noisy around 1.0 and the high anchor sits a noisy
        positive spread above it, so the raw scale wobbles substantially;
      - midpoint-model is exactly halfway between the anchors -> exactly 140
        on the ECI scale in every draw;
      - fraction-model sits at fraction f_i = i/(N_DRAWS-1) between the
        anchors -> its ECI-scale values are exactly linspace(130, 150);
      - bench_a's difficulty equals the low anchor's capability -> EDI
        exactly 130 in every draw; bench_b sits at the high anchor -> 150.
    """
    rng = np.random.default_rng(seed)
    cap_low = rng.normal(1.0, 0.3, N_DRAWS)
    spread = rng.uniform(1.0, 3.0, N_DRAWS)
    frac = np.linspace(0.0, 1.0, N_DRAWS)

    capability_samples = [
        np.array([
            cap_low[i],
            cap_low[i] + spread[i],
            cap_low[i] + 0.5 * spread[i],
            cap_low[i] + frac[i] * spread[i],
        ])
        for i in range(N_DRAWS)
    ]
    difficulty_samples = [
        np.array([cap_low[i], cap_low[i] + spread[i]]) for i in range(N_DRAWS)
    ]
    discriminability_samples = [
        np.array([1.0, rng.uniform(0.5, 2.0)]) for i in range(N_DRAWS)
    ]
    return {
        "model_ids": MODEL_IDS,
        "model_names": MODEL_NAMES,
        "benchmark_ids": BENCH_IDS,
        "benchmark_names": BENCH_NAMES,
        "capability_samples": capability_samples,
        "difficulty_samples": difficulty_samples,
        "discriminability_samples": discriminability_samples,
    }


@pytest.fixture()
def central_frames():
    return make_central_frames()


@pytest.fixture()
def bootstrap_data():
    return make_bootstrap_data()


@pytest.fixture()
def results(central_frames, bootstrap_data):
    model_df, bench_df = central_frames
    return compute_eci_scores(model_df, bench_df, bootstrap_data, **ANCHOR_KWARGS)


class TestCentralScaling:
    def test_central_map_and_anchor_values(self, results):
        assert results.scaling["a"] == pytest.approx(120.0)
        assert results.scaling["b"] == pytest.approx(10.0)
        eci = results.eci_df.set_index("Model")["eci"]
        assert eci[ANCHOR_LOW] == pytest.approx(LOW_ECI)
        assert eci[ANCHOR_HIGH] == pytest.approx(HIGH_ECI)
        assert eci["midpoint-model"] == pytest.approx(140.0)
        assert eci["fraction-model"] == pytest.approx(145.0)

    def test_edi_and_scaled_slope(self, results):
        edi = results.edi_df.set_index("benchmark")
        assert edi.loc["bench_a", "edi"] == pytest.approx(120.0)
        assert edi.loc["bench_b", "edi"] == pytest.approx(130.0)
        assert edi.loc["bench_a", "discriminability_scaled"] == pytest.approx(0.1)
        assert edi.loc["bench_b", "discriminability_scaled"] == pytest.approx(0.2)

    def test_missing_anchor_raises(self, central_frames):
        model_df, bench_df = central_frames
        with pytest.raises(ValueError, match="not found"):
            compute_eci_scores(
                model_df, bench_df,
                anchor_model_low="No Such Model",
                anchor_model_high=ANCHOR_HIGH,
            )

    def test_inverted_central_anchors_raise(self, central_frames):
        model_df, bench_df = central_frames
        with pytest.raises(ValueError, match="do not define a scale"):
            compute_eci_scores(
                model_df, bench_df,
                anchor_model_low=ANCHOR_HIGH,
                anchor_model_high=ANCHOR_LOW,
            )


class TestPerDrawScaling:
    def test_anchors_pinned_in_every_draw(self, results):
        for draw in results.samples.eci_samples:
            assert draw[0] == pytest.approx(LOW_ECI, abs=1e-9)
            assert draw[1] == pytest.approx(HIGH_ECI, abs=1e-9)

    def test_anchor_ci_cells_are_nan(self, results):
        by_model = results.eci_df.set_index("Model")
        anchors = by_model.loc[[ANCHOR_LOW, ANCHOR_HIGH]]
        assert anchors["eci_ci_low"].isna().all()
        assert anchors["eci_ci_high"].isna().all()
        others = by_model.drop([ANCHOR_LOW, ANCHOR_HIGH])
        assert others["eci_ci_low"].notna().all()
        assert others["eci_ci_high"].notna().all()

    def test_cis_match_closed_form(self, results):
        """A global-scaling implementation fails this test.

        midpoint-model has zero uncertainty relative to the anchors, so its
        CI is exactly [140, 140]; fraction-model's scaled draws are exactly
        linspace(130, 150, N_DRAWS), so its 5th/95th percentiles are exactly
        131 and 149.
        """
        by_model = results.eci_df.set_index("Model")
        assert by_model.loc["midpoint-model", "eci_ci_low"] == pytest.approx(140.0, abs=1e-9)
        assert by_model.loc["midpoint-model", "eci_ci_high"] == pytest.approx(140.0, abs=1e-9)
        assert by_model.loc["fraction-model", "eci_ci_low"] == pytest.approx(131.0, abs=1e-9)
        assert by_model.loc["fraction-model", "eci_ci_high"] == pytest.approx(149.0, abs=1e-9)

    def test_global_scaling_would_disagree(self, results, bootstrap_data):
        """Demonstrate the gap this module exists to close: pushing raw draws
        through the central (a, b) leaks anchor noise into midpoint-model's
        CI, which is exactly [140, 140] under per-draw scaling."""
        a, b = results.scaling["a"], results.scaling["b"]
        global_scaled = np.vstack([
            a + b * np.asarray(draw)
            for draw in bootstrap_data["capability_samples"]
        ])
        lo, hi = np.quantile(global_scaled[:, 2], [0.05, 0.95])
        assert hi - lo > 1.0

    def test_ci_alignment_survives_model_df_ordering(self, central_frames, bootstrap_data):
        """model_df arrives sorted by capability; sample arrays are in fit
        order. CIs must land on the right models regardless."""
        model_df, bench_df = central_frames
        shuffled = model_df.sample(frac=1, random_state=7)
        res = compute_eci_scores(shuffled, bench_df, bootstrap_data, **ANCHOR_KWARGS)
        by_model = res.eci_df.set_index("Model")
        assert by_model.loc["fraction-model", "eci_ci_low"] == pytest.approx(131.0, abs=1e-9)
        assert by_model.loc["midpoint-model", "eci_ci_high"] == pytest.approx(140.0, abs=1e-9)

    def test_ci_level_respected(self, central_frames, bootstrap_data):
        model_df, bench_df = central_frames
        res = compute_eci_scores(
            model_df, bench_df, bootstrap_data, ci_level=0.5, **ANCHOR_KWARGS
        )
        by_model = res.eci_df.set_index("Model")
        assert by_model.loc["fraction-model", "eci_ci_low"] == pytest.approx(135.0, abs=1e-9)
        assert by_model.loc["fraction-model", "eci_ci_high"] == pytest.approx(145.0, abs=1e-9)
        assert res.scaling["ci_level"] == 0.5

    def test_round_trip_via_recorded_transforms(self, results, bootstrap_data):
        for s in range(results.samples.num_samples):
            raw = (results.samples.eci_samples[s] - results.samples.a_samples[s]) \
                / results.samples.b_samples[s]
            np.testing.assert_allclose(
                raw, bootstrap_data["capability_samples"][s], atol=1e-9
            )


class TestBenchmarkParamScaling:
    def test_benchmark_positions_pinned_by_construction(self, results):
        """bench_a tracks the low anchor and bench_b the high anchor in every
        draw, so their EDIs are exactly 130/150 with zero-width CIs."""
        for draw in results.samples.edi_samples:
            assert draw[0] == pytest.approx(LOW_ECI, abs=1e-9)
            assert draw[1] == pytest.approx(HIGH_ECI, abs=1e-9)
        edi = results.edi_df.set_index("benchmark")
        assert edi.loc["bench_a", "edi_ci_low"] == pytest.approx(LOW_ECI, abs=1e-9)
        assert edi.loc["bench_b", "edi_ci_high"] == pytest.approx(HIGH_ECI, abs=1e-9)

    def test_prediction_invariance_per_draw(self, results, bootstrap_data):
        """The IRT prediction disc*(cap - diff) must be preserved by the
        scale change: slope_scaled*(eci - edi) gives the same values."""
        for s in range(results.samples.num_samples):
            cap = np.asarray(bootstrap_data["capability_samples"][s])
            diff = np.asarray(bootstrap_data["difficulty_samples"][s])
            disc = np.asarray(bootstrap_data["discriminability_samples"][s])
            raw_pred = disc[None, :] * (cap[:, None] - diff[None, :])

            eci = results.samples.eci_samples[s]
            edi = results.samples.edi_samples[s]
            slope = results.samples.slope_samples[s]
            scaled_pred = slope[None, :] * (eci[:, None] - edi[None, :])

            np.testing.assert_allclose(scaled_pred, raw_pred, atol=1e-9)


class TestDegenerateDraws:
    def _data_with_bad_draws(self):
        data = make_bootstrap_data()
        coincident = np.array([2.0, 2.0, 1.5, 2.5])
        inverted = np.array([3.0, 1.0, 2.0, 2.5])
        data["capability_samples"] = (
            [coincident, inverted] + data["capability_samples"]
        )
        pad = [np.array([0.0, 1.0])] * 2
        data["difficulty_samples"] = pad + data["difficulty_samples"]
        data["discriminability_samples"] = pad + data["discriminability_samples"]
        return data

    def test_bad_draws_dropped_with_warning(self, central_frames):
        model_df, bench_df = central_frames
        data = self._data_with_bad_draws()
        with pytest.warns(UserWarning, match="do not define a scale"):
            res = compute_eci_scores(model_df, bench_df, data, **ANCHOR_KWARGS)

        assert res.diagnostics["n_draws_total"] == N_DRAWS + 2
        assert res.diagnostics["n_draws_used"] == N_DRAWS
        assert res.diagnostics["n_draws_dropped"] == 2
        assert len(res.diagnostics["dropped_reasons"]) == 2
        assert res.samples.num_samples == N_DRAWS

    def test_quantiles_unaffected_by_dropped_draws(self, central_frames):
        model_df, bench_df = central_frames
        data = self._data_with_bad_draws()
        with pytest.warns(UserWarning):
            res = compute_eci_scores(model_df, bench_df, data, **ANCHOR_KWARGS)
        by_model = res.eci_df.set_index("Model")
        assert by_model.loc["fraction-model", "eci_ci_low"] == pytest.approx(131.0, abs=1e-9)
        assert by_model.loc["fraction-model", "eci_ci_high"] == pytest.approx(149.0, abs=1e-9)


class TestWithoutDraws:
    def test_no_bootstrap_data_means_no_ci_columns(self, central_frames):
        model_df, bench_df = central_frames
        res = compute_eci_scores(model_df, bench_df, **ANCHOR_KWARGS)
        assert res.samples is None
        assert "eci_ci_low" not in res.eci_df.columns
        assert "eci_ci_high" not in res.eci_df.columns
        assert "edi_ci_low" not in res.edi_df.columns
        assert res.diagnostics["n_draws_total"] == 0

    def test_empty_draw_list_means_no_ci_columns(self, central_frames, bootstrap_data):
        model_df, bench_df = central_frames
        for key in ("capability_samples", "difficulty_samples", "discriminability_samples"):
            bootstrap_data[key] = []
        res = compute_eci_scores(model_df, bench_df, bootstrap_data, **ANCHOR_KWARGS)
        assert res.samples is None
        assert "eci_ci_low" not in res.eci_df.columns

    def test_single_draw_keeps_samples_but_no_cis(self, central_frames, bootstrap_data):
        model_df, bench_df = central_frames
        for key in ("capability_samples", "difficulty_samples", "discriminability_samples"):
            bootstrap_data[key] = bootstrap_data[key][:1]
        with pytest.warns(UserWarning, match="skipping confidence intervals"):
            res = compute_eci_scores(model_df, bench_df, bootstrap_data, **ANCHOR_KWARGS)
        assert res.samples.num_samples == 1
        assert "eci_ci_low" not in res.eci_df.columns

    def test_anchor_missing_from_samples_raises(self, central_frames, bootstrap_data):
        model_df, bench_df = central_frames
        bootstrap_data["model_names"] = ["x0", "x1", "x2", "x3"]
        with pytest.raises(ValueError, match="missing from bootstrap samples"):
            compute_eci_scores(model_df, bench_df, bootstrap_data, **ANCHOR_KWARGS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
