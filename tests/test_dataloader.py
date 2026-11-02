"""
Tests for the metadata-driven dataloader.

Synthetic tests build a small benchmark_data.zip in-process and verify each
loading rule. Live tests download the published zip and require its output
to match the published fit input (processed_data_for_eci.csv) exactly.
"""

import zipfile

import pandas as pd
import pytest

from eci.dataloader import (
    BENCHMARK_METADATA_FILENAME,
    MODEL_METADATA_FILENAME,
    download_benchmark_data,
    get_all_benchmark_names,
    prepare_benchmark_data,
)

EXPECTED_URL = "https://epoch.ai/data/processed_data_for_eci.csv"


#########################
# Synthetic fixtures
#########################

METADATA = pd.DataFrame([
    ("Alpha", "alpha.csv", "Score", 1.0, 0.25, 1.0, "2020-01-01", "", True),
    ("Beta", "beta.csv", "Pct", 0.01, 0.0, 0.9, "2021-06-15", "", True),
    ("Gamma", "gamma.csv", "Score", 1.0, 0.0, 1.0, "2024-01-01", "", True),
    ("Old Bench", "old.csv", "Score", 1.0, 0.0, 1.0, "2022-01-01", "New Bench", True),
    ("New Bench", "new.csv", "Score", 1.0, 0.0, 1.0, "2025-01-01", "", True),
    ("Ghost", "ghost.csv", "Score", 1.0, 0.0, 1.0, "2025-01-01", "", True),
    # Display-only row: carries a baseline/ceiling for site charts, not fit
    ("DisplayOnly", "alpha.csv", "Score", 1.0, 0.5, 1.0, "2020-01-01", "", False),
], columns=["benchmark", "source_file", "score_column", "scale",
            "random_baseline", "score_ceiling", "release_date", "superseded_by",
            "in_eci"])

# The organization column stands in for the optional display/analysis
# columns model_metadata.csv carries beyond the three the loader uses.
MODEL_METADATA = pd.DataFrame([
    ("v1", "Group One", "2024-01-01", "Lab A"),
    ("v1b", "Group One", "2024-06-01", "Lab A"),
    ("v2", "Group Two", "2023-05-01", "Lab B"),
    ("v-mix-old", "Mixed", "2022-05-01", "Lab C"),
    ("v-mix-new", "Mixed", "2024-03-01", "Lab C"),
    ("v-nodate", "No Date", None, "Lab D"),
], columns=["model_version", "model_group", "date", "organization"])

SOURCE_FILES = {
    "alpha.csv": pd.DataFrame({
        "Model version": ["v1", "v1b", "v2", "v-mix-old", "v-nodate", "v1"],
        "Score": [0.75, 0.85, 0.1, 0.7, 0.9, "N/A"],
        "Source": ["src-a"] * 6,
    }),
    "beta.csv": pd.DataFrame({
        # Pct is on a 0-100 scale (scale 0.01), ceiling 0.9
        "Model version": ["v1", "v-mix-old", "v2"],
        "Pct": [45.0, 90.0, 120.0],
    }),
    "gamma.csv": pd.DataFrame({
        "Model version": ["v1", "v-mix-new"],
        "Score": [0.5, 0.6],
    }),
    "old.csv": pd.DataFrame({
        "Model version": ["v1", "v-mix-new"],
        "Score": [0.4, 0.3],
    }),
    "new.csv": pd.DataFrame({
        "Model version": ["v1b"],
        "Score": [0.55],
    }),
}


def _prepare(source, **kwargs):
    """prepare_benchmark_data on the synthetic zip, whose models have too few
    scores for the mid-range rule, with that rule off by default."""
    kwargs.setdefault("min_mid_range_benchmarks_per_model", 0)
    return prepare_benchmark_data(source, **kwargs)


@pytest.fixture(scope="module")
def synthetic_zip(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "benchmark_data.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, df in SOURCE_FILES.items():
            zf.writestr(name, df.to_csv(index=False))
        zf.writestr(BENCHMARK_METADATA_FILENAME, METADATA.to_csv(index=False))
        zf.writestr(MODEL_METADATA_FILENAME, MODEL_METADATA.to_csv(index=False))
        # ECI outputs shipped alongside the raw data
        zf.writestr("epoch_capabilities_index/eci_scores.csv", "Model,eci\nA1,100\n")
        zf.writestr("epoch_capabilities_index/eci_bootstraps.json", "{}")
        zf.writestr("README.md", "raw benchmark data")
    return path


@pytest.fixture(scope="module")
def loaded(synthetic_zip):
    with pytest.warns(UserWarning, match="Ghost"):
        return _prepare(synthetic_zip, min_benchmarks_per_model=2)


def _row(df, model, benchmark):
    rows = df[(df["model"] == model) & (df["benchmark"] == benchmark)]
    assert len(rows) <= 1
    return rows.iloc[0] if len(rows) else None


#########################
# Mechanism tests
#########################

class TestLoading:
    def test_only_raw_csvs_are_extracted(self, synthetic_zip):
        dfs = download_benchmark_data(synthetic_zip)
        assert set(dfs) == set(SOURCE_FILES) | {BENCHMARK_METADATA_FILENAME, MODEL_METADATA_FILENAME}

    def test_output_schema(self, loaded):
        assert list(loaded.columns) == [
            "model_id", "benchmark_id", "performance", "benchmark",
            "benchmark_release_date", "model", "model_version", "Model",
            "date", "source",
        ]
        assert (loaded["Model"] == loaded["model"]).all()
        assert loaded["performance"].between(0, 1).all()

    def test_baseline_and_ceiling_normalization(self, loaded):
        # Alpha: baseline 0.25 -> (0.85 - 0.25) / 0.75 for Group One's best
        alpha = _row(loaded, "Group One", "Alpha")
        assert alpha["performance"] == pytest.approx((0.85 - 0.25) / 0.75)
        # Beta: scale 0.01 then ceiling 0.9 -> 45 -> 0.45 -> 0.5
        beta = _row(loaded, "Group One", "Beta")
        assert beta["performance"] == pytest.approx(0.45 / 0.9)

    def test_clipping(self, loaded):
        # Group Two's Alpha score 0.1 is below the 0.25 baseline: clips to 0
        assert _row(loaded, "Group Two", "Alpha")["performance"] == 0.0
        # Group Two's Beta score 120 -> 1.2 -> above ceiling: clips to 1
        assert _row(loaded, "Group Two", "Beta")["performance"] == 1.0

    def test_non_numeric_scores_dropped(self, loaded):
        # v1's second Alpha row ("N/A") must not crash or beat the 0.85
        alpha = _row(loaded, "Group One", "Alpha")
        assert alpha["performance"] == pytest.approx((0.85 - 0.25) / 0.75)

    def test_supersession(self, loaded):
        # Group One has New Bench, so its Old Bench score is dropped
        assert _row(loaded, "Group One", "Old Bench") is None
        assert _row(loaded, "Group One", "New Bench") is not None
        # Mixed has no New Bench score, so it keeps Old Bench
        assert _row(loaded, "Mixed", "Old Bench") is not None

    def test_qualification_counts_only_fit_rows(self, synthetic_zip):
        # Mixed spans 4 distinct benchmarks, but Alpha and Beta come from its
        # 2022 version: only Old Bench and Gamma survive the date filter, so
        # at min 3 the model is excluded outright.
        with pytest.warns(UserWarning, match="Ghost"):
            out = _prepare(synthetic_zip, min_benchmarks_per_model=3)
        assert "Mixed" not in set(out["model"])

    def test_date_filter_trims_old_rows(self, loaded):
        # At min 2, Mixed qualifies on its two post-2023 rows; the 2022 rows
        # stay out of the output either way.
        mixed = loaded[loaded["model"] == "Mixed"]
        assert set(mixed["benchmark"]) == {"Gamma", "Old Bench"}

    def test_rows_without_dates_are_dropped(self, loaded):
        assert "No Date" not in set(loaded["model"])

    def test_aggregation_takes_max_and_newest_version(self, loaded):
        # Group One scored Alpha with v1 (0.75) and v1b (0.85): max wins,
        # and the newest-dated version is reported
        alpha = _row(loaded, "Group One", "Alpha")
        assert alpha["model_version"] == "v1b"
        assert alpha["date"] == pd.Timestamp("2024-06-01")

    def test_aggregation_reports_the_best_scoring_version(self, synthetic_zip):
        # v1b is newer than v1 but scores lower on Gamma: the row keeps v1's
        # score, version and source, dated by the pair's latest version.
        extra = pd.DataFrame({
            "model_version": ["v1b"], "benchmark": ["Gamma"],
            "performance": [0.3], "source": ["worse run"],
        })
        out = _prepare(synthetic_zip, min_benchmarks_per_model=1, extra_scores=extra)
        gamma = _row(out, "Group One", "Gamma")
        assert gamma["performance"] == pytest.approx(0.5)
        assert gamma["model_version"] == "v1"
        assert pd.isna(gamma["source"])
        assert gamma["date"] == pd.Timestamp("2024-06-01")

    def test_groups_with_mixed_release_dates_warn(self, synthetic_zip):
        # Group One's v1 and v1b have different dates
        with pytest.warns(UserWarning, match=r"different release dates.*'Group One'"):
            _prepare(synthetic_zip, min_benchmarks_per_model=1)

    def test_release_dates_from_config(self, loaded):
        beta = _row(loaded, "Group One", "Beta")
        assert beta["benchmark_release_date"] == pd.Timestamp("2021-06-15")

    def test_min_benchmarks_per_model(self, synthetic_zip):
        out = _prepare(synthetic_zip, min_benchmarks_per_model=3)
        assert "Group Two" not in set(out["model"])  # only Alpha + Beta

    def test_min_mid_range_benchmarks_per_model(self, loaded, synthetic_zip):
        best = loaded.groupby(["model", "benchmark"])["performance"].max()
        mid = best[best.between(0.1, 0.9)].groupby(level="model").size()
        out = _prepare(
            synthetic_zip, min_benchmarks_per_model=2, min_mid_range_benchmarks_per_model=2
        )
        kept = set(mid[mid >= 2].index)
        assert set(out["model"]) == kept
        # Qualifying models keep their extreme scores too
        for model in kept:
            assert len(out[out["model"] == model]) == len(loaded[loaded["model"] == model])


class TestExtraScores:
    def test_extra_scores_are_normalized_and_grouped(self, synthetic_zip):
        extra = pd.DataFrame({
            "model_version": ["stage-v", "stage-v", "stage-v"],
            "benchmark": ["Alpha", "Beta", "Gamma"],
            "performance": [0.85, 45.0, 0.7],
            "source": ["staging"] * 3,
        })
        out = _prepare(
            synthetic_zip, min_benchmarks_per_model=2, extra_scores=extra,
        )
        row = _row(out, "stage-v", "Alpha")
        assert row["performance"] == pytest.approx((0.85 - 0.25) / 0.75)
        assert _row(out, "stage-v", "Beta")["performance"] == pytest.approx(0.5)

    def test_extra_scores_are_clipped(self, synthetic_zip):
        # Below Alpha's 0.25 baseline and above Beta's ceiling
        extra = pd.DataFrame({
            "model_version": ["stage-v", "stage-v"],
            "benchmark": ["Alpha", "Beta"],
            "performance": [0.1, 120.0],
        })
        out = _prepare(synthetic_zip, min_benchmarks_per_model=2, extra_scores=extra)
        assert _row(out, "stage-v", "Alpha")["performance"] == 0.0
        assert _row(out, "stage-v", "Beta")["performance"] == 1.0

    def test_extra_scores_undated_known_version_is_dated_today(self, synthetic_zip):
        # A staging model can already exist in the model metadata with its
        # dates unset; its rows (including ones read from the zip) must be
        # dated today rather than dropped by the date filter.
        extra = pd.DataFrame({
            "model_version": ["v-nodate", "v-nodate"],
            "benchmark": ["Beta", "Gamma"],
            "performance": [45.0, 0.6],
        })
        out = _prepare(
            synthetic_zip, min_benchmarks_per_model=2, extra_scores=extra,
        )
        rows = out[out["model"] == "No Date"]
        assert set(rows["benchmark"]) == {"Alpha", "Beta", "Gamma"}
        assert (rows["date"] == pd.Timestamp.now().normalize()).all()

    def test_extra_scores_unknown_benchmark_warns(self, synthetic_zip):
        extra = pd.DataFrame({
            "model_version": ["stage-v"] * 2,
            "benchmark": ["Alpha", "Nope"],
            "performance": [0.9, 0.9],
        })
        with pytest.warns(UserWarning, match="Nope"):
            out = _prepare(
                synthetic_zip, min_benchmarks_per_model=1, extra_scores=extra,
            )
        assert "Nope" not in set(out["benchmark"])


class TestDisplayOnlyRows:
    def test_display_only_benchmarks_are_not_loaded(self, loaded):
        assert "DisplayOnly" not in set(loaded["benchmark"])

    def test_blank_in_eci_raises(self, synthetic_zip):
        metadata = METADATA.astype({"in_eci": object})
        metadata.loc[metadata["benchmark"] == "Beta", "in_eci"] = None
        with pytest.raises(ValueError, match=r"in_eci is blank .*'Beta'"):
            _prepare(synthetic_zip, benchmark_metadata=metadata)

    def test_missing_in_eci_column_fits_every_benchmark(self, synthetic_zip):
        metadata = METADATA.drop(columns="in_eci")
        with pytest.warns(UserWarning, match="Ghost"):
            out = _prepare(synthetic_zip, min_benchmarks_per_model=1,
                           benchmark_metadata=metadata)
        assert "DisplayOnly" in set(out["benchmark"])


class TestBenchmarkFiltering:
    def test_get_all_benchmark_names(self, synthetic_zip):
        names = get_all_benchmark_names(synthetic_zip)
        assert names == set(METADATA.loc[METADATA["in_eci"], "benchmark"])
        assert "DisplayOnly" not in names

    def test_include_benchmarks(self, synthetic_zip):
        out = _prepare(
            synthetic_zip, min_benchmarks_per_model=1,
            include_benchmarks={"Alpha", "Beta"},
        )
        assert set(out["benchmark"]) <= {"Alpha", "Beta"}

    def test_exclude_benchmarks(self, synthetic_zip):
        out = _prepare(
            synthetic_zip, min_benchmarks_per_model=1,
            exclude_benchmarks={"Alpha"},
        )
        assert "Alpha" not in set(out["benchmark"])

    def test_cannot_use_both_include_and_exclude(self, synthetic_zip):
        with pytest.raises(ValueError, match="Cannot specify both"):
            _prepare(
                synthetic_zip,
                include_benchmarks={"Alpha"}, exclude_benchmarks={"Beta"},
            )

    def test_unknown_benchmark_raises(self, synthetic_zip):
        with pytest.raises(ValueError, match="Unknown benchmark names"):
            _prepare(synthetic_zip, include_benchmarks={"Alpha", "Nonexistent"})
        with pytest.raises(ValueError, match="Unknown benchmark names"):
            _prepare(synthetic_zip, exclude_benchmarks={"Nonexistent"})


class TestMissingTables:
    def _zip_without(self, tmp_path, *omit):
        path = tmp_path / "partial.zip"
        with zipfile.ZipFile(path, "w") as zf:
            for name, df in SOURCE_FILES.items():
                zf.writestr(name, df.to_csv(index=False))
            if BENCHMARK_METADATA_FILENAME not in omit:
                zf.writestr(BENCHMARK_METADATA_FILENAME, METADATA.to_csv(index=False))
            if MODEL_METADATA_FILENAME not in omit:
                zf.writestr(MODEL_METADATA_FILENAME, MODEL_METADATA.to_csv(index=False))
        return path

    def test_missing_benchmark_metadata_raises(self, tmp_path):
        with pytest.raises(ValueError, match=BENCHMARK_METADATA_FILENAME):
            _prepare(self._zip_without(tmp_path, BENCHMARK_METADATA_FILENAME))

    def test_missing_model_metadata_raises(self, tmp_path):
        with pytest.raises(ValueError, match=MODEL_METADATA_FILENAME):
            _prepare(self._zip_without(tmp_path, MODEL_METADATA_FILENAME))

    def test_overrides_stand_in_for_zip_tables(self, tmp_path):
        path = self._zip_without(tmp_path, BENCHMARK_METADATA_FILENAME, MODEL_METADATA_FILENAME)
        with pytest.warns(UserWarning, match="Ghost"):
            out = _prepare(
                path, min_benchmarks_per_model=2,
                benchmark_metadata=METADATA, model_metadata=MODEL_METADATA,
            )
        assert len(out) > 0


#########################
# Live-data equivalence
#########################

@pytest.fixture(scope="module")
def live_output():
    # The published fit input predates the mid-range rule.
    return prepare_benchmark_data(min_mid_range_benchmarks_per_model=0)


@pytest.fixture(scope="module")
def expected():
    df = pd.read_csv(EXPECTED_URL)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


class TestLiveEquivalence:
    """The published zip must reproduce the published fit input."""

    def test_columns_match_exactly(self, live_output, expected):
        assert list(live_output.columns) == list(expected.columns)

    def test_models_match_exactly(self, live_output, expected):
        assert set(live_output["model"]) == set(expected["model"])

    def test_benchmarks_match_exactly(self, live_output, expected):
        assert set(live_output["benchmark"]) == set(expected["benchmark"])

    def test_performance_matches_exactly(self, live_output, expected):
        merged = live_output.merge(
            expected, on=["model", "benchmark"], suffixes=("_new", "_live"),
        )
        assert len(merged) == len(live_output) == len(expected)
        diff = (merged["performance_new"] - merged["performance_live"]).abs()
        assert (diff < 1e-9).all(), (
            merged.loc[diff >= 1e-9, ["model", "benchmark",
                                      "performance_new", "performance_live"]]
            .head(20)
            .to_string(index=False)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
