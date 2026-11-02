"""
Data loader for ECI benchmark data.

Loads raw benchmark data from https://epoch.ai/data/benchmark_data.zip and
processes it into the fit-input format (one row per model x benchmark).

All benchmark policy lives in two files shipped inside the zip, so the same
loader serves both the public zip and Epoch's production pipeline (which
builds the zip and then consumes it):

- ``benchmark_metadata.csv``: one row per benchmark — where its scores live
  (source_file, score_column, scale), how to normalize them
  (random_baseline, score_ceiling), its release_date, and optionally a
  superseded_by benchmark that replaces it model-for-model.
- ``model_metadata.csv``: one row per model version. The loader uses
  model_version, model_group (the release-dated aggregation the fit treats
  as one model), and date (the group-resolved release date); additional
  columns (display_name, organization, country, accessibility,
  training_compute_flop) describe the version for display and analysis.
"""

import io
import warnings
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import numpy as np
import pandas as pd

BENCHMARK_DATA_URL = "https://epoch.ai/data/benchmark_data.zip"
BENCHMARK_METADATA_FILENAME = "benchmark_metadata.csv"
MODEL_METADATA_FILENAME = "model_metadata.csv"


def download_benchmark_data(source: str | Path = BENCHMARK_DATA_URL, cache_dir: Optional[Path] = None) -> dict[str, pd.DataFrame]:
    """
    Download and extract benchmark data from zip file.

    Args:
        source: URL of benchmark_data.zip, or path to a local copy
        cache_dir: Optional directory to cache downloaded files (URLs only)

    Returns:
        Dictionary mapping filename to DataFrame
    """
    if not str(source).startswith(("http://", "https://")):
        with zipfile.ZipFile(source, "r") as zf:
            return _extract_csvs(zf)

    if cache_dir:
        cache_path = cache_dir / "benchmark_data.zip"
        if cache_path.exists():
            with zipfile.ZipFile(cache_path, "r") as zf:
                return _extract_csvs(zf)

    with urlopen(source) as response:
        data = response.read()

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)

    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        return _extract_csvs(zf)


def _extract_csvs(zf: zipfile.ZipFile) -> dict[str, pd.DataFrame]:
    """Extract CSV files from zipfile into DataFrames (other files are ignored)."""
    dfs = {}
    for name in zf.namelist():
        if name.endswith(".csv") and not name.startswith("epoch_capabilities_index/"):
            with zf.open(name) as f:
                basename = Path(name).name
                dfs[basename] = pd.read_csv(f)
    return dfs


def load_benchmark_metadata(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the benchmark policy table shipped in the zip."""
    if BENCHMARK_METADATA_FILENAME not in dfs:
        raise ValueError(
            f"{BENCHMARK_METADATA_FILENAME} not found in the benchmark data. "
            f"The zip at {BENCHMARK_DATA_URL} ships this table; pass "
            f"benchmark_metadata= to override."
        )
    metadata = dfs[BENCHMARK_METADATA_FILENAME].copy()
    required = {
        "benchmark", "source_file", "score_column", "scale",
        "random_baseline", "score_ceiling", "release_date",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"{BENCHMARK_METADATA_FILENAME} is missing columns: {sorted(missing)}")
    if "superseded_by" not in metadata.columns:
        metadata["superseded_by"] = pd.NA
    return _with_in_eci(metadata)


def _with_in_eci(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return ``metadata`` with a boolean in_eci column.

    Rows with in_eci false describe benchmarks the site displays but the fit
    does not use (they still carry baselines and ceilings). A table without
    the column predates the flag and fits every benchmark; a blank in a table
    that has it is an error, since either default would silently change which
    benchmarks the fit uses.
    """
    if "in_eci" not in metadata.columns:
        return metadata.assign(in_eci=True)
    blank = metadata["in_eci"].isna()
    if blank.any():
        raise ValueError(
            f"in_eci is blank for benchmarks: {sorted(metadata.loc[blank, 'benchmark'])}. "
            f"Set it to True or False."
        )
    return metadata.assign(in_eci=metadata["in_eci"].astype(bool))


def load_model_metadata(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the model version table shipped in the zip.

    Guarantees the model_version, model_group, and date columns the loader
    needs; any additional display/analysis columns are passed through.
    """
    if MODEL_METADATA_FILENAME not in dfs:
        raise ValueError(
            f"{MODEL_METADATA_FILENAME} not found in the benchmark data. The "
            f"zip at {BENCHMARK_DATA_URL} ships this table; pass "
            f"model_metadata= to override."
        )
    table = dfs[MODEL_METADATA_FILENAME].copy()
    required = {"model_version", "model_group", "date"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{MODEL_METADATA_FILENAME} is missing columns: {sorted(missing)}")
    table["date"] = pd.to_datetime(table["date"], errors="coerce")
    return table.dropna(subset=["model_version"])


def get_all_benchmark_names(
    source: str | Path = BENCHMARK_DATA_URL,
    cache_dir: Optional[Path] = None,
    benchmark_metadata: Optional[pd.DataFrame] = None,
) -> set[str]:
    """
    Get the names of all benchmarks available to the fit.

    Returns:
        Set of in-ECI benchmark names from the benchmark metadata.
    """
    if benchmark_metadata is None:
        benchmark_metadata = load_benchmark_metadata(download_benchmark_data(source, cache_dir))
    return set(benchmark_metadata.loc[benchmark_metadata["in_eci"], "benchmark"])


def _load_raw_scores(dfs: dict[str, pd.DataFrame], metadata: pd.DataFrame) -> pd.DataFrame:
    """Load per-benchmark scores as directed by the metadata, as reported
    (normalized later by _normalize_scores)."""
    frames = []
    unavailable = []
    for cfg in metadata.itertuples():
        if cfg.source_file not in dfs:
            unavailable.append(cfg.benchmark)
            continue
        df = dfs[cfg.source_file]

        cols = ["Model version", cfg.score_column]
        if "Source" in df.columns:
            cols.append("Source")
        df = df[cols].rename(columns={
            "Model version": "model_version",
            cfg.score_column: "performance",
            "Source": "source",
        })
        if "source" not in df.columns:
            df["source"] = pd.NA
        df = df.dropna(subset=["model_version", "performance"])
        df["benchmark"] = cfg.benchmark
        frames.append(df[["model_version", "benchmark", "performance", "source"]])

    if unavailable:
        warnings.warn(
            f"Benchmarks skipped because their source files are not in the "
            f"benchmark data: {unavailable}"
        )
    if not frames:
        raise ValueError("No benchmark scores could be loaded")
    return pd.concat(frames, ignore_index=True)


def _normalize_scores(scores: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw scores so each benchmark's random-guessing baseline maps
    to 0 and its max-achievable score (the ceiling) maps to 1, then clip into
    [0, 1]: sub-baseline scores clip up to 0, above-ceiling ones down to 1.
    Rows whose score is not a number, or whose benchmark's baseline equals its
    ceiling, are dropped.
    """
    cfg = metadata.set_index("benchmark")
    benchmark = scores["benchmark"]
    raw = pd.to_numeric(scores["performance"], errors="coerce") * benchmark.map(cfg["scale"])
    baseline = benchmark.map(cfg["random_baseline"])
    ceiling = benchmark.map(cfg["score_ceiling"])
    performance = ((raw - baseline) / (ceiling - baseline)).replace([np.inf, -np.inf], np.nan)
    scores = scores.assign(performance=performance)
    scores = scores[scores["performance"].notna()]
    return scores.assign(performance=scores["performance"].clip(lower=0.0, upper=1.0))


def filter_mid_range_models(
    scores: pd.DataFrame,
    min_benchmarks: int,
    score_range: tuple[float, float] = (0.1, 0.9),
    model_col: str = "model",
) -> pd.DataFrame:
    """Keep the models with at least ``min_benchmarks`` distinct benchmarks
    whose score lies in ``score_range`` (inclusive).

    Expects one row per (model, benchmark): the normalized score the fit
    sees. A qualifying model keeps all of its scores, extreme ones included;
    this only drops models whose scores sit almost entirely at the floor or
    the ceiling.
    """
    if min_benchmarks <= 0:
        return scores
    low, high = score_range
    in_range = scores["performance"].between(low, high)
    mid = scores[in_range].groupby(model_col)["benchmark"].nunique()
    return scores[scores[model_col].isin(mid[mid >= min_benchmarks].index)]


def prepare_benchmark_data(
    source: str | Path = BENCHMARK_DATA_URL,
    cache_dir: Optional[Path] = None,
    min_benchmarks_per_model: int = 4,
    min_mid_range_benchmarks_per_model: int = 4,
    mid_range: tuple[float, float] = (0.1, 0.9),
    min_date: str = "2023-01-01",
    include_benchmarks: Optional[set[str]] = None,
    exclude_benchmarks: Optional[set[str]] = None,
    extra_scores: Optional[pd.DataFrame] = None,
    benchmark_metadata: Optional[pd.DataFrame] = None,
    model_metadata: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Load and process benchmark data for ECI fitting.

    1. Load per-benchmark scores as directed by benchmark_metadata.csv,
       plus any extra_scores
    2. Normalize with each benchmark's random baseline and score ceiling,
       clipping out-of-range scores into [0, 1]
    3. Map model versions to model groups via model_metadata.csv
    4. Drop rows dated before min_date
    5. Apply benchmark supersession (e.g. FrontierMath v2 replaces v1
       tier-for-tier where a group has both)
    6. Keep one row per (model group, benchmark): the version with the best
       score, warning about groups whose versions have different dates
    7. Filter models with too few benchmarks (and optionally with too few
       benchmarks scored inside mid_range)

    Args:
        source: URL or path of benchmark_data.zip
        cache_dir: Optional directory to cache downloaded files
        min_benchmarks_per_model: Minimum distinct benchmarks required per model
        min_mid_range_benchmarks_per_model: Minimum distinct benchmarks whose
            (normalized, clipped, best-version) score lies in
            mid_range; qualifying models keep all their scores. 0 disables it.
        mid_range: Inclusive (low, high) score range for the rule above.
        min_date: Rows dated before this are dropped
        include_benchmarks: If provided, only include these benchmarks (by name).
        exclude_benchmarks: If provided, exclude these benchmarks (by name).
            Cannot be used together with include_benchmarks.
        extra_scores: Optional additional score rows (model_version, benchmark,
            performance, source) with raw scores, normalized here like rows
            from the benchmark's source file. Model versions absent from the
            model metadata are treated as their own group dated today. Used by
            Epoch's pipeline for staging (pre-release) models.
        benchmark_metadata: Optional benchmark table overriding the zip's copy.
        model_metadata: Optional model version table overriding the zip's copy.

    Returns:
        DataFrame with one row per (model group, benchmark)

    Raises:
        ValueError: If both include_benchmarks and exclude_benchmarks are
            specified, or if any specified benchmark names are not recognized.
    """
    if include_benchmarks is not None and exclude_benchmarks is not None:
        raise ValueError(
            "Cannot specify both include_benchmarks and exclude_benchmarks. "
            "Use one or the other."
        )

    dfs = download_benchmark_data(source, cache_dir)
    if benchmark_metadata is None:
        metadata = load_benchmark_metadata(dfs)
    else:
        metadata = _with_in_eci(benchmark_metadata)
    metadata = metadata[metadata["in_eci"]]
    if model_metadata is None:
        model_metadata = load_model_metadata(dfs)
    else:
        model_metadata = model_metadata.copy()
        model_metadata["date"] = pd.to_datetime(model_metadata["date"], errors="coerce")
    # Only the mapping columns take part in the fit; display/analysis columns
    # must not collide with score columns downstream.
    model_table = model_metadata[["model_version", "model_group", "date"]]

    all_benchmarks = set(metadata["benchmark"])
    for names, label in ((include_benchmarks, "include_benchmarks"),
                         (exclude_benchmarks, "exclude_benchmarks")):
        if names is not None:
            unknown = names - all_benchmarks
            if unknown:
                raise ValueError(
                    f"Unknown benchmark names in {label}: {sorted(unknown)}. "
                    f"Use get_all_benchmark_names() to see available options."
                )
    if include_benchmarks is not None:
        metadata = metadata[metadata["benchmark"].isin(include_benchmarks)]
    elif exclude_benchmarks is not None:
        metadata = metadata[~metadata["benchmark"].isin(exclude_benchmarks)]

    scores = _load_raw_scores(dfs, metadata)

    if extra_scores is not None and len(extra_scores):
        extra = extra_scores.copy()
        extra["benchmark"] = extra["benchmark"].astype(str).str.strip()
        known = extra["benchmark"].isin(set(metadata["benchmark"]))
        if (~known).any():
            warnings.warn(
                f"Dropping extra_scores rows with unknown benchmarks: "
                f"{sorted(extra.loc[~known, 'benchmark'].unique())}"
            )
        extra = extra[known]
        if "source" not in extra.columns:
            extra["source"] = pd.NA
        # Extra rows carry raw scores, normalized below with the rest.
        scores = pd.concat(
            [scores, extra[["model_version", "benchmark", "performance", "source"]]],
            ignore_index=True,
        )
        # Versions the model metadata doesn't know become their own group,
        # and known versions with no date (e.g. a staging model already in
        # Airtable with its dates unset) keep their group; both are dated
        # today so no date filter can drop them.
        today = pd.Timestamp.now().normalize()
        undated = (model_table["model_version"].isin(set(extra["model_version"]))
                   & model_table["date"].isna())
        model_table.loc[undated, "date"] = today
        missing = set(extra["model_version"]) - set(model_table["model_version"])
        if missing:
            model_table = pd.concat([model_table, pd.DataFrame({
                "model_version": sorted(missing),
                "model_group": sorted(missing),
                "date": today,
            })], ignore_index=True)

    scores = _normalize_scores(scores, metadata)
    scores = scores.merge(model_table, on="model_version", how="inner")

    # Rows dated before min_date (or with no date) never enter the fit; all
    # decisions below are made on the data that does.
    scores = scores[scores["date"] >= pd.Timestamp(min_date)]

    # Supersession: where a model group has scores on a replacement benchmark,
    # drop its scores on the benchmark it replaces so the pair isn't
    # double-counted as two separate items.
    supersessions = metadata.dropna(subset=["superseded_by"])
    supersessions = supersessions[supersessions["superseded_by"].astype(str) != ""]
    for old_name, new_name in zip(supersessions["benchmark"], supersessions["superseded_by"]):
        groups_with_new = set(scores.loc[scores["benchmark"] == new_name, "model_group"])
        drop = (scores["benchmark"] == old_name) & scores["model_group"].isin(groups_with_new)
        scores = scores[~drop]

    # One row per (model group, benchmark): the version with the best score,
    # the most recent of any tied, so its model_version and source describe
    # the score kept. The row's date is the latest of the pair's versions.
    group_dates = scores.groupby("model_group")["date"].nunique()
    scores = scores.sort_values(["model_group", "date"], ascending=[True, False])
    pairs = scores.groupby(["model_group", "benchmark"], sort=False)
    latest = pairs["date"].max()
    scores = scores.loc[pairs["performance"].idxmax()].assign(date=latest.to_numpy())

    counts = scores.groupby("model_group")["benchmark"].nunique()
    valid_models = counts[counts >= min_benchmarks_per_model].index
    scores = scores[scores["model_group"].isin(valid_models)]
    scores = filter_mid_range_models(
        scores, min_mid_range_benchmarks_per_model, mid_range, model_col="model_group"
    )

    mixed = sorted(set(group_dates[group_dates > 1].index) & set(scores["model_group"]))
    if mixed:
        warnings.warn(
            f"{len(mixed)} model groups combine versions with different release "
            f"dates; each row is dated by the latest version scored on that "
            f"benchmark: {mixed}"
        )

    release_dates = metadata.set_index("benchmark")["release_date"]
    scores["benchmark_release_date"] = pd.to_datetime(scores["benchmark"].map(release_dates))

    benchmark_ids = {b: f"b{i+1}" for i, b in enumerate(scores["benchmark"].unique())}
    model_ids = {m: f"m{i+1}" for i, m in enumerate(scores["model_group"].unique())}
    scores["benchmark_id"] = scores["benchmark"].map(benchmark_ids)
    scores["model_id"] = scores["model_group"].map(model_ids)
    scores["model"] = scores["model_group"]
    scores["Model"] = scores["model_group"]

    return scores.sort_values(["model_id", "benchmark_id"]).reset_index(drop=True)[[
        "model_id", "benchmark_id", "performance", "benchmark",
        "benchmark_release_date",
        "model", "model_version", "Model", "date", "source"
    ]]
