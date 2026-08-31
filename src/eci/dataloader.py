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
    """Extract CSV files from zipfile into DataFrames."""
    dfs = {}
    for name in zf.namelist():
        if name.endswith(".csv") and not name.startswith("additional_eci_data/"):
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
    # Rows with in_eci false describe benchmarks the site displays but the
    # fit does not use (they still carry baselines and ceilings).
    if "in_eci" not in metadata.columns:
        metadata["in_eci"] = True
    metadata["in_eci"] = metadata["in_eci"].fillna(True).astype(bool)
    return metadata


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


def _load_scores(dfs: dict[str, pd.DataFrame], metadata: pd.DataFrame) -> pd.DataFrame:
    """Load and normalize per-benchmark scores as directed by the metadata."""
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

        raw = pd.to_numeric(df["performance"], errors="coerce") * cfg.scale
        # Normalize so the random-guessing baseline maps to 0 and the
        # max-achievable score (the ceiling, defaulting to 1.0) maps to 1.
        df["performance"] = (raw - cfg.random_baseline) / (cfg.score_ceiling - cfg.random_baseline)
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


def prepare_benchmark_data(
    source: str | Path = BENCHMARK_DATA_URL,
    cache_dir: Optional[Path] = None,
    min_benchmarks_per_model: int = 4,
    min_date: str = "2023-01-01",
    include_benchmarks: Optional[set[str]] = None,
    exclude_benchmarks: Optional[set[str]] = None,
    extra_scores: Optional[pd.DataFrame] = None,
    benchmark_metadata: Optional[pd.DataFrame] = None,
    model_metadata: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Load and process benchmark data for ECI fitting.

    1. Load per-benchmark scores as directed by benchmark_metadata.csv
    2. Normalize with each benchmark's random baseline and score ceiling,
       clipping out-of-range scores into [0, 1]
    3. Map model versions to model groups via model_metadata.csv
    4. Drop rows dated before min_date
    5. Apply benchmark supersession (e.g. FrontierMath v2 replaces v1
       tier-for-tier where a group has both)
    6. Filter models with too few benchmarks
    7. Aggregate by (model group, benchmark) taking max performance

    Args:
        source: URL or path of benchmark_data.zip
        cache_dir: Optional directory to cache downloaded files
        min_benchmarks_per_model: Minimum distinct benchmarks required per model
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
        metadata = benchmark_metadata.copy()
        if "in_eci" not in metadata.columns:
            metadata["in_eci"] = True
    metadata = metadata[metadata["in_eci"].fillna(True).astype(bool)]
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

    scores = _load_scores(dfs, metadata)

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
        # Extra rows carry raw scores; normalize them exactly like rows read
        # from the benchmark's source file.
        cfg = metadata.set_index("benchmark")
        raw = (pd.to_numeric(extra["performance"], errors="coerce")
               * extra["benchmark"].map(cfg["scale"]))
        baseline = extra["benchmark"].map(cfg["random_baseline"])
        ceiling = extra["benchmark"].map(cfg["score_ceiling"])
        extra["performance"] = (raw - baseline) / (ceiling - baseline)
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

    # Sub-baseline scores clip up to 0, above-ceiling scores clip down to 1;
    # NaN performances (failed numeric coercion or degenerate config rows)
    # are removed.
    scores["performance"] = scores["performance"].replace([np.inf, -np.inf], np.nan)
    scores = scores[scores["performance"].notna()]
    scores["performance"] = scores["performance"].clip(lower=0.0, upper=1.0)

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

    # Sort so "first" aggregations prefer the most recent version metadata
    scores = scores.sort_values(["model_group", "date"], ascending=[True, False])

    counts = scores.groupby("model_group")["benchmark"].nunique()
    valid_models = counts[counts >= min_benchmarks_per_model].index
    scores = scores[scores["model_group"].isin(valid_models)]

    release_dates = metadata.set_index("benchmark")["release_date"]
    scores["benchmark_release_date"] = pd.to_datetime(scores["benchmark"].map(release_dates))

    benchmark_ids = {b: f"b{i+1}" for i, b in enumerate(scores["benchmark"].unique())}
    model_ids = {m: f"m{i+1}" for i, m in enumerate(scores["model_group"].unique())}
    scores["benchmark_id"] = scores["benchmark"].map(benchmark_ids)
    scores["model_id"] = scores["model_group"].map(model_ids)

    aggregated = scores.groupby(["model_id", "benchmark_id"]).agg(
        performance=("performance", "max"),
        benchmark=("benchmark", "first"),
        benchmark_release_date=("benchmark_release_date", "first"),
        model=("model_group", "first"),
        model_version=("model_version", "first"),
        Model=("model_group", "first"),
        date=("date", "max"),
        source=("source", "first"),
    ).reset_index()

    return aggregated[[
        "model_id", "benchmark_id", "performance", "benchmark",
        "benchmark_release_date",
        "model", "model_version", "Model", "date", "source"
    ]]
