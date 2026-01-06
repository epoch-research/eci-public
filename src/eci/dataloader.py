"""
Data loader for ECI benchmark data.

This module loads raw benchmark data from https://epoch.ai/data/benchmark_data.zip
and processes it into the format needed for ECI fitting.
"""

import io
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import pandas as pd

BENCHMARK_DATA_URL = "https://epoch.ai/data/benchmark_data.zip"

# Internal benchmarks (evaluated by Epoch)
INTERNAL_BENCHMARKS = {
    "gpqa_diamond.csv": "GPQA diamond",
    "math_level_5.csv": "MATH level 5",
    "otis_mock_aime_2024_2025.csv": "OTIS Mock AIME 2024-2025",
    "frontiermath.csv": "FrontierMath-2025-02-28-Private",
    "frontiermath_tier_4.csv": "FrontierMath-Tier-4-2025-07-01-Private",
    "simpleqa_verified.csv": "SimpleQA Verified",
    "chess_puzzles.csv": "Chess Puzzles",
    # Note: swe_bench_verified.csv exists but we use swe_bench_bash.csv instead
}

# External benchmarks with their score column mappings
EXTERNAL_BENCHMARKS = {
    "aider_polyglot_external.csv": {"name": "Aider polyglot", "score_col": "Percent correct", "scale": 1/100},
    "adversarial_nli_external.csv": {"name": "ANLI", "score_col": "Score"},
    "arc_agi_external.csv": {"name": "ARC-AGI", "score_col": "Score"},
    "arc_ai2_external.csv": {"name": "ARC AI2", "score_col": "Challenge score"},
    "balrog_external.csv": {"name": "Balrog", "score_col": "Average progress"},
    "bbh_external.csv": {"name": "BBH", "score_col": "Average"},
    "cad_eval_external.csv": {"name": "CadEval", "score_col": "Overall pass (%)"},
    "common_sense_qa_2_external.csv": {"name": "CSQA2", "score_col": "Score"},
    "cybench_external.csv": {"name": "Cybench", "score_col": "Unguided % Solved"},
    "deepresearchbench_external.csv": {"name": "DeepResearch Bench", "score_col": "Average score"},
    "fictionlivebench_external.csv": {"name": "Fiction.LiveBench", "score_col": "16k token score"},
    "geobench_external.csv": {"name": "GeoBench", "score_col": "ACW Country %"},
    "gsm8k_external.csv": {"name": "GSM8K", "score_col": "EM"},
    "gso_external.csv": {"name": "GSO-Bench", "score_col": "Score OPT@1"},
    "hella_swag_external.csv": {"name": "HellaSwag", "score_col": "Overall accuracy"},
    "lambada_external.csv": {"name": "LAMBADA", "score_col": "Score"},
    "lech_mazur_writing_external.csv": {"name": "Lech Mazur Writing", "score_col": "Mean score", "scale": 1/10},
    "mmlu_external.csv": {"name": "MMLU", "score_col": "EM"},
    "open_book_qa_external.csv": {"name": "OpenBookQA", "score_col": "Accuracy"},
    "os_world_external.csv": {"name": "OSWorld", "score_col": "Score", "scale": 1/100},
    "piqa_external.csv": {"name": "PIQA", "score_col": "Score"},
    "science_qa_external.csv": {"name": "ScienceQA", "score_col": "Score"},
    "simplebench_external.csv": {"name": "SimpleBench", "score_col": "Score (AVG@5)"},
    "swe_bench_bash.csv": {"name": "SWE-Bench Verified (Bash Only)", "score_col": "% Resolved"},
    "terminalbench_external.csv": {"name": "Terminal Bench", "score_col": "Accuracy mean"},
    "the_agent_company_external.csv": {"name": "The Agent Company", "score_col": "% Resolved"},
    "trivia_qa_external.csv": {"name": "TriviaQA", "score_col": "EM"},
    "video_mme_external.csv": {"name": "VideoMME", "score_col": "Overall (no subtitles)"},
    "vpct_external.csv": {"name": "VPCT", "score_col": "Correct"},
    "weirdml_external.csv": {"name": "WeirdML", "score_col": "Accuracy"},
    "wino_grande_external.csv": {"name": "Winogrande", "score_col": "Accuracy"},
}

# Random baseline for chance-level correction
RANDOM_BASELINES = {
    "GPQA diamond": 0.25,
    "FrontierMath-2025-02-28-Private": 0.0,
    "FrontierMath-Tier-4-2025-07-01-Private": 0.0,
    "MATH level 5": 0.0,
    "OTIS Mock AIME 2024-2025": 0.001,
    "SWE-Bench Verified (Bash Only)": 0.0,
    "Aider polyglot": 0.0,
    "ANLI": 1/3,
    "ARC-AGI": 0.0,
    "ARC AI2": 0.25,
    "Balrog": 0.0,
    "BBH": 0.25,
    "CadEval": 0.0,
    "CSQA2": 0.5,
    "Cybench": 0.0,
    "DeepResearch Bench": 0.0,
    "Fiction.LiveBench": 0.0,
    "GeoBench": 0.0,
    "GSM8K": 0.0,
    "GSO-Bench": 0.0,
    "HellaSwag": 0.25,
    "LAMBADA": 0.0,
    "Lech Mazur Writing": 0.0,
    "MMLU": 0.25,
    "OpenBookQA": 0.25,
    "OSWorld": 0.0,
    "PIQA": 0.5,
    "ScienceQA": 0.25,
    "SimpleBench": 1/6,
    "Terminal Bench": 0.0,
    "TriviaQA": 0.0,
    "VideoMME": 0.25,
    "VPCT": 0.0,
    "WeirdML": 0.0,
    "Winogrande": 0.5,
    "The Agent Company": 0.0,
}

# Benchmark metadata
BENCHMARK_METADATA = {
    "ARC AI2": {"release_date": "2018-03-14", "is_math": False, "is_coding": False},
    "Chess Puzzles": {"release_date": "2025-01-01", "is_math": False, "is_coding": False},
    "SimpleQA Verified": {"release_date": "2024-10-30", "is_math": False, "is_coding": False},
    "BBH": {"release_date": "2022-10-17", "is_math": False, "is_coding": False},
    "GSM8K": {"release_date": "2021-10-27", "is_math": True, "is_coding": False},
    "HellaSwag": {"release_date": "2019-05-19", "is_math": False, "is_coding": False},
    "LAMBADA": {"release_date": "2016-06-20", "is_math": False, "is_coding": False},
    "MMLU": {"release_date": "2020-09-07", "is_math": False, "is_coding": False},
    "GPQA diamond": {"release_date": "2023-11-20", "is_math": False, "is_coding": False},
    "MATH level 5": {"release_date": "2021-03-05", "is_math": True, "is_coding": False},
    "OTIS Mock AIME 2024-2025": {"release_date": "2024-12-19", "is_math": True, "is_coding": False},
    "WeirdML": {"release_date": "2025-01-16", "is_math": False, "is_coding": True},
    "Winogrande": {"release_date": "2019-07-24", "is_math": False, "is_coding": False},
    "PIQA": {"release_date": "2019-11-26", "is_math": False, "is_coding": False},
    "TriviaQA": {"release_date": "2017-05-09", "is_math": False, "is_coding": False},
    "OpenBookQA": {"release_date": "2018-09-08", "is_math": False, "is_coding": False},
    "ScienceQA": {"release_date": "2022-09-20", "is_math": False, "is_coding": False},
    "ANLI": {"release_date": "2019-10-31", "is_math": False, "is_coding": False},
    "FrontierMath-2025-02-28-Private": {"release_date": "2024-11-08", "is_math": True, "is_coding": False},
    "FrontierMath-Tier-4-2025-07-01-Private": {"release_date": "2024-11-08", "is_math": True, "is_coding": False},
    "Aider polyglot": {"release_date": "2024-12-21", "is_math": False, "is_coding": True},
    "SWE-Bench Verified (Bash Only)": {"release_date": "2024-04-01", "is_math": False, "is_coding": True},
    "ARC-AGI": {"release_date": "2019-11-05", "is_math": False, "is_coding": False},
    "Balrog": {"release_date": "2025-03-20", "is_math": False, "is_coding": False},
    "VideoMME": {"release_date": "2024-05-31", "is_math": False, "is_coding": False},
    "GeoBench": {"release_date": "2025-03-01", "is_math": False, "is_coding": False},
    "Fiction.LiveBench": {"release_date": "2025-02-21", "is_math": False, "is_coding": False},
    "Lech Mazur Writing": {"release_date": "2025-01-31", "is_math": False, "is_coding": False},
    "SimpleBench": {"release_date": "2024-11-04", "is_math": False, "is_coding": False},
    "Terminal Bench": {"release_date": "2025-02-12", "is_math": False, "is_coding": True},
    "The Agent Company": {"release_date": "2024-12-18", "is_math": False, "is_coding": True},
    "GSO-Bench": {"release_date": "2025-03-05", "is_math": True, "is_coding": False},
    "CadEval": {"release_date": "2025-04-08", "is_math": False, "is_coding": True},
    "CSQA2": {"release_date": "2020-04-01", "is_math": False, "is_coding": False},
    "Cybench": {"release_date": "2024-08-14", "is_math": False, "is_coding": True},
    "OSWorld": {"release_date": "2024-04-11", "is_math": False, "is_coding": True},
    "VPCT": {"release_date": "2025-04-30", "is_math": False, "is_coding": False},
    "DeepResearch Bench": {"release_date": "2025-04-08", "is_math": False, "is_coding": False},
}


def get_all_benchmark_names() -> set[str]:
    """
    Get the names of all available benchmarks.

    Returns:
        Set of benchmark names from both internal and external sources.
    """
    internal_names = set(INTERNAL_BENCHMARKS.values())
    external_names = {spec["name"] for spec in EXTERNAL_BENCHMARKS.values()}
    return internal_names | external_names


def download_benchmark_data(url: str = BENCHMARK_DATA_URL, cache_dir: Optional[Path] = None) -> dict[str, pd.DataFrame]:
    """
    Download and extract benchmark data from zip file.

    Args:
        url: URL to benchmark_data.zip
        cache_dir: Optional directory to cache downloaded files

    Returns:
        Dictionary mapping filename to DataFrame
    """
    if cache_dir:
        cache_path = cache_dir / "benchmark_data.zip"
        if cache_path.exists():
            with zipfile.ZipFile(cache_path, "r") as zf:
                return _extract_csvs(zf)

    with urlopen(url) as response:
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


def load_model_versions(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Load model version mapping from epoch_capabilities_index.csv.

    Returns DataFrame with columns: model_version, Model, date
    """
    eci = dfs["epoch_capabilities_index.csv"]
    versions = eci[["Model version", "Model name", "Release date"]].copy()
    versions = versions.rename(columns={
        "Model version": "model_version",
        "Model name": "Model",
        "Release date": "date",
    })
    versions["date"] = pd.to_datetime(versions["date"], errors="coerce")
    return versions.dropna(subset=["model_version"])


def load_internal_benchmarks(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Load scores from internal (Epoch-evaluated) benchmarks."""
    frames = []
    for filename, benchmark_name in INTERNAL_BENCHMARKS.items():
        if filename not in dfs:
            continue
        df = dfs[filename][["Model version", "Best score (across scorers)"]].copy()
        df = df.rename(columns={
            "Model version": "model_version",
            "Best score (across scorers)": "performance",
        })
        df["benchmark"] = benchmark_name
        df["source"] = "Epoch evaluations"
        df = df.dropna(subset=["performance"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_external_benchmarks(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Load scores from external benchmarks."""
    frames = []
    for filename, spec in EXTERNAL_BENCHMARKS.items():
        if filename not in dfs:
            continue
        df = dfs[filename]

        cols = ["Model version", spec["score_col"]]
        if "Source" in df.columns:
            cols.append("Source")

        df = df[cols].copy()
        rename_map = {
            "Model version": "model_version",
            spec["score_col"]: "performance",
        }
        if "Source" in df.columns:
            rename_map["Source"] = "source"
        df = df.rename(columns=rename_map)

        # Only drop rows missing essential columns (model_version or performance)
        df = df.dropna(subset=["model_version", "performance"])

        if "source" not in df.columns:
            df["source"] = pd.NA

        # Apply scaling if specified
        scale = spec.get("scale", 1.0)
        df["performance"] = pd.to_numeric(df["performance"], errors="coerce") * scale

        df["benchmark"] = spec["name"]
        frames.append(df[["model_version", "benchmark", "performance", "source"]])

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def apply_random_baseline_correction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply random baseline correction to normalize scores.

    Transforms scores from [baseline, 1] to [0, 1] range.
    """
    df = df.copy()
    df["random_baseline"] = df["benchmark"].map(RANDOM_BASELINES).fillna(0.0)
    df["performance"] = (df["performance"] - df["random_baseline"]) / (1.0 - df["random_baseline"])
    return df


def add_benchmark_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add benchmark release dates and category flags."""
    df = df.copy()

    def get_metadata(benchmark, key, default):
        return BENCHMARK_METADATA.get(benchmark, {}).get(key, default)

    df["benchmark_release_date"] = df["benchmark"].apply(
        lambda b: get_metadata(b, "release_date", None)
    )
    df["benchmark_release_date"] = pd.to_datetime(df["benchmark_release_date"], errors="coerce")
    df["is_math"] = df["benchmark"].apply(lambda b: get_metadata(b, "is_math", False))
    df["is_coding"] = df["benchmark"].apply(lambda b: get_metadata(b, "is_coding", False))
    df["optimized"] = True  # All models in ECI are considered "optimized" for their benchmarks

    return df


def prepare_benchmark_data(
    url: str = BENCHMARK_DATA_URL,
    cache_dir: Optional[Path] = None,
    min_benchmarks_per_model: int = 4,
    min_date: str = "2023-01-01",
    include_benchmarks: Optional[set[str]] = None,
    exclude_benchmarks: Optional[set[str]] = None,
) -> pd.DataFrame:
    """
    Load and process benchmark data for ECI fitting.

    This replicates the processing from eci_benchmarks.csv:
    1. Load internal and external benchmark scores
    2. Apply random baseline correction
    3. Merge with model version metadata
    4. Filter models with too few benchmarks
    5. Aggregate by (Model, benchmark) taking max performance

    Args:
        url: URL to benchmark_data.zip
        cache_dir: Optional directory to cache downloaded files
        min_benchmarks_per_model: Minimum benchmarks required per model
        min_date: Minimum model release date to include
        include_benchmarks: If provided, only include these benchmarks (by name).
            Use get_all_benchmark_names() to see available options.
        exclude_benchmarks: If provided, exclude these benchmarks (by name).
            Cannot be used together with include_benchmarks.

    Returns:
        DataFrame matching the format of eci_benchmarks.csv

    Raises:
        ValueError: If both include_benchmarks and exclude_benchmarks are specified,
            or if any specified benchmark names are not recognized.
    """
    # Validate benchmark filtering parameters
    if include_benchmarks is not None and exclude_benchmarks is not None:
        raise ValueError(
            "Cannot specify both include_benchmarks and exclude_benchmarks. "
            "Use one or the other."
        )

    all_benchmarks = get_all_benchmark_names()

    if include_benchmarks is not None:
        unknown = include_benchmarks - all_benchmarks
        if unknown:
            raise ValueError(
                f"Unknown benchmark names in include_benchmarks: {sorted(unknown)}. "
                f"Use get_all_benchmark_names() to see available options."
            )

    if exclude_benchmarks is not None:
        unknown = exclude_benchmarks - all_benchmarks
        if unknown:
            raise ValueError(
                f"Unknown benchmark names in exclude_benchmarks: {sorted(unknown)}. "
                f"Use get_all_benchmark_names() to see available options."
            )

    # Download data
    dfs = download_benchmark_data(url, cache_dir)

    # Load model version mapping
    versions = load_model_versions(dfs)

    # Load benchmark scores
    internal = load_internal_benchmarks(dfs)
    external = load_external_benchmarks(dfs)
    scores = pd.concat([internal, external], ignore_index=True)

    # Apply benchmark filtering
    if include_benchmarks is not None:
        scores = scores[scores["benchmark"].isin(include_benchmarks)]
    elif exclude_benchmarks is not None:
        scores = scores[~scores["benchmark"].isin(exclude_benchmarks)]

    # Apply random baseline correction
    scores = apply_random_baseline_correction(scores)

    # Filter valid performance values
    scores = scores[
        (scores["performance"] >= 0) &
        (scores["performance"] <= 1) &
        scores["performance"].notna()
    ]

    # Merge with model versions
    scores = scores.merge(versions, on="model_version", how="inner")

    # Filter by date (keep rows with missing dates)
    min_date_ts = pd.Timestamp(min_date)
    scores = scores[(scores["date"] >= min_date_ts) | scores["date"].isna()]

    # Sort so most recent comes first (for aggregation)
    scores = scores.sort_values(["Model", "date"], ascending=[True, False])

    # Filter models with enough benchmarks
    benchmark_counts = scores.groupby("Model")["benchmark"].nunique()
    valid_models = benchmark_counts[benchmark_counts >= min_benchmarks_per_model].index
    scores = scores[scores["Model"].isin(valid_models)]

    # Add benchmark metadata
    scores = add_benchmark_metadata(scores)

    # Create IDs
    benchmark_ids = {b: f"b{i+1}" for i, b in enumerate(scores["benchmark"].unique())}
    model_ids = {m: f"m{i+1}" for i, m in enumerate(scores["Model"].unique())}

    scores["benchmark_id"] = scores["benchmark"].map(benchmark_ids)
    scores["model_id"] = scores["Model"].map(model_ids)

    # Aggregate by (Model, benchmark) - take max performance
    aggregated = scores.groupby(["model_id", "benchmark_id"]).agg(
        performance=("performance", "max"),
        benchmark=("benchmark", "first"),
        benchmark_release_date=("benchmark_release_date", "first"),
        optimized=("optimized", "first"),
        is_math=("is_math", "first"),
        is_coding=("is_coding", "first"),
        model=("Model", "first"),  # model column = Model value
        model_version=("model_version", "first"),
        Model=("Model", "first"),
        date=("date", "max"),
        source=("source", "first"),
    ).reset_index()

    # Reorder columns to match expected format
    return aggregated[[
        "model_id", "benchmark_id", "performance", "benchmark",
        "benchmark_release_date", "optimized", "is_math", "is_coding",
        "model", "model_version", "Model", "date", "source"
    ]]
