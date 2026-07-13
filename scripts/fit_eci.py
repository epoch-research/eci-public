#!/usr/bin/env python3
"""
Fit ECI model and save scores to outputs directory.

Usage:
    python scripts/fit_eci.py
    python scripts/fit_eci.py --input path/to/benchmarks.csv
    python scripts/fit_eci.py --bootstrap-samples 200
    python scripts/fit_eci.py --numeric-jacobian
"""

import argparse
from pathlib import Path

import pandas as pd

from eci import load_benchmark_data, fit_eci_model, compute_eci_scores, BOOTSTRAP_METHODS


DEFAULT_INPUT_URL = "https://epoch.ai/data/eci_benchmarks.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Input columns carried over to the ECI output when present (the fit itself
# returns bare statistical results)
MODEL_METADATA_COLS = ["date", "Organization", "model_version", "source"]


def join_model_metadata(eci_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-model metadata columns from the input data, first value wins."""
    cols = [c for c in MODEL_METADATA_COLS if c in df.columns]
    if not cols:
        return eci_df
    metadata = df.drop_duplicates("model_id").set_index("model_id")[cols]
    return eci_df.join(metadata, on="model_id")


def main():
    parser = argparse.ArgumentParser(description="Fit ECI model and save scores")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_URL,
        help=f"Path or URL to benchmark data CSV (default: {DEFAULT_INPUT_URL})",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=500,
        help="Number of bootstrap samples for confidence intervals (default: 500)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for scores (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--numeric-jacobian",
        action="store_true",
        help="Use numerical Jacobian instead of analytical (slower, matches website exactly)",
    )
    parser.add_argument(
        "--bootstrap-method",
        choices=BOOTSTRAP_METHODS,
        default="hierarchical",
        help="Bootstrap resampling scheme for confidence intervals (default: hierarchical)",
    )
    args = parser.parse_args()

    print(f"Loading benchmark data from {args.input}...")
    df = load_benchmark_data(args.input)
    print(f"  Loaded {len(df)} performance records")
    print(f"  {df['model_id'].nunique()} models, {df['benchmark_id'].nunique()} benchmarks")

    jacobian_type = "numerical" if args.numeric_jacobian else "analytical"
    print(f"\nFitting IRT model ({jacobian_type} Jacobian, {args.bootstrap_samples} "
          f"'{args.bootstrap_method}' bootstrap samples)...")
    model_df, bench_df, bootstrap_data = fit_eci_model(
        df,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=12345,
        bootstrap_method=args.bootstrap_method,
        use_analytical_jacobian=not args.numeric_jacobian,
    )

    print("Computing ECI/EDI scores...")
    results = compute_eci_scores(model_df, bench_df, bootstrap_data)
    if results.diagnostics["n_draws_dropped"]:
        print(f"  WARNING: {results.diagnostics['n_draws_dropped']} of "
              f"{results.diagnostics['n_draws_total']} bootstrap draws dropped:")
        for reason in results.diagnostics["dropped_reasons"]:
            print(f"    {reason}")

    eci_df = join_model_metadata(results.eci_df, df)

    # Prepare output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save ECI scores
    eci_output = args.output_dir / "eci_scores.csv"
    eci_cols = ["Model", "eci"]
    if "eci_ci_low" in eci_df.columns:
        eci_cols += ["eci_ci_low", "eci_ci_high"]
    eci_cols += [col for col in MODEL_METADATA_COLS if col in eci_df.columns]
    eci_df[eci_cols].to_csv(eci_output, index=False)
    print(f"\nSaved ECI scores to {eci_output}")

    # Save EDI scores
    edi_df = results.edi_df
    if "benchmark_release_date" in df.columns:
        release_dates = df.drop_duplicates("benchmark_id").set_index("benchmark_id")["benchmark_release_date"]
        edi_df = edi_df.join(release_dates, on="benchmark_id")
    edi_output = args.output_dir / "edi_scores.csv"
    edi_cols = ["benchmark", "edi", "discriminability_scaled", "is_anchor"]
    if "benchmark_release_date" in edi_df.columns:
        edi_cols.insert(3, "benchmark_release_date")
    if "edi_ci_low" in edi_df.columns:
        edi_cols += ["edi_ci_low", "edi_ci_high"]
    edi_df[edi_cols].to_csv(edi_output, index=False)
    print(f"Saved EDI scores to {edi_output}")

    # Print summary
    print("\n=== Top 10 Models by ECI ===")
    print(eci_df[["Model", "eci"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
