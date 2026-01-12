#!/usr/bin/env python3
"""
Fit ECI model for different benchmark baskets.

This script fits separate ECI models for domain-specific benchmark subsets:
- SWE-ECI: Software engineering benchmarks
- Knowledge-ECI: Knowledge and reasoning benchmarks
- Math-ECI: Mathematics benchmarks

Two fitting modes are available:

1. Projection mode (default):
   - First fits a full model on ALL available benchmarks
   - Then for each basket, freezes the benchmark parameters (difficulty, slope)
   - Fits only model capabilities to best explain basket performance scores
   - This approach uses more data to estimate benchmark characteristics

2. Refit mode (--refit flag):
   - Fits each basket from scratch independently
   - Estimates both model capabilities AND benchmark parameters per basket
   - Uses only the benchmarks in that basket for fitting

Usage:
    python scripts/fit_baskets.py                    # projection mode (default)
    python scripts/fit_baskets.py --refit            # refit mode
    python scripts/fit_baskets.py --baskets swe math
    python scripts/fit_baskets.py --bootstrap-samples 200
"""

import argparse
from pathlib import Path

import pandas as pd

from eci.dataloader import prepare_benchmark_data
from eci.fitting import fit_eci_model, compute_eci_scores, fit_capabilities_given_benchmarks


OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Define benchmark baskets with their configurations
BASKETS = {
    "swe": {
        "name": "SWE-ECI",
        "benchmarks": {
            "SWE-Bench Verified (Bash Only)",
            "Aider polyglot",
            "GSO-Bench",
            "WeirdML",
            "CadEval",
            "Terminal Bench",
            "Cybench",
        },
        "anchor_benchmark": "SWE-Bench Verified (Bash Only)",
    },
    "knowledge": {
        "name": "Knowledge-ECI",
        "benchmarks": {
            "TriviaQA",
            "GPQA diamond",
            "ARC AI2",
            "MMLU",
            "OpenBookQA",
            "SimpleQA Verified",
            "ScienceQA",
        },
        "anchor_benchmark": "GPQA diamond",
    },
    "math": {
        "name": "Math-ECI",
        "benchmarks": {
            "FrontierMath-2025-02-28-Private",
            "FrontierMath-Tier-4-2025-07-01-Private",
            "MATH level 5",
            "OTIS Mock AIME 2024-2025",
            "GSM8K",
        },
        "anchor_benchmark": "MATH level 5",
    },
}


def fit_full_model(
    bootstrap_samples: int = 100,
    min_benchmarks_per_model: int = 3,
    use_analytical_jacobian: bool = True,
) -> tuple:
    """
    Fit ECI model on ALL available benchmarks.

    Args:
        bootstrap_samples: Number of bootstrap samples for confidence intervals
        min_benchmarks_per_model: Minimum benchmarks required per model
        use_analytical_jacobian: Use analytical Jacobian for faster optimization

    Returns:
        Tuple of (model_df, bench_df, full_df) - model capabilities, benchmark params, and raw data
    """
    print(f"\n{'='*60}")
    print("Fitting full ECI model on all benchmarks")
    print(f"{'='*60}")

    # Load ALL benchmark data (no filtering)
    print(f"\nLoading all benchmark data...")
    df = prepare_benchmark_data(
        cache_dir=Path(".cache"),
        min_benchmarks_per_model=min_benchmarks_per_model,
    )

    print(f"  Loaded {len(df)} performance records")
    print(f"  {df['model_id'].nunique()} models, {df['benchmark_id'].nunique()} benchmarks")

    # Fit the full model
    jacobian_type = "analytical" if use_analytical_jacobian else "numerical"
    print(f"\nFitting IRT model ({jacobian_type} Jacobian, {bootstrap_samples} bootstrap samples)...")

    model_df, bench_df = fit_eci_model(
        df,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=12345,
        use_analytical_jacobian=use_analytical_jacobian,
    )

    print(f"  Fitted {len(model_df)} models and {len(bench_df)} benchmarks")

    return model_df, bench_df, df


def fit_basket(
    basket_key: str,
    bootstrap_samples: int = 100,
    min_benchmarks_per_model: int = 3,
    output_dir: Path = OUTPUT_DIR,
    use_analytical_jacobian: bool = True,
    raw: bool = False,
    full_bench_df: pd.DataFrame | None = None,
    full_data_df: pd.DataFrame | None = None,
) -> tuple:
    """
    Fit ECI model for a specific benchmark basket.

    Args:
        basket_key: Key in BASKETS dict (e.g., 'swe', 'knowledge', 'math')
        bootstrap_samples: Number of bootstrap samples for confidence intervals
        min_benchmarks_per_model: Minimum benchmarks required per model
        output_dir: Directory to save output files
        use_analytical_jacobian: Use analytical Jacobian for faster optimization
        raw: If True, output raw capability/difficulty scores without ECI scaling
        full_bench_df: If provided, use these benchmark parameters (from a full model fit)
            and only fit model capabilities. This is the "projection" approach.
        full_data_df: If provided along with full_bench_df, use this data (filtered to
            basket benchmarks) instead of reloading.

    Returns:
        Tuple of (eci_df, edi_df) DataFrames
    """
    basket = BASKETS[basket_key]
    basket_name = basket["name"]
    benchmarks = basket["benchmarks"]
    anchor_benchmark = basket["anchor_benchmark"]

    # Determine mode
    projection_mode = full_bench_df is not None

    print(f"\n{'='*60}")
    print(f"Fitting {basket_name}" + (" (projection mode)" if projection_mode else " (refit mode)"))
    print(f"{'='*60}")
    print(f"Benchmarks ({len(benchmarks)}):")
    for b in sorted(benchmarks):
        marker = " (anchor)" if b == anchor_benchmark else ""
        print(f"  - {b}{marker}")

    if projection_mode:
        # Use pre-loaded data filtered to basket benchmarks
        if full_data_df is not None:
            df = full_data_df[full_data_df["benchmark"].isin(benchmarks)].copy()
            # Re-filter models that have enough benchmarks in this basket
            benchmark_counts = df.groupby("Model")["benchmark"].nunique()
            valid_models = benchmark_counts[benchmark_counts >= min_benchmarks_per_model].index
            df = df[df["Model"].isin(valid_models)]
        else:
            # Fall back to loading data for this basket
            print(f"\nLoading benchmark data...")
            df = prepare_benchmark_data(
                cache_dir=Path(".cache"),
                include_benchmarks=benchmarks,
                min_benchmarks_per_model=min_benchmarks_per_model,
            )
    else:
        # Load data for this basket only (refit mode)
        print(f"\nLoading benchmark data...")
        df = prepare_benchmark_data(
            cache_dir=Path(".cache"),
            include_benchmarks=benchmarks,
            min_benchmarks_per_model=min_benchmarks_per_model,
        )

    if len(df) == 0:
        print(f"  WARNING: No data available for {basket_name}")
        return None, None

    print(f"  Using {len(df)} performance records")
    print(f"  {df['model_id'].nunique()} models, {df['benchmark_id'].nunique()} benchmarks")

    # Check which benchmarks are actually present
    present_benchmarks = set(df["benchmark"].unique())
    missing = benchmarks - present_benchmarks
    if missing:
        print(f"  WARNING: Missing benchmarks: {sorted(missing)}")

    if projection_mode:
        # Fit only capabilities using fixed benchmark parameters
        print(f"\nFitting model capabilities (benchmark params frozen, {bootstrap_samples} bootstrap samples)...")

        # Filter bench_df to basket benchmarks
        basket_bench_df = full_bench_df[full_bench_df["benchmark"].isin(benchmarks)].copy()

        model_df = fit_capabilities_given_benchmarks(
            df,
            basket_bench_df,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=12345,
        )

        # Use the filtered benchmark params for output
        bench_df = basket_bench_df.copy()
        bench_df["is_anchor"] = bench_df["benchmark"] == anchor_benchmark
    else:
        # Fit the full model from scratch (refit mode)
        jacobian_type = "analytical" if use_analytical_jacobian else "numerical"
        print(f"\nFitting IRT model ({jacobian_type} Jacobian, {bootstrap_samples} bootstrap samples)...")

        model_df, bench_df = fit_eci_model(
            df,
            anchor_benchmark=anchor_benchmark,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=12345,
            use_analytical_jacobian=use_analytical_jacobian,
        )

    # Compute ECI scores (or use raw scores)
    def use_raw_scores():
        """Convert raw capability/difficulty to output format."""
        eci = model_df.copy()
        eci["eci"] = eci["capability"]
        if "capability_ci_low" in eci.columns:
            eci["eci_ci_low"] = eci["capability_ci_low"]
            eci["eci_ci_high"] = eci["capability_ci_high"]
        edi = bench_df.copy()
        edi["edi"] = edi["difficulty"]
        edi["discriminability_scaled"] = edi["discriminability"]
        return eci, edi

    if raw:
        print("Using raw capability/difficulty scores (--raw flag)")
        eci_df, edi_df = use_raw_scores()
    else:
        print("Computing ECI/EDI scores...")
        try:
            eci_df, edi_df = compute_eci_scores(model_df, bench_df)
        except ValueError as e:
            print(f"  WARNING: Could not compute scaled ECI scores: {e}")
            print("  Returning raw capability scores instead (use --raw to suppress this warning)")
            eci_df, edi_df = use_raw_scores()

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    eci_output = output_dir / f"{basket_key}_eci_scores.csv"
    eci_cols = ["Model", "eci"]
    if "eci_ci_low" in eci_df.columns:
        eci_cols.extend(["eci_ci_low", "eci_ci_high"])
    eci_df[eci_cols].to_csv(eci_output, index=False)
    print(f"\nSaved ECI scores to {eci_output}")

    edi_output = output_dir / f"{basket_key}_edi_scores.csv"
    edi_cols = ["benchmark", "edi", "discriminability_scaled", "is_anchor"]
    if "benchmark_release_date" in edi_df.columns:
        edi_cols.insert(3, "benchmark_release_date")
    edi_df[edi_cols].to_csv(edi_output, index=False)
    print(f"Saved EDI scores to {edi_output}")

    # Print summary
    print(f"\n=== Top 10 Models by {basket_name} ===")
    print(eci_df[["Model", "eci"]].head(10).to_string(index=False))

    return eci_df, edi_df


def main():
    parser = argparse.ArgumentParser(
        description="Fit ECI model for benchmark baskets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available baskets:
  swe        Software engineering (SWE-Bench, Aider, etc.)
  knowledge  Knowledge and reasoning (MMLU, GPQA, etc.)
  math       Mathematics (FrontierMath, MATH, etc.)

Fitting modes:
  Default (projection): Fit a full model on all benchmarks first, then
    estimate model capabilities for each basket using fixed benchmark
    parameters from the full fit.

  --refit: Fit each basket from scratch, estimating both model capabilities
    and benchmark parameters using only the benchmarks in that basket.
        """,
    )
    parser.add_argument(
        "--baskets",
        nargs="+",
        choices=list(BASKETS.keys()),
        default=list(BASKETS.keys()),
        help="Which baskets to fit (default: all)",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=100,
        help="Number of bootstrap samples for confidence intervals (default: 100)",
    )
    parser.add_argument(
        "--min-benchmarks",
        type=int,
        default=3,
        help="Minimum benchmarks required per model (default: 3)",
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
        help="Use numerical Jacobian instead of analytical (slower)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Output raw capability/difficulty scores without ECI scaling",
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Refit from scratch for each basket (estimate benchmark params per basket)",
    )
    args = parser.parse_args()

    if args.refit:
        # Refit mode: fit each basket from scratch
        for basket_key in args.baskets:
            fit_basket(
                basket_key,
                bootstrap_samples=args.bootstrap_samples,
                min_benchmarks_per_model=args.min_benchmarks,
                output_dir=args.output_dir,
                use_analytical_jacobian=not args.numeric_jacobian,
                raw=args.raw,
            )
    else:
        # Projection mode (default): fit full model first, then project onto baskets
        _, full_bench_df, full_data_df = fit_full_model(
            bootstrap_samples=args.bootstrap_samples,
            min_benchmarks_per_model=args.min_benchmarks,
            use_analytical_jacobian=not args.numeric_jacobian,
        )

        for basket_key in args.baskets:
            fit_basket(
                basket_key,
                bootstrap_samples=args.bootstrap_samples,
                min_benchmarks_per_model=args.min_benchmarks,
                output_dir=args.output_dir,
                use_analytical_jacobian=not args.numeric_jacobian,
                raw=args.raw,
                full_bench_df=full_bench_df,
                full_data_df=full_data_df,
            )

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
