#!/usr/bin/env python3
"""
Fit ECI model for different benchmark baskets.

This script fits separate ECI models for domain-specific benchmark subsets:
- SWE-ECI: Software engineering benchmarks
- Knowledge-ECI: Knowledge and reasoning benchmarks
- Math-ECI: Mathematics benchmarks

Usage:
    python scripts/fit_baskets.py
    python scripts/fit_baskets.py --baskets swe knowledge
    python scripts/fit_baskets.py --bootstrap-samples 200
"""

import argparse
from pathlib import Path

from eci.dataloader import prepare_benchmark_data
from eci.fitting import fit_eci_model, compute_eci_scores


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
        "anchor_benchmark": "MMLU",
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
        "anchor_benchmark": "GSM8K",
    },
}


def fit_basket(
    basket_key: str,
    bootstrap_samples: int = 100,
    min_benchmarks_per_model: int = 3,
    output_dir: Path = OUTPUT_DIR,
    use_analytical_jacobian: bool = True,
) -> tuple:
    """
    Fit ECI model for a specific benchmark basket.

    Args:
        basket_key: Key in BASKETS dict (e.g., 'swe', 'knowledge', 'math')
        bootstrap_samples: Number of bootstrap samples for confidence intervals
        min_benchmarks_per_model: Minimum benchmarks required per model
        output_dir: Directory to save output files
        use_analytical_jacobian: Use analytical Jacobian for faster optimization

    Returns:
        Tuple of (eci_df, edi_df) DataFrames
    """
    basket = BASKETS[basket_key]
    basket_name = basket["name"]
    benchmarks = basket["benchmarks"]
    anchor_benchmark = basket["anchor_benchmark"]

    print(f"\n{'='*60}")
    print(f"Fitting {basket_name}")
    print(f"{'='*60}")
    print(f"Benchmarks ({len(benchmarks)}):")
    for b in sorted(benchmarks):
        marker = " (anchor)" if b == anchor_benchmark else ""
        print(f"  - {b}{marker}")

    # Load data for this basket
    print(f"\nLoading benchmark data...")
    df = prepare_benchmark_data(
        cache_dir=Path(".cache"),
        include_benchmarks=benchmarks,
        min_benchmarks_per_model=min_benchmarks_per_model,
    )

    if len(df) == 0:
        print(f"  WARNING: No data available for {basket_name}")
        return None, None

    print(f"  Loaded {len(df)} performance records")
    print(f"  {df['model_id'].nunique()} models, {df['benchmark_id'].nunique()} benchmarks")

    # Check which benchmarks are actually present
    present_benchmarks = set(df["benchmark"].unique())
    missing = benchmarks - present_benchmarks
    if missing:
        print(f"  WARNING: Missing benchmarks: {sorted(missing)}")

    # Fit the model
    jacobian_type = "analytical" if use_analytical_jacobian else "numerical"
    print(f"\nFitting IRT model ({jacobian_type} Jacobian, {bootstrap_samples} bootstrap samples)...")

    model_df, bench_df = fit_eci_model(
        df,
        anchor_benchmark=anchor_benchmark,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=12345,
        use_analytical_jacobian=use_analytical_jacobian,
    )

    # Compute ECI scores
    # Note: This may fail if anchor models aren't in the data
    print("Computing ECI/EDI scores...")
    try:
        eci_df, edi_df = compute_eci_scores(model_df, bench_df)
    except ValueError as e:
        print(f"  WARNING: Could not compute scaled ECI scores: {e}")
        print("  Returning raw capability scores instead")
        eci_df = model_df.copy()
        eci_df["eci"] = eci_df["capability"]
        if "capability_ci_low" in eci_df.columns:
            eci_df["eci_ci_low"] = eci_df["capability_ci_low"]
            eci_df["eci_ci_high"] = eci_df["capability_ci_high"]
        edi_df = bench_df.copy()
        edi_df["edi"] = edi_df["difficulty"]
        edi_df["discriminability_scaled"] = edi_df["discriminability"]

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
    args = parser.parse_args()

    for basket_key in args.baskets:
        fit_basket(
            basket_key,
            bootstrap_samples=args.bootstrap_samples,
            min_benchmarks_per_model=args.min_benchmarks,
            output_dir=args.output_dir,
            use_analytical_jacobian=not args.numeric_jacobian,
        )

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
