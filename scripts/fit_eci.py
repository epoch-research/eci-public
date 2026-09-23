#!/usr/bin/env python3
"""
Fit ECI model and save scores to outputs directory.

Usage:
    python scripts/fit_eci.py
    python scripts/fit_eci.py --input path/to/benchmark_data.zip
    python scripts/fit_eci.py --evaluate
    python scripts/fit_eci.py --draws 8000 --chains 4
    python scripts/fit_eci.py --method least-squares
    python scripts/fit_eci.py --method least-squares --bootstrap-samples 200
"""

import argparse
from pathlib import Path

import pandas as pd

from eci import fit_eci_model, prepare_benchmark_data
from eci.dataloader import BENCHMARK_DATA_URL


OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Input columns carried over to the ECI output when present (the fit itself
# returns bare statistical results)
MODEL_METADATA_COLS = ["date", "Organization", "model_version", "source"]

# Per-benchmark columns written out, in order, when the fit produces them
EDI_VALUE_COLS = ["edi", "discriminability_scaled", "kappa"]


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
        default=BENCHMARK_DATA_URL,
        help=f"Path or URL of benchmark_data.zip (default: {BENCHMARK_DATA_URL})",
    )
    parser.add_argument(
        "--method",
        choices=["bayesian", "least-squares"],
        default="bayesian",
        help="Fitting method: Bayesian MCMC with posterior credible intervals "
             "(default), or least squares with bootstrap CIs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for scores (default: {OUTPUT_DIR})",
    )

    bayesian = parser.add_argument_group("bayesian options")
    bayesian.add_argument(
        "--draws", type=int, default=4000,
        help="Posterior draws kept per chain (default: 4000)",
    )
    bayesian.add_argument(
        "--tune", type=int, default=2000,
        help="Tuning steps per chain, discarded (default: 2000)",
    )
    bayesian.add_argument(
        "--chains", type=int, default=4, help="Number of chains (default: 4)",
    )
    bayesian.add_argument(
        "--target-accept", type=float, default=0.95,
        help="NUTS target acceptance rate; raise towards 0.99 if the fit "
             "reports divergences (default: 0.95)",
    )
    bayesian.add_argument(
        "--seed", type=int, default=12345, help="Sampler seed (default: 12345)",
    )
    bayesian.add_argument(
        "--non-centred",
        action="store_true",
        help="Sample the item parameters as unit-scale offsets rescaled by "
             "their hyperparameter, instead of directly. Same posterior, "
             "different geometry for the sampler",
    )
    bayesian.add_argument(
        "--evaluate",
        action="store_true",
        help="Also run the diagnostic suite (convergence, accuracy, LOO, "
             "calibration, posterior predictive checks) and write it, with a "
             "PDF of plots, to <output-dir>/evaluation/",
    )

    least_squares = parser.add_argument_group("least-squares options")
    least_squares.add_argument(
        "--bootstrap-samples",
        type=int,
        default=500,
        help="Number of bootstrap samples for confidence intervals (default: 500)",
    )
    least_squares.add_argument(
        "--numeric-jacobian",
        action="store_true",
        help="Use numerical Jacobian instead of analytical (slower)",
    )

    args = parser.parse_args()

    if args.evaluate and args.method != "bayesian":
        parser.error(
            "--evaluate scores a posterior, so it needs --method bayesian; "
            "the least-squares fit has no diagnostic suite"
        )

    print(f"Preparing benchmark data from {args.input}...")
    df = prepare_benchmark_data(args.input)
    print(f"  Loaded {len(df)} performance records")
    print(f"  {df['model_id'].nunique()} models, {df['benchmark_id'].nunique()} benchmarks")

    if args.method == "bayesian":
        from eci import fit_bayesian_eci

        print(f"\nFitting Bayesian IRT model ({args.chains} chains, "
              f"{args.draws} draws, {args.tune} tuning steps)...")
        eci_df, edi_df, draws = fit_bayesian_eci(
            df,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept,
            random_seed=args.seed,
            centred=not args.non_centred,
            evaluate=args.evaluate,
        )
    else:
        jacobian_type = "numerical" if args.numeric_jacobian else "analytical"
        print(f"\nFitting IRT model ({jacobian_type} Jacobian, "
              f"{args.bootstrap_samples} bootstrap samples)...")
        eci_df, edi_df, _ = fit_eci_model(
            df,
            bootstrap_samples=args.bootstrap_samples,
            use_analytical_jacobian=not args.numeric_jacobian,
        )

    eci_df = join_model_metadata(eci_df, df)

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
    if "benchmark_release_date" in df.columns:
        release_dates = df.drop_duplicates("benchmark_id").set_index("benchmark_id")["benchmark_release_date"]
        edi_df = edi_df.join(release_dates, on="benchmark_id")
    edi_output = args.output_dir / "edi_scores.csv"
    edi_cols = ["benchmark", "edi", "discriminability_scaled"]
    if "benchmark_release_date" in edi_df.columns:
        edi_cols.append("benchmark_release_date")
    # is_anchor is least-squares only (the Bayesian fit pins no benchmark);
    # kappa is Bayesian only
    edi_cols += [c for c in ("is_anchor", "kappa") if c in edi_df.columns]
    edi_cols += [
        f"{name}_{bound}"
        for name in EDI_VALUE_COLS
        for bound in ("ci_low", "ci_high")
        if f"{name}_ci_low" in edi_df.columns
    ]
    edi_df[edi_cols].to_csv(edi_output, index=False)
    print(f"Saved EDI scores to {edi_output}")

    if args.method == "bayesian" and args.evaluate:
        from eci import save_evaluation, summarise_evaluation

        evaluation = draws["evaluation"]
        written = save_evaluation(evaluation, args.output_dir / "evaluation")
        print(f"\nSaved {len(written)} evaluation files to "
              f"{args.output_dir / 'evaluation'}")
        print("\n=== Evaluation ===")
        print(summarise_evaluation(evaluation))

    # Print summary
    print("\n=== Top 10 Models by ECI ===")
    print(eci_df[["Model", "eci"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
