#!/usr/bin/env python3
"""
Analysis of α parametrization and regularization effects on ECI model.

This script compares two parametrizations:
1. Standard: regularize α directly (penalize α²)
2. Log-parametrized: regularize log(α) (penalize log(α)²)

And analyzes how they affect:
- Parameter stability across regularization strengths
- Capability slope over time (algorithmic progress)
- Individual model and benchmark estimates
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import least_squares
from scipy import stats

from eci import load_benchmark_data
from eci.fitting import (
    DEFAULT_ANCHOR_BENCHMARK,
    DEFAULT_ANCHOR_DIFFICULTY,
    DEFAULT_ANCHOR_DISCRIMINABILITY,
)


# ============================================================================
# STYLING SETUP
# ============================================================================


def setup_custom_style():
    """Set up custom graph styling for all plots."""
    custom_colors = [
        "#00A5A6",  # teal
        "#E03D90",  # pink
        "#FC6538",  # orange
        "#6A3ECB",  # purple
        "#0058DC",  # blue
        "#EA8D00",  # yellow
        "#B087F4",  # lightPurple
        "#279E27",  # green
        "#009AF1",  # lightBlue
        "#015D90",  # darkBlue
        "#EA4831",  # red
        "#E1C700",  # yellow2
        "#46FFFF",  # turquoise
        "#63F039",  # lightGreen
    ]

    sns.set_palette(custom_colors)
    sns.set_theme(style="whitegrid", palette=custom_colors, context="notebook")

    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 120,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
    })

    return custom_colors


def save_plot(output_path: Path, dpi: int = 300, bbox_inches: str = "tight"):
    """Save the current plot as both PNG and PDF."""
    png_path = output_path.with_suffix(".png")
    plt.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches=bbox_inches)
    return png_path, pdf_path


# ============================================================================
# FITTING FUNCTIONS
# ============================================================================


def fit_eci_model_with_parametrization(
    df: pd.DataFrame,
    parametrization: str = "standard",  # "standard" or "log"
    anchor_benchmark: str = DEFAULT_ANCHOR_BENCHMARK,
    anchor_difficulty: float = DEFAULT_ANCHOR_DIFFICULTY,
    anchor_discriminability: float = DEFAULT_ANCHOR_DISCRIMINABILITY,
    regularization_strength: float = 0.1,
    performance_clip_eps: float = 1e-3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit ECI model with specified parametrization.

    Args:
        parametrization: "standard" (penalize α²) or "log" (penalize log(α)²)
    """
    df = df.copy()

    if performance_clip_eps > 0:
        df["performance"] = df["performance"].clip(
            performance_clip_eps, 1 - performance_clip_eps
        )

    model_ids = df["model_id"].unique()
    benchmark_ids = df["benchmark_id"].unique()

    model_to_idx = {m: i for i, m in enumerate(model_ids)}
    bench_to_idx = {b: i for i, b in enumerate(benchmark_ids)}

    n_models = len(model_ids)
    n_benchmarks = len(benchmark_ids)

    model_idx = np.array([model_to_idx[m] for m in df["model_id"]])
    bench_idx = np.array([bench_to_idx[b] for b in df["benchmark_id"]])
    performance = df["performance"].values

    id_to_model_name = df.drop_duplicates("model_id").set_index("model_id")["Model"].to_dict()
    id_to_bench_name = df.drop_duplicates("benchmark_id").set_index("benchmark_id")["benchmark"].to_dict()

    # Get model dates
    model_dates = df.drop_duplicates("model_id").set_index("model_id")["date"].to_dict()

    try:
        anchor_bench_id = df.loc[df["benchmark"] == anchor_benchmark, "benchmark_id"].iloc[0]
    except IndexError:
        raise ValueError(f"Anchor benchmark '{anchor_benchmark}' not found in data")
    anchor_idx = bench_to_idx[anchor_bench_id]

    def sigmoid(x: np.ndarray) -> np.ndarray:
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    n_params = n_models + n_benchmarks + (n_benchmarks - 1)

    if parametrization == "log":
        # Log parametrization: optimize log(α), regularize log(α)²
        anchor_log_alpha = np.log(anchor_discriminability)

        def unpack_params(params):
            capability = params[:n_models]
            difficulty = params[n_models:n_models + n_benchmarks]
            log_alpha_free = params[n_models + n_benchmarks:]
            log_alpha = np.insert(log_alpha_free, anchor_idx, anchor_log_alpha)
            discriminability = np.exp(log_alpha)
            return capability, difficulty, discriminability, log_alpha

        def residuals(params):
            capability, difficulty, discriminability, log_alpha = unpack_params(params)
            pred = sigmoid(discriminability[bench_idx] * (capability[model_idx] - difficulty[bench_idx]))
            resid = pred - performance

            if regularization_strength > 0:
                log_alpha_free = log_alpha[np.arange(len(log_alpha)) != anchor_idx]
                reg_penalty = regularization_strength * (
                    np.sum(capability**2) +
                    np.sum(difficulty**2) +
                    np.sum(log_alpha_free**2)
                ) / n_params
                resid = np.append(resid, np.sqrt(reg_penalty))

            return resid

        np.random.seed(42)
        init_capability = np.random.randn(n_models) * 0.1
        init_difficulty = np.random.randn(n_benchmarks) * 0.1
        init_log_alpha = np.zeros(n_benchmarks - 1)
        init_params = np.concatenate([init_capability, init_difficulty, init_log_alpha])

        lower = np.concatenate([
            np.full(n_models, -10),
            np.full(n_benchmarks, -10),
            np.full(n_benchmarks - 1, -2.3)
        ])
        upper = np.concatenate([
            np.full(n_models, 10),
            np.full(n_benchmarks, 10),
            np.full(n_benchmarks - 1, 2.3)
        ])

    else:  # standard
        def unpack_params(params):
            capability = params[:n_models]
            difficulty = params[n_models:n_models + n_benchmarks]
            discrim_free = params[n_models + n_benchmarks:]
            discriminability = np.insert(discrim_free, anchor_idx, anchor_discriminability)
            return capability, difficulty, discriminability, None

        def residuals(params):
            capability, difficulty, discriminability, _ = unpack_params(params)
            pred = sigmoid(discriminability[bench_idx] * (capability[model_idx] - difficulty[bench_idx]))
            resid = pred - performance

            if regularization_strength > 0:
                reg_penalty = regularization_strength * (
                    np.sum(capability**2) +
                    np.sum(difficulty**2) +
                    np.sum(discriminability[discriminability != anchor_discriminability]**2)
                ) / n_params
                resid = np.append(resid, np.sqrt(reg_penalty))

            return resid

        np.random.seed(42)
        init_capability = np.random.randn(n_models) * 0.1
        init_difficulty = np.random.randn(n_benchmarks) * 0.1
        init_discrim = np.full(n_benchmarks - 1, 1.0)
        init_params = np.concatenate([init_capability, init_difficulty, init_discrim])

        lower = np.concatenate([
            np.full(n_models, -10),
            np.full(n_benchmarks, -10),
            np.full(n_benchmarks - 1, 0.1)
        ])
        upper = np.concatenate([
            np.full(n_models, 10),
            np.full(n_benchmarks, 10),
            np.full(n_benchmarks - 1, 10)
        ])

    result = least_squares(
        residuals,
        init_params,
        bounds=(lower, upper),
        method="trf",
        verbose=0
    )

    capability_hat, difficulty_hat, discriminability_hat, _ = unpack_params(result.x)

    # Shift to anchor the benchmark difficulty
    shift = difficulty_hat[anchor_idx] - anchor_difficulty
    capability_hat = capability_hat - shift
    difficulty_hat = difficulty_hat - shift

    # Build output DataFrames
    model_names = [id_to_model_name[m] for m in model_ids]
    dates = [model_dates.get(m) for m in model_ids]
    model_df = pd.DataFrame({
        "model_id": model_ids,
        "Model": model_names,
        "capability": capability_hat,
        "date": dates,
    }).sort_values("capability", ascending=False)

    bench_names = [id_to_bench_name[b] for b in benchmark_ids]
    bench_df = pd.DataFrame({
        "benchmark_id": benchmark_ids,
        "benchmark": bench_names,
        "difficulty": difficulty_hat,
        "discriminability": discriminability_hat,
        "is_anchor": [b == anchor_bench_id for b in benchmark_ids],
    }).sort_values("difficulty")

    return model_df, bench_df


def compute_capability_slope(model_df: pd.DataFrame) -> dict:
    """
    Compute the slope of capability over time (algorithmic progress).

    Returns dict with slope, intercept, r_squared, p_value, std_err
    """
    df = model_df.dropna(subset=["date", "capability"]).copy()

    if len(df) < 3:
        return {"slope": np.nan, "intercept": np.nan, "r_squared": np.nan,
                "p_value": np.nan, "std_err": np.nan, "n_models": len(df)}

    # Convert dates to numeric (years since 2020)
    df["date"] = pd.to_datetime(df["date"])
    df["years"] = (df["date"] - pd.Timestamp("2020-01-01")).dt.days / 365.25

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df["years"], df["capability"]
    )

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "p_value": p_value,
        "std_err": std_err,
        "n_models": len(df),
    }


# ============================================================================
# ANALYSIS
# ============================================================================


def run_full_analysis(
    df: pd.DataFrame,
    lambda_min: float = -5,
    lambda_max: float = 0,
    n_points: int = 51,
    output_dir: Path = None,
    verbose: bool = True,
) -> dict:
    """
    Run full analysis across regularization strengths and parametrizations.

    Saves all individual model/benchmark values for each configuration.
    """
    lambda_exps = np.linspace(lambda_min, lambda_max, n_points)

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs" / "parametrization"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for detailed results
    models_dir = output_dir / "models"
    benchmarks_dir = output_dir / "benchmarks"
    models_dir.mkdir(exist_ok=True)
    benchmarks_dir.mkdir(exist_ok=True)

    summary_results = []
    all_model_results = []
    all_bench_results = []

    if verbose:
        print(f"Running analysis from λ=10^{lambda_min} to λ=10^{lambda_max}")
        print(f"Number of points: {n_points}")
        print()

    for parametrization in ["standard", "log"]:
        if verbose:
            print(f"\n=== Parametrization: {parametrization} ===")

        for i, lambda_exp in enumerate(lambda_exps):
            if verbose and (i % 10 == 0 or i == len(lambda_exps) - 1):
                print(f"  Progress: {i + 1}/{n_points} (λ=10^{lambda_exp:.2f})")

            reg_strength = 10**lambda_exp

            model_df, bench_df = fit_eci_model_with_parametrization(
                df,
                parametrization=parametrization,
                regularization_strength=reg_strength
            )

            # Compute slope over time
            slope_stats = compute_capability_slope(model_df)

            # Summary statistics
            summary_results.append({
                "parametrization": parametrization,
                "regularizer_exp": lambda_exp,
                "regularization_strength": reg_strength,
                "avg_C_magnitude": model_df["capability"].abs().mean(),
                "avg_D_magnitude": bench_df["difficulty"].abs().mean(),
                "avg_alpha_magnitude": bench_df["discriminability"].abs().mean(),
                "std_C": model_df["capability"].std(),
                "std_D": bench_df["difficulty"].std(),
                "std_alpha": bench_df["discriminability"].std(),
                "capability_slope": slope_stats["slope"],
                "capability_slope_stderr": slope_stats["std_err"],
                "capability_slope_r2": slope_stats["r_squared"],
                "capability_slope_pvalue": slope_stats["p_value"],
            })

            # Store individual model results
            model_df_extended = model_df.copy()
            model_df_extended["parametrization"] = parametrization
            model_df_extended["regularizer_exp"] = lambda_exp
            model_df_extended["regularization_strength"] = reg_strength
            all_model_results.append(model_df_extended)

            # Store individual benchmark results
            bench_df_extended = bench_df.copy()
            bench_df_extended["parametrization"] = parametrization
            bench_df_extended["regularizer_exp"] = lambda_exp
            bench_df_extended["regularization_strength"] = reg_strength
            all_bench_results.append(bench_df_extended)

    # Combine and save results
    summary_df = pd.DataFrame(summary_results)
    all_models_df = pd.concat(all_model_results, ignore_index=True)
    all_benchmarks_df = pd.concat(all_bench_results, ignore_index=True)

    # Save summary
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    # Save all model results
    all_models_df.to_csv(output_dir / "all_model_capabilities.csv", index=False)

    # Save all benchmark results
    all_benchmarks_df.to_csv(output_dir / "all_benchmark_params.csv", index=False)

    if verbose:
        print(f"\nSaved results to {output_dir}")

    return {
        "summary": summary_df,
        "all_models": all_models_df,
        "all_benchmarks": all_benchmarks_df,
        "output_dir": output_dir,
    }


def create_plots(results: dict, colors: list = None) -> dict:
    """Create all analysis plots."""
    if colors is None:
        colors = setup_custom_style()

    output_dir = results["output_dir"]
    summary = results["summary"]

    df_standard = summary[summary["parametrization"] == "standard"]
    df_log = summary[summary["parametrization"] == "log"]

    saved_plots = {}

    # Plot 1: Parameter magnitudes comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, param, ylabel in zip(
        axes,
        ["avg_C_magnitude", "avg_D_magnitude", "avg_alpha_magnitude"],
        ["Avg |C| (Capability)", "Avg |D| (Difficulty)", "Avg |α| (Discriminability)"]
    ):
        ax.plot(df_standard["regularizer_exp"], df_standard[param],
                marker="o", color=colors[0], linewidth=2, markersize=4, label="Standard (α²)")
        ax.plot(df_log["regularizer_exp"], df_log[param],
                marker="s", color=colors[1], linewidth=2, markersize=4, label="Log (log(α)²)")
        ax.set_xlabel("log10(λ)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(-1, color="red", linestyle="--", alpha=0.5, label="λ=0.1 (default)")

    axes[0].set_title("Capability Magnitude")
    axes[1].set_title("Difficulty Magnitude")
    axes[2].set_title("Discriminability Magnitude")

    plt.tight_layout()
    saved_plots["magnitudes"] = save_plot(output_dir / "parameter_magnitudes")
    plt.close()

    # Plot 2: Capability slope over time (THE KEY PLOT)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Slope values
    ax = axes[0]
    ax.plot(df_standard["regularizer_exp"], df_standard["capability_slope"],
            marker="o", color=colors[0], linewidth=2, markersize=4, label="Standard (α²)")
    ax.fill_between(
        df_standard["regularizer_exp"],
        df_standard["capability_slope"] - 1.96 * df_standard["capability_slope_stderr"],
        df_standard["capability_slope"] + 1.96 * df_standard["capability_slope_stderr"],
        alpha=0.2, color=colors[0]
    )
    ax.plot(df_log["regularizer_exp"], df_log["capability_slope"],
            marker="s", color=colors[1], linewidth=2, markersize=4, label="Log (log(α)²)")
    ax.fill_between(
        df_log["regularizer_exp"],
        df_log["capability_slope"] - 1.96 * df_log["capability_slope_stderr"],
        df_log["capability_slope"] + 1.96 * df_log["capability_slope_stderr"],
        alpha=0.2, color=colors[1]
    )
    ax.axvline(-1, color="red", linestyle="--", alpha=0.5, label="λ=0.1 (default)")
    ax.set_xlabel("log10(λ)")
    ax.set_ylabel("Capability Slope (units/year)")
    ax.set_title("Algorithmic Progress: Capability Slope Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R² values
    ax = axes[1]
    ax.plot(df_standard["regularizer_exp"], df_standard["capability_slope_r2"],
            marker="o", color=colors[0], linewidth=2, markersize=4, label="Standard (α²)")
    ax.plot(df_log["regularizer_exp"], df_log["capability_slope_r2"],
            marker="s", color=colors[1], linewidth=2, markersize=4, label="Log (log(α)²)")
    ax.axvline(-1, color="red", linestyle="--", alpha=0.5, label="λ=0.1 (default)")
    ax.set_xlabel("log10(λ)")
    ax.set_ylabel("R² (variance explained)")
    ax.set_title("Linear Fit Quality: R² of Capability vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    saved_plots["capability_slope"] = save_plot(output_dir / "capability_slope_over_time")
    plt.close()

    # Plot 3: Slope sensitivity analysis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate relative change in slope from the stable region
    stable_slope_std = df_standard[df_standard["regularizer_exp"] <= -3]["capability_slope"].mean()
    stable_slope_log = df_log[df_log["regularizer_exp"] <= -3]["capability_slope"].mean()

    relative_change_std = (df_standard["capability_slope"] - stable_slope_std) / abs(stable_slope_std) * 100
    relative_change_log = (df_log["capability_slope"] - stable_slope_log) / abs(stable_slope_log) * 100

    ax.plot(df_standard["regularizer_exp"], relative_change_std,
            marker="o", color=colors[0], linewidth=2, markersize=4, label="Standard (α²)")
    ax.plot(df_log["regularizer_exp"], relative_change_log,
            marker="s", color=colors[1], linewidth=2, markersize=4, label="Log (log(α)²)")
    ax.axvline(-1, color="red", linestyle="--", alpha=0.5, label="λ=0.1 (default)")
    ax.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("log10(λ)")
    ax.set_ylabel("% Change in Slope (from stable region)")
    ax.set_title("Slope Sensitivity to Regularization")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    saved_plots["slope_sensitivity"] = save_plot(output_dir / "slope_sensitivity")
    plt.close()

    # Plot 4: Example capability vs time for different lambda values
    all_models = results["all_models"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    lambda_examples = [-4, -2, -1]  # Three representative lambda values

    for row, parametrization in enumerate(["standard", "log"]):
        for col, lambda_exp in enumerate(lambda_examples):
            ax = axes[row, col]

            # Get models for this configuration
            mask = (all_models["parametrization"] == parametrization) & \
                   (np.abs(all_models["regularizer_exp"] - lambda_exp) < 0.1)
            models = all_models[mask].copy()
            models["date"] = pd.to_datetime(models["date"])
            models = models.dropna(subset=["date"])

            ax.scatter(models["date"], models["capability"], alpha=0.5, s=20, color=colors[col])

            # Add trend line
            if len(models) > 2:
                years = (models["date"] - pd.Timestamp("2020-01-01")).dt.days / 365.25
                slope, intercept, _, _, _ = stats.linregress(years, models["capability"])
                x_line = np.array([years.min(), years.max()])
                y_line = intercept + slope * x_line
                dates_line = [pd.Timestamp("2020-01-01") + pd.Timedelta(days=d*365.25) for d in x_line]
                ax.plot(dates_line, y_line, color="red", linewidth=2,
                       label=f"slope={slope:.3f}/yr")

            ax.set_xlabel("Date")
            ax.set_ylabel("Capability")
            ax.set_title(f"{parametrization.title()}, λ=10^{lambda_exp}")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    saved_plots["capability_vs_time"] = save_plot(output_dir / "capability_vs_time_examples")
    plt.close()

    return saved_plots


def write_summary_report(results: dict) -> Path:
    """Write a detailed summary report."""
    output_dir = results["output_dir"]
    summary = results["summary"]

    df_standard = summary[summary["parametrization"] == "standard"]
    df_log = summary[summary["parametrization"] == "log"]

    report_path = output_dir / "analysis_report.txt"

    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("ECI REGULARIZATION AND PARAMETRIZATION ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Parameter magnitude ranges
        f.write("1. PARAMETER MAGNITUDE RANGES\n")
        f.write("-" * 40 + "\n\n")

        f.write("Standard Parametrization (penalize α²):\n")
        f.write(f"  |C| range: {df_standard['avg_C_magnitude'].min():.4f} - {df_standard['avg_C_magnitude'].max():.4f}\n")
        f.write(f"  |D| range: {df_standard['avg_D_magnitude'].min():.4f} - {df_standard['avg_D_magnitude'].max():.4f}\n")
        f.write(f"  |α| range: {df_standard['avg_alpha_magnitude'].min():.4f} - {df_standard['avg_alpha_magnitude'].max():.4f}\n\n")

        f.write("Log Parametrization (penalize log(α)²):\n")
        f.write(f"  |C| range: {df_log['avg_C_magnitude'].min():.4f} - {df_log['avg_C_magnitude'].max():.4f}\n")
        f.write(f"  |D| range: {df_log['avg_D_magnitude'].min():.4f} - {df_log['avg_D_magnitude'].max():.4f}\n")
        f.write(f"  |α| range: {df_log['avg_alpha_magnitude'].min():.4f} - {df_log['avg_alpha_magnitude'].max():.4f}\n\n")

        # Capability slope analysis
        f.write("2. CAPABILITY SLOPE OVER TIME (ALGORITHMIC PROGRESS)\n")
        f.write("-" * 40 + "\n\n")

        f.write("Standard Parametrization:\n")
        f.write(f"  Slope range: {df_standard['capability_slope'].min():.4f} - {df_standard['capability_slope'].max():.4f} units/year\n")
        f.write(f"  At λ=0.1 (default): {df_standard[df_standard['regularizer_exp'] == -1]['capability_slope'].values[0]:.4f} units/year\n")
        f.write(f"  At λ=10^-4 (stable): {df_standard[np.abs(df_standard['regularizer_exp'] + 4) < 0.1]['capability_slope'].values[0]:.4f} units/year\n\n")

        f.write("Log Parametrization:\n")
        f.write(f"  Slope range: {df_log['capability_slope'].min():.4f} - {df_log['capability_slope'].max():.4f} units/year\n")
        f.write(f"  At λ=0.1 (default): {df_log[df_log['regularizer_exp'] == -1]['capability_slope'].values[0]:.4f} units/year\n")
        f.write(f"  At λ=10^-4 (stable): {df_log[np.abs(df_log['regularizer_exp'] + 4) < 0.1]['capability_slope'].values[0]:.4f} units/year\n\n")

        # Impact of regularization on slope
        f.write("3. IMPACT OF REGULARIZATION ON SLOPE ESTIMATES\n")
        f.write("-" * 40 + "\n\n")

        stable_slope_std = df_standard[df_standard["regularizer_exp"] <= -3]["capability_slope"].mean()
        default_slope_std = df_standard[df_standard["regularizer_exp"] == -1]["capability_slope"].values[0]
        change_std = (default_slope_std - stable_slope_std) / abs(stable_slope_std) * 100

        stable_slope_log = df_log[df_log["regularizer_exp"] <= -3]["capability_slope"].mean()
        default_slope_log = df_log[df_log["regularizer_exp"] == -1]["capability_slope"].values[0]
        change_log = (default_slope_log - stable_slope_log) / abs(stable_slope_log) * 100

        f.write("Standard Parametrization:\n")
        f.write(f"  Slope in stable region (λ ≤ 10^-3): {stable_slope_std:.4f} units/year\n")
        f.write(f"  Slope at default λ=0.1: {default_slope_std:.4f} units/year\n")
        f.write(f"  Change: {change_std:+.1f}%\n\n")

        f.write("Log Parametrization:\n")
        f.write(f"  Slope in stable region (λ ≤ 10^-3): {stable_slope_log:.4f} units/year\n")
        f.write(f"  Slope at default λ=0.1: {default_slope_log:.4f} units/year\n")
        f.write(f"  Change: {change_log:+.1f}%\n\n")

        # Key findings
        f.write("4. KEY FINDINGS\n")
        f.write("-" * 40 + "\n\n")

        alpha_sens_std = df_standard['avg_alpha_magnitude'].max() - df_standard['avg_alpha_magnitude'].min()
        alpha_sens_log = df_log['avg_alpha_magnitude'].max() - df_log['avg_alpha_magnitude'].min()

        f.write(f"• α sensitivity reduced by {alpha_sens_std/alpha_sens_log:.1f}x with log parametrization\n")
        f.write(f"• Standard slope change with regularization: {change_std:+.1f}%\n")
        f.write(f"• Log slope change with regularization: {change_log:+.1f}%\n")
        f.write(f"• Default λ=0.1 is outside the stable region (λ ≤ 10^-3) for both parametrizations\n")

    return report_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze ECI regularization and α parametrization effects"
    )
    parser.add_argument("--lambda-min", type=float, default=-5)
    parser.add_argument("--lambda-max", type=float, default=0)
    parser.add_argument("--n-points", type=int, default=51)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-url", type=str, default="https://epoch.ai/data/eci_benchmarks.csv")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    print("=" * 60)
    print("ECI REGULARIZATION AND PARAMETRIZATION ANALYSIS")
    print("=" * 60)
    print()

    print(f"Loading data from {args.data_url}...")
    df = load_benchmark_data(args.data_url)
    print(f"  Loaded {len(df)} records")
    print(f"  {df['model_id'].nunique()} models, {df['benchmark_id'].nunique()} benchmarks")

    results = run_full_analysis(
        df,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        n_points=args.n_points,
        output_dir=output_dir,
    )

    print("\nCreating visualizations...")
    colors = setup_custom_style()
    saved_plots = create_plots(results, colors)

    for name, (png, pdf) in saved_plots.items():
        print(f"  - {name}: {png.name}")

    print("\nWriting summary report...")
    report_path = write_summary_report(results)
    print(f"  - {report_path.name}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved to: {results['output_dir']}")


if __name__ == "__main__":
    main()
