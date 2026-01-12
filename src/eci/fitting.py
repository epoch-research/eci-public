"""
ECI Model Fitting

This module implements Item Response Theory (IRT) fitting for computing:
- ECI (Epoch Capability Index): Model capability scores
- EDI (Epoch Difficulty Index): Benchmark difficulty scores

The model assumes benchmark performance follows a logistic function:
    P(correct) = sigmoid(discriminability * (capability - difficulty))

where:
- capability (C): How capable a model is (higher = more capable)
- difficulty (D): How hard a benchmark is (higher = harder)
- discriminability (α): How sharply performance transitions (higher = sharper)
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from tqdm import tqdm


# Default scaling anchors - these define the ECI scale
DEFAULT_ANCHOR_MODEL_LOW = "Claude 3.5 Sonnet"
DEFAULT_ANCHOR_ECI_LOW = 130.0
DEFAULT_ANCHOR_MODEL_HIGH = "GPT-5"
DEFAULT_ANCHOR_ECI_HIGH = 150.0

# Default benchmark anchor for model identification
DEFAULT_ANCHOR_BENCHMARK = "Winogrande"
DEFAULT_ANCHOR_DIFFICULTY = 0.0
DEFAULT_ANCHOR_DISCRIMINABILITY = 1.0


def load_benchmark_data(url: str = "https://epoch.ai/data/eci_benchmarks.csv") -> pd.DataFrame:
    """
    Load benchmark performance data from CSV.

    Args:
        url: URL or file path to the benchmark data CSV.

    Returns:
        DataFrame with columns: model_id, benchmark_id, performance, benchmark,
        Model, model, date, and other metadata.
    """
    df = pd.read_csv(url)
    required_cols = ["model_id", "benchmark_id", "performance", "benchmark", "Model"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def fit_eci_model(
    df: pd.DataFrame,
    anchor_benchmark: str = DEFAULT_ANCHOR_BENCHMARK,
    anchor_difficulty: float = DEFAULT_ANCHOR_DIFFICULTY,
    anchor_discriminability: float = DEFAULT_ANCHOR_DISCRIMINABILITY,
    regularization_strength: float = 0.1,
    performance_clip_eps: float = 1e-3,
    bootstrap_samples: int = 100,
    bootstrap_seed: int = 12345,
    ci_level: float = 0.90,
    use_analytical_jacobian: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit the IRT model to estimate model capabilities and benchmark difficulties.

    The model uses a logistic (sigmoid) function:
        performance = sigmoid(discriminability * (capability - difficulty))

    To identify the model (avoid infinite solutions), we anchor one benchmark's
    difficulty and discriminability to fixed values.

    Args:
        df: DataFrame with columns model_id, benchmark_id, performance, benchmark, Model.
        anchor_benchmark: Name of benchmark to anchor (fixes scale location).
        anchor_difficulty: Fixed difficulty value for anchor benchmark.
        anchor_discriminability: Fixed discriminability for anchor benchmark.
        regularization_strength: L2 regularization to prevent extreme values (0-1).
        performance_clip_eps: Clip performance to [eps, 1-eps] to avoid degeneracy.
        bootstrap_samples: Number of bootstrap resamples for confidence intervals.
        bootstrap_seed: Random seed for reproducibility.
        ci_level: Confidence interval level (e.g., 0.90 for 90% CI).
        use_analytical_jacobian: If True, use analytical Jacobian for faster optimization.
            If False, use numerical differentiation (slower but may give slightly
            different results due to optimizer path differences).

    Returns:
        Tuple of (model_capabilities_df, benchmark_params_df):
        - model_capabilities_df: Model names and estimated capabilities
        - benchmark_params_df: Benchmark names, difficulties, and discriminabilities
    """
    df = df.copy()

    # Validate inputs
    if df["performance"].isna().any():
        raise ValueError("Performance data contains NaN values")
    if (df["performance"] < 0).any() or (df["performance"] > 1).any():
        raise ValueError("Performance scores must be in [0, 1] range")

    # Clip extreme performance values to avoid degenerate fits
    if performance_clip_eps > 0:
        df["performance"] = df["performance"].clip(
            performance_clip_eps, 1 - performance_clip_eps
        )

    # Build index mappings
    model_ids = df["model_id"].unique()
    benchmark_ids = df["benchmark_id"].unique()

    model_to_idx = {m: i for i, m in enumerate(model_ids)}
    bench_to_idx = {b: i for i, b in enumerate(benchmark_ids)}

    n_models = len(model_ids)
    n_benchmarks = len(benchmark_ids)

    # Convert to index arrays for efficient computation
    model_idx = np.array([model_to_idx[m] for m in df["model_id"]])
    bench_idx = np.array([bench_to_idx[b] for b in df["benchmark_id"]])
    performance = df["performance"].values

    # Map IDs to names
    id_to_model_name = df.drop_duplicates("model_id").set_index("model_id")["Model"].to_dict()
    id_to_bench_name = df.drop_duplicates("benchmark_id").set_index("benchmark_id")["benchmark"].to_dict()

    # Find anchor benchmark index
    try:
        anchor_bench_id = df.loc[df["benchmark"] == anchor_benchmark, "benchmark_id"].iloc[0]
    except IndexError:
        raise ValueError(f"Anchor benchmark '{anchor_benchmark}' not found in data")
    anchor_idx = bench_to_idx[anchor_bench_id]

    # Define the model
    def sigmoid(x: np.ndarray) -> np.ndarray:
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def unpack_params(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract capability, difficulty, discriminability from flat parameter vector."""
        capability = params[:n_models]
        difficulty = params[n_models:n_models + n_benchmarks]
        # Discriminability: all free except anchor
        discrim_free = params[n_models + n_benchmarks:]
        discriminability = np.insert(discrim_free, anchor_idx, anchor_discriminability)
        return capability, difficulty, discriminability

    n_params = n_models + n_benchmarks + (n_benchmarks - 1)
    n_obs = len(performance)

    def residuals(params: np.ndarray) -> np.ndarray:
        capability, difficulty, discriminability = unpack_params(params)
        pred = sigmoid(discriminability[bench_idx] * (capability[model_idx] - difficulty[bench_idx]))
        resid = pred - performance

        # L2 regularization
        if regularization_strength > 0:
            reg_penalty = regularization_strength * (
                np.sum(capability**2) +
                np.sum(difficulty**2) +
                np.sum(discriminability[discriminability != anchor_discriminability]**2)
            ) / n_params
            resid = np.append(resid, np.sqrt(reg_penalty))

        return resid

    def jacobian(params: np.ndarray) -> np.ndarray:
        """Analytical Jacobian for faster optimization."""
        capability, difficulty, discriminability = unpack_params(params)

        # Compute predictions and sigmoid derivative
        z = discriminability[bench_idx] * (capability[model_idx] - difficulty[bench_idx])
        s = sigmoid(z)
        ds = s * (1 - s)  # sigmoid'(z)

        # Number of rows: n_obs + 1 (for regularization)
        n_rows = n_obs + 1 if regularization_strength > 0 else n_obs
        jac = lil_matrix((n_rows, n_params))

        # Derivatives w.r.t. capability
        # d(resid_i)/d(cap_m) = ds[i] * discrim[b] for obs where model_idx[i] == m
        cap_derivs = ds * discriminability[bench_idx]
        for i in range(n_obs):
            jac[i, model_idx[i]] = cap_derivs[i]

        # Derivatives w.r.t. difficulty
        # d(resid_i)/d(diff_b) = -ds[i] * discrim[b] for obs where bench_idx[i] == b
        diff_derivs = -ds * discriminability[bench_idx]
        for i in range(n_obs):
            jac[i, n_models + bench_idx[i]] = diff_derivs[i]

        # Derivatives w.r.t. discriminability (free parameters only)
        # d(resid_i)/d(discrim_b) = ds[i] * (cap[m] - diff[b])
        discrim_derivs = ds * (capability[model_idx] - difficulty[bench_idx])
        for i in range(n_obs):
            b = bench_idx[i]
            if b == anchor_idx:
                continue  # anchor discriminability is fixed
            # Map benchmark index to parameter index
            param_idx = n_models + n_benchmarks + (b if b < anchor_idx else b - 1)
            jac[i, param_idx] = discrim_derivs[i]

        # Regularization term derivatives
        if regularization_strength > 0:
            reg_penalty = regularization_strength * (
                np.sum(capability**2) +
                np.sum(difficulty**2) +
                np.sum(discriminability[discriminability != anchor_discriminability]**2)
            ) / n_params

            if reg_penalty > 0:
                scale = regularization_strength / (n_params * np.sqrt(reg_penalty))
                # d/d(cap_m) of sqrt(reg) = scale * cap_m
                for m in range(n_models):
                    jac[n_obs, m] = scale * capability[m]
                # d/d(diff_b) of sqrt(reg) = scale * diff_b
                for b in range(n_benchmarks):
                    jac[n_obs, n_models + b] = scale * difficulty[b]
                # d/d(discrim_b) of sqrt(reg) = scale * discrim_b (free only)
                for b in range(n_benchmarks):
                    if b == anchor_idx:
                        continue
                    param_idx = n_models + n_benchmarks + (b if b < anchor_idx else b - 1)
                    jac[n_obs, param_idx] = scale * discriminability[b]

        return jac.tocsr()

    # Initial values
    np.random.seed(42)
    init_capability = np.random.randn(n_models) * 0.1
    init_difficulty = np.random.randn(n_benchmarks) * 0.1
    init_discrim = np.full(n_benchmarks - 1, 1.0)
    init_params = np.concatenate([init_capability, init_difficulty, init_discrim])

    # Bounds to prevent extreme values
    lower = np.concatenate([
        np.full(n_models, -10),      # capability
        np.full(n_benchmarks, -10),  # difficulty
        np.full(n_benchmarks - 1, 0.1)  # discriminability (positive)
    ])
    upper = np.concatenate([
        np.full(n_models, 10),
        np.full(n_benchmarks, 10),
        np.full(n_benchmarks - 1, 10)
    ])

    # Fit the model
    result = least_squares(
        residuals,
        init_params,
        jac=jacobian if use_analytical_jacobian else "2-point",
        bounds=(lower, upper),
        method="trf",
        verbose=0
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    # Extract fitted parameters
    capability_hat, difficulty_hat, discriminability_hat = unpack_params(result.x)

    # Shift to anchor the benchmark difficulty
    shift = difficulty_hat[anchor_idx] - anchor_difficulty
    capability_hat = capability_hat - shift
    difficulty_hat = difficulty_hat - shift

    # Bootstrap for confidence intervals
    se_capability = np.full(n_models, np.nan)
    se_difficulty = np.full(n_benchmarks, np.nan)
    ci_capability_low = np.full(n_models, np.nan)
    ci_capability_high = np.full(n_models, np.nan)
    ci_difficulty_low = np.full(n_benchmarks, np.nan)
    ci_difficulty_high = np.full(n_benchmarks, np.nan)

    if bootstrap_samples > 0:
        rng = np.random.default_rng(bootstrap_seed)
        capability_samples = []
        difficulty_samples = []

        for _ in tqdm(range(bootstrap_samples), desc="Bootstrap", unit="sample"):
            # Resample with replacement
            idx = rng.integers(0, len(performance), size=len(performance))
            boot_performance = performance[idx]
            boot_model_idx = model_idx[idx]
            boot_bench_idx = bench_idx[idx]
            boot_n_obs = len(boot_performance)

            def boot_residuals(params):
                cap, diff, disc = unpack_params(params)
                pred = sigmoid(disc[boot_bench_idx] * (cap[boot_model_idx] - diff[boot_bench_idx]))
                resid = pred - boot_performance
                if regularization_strength > 0:
                    reg = regularization_strength * (
                        np.sum(cap**2) + np.sum(diff**2) +
                        np.sum(disc[disc != anchor_discriminability]**2)
                    ) / n_params
                    resid = np.append(resid, np.sqrt(reg))
                return resid

            def boot_jacobian(params):
                cap, diff, disc = unpack_params(params)
                z = disc[boot_bench_idx] * (cap[boot_model_idx] - diff[boot_bench_idx])
                s = sigmoid(z)
                ds = s * (1 - s)

                n_rows = boot_n_obs + 1 if regularization_strength > 0 else boot_n_obs
                jac = lil_matrix((n_rows, n_params))

                cap_derivs = ds * disc[boot_bench_idx]
                diff_derivs = -ds * disc[boot_bench_idx]
                discrim_derivs = ds * (cap[boot_model_idx] - diff[boot_bench_idx])

                for i in range(boot_n_obs):
                    jac[i, boot_model_idx[i]] = cap_derivs[i]
                    jac[i, n_models + boot_bench_idx[i]] = diff_derivs[i]
                    b = boot_bench_idx[i]
                    if b != anchor_idx:
                        param_idx = n_models + n_benchmarks + (b if b < anchor_idx else b - 1)
                        jac[i, param_idx] = discrim_derivs[i]

                if regularization_strength > 0:
                    reg = regularization_strength * (
                        np.sum(cap**2) + np.sum(diff**2) +
                        np.sum(disc[disc != anchor_discriminability]**2)
                    ) / n_params
                    if reg > 0:
                        scale = regularization_strength / (n_params * np.sqrt(reg))
                        for m in range(n_models):
                            jac[boot_n_obs, m] = scale * cap[m]
                        for b in range(n_benchmarks):
                            jac[boot_n_obs, n_models + b] = scale * diff[b]
                            if b != anchor_idx:
                                param_idx = n_models + n_benchmarks + (b if b < anchor_idx else b - 1)
                                jac[boot_n_obs, param_idx] = scale * disc[b]

                return jac.tocsr()

            try:
                boot_result = least_squares(
                    boot_residuals,
                    result.x.copy(),
                    jac=boot_jacobian if use_analytical_jacobian else "2-point",
                    bounds=(lower, upper),
                    method="trf",
                    verbose=0
                )
                if boot_result.success:
                    cap, diff, _ = unpack_params(boot_result.x)
                    shift_b = diff[anchor_idx] - anchor_difficulty
                    capability_samples.append(cap - shift_b)
                    difficulty_samples.append(diff - shift_b)
            except Exception:
                continue

        if len(capability_samples) > 1:
            cap_arr = np.vstack(capability_samples)
            diff_arr = np.vstack(difficulty_samples)

            se_capability = np.std(cap_arr, axis=0, ddof=1)
            se_difficulty = np.std(diff_arr, axis=0, ddof=1)

            tail = (1 - ci_level) / 2
            ci_capability_low = np.quantile(cap_arr, tail, axis=0)
            ci_capability_high = np.quantile(cap_arr, 1 - tail, axis=0)
            ci_difficulty_low = np.quantile(diff_arr, tail, axis=0)
            ci_difficulty_high = np.quantile(diff_arr, 1 - tail, axis=0)

    # Build output DataFrames
    model_names = [id_to_model_name[m] for m in model_ids]
    model_df = pd.DataFrame({
        "model_id": model_ids,
        "Model": model_names,
        "capability": capability_hat,
        "capability_se": se_capability,
        "capability_ci_low": ci_capability_low,
        "capability_ci_high": ci_capability_high,
    }).sort_values("capability", ascending=False)

    bench_names = [id_to_bench_name[b] for b in benchmark_ids]
    bench_df = pd.DataFrame({
        "benchmark_id": benchmark_ids,
        "benchmark": bench_names,
        "difficulty": difficulty_hat,
        "discriminability": discriminability_hat,
        "difficulty_se": se_difficulty,
        "difficulty_ci_low": ci_difficulty_low,
        "difficulty_ci_high": ci_difficulty_high,
        "is_anchor": [b == anchor_bench_id for b in benchmark_ids],
    }).sort_values("difficulty")

    # Add benchmark release dates if available
    if "benchmark_release_date" in df.columns:
        date_map = df.drop_duplicates("benchmark_id").set_index("benchmark_id")["benchmark_release_date"].to_dict()
        bench_df["benchmark_release_date"] = bench_df["benchmark_id"].map(date_map)

    return model_df, bench_df


def fit_capabilities_given_benchmarks(
    df: pd.DataFrame,
    bench_df: pd.DataFrame,
    regularization_strength: float = 0.1,
    performance_clip_eps: float = 1e-3,
    bootstrap_samples: int = 100,
    bootstrap_seed: int = 12345,
    ci_level: float = 0.90,
) -> pd.DataFrame:
    """
    Fit model capabilities while holding benchmark parameters fixed.

    This is useful for "projecting" models onto a pre-fit benchmark space.
    Given fixed benchmark difficulties and discriminabilities from a full model fit,
    this function estimates only the model capabilities that best explain
    the observed performance on a subset of benchmarks.

    Args:
        df: DataFrame with columns model_id, benchmark_id, performance, benchmark, Model.
        bench_df: DataFrame with benchmark parameters from a previous fit.
            Must contain columns: benchmark, difficulty, discriminability.
        regularization_strength: L2 regularization on capabilities (0-1).
        performance_clip_eps: Clip performance to [eps, 1-eps] to avoid degeneracy.
        bootstrap_samples: Number of bootstrap resamples for confidence intervals.
        bootstrap_seed: Random seed for reproducibility.
        ci_level: Confidence interval level (e.g., 0.90 for 90% CI).

    Returns:
        DataFrame with model capabilities and confidence intervals.
    """
    df = df.copy()

    # Validate inputs
    if df["performance"].isna().any():
        raise ValueError("Performance data contains NaN values")
    if (df["performance"] < 0).any() or (df["performance"] > 1).any():
        raise ValueError("Performance scores must be in [0, 1] range")

    # Clip extreme performance values
    if performance_clip_eps > 0:
        df["performance"] = df["performance"].clip(
            performance_clip_eps, 1 - performance_clip_eps
        )

    # Filter to benchmarks that exist in bench_df
    bench_params = bench_df.set_index("benchmark")[["difficulty", "discriminability"]].to_dict("index")
    available_benchmarks = set(bench_params.keys())
    df = df[df["benchmark"].isin(available_benchmarks)]

    if len(df) == 0:
        raise ValueError("No benchmark data matches the provided benchmark parameters")

    # Build index mappings
    model_ids = df["model_id"].unique()
    n_models = len(model_ids)
    model_to_idx = {m: i for i, m in enumerate(model_ids)}

    # Map IDs to names
    id_to_model_name = df.drop_duplicates("model_id").set_index("model_id")["Model"].to_dict()

    # Extract fixed benchmark parameters for each observation
    benchmark_names = df["benchmark"].values
    difficulty = np.array([bench_params[b]["difficulty"] for b in benchmark_names])
    discriminability = np.array([bench_params[b]["discriminability"] for b in benchmark_names])

    # Convert to index arrays
    model_idx = np.array([model_to_idx[m] for m in df["model_id"]])
    performance = df["performance"].values

    def sigmoid(x: np.ndarray) -> np.ndarray:
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def residuals(capability: np.ndarray) -> np.ndarray:
        pred = sigmoid(discriminability * (capability[model_idx] - difficulty))
        resid = pred - performance

        if regularization_strength > 0:
            reg_penalty = regularization_strength * np.sum(capability**2) / n_models
            resid = np.append(resid, np.sqrt(reg_penalty))

        return resid

    # Initial values
    np.random.seed(42)
    init_capability = np.random.randn(n_models) * 0.1

    # Bounds
    lower = np.full(n_models, -10)
    upper = np.full(n_models, 10)

    # Fit
    result = least_squares(
        residuals,
        init_capability,
        bounds=(lower, upper),
        method="trf",
        verbose=0
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    capability_hat = result.x

    # Bootstrap for confidence intervals
    se_capability = np.full(n_models, np.nan)
    ci_capability_low = np.full(n_models, np.nan)
    ci_capability_high = np.full(n_models, np.nan)

    if bootstrap_samples > 0:
        rng = np.random.default_rng(bootstrap_seed)
        capability_samples = []

        for _ in tqdm(range(bootstrap_samples), desc="Bootstrap", unit="sample"):
            idx = rng.integers(0, len(performance), size=len(performance))
            boot_performance = performance[idx]
            boot_model_idx = model_idx[idx]
            boot_difficulty = difficulty[idx]
            boot_discriminability = discriminability[idx]

            def boot_residuals(cap):
                pred = sigmoid(boot_discriminability * (cap[boot_model_idx] - boot_difficulty))
                resid = pred - boot_performance
                if regularization_strength > 0:
                    reg = regularization_strength * np.sum(cap**2) / n_models
                    resid = np.append(resid, np.sqrt(reg))
                return resid

            try:
                boot_result = least_squares(
                    boot_residuals,
                    result.x.copy(),
                    bounds=(lower, upper),
                    method="trf",
                    verbose=0
                )
                if boot_result.success:
                    capability_samples.append(boot_result.x)
            except Exception:
                continue

        if len(capability_samples) > 1:
            cap_arr = np.vstack(capability_samples)
            se_capability = np.std(cap_arr, axis=0, ddof=1)
            tail = (1 - ci_level) / 2
            ci_capability_low = np.quantile(cap_arr, tail, axis=0)
            ci_capability_high = np.quantile(cap_arr, 1 - tail, axis=0)

    # Build output DataFrame
    model_names = [id_to_model_name[m] for m in model_ids]
    model_df = pd.DataFrame({
        "model_id": model_ids,
        "Model": model_names,
        "capability": capability_hat,
        "capability_se": se_capability,
        "capability_ci_low": ci_capability_low,
        "capability_ci_high": ci_capability_high,
    }).sort_values("capability", ascending=False)

    return model_df


def compute_eci_scores(
    model_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    anchor_model_low: str = DEFAULT_ANCHOR_MODEL_LOW,
    anchor_eci_low: float = DEFAULT_ANCHOR_ECI_LOW,
    anchor_model_high: str = DEFAULT_ANCHOR_MODEL_HIGH,
    anchor_eci_high: float = DEFAULT_ANCHOR_ECI_HIGH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert raw capabilities and difficulties to ECI/EDI scale.

    The ECI scale is defined by two anchor points:
    - anchor_model_low is assigned anchor_eci_low (default: Claude 3.5 Sonnet = 130)
    - anchor_model_high is assigned anchor_eci_high (default: GPT-5 = 150)

    All other scores are linearly interpolated/extrapolated.

    Args:
        model_df: DataFrame with 'Model' and 'capability' columns from fit_eci_model.
        bench_df: DataFrame with 'difficulty' columns from fit_eci_model.
        anchor_model_low: Name of model for lower anchor point.
        anchor_eci_low: ECI value for lower anchor.
        anchor_model_high: Name of model for upper anchor point.
        anchor_eci_high: ECI value for upper anchor.

    Returns:
        Tuple of (eci_df, edi_df) with scores on the ECI/EDI scale.
    """
    # Find anchor capabilities
    low_cap = model_df.loc[model_df["Model"] == anchor_model_low, "capability"]
    high_cap = model_df.loc[model_df["Model"] == anchor_model_high, "capability"]

    if low_cap.empty:
        raise ValueError(f"Anchor model '{anchor_model_low}' not found")
    if high_cap.empty:
        raise ValueError(f"Anchor model '{anchor_model_high}' not found")

    cap_low = low_cap.iloc[0]
    cap_high = high_cap.iloc[0]

    # Compute linear scaling: eci = a + b * capability
    b = (anchor_eci_high - anchor_eci_low) / (cap_high - cap_low)
    a = anchor_eci_low - b * cap_low

    # Apply scaling to model capabilities
    eci_df = model_df.copy()
    eci_df["eci"] = a + b * eci_df["capability"]

    # Scale confidence intervals if present
    if "capability_ci_low" in eci_df.columns:
        eci_df["eci_ci_low"] = a + b * eci_df["capability_ci_low"]
        eci_df["eci_ci_high"] = a + b * eci_df["capability_ci_high"]

    # Apply same scaling to benchmark difficulties (EDI)
    edi_df = bench_df.copy()
    edi_df["edi"] = a + b * edi_df["difficulty"]

    # Scale discriminability to match ECI scale
    edi_df["discriminability_scaled"] = edi_df["discriminability"] / b

    # Scale confidence intervals if present
    if "difficulty_ci_low" in edi_df.columns:
        edi_df["edi_ci_low"] = a + b * edi_df["difficulty_ci_low"]
        edi_df["edi_ci_high"] = a + b * edi_df["difficulty_ci_high"]

    return eci_df, edi_df
