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

Everything in this module lives on the RAW scale, identified by the anchor
benchmark below. Conversion to the public ECI/EDI scale - including all
confidence-interval construction - lives in scaling.py; this module
deliberately exposes bootstrap draws rather than CIs, so that CIs can only
be built with the correct per-draw re-anchoring.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from tqdm import tqdm


# Default benchmark anchor for model identification
DEFAULT_ANCHOR_BENCHMARK = "Winogrande"
DEFAULT_ANCHOR_DIFFICULTY = 0.0
DEFAULT_ANCHOR_DISCRIMINABILITY = 1.0

# Supported bootstrap resampling schemes
BOOTSTRAP_METHODS = ("hierarchical", "observation")


def _validate_bootstrap_method(bootstrap_method: str) -> None:
    if bootstrap_method not in BOOTSTRAP_METHODS:
        raise ValueError(
            f"Unknown bootstrap_method '{bootstrap_method}'; "
            f"expected one of {BOOTSTRAP_METHODS}"
        )


def _bootstrap_indices(
    rng: np.random.Generator,
    bootstrap_method: str,
    n_obs: int,
    rows_by_model: list[np.ndarray] | None,
) -> np.ndarray:
    """Draw one bootstrap resample of observation indices."""
    if bootstrap_method == "hierarchical":
        # Hold models fixed; resample each model's results with replacement
        return np.concatenate([
            rng.choice(rows, size=rows.size, replace=True)
            for rows in rows_by_model
        ])
    return rng.integers(0, n_obs, size=n_obs)


def _sigmoid_derivative(
    capability: np.ndarray,
    difficulty: np.ndarray,
    discriminability: np.ndarray,
    model_idx: np.ndarray,
    bench_idx: np.ndarray,
) -> np.ndarray:
    z = discriminability[bench_idx] * (capability[model_idx] - difficulty[bench_idx])
    s = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    return s * (1 - s)


def _irt_jacobian(
    capability: np.ndarray,
    difficulty: np.ndarray,
    discriminability: np.ndarray,
    model_idx: np.ndarray,
    bench_idx: np.ndarray,
    anchor_idx: int,
    anchor_discriminability: float,
    n_models: int,
    n_benchmarks: int,
    n_params: int,
    regularization_strength: float,
):
    """
    Sparse Jacobian of the residual vector for the full IRT model.

    Vectorized COO assembly; numerically identical to looping over
    observations, but without Python-level per-observation cost.
    """
    n_obs = len(model_idx)
    ds = _sigmoid_derivative(capability, difficulty, discriminability, model_idx, bench_idx)
    obs_rows = np.arange(n_obs)

    # d(resid_i)/d(cap_m), d(resid_i)/d(diff_b), d(resid_i)/d(discrim_b)
    cap_vals = ds * discriminability[bench_idx]
    diff_vals = -ds * discriminability[bench_idx]
    discrim_vals = ds * (capability[model_idx] - difficulty[bench_idx])

    # Free-discriminability parameter columns (anchor benchmark is fixed)
    free = bench_idx != anchor_idx
    discrim_cols = (
        n_models + n_benchmarks
        + np.where(bench_idx < anchor_idx, bench_idx, bench_idx - 1)
    )

    rows = [obs_rows, obs_rows, obs_rows[free]]
    cols = [model_idx, n_models + bench_idx, discrim_cols[free]]
    vals = [cap_vals, diff_vals, discrim_vals[free]]

    n_rows = n_obs
    if regularization_strength > 0:
        n_rows = n_obs + 1
        reg_penalty = regularization_strength * (
            np.sum(capability**2) +
            np.sum(difficulty**2) +
            np.sum(discriminability[discriminability != anchor_discriminability]**2)
        ) / n_params

        if reg_penalty > 0:
            scale = regularization_strength / (n_params * np.sqrt(reg_penalty))
            free_bench = np.flatnonzero(np.arange(n_benchmarks) != anchor_idx)
            free_bench_cols = (
                n_models + n_benchmarks
                + np.where(free_bench < anchor_idx, free_bench, free_bench - 1)
            )
            reg_cols = np.concatenate([
                np.arange(n_models),
                n_models + np.arange(n_benchmarks),
                free_bench_cols,
            ])
            reg_vals = np.concatenate([
                scale * capability,
                scale * difficulty,
                scale * discriminability[free_bench],
            ])
            rows.append(np.full(reg_cols.size, n_obs))
            cols.append(reg_cols)
            vals.append(reg_vals)

    jac = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_rows, n_params),
    )
    return jac.tocsr()


def _capability_jacobian(
    capability: np.ndarray,
    model_idx: np.ndarray,
    difficulty: np.ndarray,
    discriminability: np.ndarray,
    n_models: int,
    regularization_strength: float,
):
    """Sparse Jacobian for the capabilities-only fit (benchmark params fixed)."""
    n_obs = len(model_idx)
    z = discriminability * (capability[model_idx] - difficulty)
    s = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    ds = s * (1 - s)

    rows = [np.arange(n_obs)]
    cols = [model_idx]
    vals = [ds * discriminability]

    n_rows = n_obs
    if regularization_strength > 0:
        n_rows = n_obs + 1
        reg_penalty = regularization_strength * np.sum(capability**2) / n_models
        if reg_penalty > 0:
            scale = regularization_strength / (n_models * np.sqrt(reg_penalty))
            rows.append(np.full(n_models, n_obs))
            cols.append(np.arange(n_models))
            vals.append(scale * capability)

    jac = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_rows, n_models),
    )
    return jac.tocsr()


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
    bootstrap_samples: int = 500,
    bootstrap_seed: int = 12345,
    bootstrap_method: str = "hierarchical",
    use_analytical_jacobian: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Fit the IRT model to estimate model capabilities and benchmark difficulties.

    The model uses a logistic (sigmoid) function:
        performance = sigmoid(discriminability * (capability - difficulty))

    To identify the model (avoid infinite solutions), we anchor one benchmark's
    difficulty and discriminability to fixed values.

    All outputs are on the raw scale and contain point estimates plus raw
    bootstrap draws - no confidence intervals. CIs are only meaningful on
    the ECI scale, with each draw re-anchored individually; pass the returned
    bootstrap_data to scaling.compute_eci_scores to construct them.

    Args:
        df: DataFrame with columns model_id, benchmark_id, performance, benchmark, Model.
        anchor_benchmark: Name of benchmark to anchor (fixes scale location).
        anchor_difficulty: Fixed difficulty value for anchor benchmark.
        anchor_discriminability: Fixed discriminability for anchor benchmark.
        regularization_strength: L2 regularization to prevent extreme values (0-1).
        performance_clip_eps: Clip performance to [eps, 1-eps] to avoid degeneracy.
        bootstrap_samples: Number of bootstrap resamples to draw (0 to skip).
        bootstrap_seed: Random seed for reproducibility.
        bootstrap_method: Resampling scheme for the bootstrap draws:
            - "hierarchical" (default): hold the set of models fixed and
              resample each model's benchmark results with replacement, so
              every model keeps its observation count in every resample.
            - "observation": resample all (model, benchmark) observations with
              replacement from the pooled data. A model can lose all of its
              observations in a resample.
        use_analytical_jacobian: If True, use analytical Jacobian for faster optimization.
            If False, use numerical differentiation (slower but may give slightly
            different results due to optimizer path differences).

    Returns:
        Tuple of (model_df, bench_df, bootstrap_data):
        - model_df: model_id, Model, capability (sorted by capability desc).
        - bench_df: benchmark_id, benchmark, difficulty, discriminability,
          is_anchor (sorted by difficulty).
        - bootstrap_data: dict with keys 'model_ids', 'model_names',
          'benchmark_ids', 'benchmark_names', 'capability_samples',
          'difficulty_samples', 'discriminability_samples'. Each *_samples
          value is a list of up to bootstrap_samples 1-D numpy arrays
          (draws that fail to converge are skipped); empty lists when
          bootstrap_samples=0.
    """
    df = df.copy()

    # Validate inputs
    _validate_bootstrap_method(bootstrap_method)
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

    def jacobian(params: np.ndarray):
        """Analytical Jacobian for faster optimization."""
        capability, difficulty, discriminability = unpack_params(params)
        return _irt_jacobian(
            capability, difficulty, discriminability,
            model_idx, bench_idx,
            anchor_idx, anchor_discriminability,
            n_models, n_benchmarks, n_params,
            regularization_strength,
        )

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

    # Bootstrap draws (raw scale; CI construction happens in scaling.py)
    capability_samples: list[np.ndarray] = []
    difficulty_samples: list[np.ndarray] = []
    discriminability_samples: list[np.ndarray] = []

    if bootstrap_samples > 0:
        rng = np.random.default_rng(bootstrap_seed)

        rows_by_model = None
        if bootstrap_method == "hierarchical":
            rows_by_model = [np.flatnonzero(model_idx == m) for m in range(n_models)]

        for _ in tqdm(range(bootstrap_samples), desc="Bootstrap", unit="sample"):
            idx = _bootstrap_indices(
                rng, bootstrap_method, len(performance), rows_by_model
            )
            boot_performance = performance[idx]
            boot_model_idx = model_idx[idx]
            boot_bench_idx = bench_idx[idx]

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
                return _irt_jacobian(
                    cap, diff, disc,
                    boot_model_idx, boot_bench_idx,
                    anchor_idx, anchor_discriminability,
                    n_models, n_benchmarks, n_params,
                    regularization_strength,
                )

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
                    cap, diff, disc = unpack_params(boot_result.x)
                    shift_b = diff[anchor_idx] - anchor_difficulty
                    capability_samples.append(cap - shift_b)
                    difficulty_samples.append(diff - shift_b)
                    discriminability_samples.append(disc.copy())
            except Exception:
                continue

    # Build output DataFrames (bare statistical results; callers join their
    # own metadata - see scripts/ for examples)
    model_names = [id_to_model_name[m] for m in model_ids]
    model_df = pd.DataFrame({
        "model_id": model_ids,
        "Model": model_names,
        "capability": capability_hat,
    }).sort_values("capability", ascending=False)

    bench_names = [id_to_bench_name[b] for b in benchmark_ids]
    bench_df = pd.DataFrame({
        "benchmark_id": benchmark_ids,
        "benchmark": bench_names,
        "difficulty": difficulty_hat,
        "discriminability": discriminability_hat,
        "is_anchor": [b == anchor_bench_id for b in benchmark_ids],
    }).sort_values("difficulty")

    bootstrap_data = {
        "model_ids": list(model_ids),
        "model_names": model_names,
        "benchmark_ids": list(benchmark_ids),
        "benchmark_names": bench_names,
        "capability_samples": capability_samples,
        "difficulty_samples": difficulty_samples,
        "discriminability_samples": discriminability_samples,
    }
    return model_df, bench_df, bootstrap_data


def fit_capabilities_given_benchmarks(
    df: pd.DataFrame,
    bench_df: pd.DataFrame,
    regularization_strength: float = 0.1,
    performance_clip_eps: float = 1e-3,
    bootstrap_samples: int = 500,
    bootstrap_seed: int = 12345,
    bootstrap_method: str = "hierarchical",
    ci_level: float = 0.90,
    use_analytical_jacobian: bool = True,
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
        bootstrap_method: Resampling scheme for confidence intervals;
            see fit_eci_model for the available options.
        ci_level: Confidence interval level (e.g., 0.90 for 90% CI).
        use_analytical_jacobian: If True, use analytical Jacobian for faster
            optimization. If False, use numerical differentiation (slower,
            reproduces the original behavior of this function exactly).

    Returns:
        DataFrame with model capabilities and confidence intervals.
    """
    df = df.copy()

    # Validate inputs
    _validate_bootstrap_method(bootstrap_method)
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

    def jacobian(capability: np.ndarray):
        return _capability_jacobian(
            capability, model_idx, difficulty, discriminability,
            n_models, regularization_strength,
        )

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
        jac=jacobian if use_analytical_jacobian else "2-point",
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

        rows_by_model = None
        if bootstrap_method == "hierarchical":
            rows_by_model = [np.flatnonzero(model_idx == m) for m in range(n_models)]

        for _ in tqdm(range(bootstrap_samples), desc="Bootstrap", unit="sample"):
            idx = _bootstrap_indices(
                rng, bootstrap_method, len(performance), rows_by_model
            )
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

            def boot_jacobian(cap):
                return _capability_jacobian(
                    cap, boot_model_idx, boot_difficulty, boot_discriminability,
                    n_models, regularization_strength,
                )

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
    })

    # Preserve model metadata columns from input (date, Organization, model_version, etc.)
    metadata_cols = [c for c in df.columns if c not in [
        "model_id", "benchmark_id", "performance", "benchmark", "Model",
        "benchmark_release_date", "optimized", "is_math", "is_coding",
        "random_baseline", "model"
    ]]
    if metadata_cols:
        model_metadata = df.drop_duplicates("model_id").set_index("model_id")[metadata_cols]
        for col in metadata_cols:
            if col in model_metadata.columns:
                model_df[col] = model_df["model_id"].map(model_metadata[col].to_dict())

    model_df = model_df.sort_values("capability", ascending=False)

    return model_df

