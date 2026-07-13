"""
IRT fitting for the Epoch Capabilities Index.

The model assumes benchmark performance follows a logistic curve:

    performance = sigmoid(discriminability * (capability - difficulty))

with one capability per model and one (difficulty, discriminability) pair
per benchmark. The fit is identified by pinning the anchor benchmark's
difficulty and discriminability. Everything in this module lives on that
raw scale; conversion to the public ECI/EDI scale and all
confidence-interval construction live in scaling.py.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from tqdm import tqdm

DEFAULT_ANCHOR_BENCHMARK = "Winogrande"
DEFAULT_ANCHOR_DIFFICULTY = 0.0
DEFAULT_ANCHOR_DISCRIMINABILITY = 1.0

BOOTSTRAP_METHODS = ("hierarchical", "observation")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _bootstrap_indices(
    rng: np.random.Generator,
    bootstrap_method: str,
    n_obs: int,
    rows_by_model: "list[np.ndarray] | None",
) -> np.ndarray:
    """Draw one bootstrap resample of observation indices."""
    if bootstrap_method == "hierarchical":
        # Hold models fixed; resample each model's results with replacement
        return np.concatenate([
            rng.choice(rows, size=rows.size, replace=True)
            for rows in rows_by_model
        ])
    return rng.integers(0, n_obs, size=n_obs)


def _irt_jacobian(
    params: np.ndarray,
    capability: np.ndarray,
    difficulty: np.ndarray,
    discriminability: np.ndarray,
    model_idx: np.ndarray,
    bench_idx: np.ndarray,
    anchor_idx: int,
    n_models: int,
    n_benchmarks: int,
    regularization_strength: float,
):
    """Sparse Jacobian of the residual vector: one row per observation plus
    the regularization row. Vectorized COO assembly."""
    n_obs = len(model_idx)
    n_params = params.size
    s = sigmoid(discriminability[bench_idx] * (capability[model_idx] - difficulty[bench_idx]))
    ds = s * (1 - s)
    obs_rows = np.arange(n_obs)

    cap_vals = ds * discriminability[bench_idx]
    diff_vals = -ds * discriminability[bench_idx]
    discrim_vals = ds * (capability[model_idx] - difficulty[bench_idx])

    # Discriminability parameter columns exist only for non-anchor benchmarks
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
        n_rows += 1
        penalty = regularization_strength * np.sum(params**2) / n_params
        if penalty > 0:
            scale = regularization_strength / (n_params * np.sqrt(penalty))
            rows.append(np.full(n_params, n_obs))
            cols.append(np.arange(n_params))
            vals.append(scale * params)

    jac = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_rows, n_params),
    )
    return jac.tocsr()


def load_benchmark_data(url: str = "https://epoch.ai/data/eci_benchmarks.csv") -> pd.DataFrame:
    """Load benchmark performance data from a CSV path or URL."""
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
    Fit the IRT model: model capabilities and benchmark parameters.

    All outputs are on the raw scale; pass them to scaling.compute_eci_scores
    for ECI-scale scores and confidence intervals.

    Args:
        df: DataFrame with columns model_id, benchmark_id, performance,
            benchmark, Model.
        anchor_benchmark: benchmark whose parameters are pinned to identify
            the fit.
        anchor_difficulty / anchor_discriminability: the pinned values.
        regularization_strength: L2 penalty on the parameter vector.
        performance_clip_eps: clip performance into [eps, 1 - eps].
        bootstrap_samples: number of bootstrap refits (0 to skip).
        bootstrap_seed: random seed for the resampling.
        bootstrap_method:
            - "hierarchical" (default): hold the set of models fixed and
              resample each model's benchmark results with replacement, so
              every model keeps its observation count in every resample.
            - "observation": resample all (model, benchmark) observations
              with replacement from the pooled data.
        use_analytical_jacobian: use the analytical sparse Jacobian (fast);
            False uses finite differences (slower, and may converge along a
            slightly different optimizer path).

    Returns:
        (model_df, bench_df, bootstrap_data):
        - model_df: model_id, Model, capability (sorted by capability desc).
        - bench_df: benchmark_id, benchmark, difficulty, discriminability,
          is_anchor (sorted by difficulty).
        - bootstrap_data: dict with model_ids, model_names, benchmark_ids,
          benchmark_names, and capability_samples / difficulty_samples /
          discriminability_samples, each a list of per-draw arrays (draws
          that fail to converge are skipped; empty when bootstrap_samples=0).
    """
    if bootstrap_method not in BOOTSTRAP_METHODS:
        raise ValueError(
            f"Unknown bootstrap_method '{bootstrap_method}'; "
            f"expected one of {BOOTSTRAP_METHODS}"
        )
    df = df.copy()
    if df["performance"].isna().any():
        raise ValueError("Performance data contains NaN values")
    if (df["performance"] < 0).any() or (df["performance"] > 1).any():
        raise ValueError("Performance scores must be in [0, 1] range")
    if performance_clip_eps > 0:
        df["performance"] = df["performance"].clip(
            performance_clip_eps, 1 - performance_clip_eps
        )

    model_ids = df["model_id"].unique()
    benchmark_ids = df["benchmark_id"].unique()
    n_models = len(model_ids)
    n_benchmarks = len(benchmark_ids)
    n_params = n_models + n_benchmarks + (n_benchmarks - 1)

    model_to_idx = {m: i for i, m in enumerate(model_ids)}
    bench_to_idx = {b: i for i, b in enumerate(benchmark_ids)}
    model_idx = np.array([model_to_idx[m] for m in df["model_id"]])
    bench_idx = np.array([bench_to_idx[b] for b in df["benchmark_id"]])
    performance = df["performance"].values

    try:
        anchor_bench_id = df.loc[df["benchmark"] == anchor_benchmark, "benchmark_id"].iloc[0]
    except IndexError:
        raise ValueError(f"Anchor benchmark '{anchor_benchmark}' not found in data")
    anchor_idx = bench_to_idx[anchor_bench_id]

    def unpack_params(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split the flat vector [capabilities, difficulties, free discriminabilities]."""
        capability = params[:n_models]
        difficulty = params[n_models:n_models + n_benchmarks]
        discriminability = np.insert(
            params[n_models + n_benchmarks:], anchor_idx, anchor_discriminability
        )
        return capability, difficulty, discriminability

    def residuals(params, performance, model_idx, bench_idx):
        capability, difficulty, discriminability = unpack_params(params)
        pred = sigmoid(discriminability[bench_idx] * (capability[model_idx] - difficulty[bench_idx]))
        resid = pred - performance
        if regularization_strength > 0:
            penalty = regularization_strength * np.sum(params**2) / n_params
            resid = np.append(resid, np.sqrt(penalty))
        return resid

    def jacobian(params, performance, model_idx, bench_idx):
        capability, difficulty, discriminability = unpack_params(params)
        return _irt_jacobian(
            params, capability, difficulty, discriminability,
            model_idx, bench_idx, anchor_idx,
            n_models, n_benchmarks, regularization_strength,
        )

    lower = np.concatenate([
        np.full(n_models, -10.0),
        np.full(n_benchmarks, -10.0),
        np.full(n_benchmarks - 1, 0.1),  # discriminability stays positive
    ])
    upper = np.concatenate([
        np.full(n_models, 10.0),
        np.full(n_benchmarks, 10.0),
        np.full(n_benchmarks - 1, 10.0),
    ])

    def fit(x0, performance, model_idx, bench_idx):
        return least_squares(
            residuals,
            x0,
            jac=jacobian if use_analytical_jacobian else "2-point",
            args=(performance, model_idx, bench_idx),
            bounds=(lower, upper),
            method="trf",
        )

    np.random.seed(42)
    init_params = np.concatenate([
        np.random.randn(n_models) * 0.1,
        np.random.randn(n_benchmarks) * 0.1,
        np.full(n_benchmarks - 1, 1.0),
    ])

    result = fit(init_params, performance, model_idx, bench_idx)
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    capability_hat, difficulty_hat, discriminability_hat = unpack_params(result.x)
    # Difficulties are fit freely; translate the solution so the anchor
    # benchmark sits exactly at its pinned difficulty.
    shift = difficulty_hat[anchor_idx] - anchor_difficulty
    capability_hat = capability_hat - shift
    difficulty_hat = difficulty_hat - shift

    capability_samples: list[np.ndarray] = []
    difficulty_samples: list[np.ndarray] = []
    discriminability_samples: list[np.ndarray] = []
    if bootstrap_samples > 0:
        rng = np.random.default_rng(bootstrap_seed)
        rows_by_model = None
        if bootstrap_method == "hierarchical":
            rows_by_model = [np.flatnonzero(model_idx == m) for m in range(n_models)]

        for _ in tqdm(range(bootstrap_samples), desc="Bootstrap", unit="sample"):
            idx = _bootstrap_indices(rng, bootstrap_method, len(performance), rows_by_model)
            try:
                boot = fit(result.x.copy(), performance[idx], model_idx[idx], bench_idx[idx])
            except Exception:
                continue
            if not boot.success:
                continue
            cap, diff, disc = unpack_params(boot.x)
            shift_b = diff[anchor_idx] - anchor_difficulty
            capability_samples.append(cap - shift_b)
            difficulty_samples.append(diff - shift_b)
            discriminability_samples.append(disc)

    # Bare statistical results; callers join their own metadata (see
    # scripts/fit_eci.py for an example)
    id_to_model = df.drop_duplicates("model_id").set_index("model_id")["Model"]
    id_to_bench = df.drop_duplicates("benchmark_id").set_index("benchmark_id")["benchmark"]
    model_names = [id_to_model[m] for m in model_ids]
    bench_names = [id_to_bench[b] for b in benchmark_ids]

    model_df = pd.DataFrame({
        "model_id": model_ids,
        "Model": model_names,
        "capability": capability_hat,
    }).sort_values("capability", ascending=False)

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
