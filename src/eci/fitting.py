"""
Fitting for the Epoch Capabilities Index (ECI).

The model assumes benchmark performance follows a logistic curve:

    performance = sigmoid(discriminability * (capability - difficulty))

with one capability per model and one (difficulty, discriminability) pair
per benchmark. Two anchors identify the scales:

- The raw fit pins the anchor *benchmark* (Winogrande: difficulty = 0,
  discriminability = 1).
- The public ECI scale is defined by two anchor *models* (Claude 3.5 Sonnet
  -> 130, GPT-5 -> 150) via the affine map ``eci = a + b * capability``.
  The same map sends benchmark difficulties to EDI, and dividing
  discriminabilities by ``b`` preserves the predictions.

Every bootstrap refit lands on its own raw scale, so the affine map is
recomputed per draw from that draw's anchor capabilities before quantile
confidence intervals are taken. The anchor models therefore sit at exactly
130/150 in every draw, and their CI cells are NaN.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from tqdm import tqdm

DEFAULT_ANCHOR_BENCHMARK = "Winogrande"
DEFAULT_ANCHOR_DIFFICULTY = 0.0
DEFAULT_ANCHOR_DISCRIMINABILITY = 1.0

# The anchor model names are matched against the "Model" column of the fit
# input, i.e. the Airtable "Model aggregation" names. Renaming an aggregation
# in Airtable requires a coordinated change here (the fit raises if an anchor
# is missing).
DEFAULT_ANCHOR_MODEL_LOW = "Claude 3.5 Sonnet"
DEFAULT_ANCHOR_ECI_LOW = 130.0
DEFAULT_ANCHOR_MODEL_HIGH = "GPT-5"
DEFAULT_ANCHOR_ECI_HIGH = 150.0

_MIN_ANCHOR_SPREAD = 1e-12


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


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
    the regularization row.

    This exists purely for speed: without it, least_squares estimates the
    Jacobian by finite differences, at one residual evaluation per parameter
    per optimizer step.
    """
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


def _affine_map(cap_low, cap_high, eci_low, eci_high):
    """(a, b) of the map ``eci = a + b * capability`` sending cap_low to
    eci_low and cap_high to eci_high. Vectorized over draws."""
    spread = np.asarray(cap_high) - np.asarray(cap_low)
    bad = np.flatnonzero(~(spread > _MIN_ANCHOR_SPREAD))  # includes NaN
    if bad.size:
        where = f"draw(s) {bad.tolist()}" if spread.ndim else "the central fit"
        raise ValueError(
            f"anchor capabilities in {where} do not define a scale: "
            "the high anchor must sit above the low anchor"
        )
    b = (eci_high - eci_low) / spread
    return eci_low - b * np.asarray(cap_low), b


def _anchor_capability(model_df: pd.DataFrame, anchor_model: str) -> float:
    rows = model_df.loc[model_df["Model"] == anchor_model, "capability"]
    if rows.empty:
        raise ValueError(f"Anchor model '{anchor_model}' not found")
    return float(rows.iloc[0])


def _scale_to_eci(
    model_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    bootstrap_data: "dict | None",
    *,
    anchor_model_low: str,
    anchor_eci_low: float,
    anchor_model_high: str,
    anchor_eci_high: float,
    ci_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame, "dict | None"]:
    """Map a raw fit onto the ECI scale.

    Central estimates use the (a, b) derived from the central fit's anchor
    capabilities; each bootstrap draw uses the (a, b) derived from its own
    anchor capabilities, and the CI columns are the ci_level quantiles of
    the scaled draws.
    """
    a, b = _affine_map(
        _anchor_capability(model_df, anchor_model_low),
        _anchor_capability(model_df, anchor_model_high),
        anchor_eci_low,
        anchor_eci_high,
    )
    eci_df = model_df[["model_id", "Model"]].copy()
    eci_df["eci"] = a + b * model_df["capability"].to_numpy()
    edi_df = bench_df[["benchmark_id", "benchmark", "is_anchor"]].copy()
    edi_df["edi"] = a + b * bench_df["difficulty"].to_numpy()
    edi_df["discriminability_scaled"] = bench_df["discriminability"].to_numpy() / b

    if not bootstrap_data or not len(bootstrap_data["capability_samples"]):
        return eci_df, edi_df, None

    model_names = list(bootstrap_data["model_names"])
    try:
        idx_low = model_names.index(anchor_model_low)
        idx_high = model_names.index(anchor_model_high)
    except ValueError as e:
        raise ValueError(f"Anchor model missing from bootstrap samples: {e}") from None

    caps = np.asarray(bootstrap_data["capability_samples"], dtype=float)
    diffs = np.asarray(bootstrap_data["difficulty_samples"], dtype=float)
    discs = np.asarray(bootstrap_data["discriminability_samples"], dtype=float)
    a_s, b_s = _affine_map(
        caps[:, idx_low], caps[:, idx_high], anchor_eci_low, anchor_eci_high
    )
    draws = {
        "model_ids": list(bootstrap_data["model_ids"]),
        "model_names": model_names,
        "benchmark_ids": list(bootstrap_data["benchmark_ids"]),
        "benchmark_names": list(bootstrap_data["benchmark_names"]),
        "eci": a_s[:, None] + b_s[:, None] * caps,
        "edi": a_s[:, None] + b_s[:, None] * diffs,
        "slope": discs / b_s[:, None],
        "a": a_s,
        "b": b_s,
    }

    tail = (1.0 - ci_level) / 2.0
    # model_df is sorted by capability while the draw columns are in fit
    # insertion order, so align the quantiles by id.
    lo, hi = np.quantile(draws["eci"], [tail, 1.0 - tail], axis=0)
    eci_df["eci_ci_low"] = eci_df["model_id"].map(pd.Series(lo, index=draws["model_ids"]))
    eci_df["eci_ci_high"] = eci_df["model_id"].map(pd.Series(hi, index=draws["model_ids"]))
    # The anchor models' values are fixed by definition, not estimated
    anchor_rows = eci_df["Model"].isin([anchor_model_low, anchor_model_high])
    eci_df.loc[anchor_rows, ["eci_ci_low", "eci_ci_high"]] = np.nan

    lo, hi = np.quantile(draws["edi"], [tail, 1.0 - tail], axis=0)
    edi_df["edi_ci_low"] = edi_df["benchmark_id"].map(pd.Series(lo, index=draws["benchmark_ids"]))
    edi_df["edi_ci_high"] = edi_df["benchmark_id"].map(pd.Series(hi, index=draws["benchmark_ids"]))

    return eci_df, edi_df, draws


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
    use_analytical_jacobian: bool = True,
    *,
    anchor_model_low: str = DEFAULT_ANCHOR_MODEL_LOW,
    anchor_eci_low: float = DEFAULT_ANCHOR_ECI_LOW,
    anchor_model_high: str = DEFAULT_ANCHOR_MODEL_HIGH,
    anchor_eci_high: float = DEFAULT_ANCHOR_ECI_HIGH,
    ci_level: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame, "dict | None"]:
    """
    Fit the IRT model and return ECI-scale scores with bootstrap CIs.

    Args:
        df: DataFrame with columns model_id, benchmark_id, performance,
            benchmark, Model.
        anchor_benchmark: benchmark whose parameters are pinned to identify
            the raw fit.
        anchor_difficulty / anchor_discriminability: the pinned values.
        regularization_strength: L2 penalty on the parameter vector.
        performance_clip_eps: clip performance into [eps, 1 - eps].
        bootstrap_samples: number of bootstrap refits (0 to skip). Refits
            hold the set of models fixed and resample each model's benchmark
            results with replacement, so every model keeps its observation
            count in every resample.
        bootstrap_seed: random seed for the resampling.
        use_analytical_jacobian: use the analytical sparse Jacobian (fast);
            False uses finite differences (slower, and may converge along a
            slightly different optimizer path).
        anchor_model_low / anchor_eci_low / anchor_model_high /
            anchor_eci_high: the models that define the ECI scale and their
            fixed values.
        ci_level: central quantile mass for the CIs (0.90 -> 5th/95th).

    Returns:
        (eci_df, edi_df, draws), everything on the ECI scale:
        - eci_df: model_id, Model, eci, and eci_ci_low / eci_ci_high when
          bootstrapping (NaN for the anchor models, whose values are fixed
          by definition rather than estimated). Sorted by eci descending.
        - edi_df: benchmark_id, benchmark, is_anchor, edi,
          discriminability_scaled, and edi_ci_low / edi_ci_high when
          bootstrapping. Sorted by edi.
        - draws: None without bootstrapping, else a dict with model_ids,
          model_names, benchmark_ids, benchmark_names, the (n_draws, n)
          arrays eci, edi, slope, and the per-draw map coefficients a, b.
          Raises ValueError if any draw's anchor capabilities coincide or
          invert (such a draw defines no scale).
    """
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
        """Split the flat vector [capabilities, difficulties, free
        discriminabilities], re-inserting the anchor benchmark's pinned
        discriminability (it is not a parameter)."""
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

    result = fit(x0=init_params, performance=performance,
                 model_idx=model_idx, bench_idx=bench_idx)
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
        rows_by_model = [np.flatnonzero(model_idx == m) for m in range(n_models)]
        for _ in tqdm(range(bootstrap_samples), desc="Bootstrap", unit="sample"):
            # Resample each model's observations with replacement
            idx = np.concatenate([
                rng.choice(rows, size=rows.size, replace=True)
                for rows in rows_by_model
            ])
            try:
                # Warm-start each refit from the central solution
                boot = fit(x0=result.x, performance=performance[idx],
                           model_idx=model_idx[idx], bench_idx=bench_idx[idx])
            except Exception:
                continue
            if not boot.success:
                continue
            cap, diff, disc = unpack_params(boot.x)
            shift_b = diff[anchor_idx] - anchor_difficulty
            capability_samples.append(cap - shift_b)
            difficulty_samples.append(diff - shift_b)
            discriminability_samples.append(disc)

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
    return _scale_to_eci(
        model_df, bench_df, bootstrap_data,
        anchor_model_low=anchor_model_low,
        anchor_eci_low=anchor_eci_low,
        anchor_model_high=anchor_model_high,
        anchor_eci_high=anchor_eci_high,
        ci_level=ci_level,
    )
