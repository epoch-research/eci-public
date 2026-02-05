# Effect of Regularization and Reparametrization on ECI

This document summarizes an analysis of how regularization strength (λ) and α parametrization choices affect the ECI model estimates.

## Background

The ECI model fits: `performance = σ(α × (C - D))`

Where:
- C = capability (per model)
- D = difficulty (per benchmark)
- α = discriminability (per benchmark)

The model uses L2 regularization on parameters. Two parametrizations were compared:
- **Standard**: penalize α² directly
- **Log**: penalize log(α)²

## 1. Effect on Average Parameter Values

### Standard parametrization (penalize α²)

| Parameter | At λ = 10⁻⁴ (stable) | At λ = 0.1 (default) | Change |
|-----------|---------------------|---------------------|--------|
| avg \|C\| | 0.83 | 0.93 | +12% |
| avg \|D\| | 1.24 | 1.32 | +6% |
| avg \|α\| | 3.32 | 1.82 | **-45%** |

### Log parametrization (penalize log(α)²)

| Parameter | At λ = 10⁻⁴ (stable) | At λ = 0.1 (default) | Change |
|-----------|---------------------|---------------------|--------|
| avg \|C\| | 0.83 | 0.65 | -22% |
| avg \|D\| | 1.24 | 0.85 | -31% |
| avg \|α\| | 3.32 | 3.57 | **+8%** |

**Why the difference:** With standard parametrization, α² ≈ 9 is much larger than C² ≈ 0.7 or D² ≈ 1.5, so α takes most of the regularization pressure. With log parametrization, log(α)² ≈ 1.2 is comparable to C² and D², so the pressure is spread more evenly.

## 2. Effect on Capability Slope Over Time

| Parametrization | Slope at λ = 10⁻⁴ | Slope at λ = 0.1 | Change |
|-----------------|-------------------|------------------|--------|
| Standard | 0.77 units/year | 0.93 units/year | **+21%** |
| Log | 0.76 units/year | 0.72 units/year | -5% |

Both parametrizations agree in the stable region (~0.76-0.77 units/year), but diverge at the default λ = 0.1.

## 3. Mechanism for Slope Inflation (Standard Parametrization)

1. **High-α benchmarks exist at low λ**: Some benchmarks (FrontierMath, ARC-AGI) have α ≈ 5-10 when regularization is weak

2. **These are recent/hard benchmarks**: High-α benchmarks tend to be newer and more difficult

3. **Newer models are evaluated on more high-α benchmarks**: Correlation between model release date and fraction of evaluations on high-α benchmarks

4. **Regularization compresses α differentially**: When λ increases, high-α benchmarks shrink most (e.g., FrontierMath: α = 10 → 4.6)

5. **The model compensates by expanding (C - D)**: To maintain the same predicted performance σ(α(C-D)), when α shrinks, (C-D) must grow

6. **This inflates capabilities of models on high-α benchmarks**: Models evaluated primarily on high-α benchmarks get larger capability estimates

7. **Since those models are newer, the time slope is inflated**: The systematic capability inflation correlates with release date, steepening the slope

## 4. Tradeoffs Between Parametrizations

**Standard parametrization at default λ:**
- C and D relatively stable (change ~6-12%)
- α changes a lot (-45%)
- Slope inflated by +21%

**Log parametrization at default λ:**
- C and D change a lot (-22% and -31%)
- α relatively stable (+8%)
- Slope only changes by -5%

## 5. Key Findings

- The default λ = 0.1 is **outside the stable region** (λ ≤ 10⁻³) for both parametrizations
- With standard parametrization, the default λ inflates the capability-vs-time slope by ~21%
- Log parametrization is more robust to regularization choice for slope estimates, but less stable for absolute C/D values
- Both parametrizations agree when λ is in the stable region

## 6. Recommendations

1. **Use smaller λ**: A value in the stable region (λ ≤ 10⁻³) would give more robust estimates regardless of parametrization choice

2. **Choice of parametrization depends on use case**:
   - If tracking trends over time → log parametrization is more robust
   - If absolute capability/difficulty values matter → standard parametrization (with small λ) may be preferable

3. **Report sensitivity**: Any analysis should note sensitivity to these choices, especially if using default λ = 0.1

## Files

- `scripts/analyze_alpha_parametrization.py`: Analysis script
- `outputs/parametrization/`: Output data and plots
