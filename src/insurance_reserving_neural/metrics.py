"""
metrics.py — Evaluation metrics for individual claims reserving.

Three primary metrics from the literature, plus a chain-ladder baseline:

MALE (Mean Absolute Log Error)
    The standard metric from Avanzi et al. (2025). Measures per-claim
    accuracy in log space — less sensitive to large outliers than RMSE.
    MALE = (1/n) × Σ |log(Ŷ_i / Y_i)|
    where Y_i = ultimate_i - paid_i (actual outstanding)
    and   Ŷ_i = predicted outstanding

OCLerr (Outstanding Claims Liability Error)
    Portfolio-level bias metric. Tells you whether the model over- or
    under-reserves in aggregate — what the finance director cares about.
    OCLerr = (Σ Ŷ_i - Σ Y_i) / Σ Y_i
    OCLerr > 0 means over-reserved; OCLerr < 0 means under-reserved.
    Regulators want this close to zero; under-reserving is the capital risk.

reserve_range
    Bootstrap-based confidence interval for the total RBNS reserve.
    Returns P10/P50/P75/P90/P99.5 — the Solvency II percentiles.

chain_ladder_reserve
    Aggregate chain-ladder baseline computed directly from the individual
    claims data. Required for regulatory benchmarking under PRA SS8/24.
    We implement the simple volume-weighted development factor method —
    no external dependencies needed.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import polars as pl
from scipy import stats


def mean_absolute_log_error(
    preds_df: pl.DataFrame,
    actual_col: str = "ultimate",
    paid_col: str = "cumulative_paid",
    pred_col: str = "predicted_outstanding",
    clip_min: float = 1.0,
) -> float:
    """
    Mean absolute log error (MALE) on settled claims.

    Only rows where actual outstanding > clip_min are included — very small
    outstanding amounts create numerical instability in log space and are
    usually rounding artefacts.

    Parameters
    ----------
    preds_df : pl.DataFrame
        Output from model.predict(), must contain actual_col and paid_col.
    actual_col : str
        Column name for the ultimate settlement amount. Default 'ultimate'.
    paid_col : str
        Column name for cumulative paid. Default 'cumulative_paid'.
    pred_col : str
        Column name for the predicted outstanding. Default 'predicted_outstanding'.
    clip_min : float
        Minimum actual outstanding to include in MALE. Default 1.0.

    Returns
    -------
    float
        Mean absolute log error (lower is better; 0.0 is perfect).
    """
    df = preds_df.filter(
        (pl.col("is_open") == False)
        & (pl.col(actual_col).is_not_null())
        & ((pl.col(actual_col) - pl.col(paid_col)) >= clip_min)
        & (pl.col(pred_col) >= clip_min)
    )

    if len(df) == 0:
        raise ValueError(
            "No settled claims found for MALE computation. "
            "Ensure is_open=False rows with known ultimate are included."
        )

    actual_outstanding = (df[actual_col] - df[paid_col]).to_numpy()
    predicted = df[pred_col].to_numpy()

    log_ratio = np.abs(np.log(predicted / actual_outstanding))
    return float(np.mean(log_ratio))


def ocl_error(
    preds_df: pl.DataFrame,
    actual_col: str = "ultimate",
    paid_col: str = "cumulative_paid",
    pred_col: str = "predicted_outstanding",
) -> float:
    """
    Outstanding Claims Liability Error (OCLerr) — portfolio-level bias.

    OCLerr = (sum(predicted) - sum(actual)) / sum(actual)

    Positive: model over-reserves (conservative, capital-inefficient).
    Negative: model under-reserves (dangerous, regulatory concern).

    Parameters
    ----------
    preds_df : pl.DataFrame
        Predictions on settled claims (is_open=False, known ultimate).
    actual_col, paid_col, pred_col : str
        Column names as described above.

    Returns
    -------
    float
        Signed relative bias at portfolio level.
    """
    df = preds_df.filter(
        (pl.col("is_open") == False)
        & (pl.col(actual_col).is_not_null())
        & ((pl.col(actual_col) - pl.col(paid_col)) > 0)
    )

    if len(df) == 0:
        raise ValueError(
            "No settled claims with positive outstanding found. "
            "Cannot compute OCLerr."
        )

    actual_total = float((df[actual_col] - df[paid_col]).sum())
    predicted_total = float(df[pred_col].sum())

    if actual_total == 0:
        raise ValueError("Sum of actual outstanding is zero — cannot compute OCLerr.")

    return (predicted_total - actual_total) / actual_total


def reserve_range(
    preds_df: pl.DataFrame,
    pred_col: str = "predicted_outstanding",
    percentiles: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Bootstrap-based confidence intervals for the portfolio reserve.

    Resamples predicted outstanding values (one row per open claim) to
    approximate the sampling distribution of the total RBNS reserve.
    This is a simplified parametric bootstrap on the predictions themselves —
    for full residual bootstrap, use BootstrapReserver.

    Parameters
    ----------
    preds_df : pl.DataFrame
        Predictions for open claims (is_open=True rows).
    pred_col : str
        Column of predicted outstanding amounts.
    percentiles : list of float | None
        Percentiles to return. Default is [10, 50, 75, 90, 99.5]
        matching Solvency II disclosure requirements.

    Returns
    -------
    dict
        Keys: 'mean', 'std', 'P10', 'P50', 'P75', 'P90', 'P99_5' (and others).
    """
    if percentiles is None:
        percentiles = [10.0, 50.0, 75.0, 90.0, 99.5]

    open_preds = preds_df.filter(pl.col("is_open") == True)
    if len(open_preds) == 0:
        raise ValueError("No open claims found for reserve_range computation.")

    values = open_preds[pred_col].to_numpy()

    # Bootstrap the total reserve
    n_boot = 10_000
    rng = np.random.default_rng(42)
    n = len(values)
    boot_totals = np.array([
        rng.choice(values, size=n, replace=True).sum()
        for _ in range(n_boot)
    ])

    result = {
        "mean": float(np.mean(boot_totals)),
        "std": float(np.std(boot_totals)),
        "point_estimate": float(values.sum()),
    }

    for p in percentiles:
        key = f"P{str(p).replace('.', '_')}"
        result[key] = float(np.percentile(boot_totals, p))

    return result


def chain_ladder_reserve(
    df: pl.DataFrame,
    accident_col: str = "accident_date",
    dev_col: str = "dev_quarter",
    paid_col: str = "cumulative_paid",
    open_col: str = "is_open",
    max_dev: Optional[int] = None,
) -> Dict[str, float]:
    """
    Chain-ladder reserve estimate from individual claims data.

    Aggregates individual claims into a development triangle, then applies
    volume-weighted development factors. This is the regulatory benchmark
    required by PRA SS8/24 for model validation.

    We compute the triangle in pure NumPy/Polars — no external chainladder
    dependency. The result is the IBNR-inclusive development projection,
    which you compare against the neural model's RBNS to assess bias.

    Parameters
    ----------
    df : pl.DataFrame
        Claims panel. Uses the last observation per claim per accident quarter
        to build the triangle.
    accident_col : str
        Column containing accident date (used to derive accident quarter).
    dev_col : str
        Development quarter column (0-indexed). Default 'dev_quarter'.
    paid_col : str
        Cumulative paid column. Default 'cumulative_paid'.
    open_col : str
        Open indicator. Default 'is_open'.
    max_dev : int | None
        Maximum development quarter to include. If None, infers from data.

    Returns
    -------
    dict
        'cdf': list of CDFs by development quarter
        'dev_factors': list of age-to-age factors
        'reserve': total reserve estimate from chain-ladder
        'triangle': the raw cumulative paid triangle (as dict)
    """
    # Use only data at each valuation point — latest observation per claim/dev
    # First, need to prepare dev_quarter if not present
    if dev_col not in df.columns:
        df = df.with_columns([
            (
                (pl.col("valuation_date").dt.year() - pl.col(accident_col).dt.year()) * 4
                + (pl.col("valuation_date").dt.quarter() - pl.col(accident_col).dt.quarter())
            ).cast(pl.Int32).alias(dev_col)
        ])

    # Get accident quarter from accident_date
    if "accident_quarter_idx" not in df.columns:
        df = df.with_columns([
            (
                (pl.col(accident_col).dt.year() - pl.col(accident_col).dt.year().min()) * 4
                + pl.col(accident_col).dt.quarter()
            ).cast(pl.Int32).alias("accident_quarter_idx")
        ])

    # Build triangle: for each (accident_quarter, dev_quarter), take max cumulative paid
    triangle_df = (
        df.group_by(["accident_quarter_idx", dev_col])
        .agg(pl.col(paid_col).max().alias("cum_paid"))
        .sort(["accident_quarter_idx", dev_col])
    )

    acc_quarters = sorted(triangle_df["accident_quarter_idx"].unique().to_list())
    dev_quarters = sorted(triangle_df[dev_col].unique().to_list())
    n_acc = len(acc_quarters)
    n_dev = len(dev_quarters)

    if max_dev is not None:
        dev_quarters = [d for d in dev_quarters if d <= max_dev]

    # Build triangle matrix (n_acc x n_dev), NaN where not yet observed
    triangle = np.full((n_acc, len(dev_quarters)), np.nan)
    acc_idx_map = {a: i for i, a in enumerate(acc_quarters)}
    dev_idx_map = {d: j for j, d in enumerate(dev_quarters)}

    for row in triangle_df.iter_rows(named=True):
        i = acc_idx_map.get(row["accident_quarter_idx"])
        j = dev_idx_map.get(row[dev_col])
        if i is not None and j is not None:
            triangle[i, j] = row["cum_paid"]

    # Volume-weighted development factors
    n_factors = len(dev_quarters) - 1
    dev_factors = []
    for j in range(n_factors):
        # Sum over accident quarters where both columns j and j+1 are observed
        col_j = triangle[:, j]
        col_j1 = triangle[:, j + 1]
        mask = np.isfinite(col_j) & np.isfinite(col_j1)
        if mask.sum() >= 1:
            factor = col_j1[mask].sum() / col_j[mask].sum()
        else:
            factor = 1.0
        dev_factors.append(float(factor))

    # Chain-ladder CDFs (tail factors)
    cdfs = []
    cdf = 1.0
    for f in reversed(dev_factors):
        cdf *= f
        cdfs.insert(0, cdf)
    cdfs.append(1.0)  # Ultimate CDF = 1.0

    # Reserve for each accident quarter: CL projected ultimate - current paid
    reserve = 0.0
    for i in range(n_acc):
        row_data = triangle[i]
        # Find the latest non-null development
        valid = np.where(np.isfinite(row_data))[0]
        if len(valid) == 0:
            continue
        latest_dev_idx = valid[-1]
        current_paid = row_data[latest_dev_idx]
        cl_cdf = cdfs[latest_dev_idx] if latest_dev_idx < len(cdfs) else 1.0
        projected_ultimate = current_paid * cl_cdf
        reserve += max(0.0, projected_ultimate - current_paid)

    return {
        "reserve": reserve,
        "dev_factors": dev_factors,
        "cdfs": cdfs,
        "n_accident_quarters": n_acc,
        "n_dev_quarters": len(dev_quarters),
    }
