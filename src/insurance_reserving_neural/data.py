"""
data.py — ClaimSchema and feature preparation for individual neural reserving.

The data contract is a Polars DataFrame where each row represents one claim
at one valuation date (the "panel" or "long" format). This means a claim open
for 8 quarters appears 8 times, each row containing the history available at
that valuation date.

Required columns
----------------
claim_id         : str   — unique claim identifier
accident_date    : date  — date of loss
valuation_date   : date  — the as-at date for this observation
cumulative_paid  : float — total payments made up to valuation_date
case_estimate    : float — current case reserve at valuation_date
is_open          : bool  — True if claim still open at valuation_date
ultimate         : float — final settled amount (NaN for open claims)

Case estimate history columns (optional, required for FNN+)
-----------------------------------------------------------
n_case_revisions        : int   — number of case estimate changes to date
case_estimate_mean      : float — mean case estimate across revision history
case_estimate_std       : float — std dev of case estimate revisions
case_estimate_trend     : float — linear trend of case estimates over time
                                  (positive = reserves increasing)
largest_case_revision   : float — single largest absolute revision
prop_upward_revisions   : float — fraction of revisions that were upward

Feature columns produced by prepare_features()
----------------------------------------------
dev_quarter      : int   — development quarter (0-indexed from accident quarter)
log_cumulative_paid : float — log(cumulative_paid + 1)
claim_age_quarters  : int   — quarters since accident date
calendar_quarter    : int   — calendar quarter of valuation date (1-4)
calendar_year       : int   — calendar year of valuation date

Covariate columns (user-supplied, passed through)
-------------------------------------------------
Any additional columns starting with 'feat_' are treated as input features.
The standard SPLICE-like covariates are:
    feat_claim_type : int (0=property, 1=liability, 2=bodily_injury)
    feat_litigation : int (0/1)
    feat_fault      : float (0.0-1.0 proportion fault attributed to claimant)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import polars as pl
import numpy as np


@dataclass
class ClaimSchema:
    """
    Validates that a DataFrame conforms to the individual claims panel format.

    Use this to fail fast at ingestion rather than getting cryptic errors
    from PyTorch when you feed it a column that's actually a string.
    """

    required_columns: List[str] = field(default_factory=lambda: [
        "claim_id",
        "accident_date",
        "valuation_date",
        "cumulative_paid",
        "case_estimate",
        "is_open",
    ])

    optional_columns: List[str] = field(default_factory=lambda: [
        "ultimate",
        "reporting_date",
        "n_case_revisions",
        "case_estimate_mean",
        "case_estimate_std",
        "case_estimate_trend",
        "largest_case_revision",
        "prop_upward_revisions",
    ])

    def validate(self, df: pl.DataFrame) -> None:
        """
        Raise ValueError with a clear message if the DataFrame is missing
        required columns or has columns with wrong types.
        """
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"ClaimSchema validation failed. Missing required columns: {missing}. "
                f"Available columns: {df.columns}"
            )

        # Type checks
        if df["cumulative_paid"].dtype not in (pl.Float32, pl.Float64):
            raise ValueError(
                f"cumulative_paid must be Float32 or Float64, got {df['cumulative_paid'].dtype}"
            )
        if df["case_estimate"].dtype not in (pl.Float32, pl.Float64):
            raise ValueError(
                f"case_estimate must be Float32 or Float64, got {df['case_estimate'].dtype}"
            )

        # Non-negative payments
        neg_paid = (df["cumulative_paid"] < 0).sum()
        if neg_paid > 0:
            raise ValueError(
                f"cumulative_paid contains {neg_paid} negative values. "
                "Cumulative payments cannot decrease."
            )

    def has_case_history(self, df: pl.DataFrame) -> bool:
        """True if the DataFrame contains case estimate history columns."""
        return "n_case_revisions" in df.columns and "case_estimate_mean" in df.columns


_SCHEMA = ClaimSchema()


def prepare_features(
    df: pl.DataFrame,
    use_case_estimates: bool = True,
) -> pl.DataFrame:
    """
    Extract model-ready features from the claims panel DataFrame.

    This is the FNN+ feature set from Avanzi et al. (2025): claim-level
    summary statistics computed from the cumulative payment and case estimate
    history available at each valuation date.

    Parameters
    ----------
    df : pl.DataFrame
        Claims panel in the format described by ClaimSchema.
    use_case_estimates : bool
        If True, include case estimate summary features (requires the
        n_case_revisions / case_estimate_mean / ... columns to be present).
        FNN+ uses these; plain FNN does not.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with additional feature columns added:
        dev_quarter, log_cumulative_paid, claim_age_quarters, calendar_quarter,
        calendar_year, and optionally the case estimate summaries.
    """
    _SCHEMA.validate(df)

    # Development quarter: quarters from accident_date to valuation_date
    df = df.with_columns([
        (
            (
                (pl.col("valuation_date").dt.year() - pl.col("accident_date").dt.year()) * 4
                + (pl.col("valuation_date").dt.quarter() - pl.col("accident_date").dt.quarter())
            ).cast(pl.Int32).alias("dev_quarter")
        ),
        (
            (pl.col("cumulative_paid") + 1.0).log().alias("log_cumulative_paid")
        ),
        (
            pl.col("valuation_date").dt.quarter().cast(pl.Int32).alias("calendar_quarter")
        ),
        (
            pl.col("valuation_date").dt.year().cast(pl.Int32).alias("calendar_year")
        ),
    ])

    # claim_age_quarters: quarters from accident to valuation
    # dev_quarter is already this, but alias for clarity
    df = df.with_columns([
        pl.col("dev_quarter").alias("claim_age_quarters")
    ])

    if use_case_estimates and _SCHEMA.has_case_history(df):
        # Fill any missing case history columns with sensible defaults
        if "case_estimate_std" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("case_estimate_std"))
        if "case_estimate_trend" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("case_estimate_trend"))
        if "largest_case_revision" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("largest_case_revision"))
        if "prop_upward_revisions" not in df.columns:
            df = df.with_columns(pl.lit(0.5).alias("prop_upward_revisions"))

    return df


def get_feature_columns(
    df: pl.DataFrame,
    use_case_estimates: bool = True,
) -> List[str]:
    """
    Return the ordered list of feature column names to feed into the model.

    The order matters — it must be consistent between fit() and predict().
    """
    base_features = [
        "log_cumulative_paid",
        "case_estimate",
        "dev_quarter",
        "claim_age_quarters",
        "calendar_quarter",
    ]

    case_features = [
        "n_case_revisions",
        "case_estimate_mean",
        "case_estimate_std",
        "case_estimate_trend",
        "largest_case_revision",
        "prop_upward_revisions",
    ]

    # User-supplied covariates
    feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])

    cols = base_features + feat_cols
    if use_case_estimates and _SCHEMA.has_case_history(df):
        cols = cols + case_features

    # Return only columns that actually exist
    return [c for c in cols if c in df.columns]


def outstanding(df: pl.DataFrame) -> pl.Series:
    """
    Compute the outstanding liability for each settled claim row.

    outstanding = ultimate - cumulative_paid

    Only meaningful for settled claims (is_open == False).
    Returns NaN for open claims.
    """
    return (pl.col("ultimate") - pl.col("cumulative_paid")).alias("outstanding")


def to_numpy_features(
    df: pl.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    """Extract feature matrix as a float32 numpy array."""
    return df.select(feature_cols).to_numpy().astype(np.float32)


def to_numpy_targets(df: pl.DataFrame) -> np.ndarray:
    """
    Extract log-outstanding as a float32 numpy array.

    Target: log(ultimate - cumulative_paid)
    Only valid for settled claims with ultimate > cumulative_paid.
    Raises ValueError if any NaN targets remain.
    """
    outstanding_vals = (df["ultimate"] - df["cumulative_paid"]).to_numpy()
    log_outstanding = np.log(outstanding_vals.clip(min=1e-6))
    if np.any(~np.isfinite(log_outstanding)):
        n_bad = np.sum(~np.isfinite(log_outstanding))
        raise ValueError(
            f"Found {n_bad} non-finite log-outstanding values. "
            "Check that ultimate > cumulative_paid for all training rows "
            "and that no cumulative_paid exceeds ultimate."
        )
    return log_outstanding.astype(np.float32)
