"""
bootstrap.py — Bootstrap uncertainty quantification for RBNS reserves.

Residual bootstrap for individual claims reserving. The procedure:
1. Fit the model on training data, compute residuals in log space.
2. For each bootstrap replicate:
   a. Resample claims with replacement (preserves claim-level correlation).
   b. Add a resampled residual to each prediction.
   c. Back-transform and sum to get bootstrap portfolio reserve.
3. Report the empirical distribution: P10/P50/P75/P90/P99.5.

Why residual bootstrap rather than parametric?
    Insurance residuals are not normal. Bodily injury development is
    heavy-tailed and the residuals from log-space FNN models are
    typically right-skewed. Non-parametric residual resampling makes no
    distributional assumption.

Why resample claims rather than rows?
    Rows within a claim are correlated (same accident, same settlement
    trajectory). Resampling rows independently would understate reserve
    uncertainty by treating correlated payments as independent observations.

Solvency II context
    PRA SS8/24 requires insurers to estimate the probability distribution
    of technical provisions. The P75 of the bootstrap reserve distribution
    is a common proxy for the risk margin; P99.5 is used for SCR calculation
    via internal models. We return all standard Solvency II percentiles.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type, Union
import numpy as np
import polars as pl
from scipy import stats

from insurance_reserving_neural.models import FNNReserver, LSTMReserver


class BootstrapReserver:
    """
    Bootstrap uncertainty quantification wrapper for FNNReserver or LSTMReserver.

    Fits the base model, then generates a distribution of portfolio RBNS
    reserve estimates via residual bootstrap.

    Parameters
    ----------
    base_model : FNNReserver | LSTMReserver
        A configured (but not yet fitted) model instance.
    n_boot : int
        Number of bootstrap replicates. Default 500. For production Solvency II
        reporting use 2000+; 500 is sufficient for model development.
    percentiles : list of float | None
        Percentiles to report. Default [10, 50, 75, 90, 99.5].
    random_state : int | None
        Random seed for reproducibility.
    verbose : bool
        Print bootstrap progress every 100 replicates.

    Examples
    --------
    >>> model = FNNReserver(use_case_estimates=True, random_state=42)
    >>> boot = BootstrapReserver(model, n_boot=200, random_state=42)
    >>> boot.fit(train_df)
    >>> result = boot.reserve_distribution(open_df)
    >>> print(f"P75 reserve: {result['P75']:,.0f}")
    >>> print(f"P99.5 reserve: {result['P99_5']:,.0f}")
    """

    def __init__(
        self,
        base_model: Union[FNNReserver, LSTMReserver],
        n_boot: int = 500,
        percentiles: Optional[List[float]] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.base_model = base_model
        self.n_boot = n_boot
        self.percentiles = percentiles if percentiles is not None else [10.0, 50.0, 75.0, 90.0, 99.5]
        self.random_state = random_state
        self.verbose = verbose

        self._rng = np.random.default_rng(random_state)
        self._residuals: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    def fit(self, df: pl.DataFrame) -> "BootstrapReserver":
        """
        Fit the base model and store training residuals for bootstrap.

        Parameters
        ----------
        df : pl.DataFrame
            Training panel. Must contain settled claims with known ultimate.

        Returns
        -------
        self
        """
        # Fit base model
        self.base_model.fit(df)

        # Compute training residuals on settled claims in log space
        settled = df.filter(
            (pl.col("is_open") == False)
            & (pl.col("ultimate").is_not_null())
            & ((pl.col("ultimate") - pl.col("cumulative_paid")) > 0)
        )

        preds = self.base_model.predict(settled)

        actual_outstanding = (
            preds["ultimate"] - preds["cumulative_paid"]
        ).to_numpy()
        log_actual = np.log(np.clip(actual_outstanding, 1e-6, None))
        log_predicted = preds["predicted_log_outstanding"].to_numpy()

        # Residuals in log space: ε_i = log(y_i) - log(ŷ_i)
        self._residuals = log_actual - log_predicted

        self._is_fitted = True
        return self

    def reserve_distribution(
        self,
        open_df: pl.DataFrame,
    ) -> Dict[str, float]:
        """
        Generate the bootstrap distribution of the portfolio RBNS reserve.

        Parameters
        ----------
        open_df : pl.DataFrame
            Panel data for open claims at the valuation date.
            Should contain only is_open=True claims (or the method filters them).

        Returns
        -------
        dict
            'point_estimate': reserve from fitted model (no bootstrap noise)
            'mean': mean of bootstrap reserve distribution
            'std': standard deviation of bootstrap distribution
            'P10', 'P50', 'P75', 'P90', 'P99_5': percentile reserves
            'coeff_of_variation': std / mean
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before reserve_distribution().")

        # Point estimate from the fitted model
        open_claims = open_df.filter(pl.col("is_open") == True)
        if len(open_claims) == 0:
            open_claims = open_df  # trust the caller

        point_preds = self.base_model.predict(open_claims)
        point_estimates = point_preds["predicted_outstanding"].to_numpy()
        point_log_preds = point_preds["predicted_log_outstanding"].to_numpy()
        point_reserve = float(point_estimates.sum())

        n_open = len(point_estimates)
        n_residuals = len(self._residuals)

        # Bootstrap: resample claims + add resampled residuals
        boot_reserves = np.zeros(self.n_boot)

        for b in range(self.n_boot):
            if self.verbose and (b + 1) % 100 == 0:
                print(f"Bootstrap replicate {b + 1}/{self.n_boot}")

            # Resample claim indices (with replacement)
            claim_idx = self._rng.integers(0, n_open, size=n_open)
            boot_log_preds = point_log_preds[claim_idx]

            # Add resampled residuals
            resid_idx = self._rng.integers(0, n_residuals, size=n_open)
            boot_log_preds = boot_log_preds + self._residuals[resid_idx]

            # Back-transform using stored smearing factor
            smearing = self.base_model._smearing_factor
            boot_outstanding = np.exp(boot_log_preds) * smearing
            boot_outstanding = np.clip(boot_outstanding, 0.0, None)

            boot_reserves[b] = boot_outstanding.sum()

        result = {
            "point_estimate": point_reserve,
            "mean": float(np.mean(boot_reserves)),
            "std": float(np.std(boot_reserves)),
            "coeff_of_variation": float(np.std(boot_reserves) / max(np.mean(boot_reserves), 1.0)),
        }

        for p in self.percentiles:
            key = f"P{str(p).replace('.', '_')}"
            result[key] = float(np.percentile(boot_reserves, p))

        return result

    def residual_summary(self) -> Dict[str, float]:
        """
        Diagnostic summary of the training residuals.

        Returns mean, std, skewness, kurtosis, and normality test p-value.
        Large skewness or significant non-normality indicates the residual
        bootstrap is preferable to a parametric log-normal assumption.
        """
        if self._residuals is None:
            raise RuntimeError("Call fit() first.")

        r = self._residuals
        _, norm_p = stats.normaltest(r)

        return {
            "n_residuals": int(len(r)),
            "mean": float(np.mean(r)),
            "std": float(np.std(r)),
            "skewness": float(stats.skew(r)),
            "kurtosis": float(stats.kurtosis(r)),
            "normality_p_value": float(norm_p),
            "is_normal_at_5pct": bool(norm_p > 0.05),
        }
