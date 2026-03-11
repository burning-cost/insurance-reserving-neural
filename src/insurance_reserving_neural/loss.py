"""
loss.py — Log-space MSE loss with Duan (1983) smearing bias correction.

The problem with predicting log(outstanding) and back-transforming with exp()
is that E[exp(ŷ)] ≠ exp(E[ŷ]) — Jensen's inequality bites you. The naive
back-transform understates the mean by roughly exp(σ²/2).

Duan (1983) provides the smearing estimator:

    Ŷ = exp(ŷ) × (1/n) × Σ exp(ε_i)

where ε_i = y_i - ŷ_i are the residuals on the log scale. This is consistent
even when the residuals are not normally distributed — which they aren't for
insurance claims.

Reference: Duan N (1983). "Smearing estimate: A nonparametric retransformation
method." Journal of the American Statistical Association, 78(383), 605-610.

Avanzi et al. (2025) arXiv:2601.05274 use exactly this formulation.
"""

from __future__ import annotations

from typing import Optional
import numpy as np


def log_space_mse(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
) -> float:
    """
    Mean squared error on log-transformed outstanding amounts.

    Parameters
    ----------
    y_true_log : np.ndarray
        Log-transformed actual outstanding values: log(ultimate - paid).
    y_pred_log : np.ndarray
        Log-transformed predicted outstanding values from the model.

    Returns
    -------
    float
        Mean squared error in log space.
    """
    residuals = y_true_log - y_pred_log
    return float(np.mean(residuals ** 2))


def duan_smearing_correction(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
) -> float:
    """
    Compute the Duan (1983) smearing correction factor.

    b = (1/n) × Σ exp(y_i - ŷ_i)

    Applied as: predicted_outstanding = exp(ŷ) × b

    Parameters
    ----------
    y_true_log : np.ndarray
        Log-scale actual values used to estimate residuals.
        Typically from the training set.
    y_pred_log : np.ndarray
        Log-scale fitted values.

    Returns
    -------
    float
        The smearing correction factor b. Values > 1 indicate the raw
        exp(ŷ) understates the true mean (which is expected).
    """
    residuals = y_true_log - y_pred_log
    # Clip residuals to prevent overflow in exp()
    residuals_clipped = np.clip(residuals, -20.0, 20.0)
    return float(np.mean(np.exp(residuals_clipped)))


def apply_smearing(
    y_pred_log: np.ndarray,
    smearing_factor: float,
) -> np.ndarray:
    """
    Back-transform log-scale predictions with Duan smearing correction.

    Ŷ = exp(ŷ) × b

    Parameters
    ----------
    y_pred_log : np.ndarray
        Log-scale predictions from the model.
    smearing_factor : float
        The Duan smearing factor b computed on the training set.

    Returns
    -------
    np.ndarray
        Predicted outstanding amounts in the original scale.
    """
    return np.exp(y_pred_log) * smearing_factor
