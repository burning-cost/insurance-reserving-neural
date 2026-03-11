"""
Tests for log-space MSE loss and Duan smearing correction.
"""

import pytest
import numpy as np
from insurance_reserving_neural.loss import (
    log_space_mse,
    duan_smearing_correction,
    apply_smearing,
)


class TestLogSpaceMSE:
    def test_perfect_predictions_zero_loss(self):
        y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert log_space_mse(y, y) == pytest.approx(0.0, abs=1e-7)

    def test_symmetric_errors(self):
        y_true = np.log(np.array([1000.0, 5000.0, 20000.0]))
        y_pred = np.log(np.array([900.0, 5500.0, 18000.0]))
        loss = log_space_mse(y_true, y_pred)
        assert loss > 0.0

    def test_larger_errors_give_larger_loss(self):
        y_true = np.ones(5)
        y_pred_good = y_true + 0.1
        y_pred_bad = y_true + 0.5
        assert log_space_mse(y_true, y_pred_bad) > log_space_mse(y_true, y_pred_good)

    def test_returns_float(self):
        y = np.array([1.0, 2.0])
        assert isinstance(log_space_mse(y, y), float)


class TestDuanSmearing:
    def test_factor_equals_one_for_zero_residuals(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])  # zero residuals
        factor = duan_smearing_correction(y_true, y_pred)
        assert factor == pytest.approx(1.0, rel=1e-6)

    def test_factor_greater_than_one_with_typical_residuals(self):
        """
        With log-normal residuals, smearing factor should be > 1.
        exp(ε) where ε ~ N(0, σ²) has mean exp(σ²/2) > 1.
        """
        rng = np.random.default_rng(0)
        n = 10_000
        sigma = 0.5
        y_true = rng.normal(0, sigma, size=n)  # log-scale residuals
        y_pred = np.zeros(n)  # zero prediction
        factor = duan_smearing_correction(y_true, y_pred)
        expected = np.exp(sigma ** 2 / 2)
        assert factor == pytest.approx(expected, rel=0.05)

    def test_returns_float(self):
        y = np.array([0.0, 0.1, -0.1])
        factor = duan_smearing_correction(y, np.zeros(3))
        assert isinstance(factor, float)

    def test_handles_large_residuals(self):
        """Should not overflow with large residuals due to clipping."""
        y_true = np.array([100.0, -100.0, 0.0])
        y_pred = np.zeros(3)
        factor = duan_smearing_correction(y_true, y_pred)
        assert np.isfinite(factor)


class TestApplySmearing:
    def test_apply_factor_one_is_just_exp(self):
        y_pred_log = np.array([5.0, 7.0, 9.0])
        result = apply_smearing(y_pred_log, 1.0)
        expected = np.exp(y_pred_log)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_factor_greater_than_one_inflates_prediction(self):
        y_pred_log = np.array([5.0, 7.0])
        factor = 1.2
        result = apply_smearing(y_pred_log, factor)
        raw = np.exp(y_pred_log)
        assert np.all(result > raw)

    def test_corrected_closer_to_true_mean_than_naive(self):
        """
        For log-normal data, smearing-corrected predictions should be
        closer to the true mean than naive exp().
        """
        rng = np.random.default_rng(42)
        mu, sigma = 8.0, 1.0
        n = 5000
        log_vals = rng.normal(mu, sigma, size=n)

        # True mean of underlying log-normal
        true_mean = np.exp(mu + sigma ** 2 / 2)

        # Naive back-transform
        naive = np.exp(mu)

        # Smearing-corrected
        smearing = duan_smearing_correction(log_vals, np.full(n, mu))
        corrected = apply_smearing(np.array([mu]), smearing)[0]

        err_naive = abs(naive - true_mean)
        err_corrected = abs(corrected - true_mean)
        assert err_corrected < err_naive, (
            f"Corrected error ({err_corrected:.2f}) should be < naive error ({err_naive:.2f})"
        )
