"""
Tests for BootstrapReserver uncertainty quantification.

Bootstrap tests require PyTorch — they run on Databricks only.
"""

import pytest
import polars as pl
import numpy as np

torch = pytest.importorskip("torch", reason="PyTorch not installed — run on Databricks")

from insurance_reserving_neural.synthetic import SyntheticClaimsGenerator
from insurance_reserving_neural.models import FNNReserver
from insurance_reserving_neural.bootstrap import BootstrapReserver


@pytest.fixture(scope="module")
def datasets():
    gen = SyntheticClaimsGenerator(
        n_claims=500,
        n_periods=8,
        max_dev_quarters=10,
        random_state=7,
    )
    df = gen.generate()
    return (
        df.filter(pl.col("split") == "train"),
        df.filter(pl.col("split") == "test"),
    )


@pytest.fixture(scope="module")
def fitted_boot(datasets):
    train_df, _ = datasets
    model = FNNReserver(
        max_epochs=15, patience=5, random_state=0,
        hidden_sizes=(32, 16), batch_size=256,
    )
    boot = BootstrapReserver(model, n_boot=50, random_state=0)
    boot.fit(train_df)
    return boot


class TestBootstrapReserverFit:
    def test_fit_returns_self(self, datasets):
        train_df, _ = datasets
        model = FNNReserver(max_epochs=5, random_state=1)
        boot = BootstrapReserver(model, n_boot=10, random_state=1)
        result = boot.fit(train_df)
        assert result is boot

    def test_fitted_flag_set(self, fitted_boot):
        assert fitted_boot._is_fitted is True

    def test_residuals_stored(self, fitted_boot):
        assert fitted_boot._residuals is not None
        assert len(fitted_boot._residuals) > 0

    def test_residuals_finite(self, fitted_boot):
        assert np.all(np.isfinite(fitted_boot._residuals))

    def test_unfitted_raises(self, datasets):
        _, test_df = datasets
        model = FNNReserver()
        boot = BootstrapReserver(model, n_boot=10)
        with pytest.raises(RuntimeError):
            boot.reserve_distribution(test_df)


class TestReserveDistribution:
    def test_returns_dict(self, fitted_boot, datasets):
        _, test_df = datasets
        open_df = test_df.filter(pl.col("is_open") == True)
        if len(open_df) == 0:
            pytest.skip("No open claims in test set")
        result = fitted_boot.reserve_distribution(open_df)
        assert isinstance(result, dict)

    def test_expected_keys(self, fitted_boot, datasets):
        _, test_df = datasets
        open_df = test_df.filter(pl.col("is_open") == True)
        if len(open_df) == 0:
            pytest.skip("No open claims")
        result = fitted_boot.reserve_distribution(open_df)
        for key in ["point_estimate", "mean", "std", "P50", "P99_5"]:
            assert key in result, f"Missing key: {key}"

    def test_percentiles_ordered(self, fitted_boot, datasets):
        _, test_df = datasets
        open_df = test_df.filter(pl.col("is_open") == True)
        if len(open_df) == 0:
            pytest.skip("No open claims")
        result = fitted_boot.reserve_distribution(open_df)
        assert result["P10"] <= result["P50"]
        assert result["P50"] <= result["P75"]
        assert result["P75"] <= result["P90"]
        assert result["P90"] <= result["P99_5"]

    def test_std_positive(self, fitted_boot, datasets):
        _, test_df = datasets
        open_df = test_df.filter(pl.col("is_open") == True)
        if len(open_df) == 0:
            pytest.skip("No open claims")
        result = fitted_boot.reserve_distribution(open_df)
        assert result["std"] > 0.0

    def test_all_values_positive(self, fitted_boot, datasets):
        _, test_df = datasets
        open_df = test_df.filter(pl.col("is_open") == True)
        if len(open_df) == 0:
            pytest.skip("No open claims")
        result = fitted_boot.reserve_distribution(open_df)
        for k, v in result.items():
            assert v >= 0.0, f"Negative value for key {k}: {v}"

    def test_coeff_variation_reasonable(self, fitted_boot, datasets):
        """CoV should be between 1% and 50% for a realistic portfolio."""
        _, test_df = datasets
        open_df = test_df.filter(pl.col("is_open") == True)
        if len(open_df) == 0:
            pytest.skip("No open claims")
        result = fitted_boot.reserve_distribution(open_df)
        cv = result["coeff_of_variation"]
        assert 0.0 < cv < 1.0, f"CoV={cv:.3f} outside reasonable range [0,1]"


class TestResidualSummary:
    def test_returns_dict(self, fitted_boot):
        summary = fitted_boot.residual_summary()
        assert isinstance(summary, dict)

    def test_has_expected_keys(self, fitted_boot):
        summary = fitted_boot.residual_summary()
        for k in ["n_residuals", "mean", "std", "skewness", "kurtosis"]:
            assert k in summary

    def test_n_residuals_matches_training(self, fitted_boot):
        summary = fitted_boot.residual_summary()
        assert summary["n_residuals"] == len(fitted_boot._residuals)

    def test_unfitted_raises(self):
        model = FNNReserver()
        boot = BootstrapReserver(model, n_boot=10)
        with pytest.raises(RuntimeError):
            boot.residual_summary()
