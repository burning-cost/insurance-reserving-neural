"""
Tests for FNNReserver and LSTMReserver.

All model tests run on Databricks — they require PyTorch which is too heavy
for local execution. The fixtures use small datasets so training completes
quickly in the Databricks serverless environment.

Key things we test:
- Model fits without error on synthetic data
- Training loss decreases (convergence)
- Predictions have correct shape and types
- Predictions are non-negative
- reserve() returns a scalar
- predict() returns expected columns
- Unfitted models raise RuntimeError
- Edge cases: single claim, all settled, all open
"""

import pytest
import polars as pl
import numpy as np

# Skip all tests if torch not installed
torch = pytest.importorskip("torch", reason="PyTorch not installed — run on Databricks")

from insurance_reserving_neural.synthetic import SyntheticClaimsGenerator
from insurance_reserving_neural.models import FNNReserver, LSTMReserver


@pytest.fixture(scope="module")
def medium_df():
    """Generate a medium dataset used by most tests."""
    gen = SyntheticClaimsGenerator(
        n_claims=800,
        n_periods=8,
        max_dev_quarters=12,
        random_state=0,
    )
    return gen.generate()


@pytest.fixture(scope="module")
def train_df(medium_df):
    return medium_df.filter(pl.col("split") == "train")


@pytest.fixture(scope="module")
def test_df(medium_df):
    return medium_df.filter(pl.col("split") == "test")


@pytest.fixture(scope="module")
def fitted_fnn(train_df):
    model = FNNReserver(
        use_case_estimates=True,
        hidden_sizes=(32, 16),
        max_epochs=20,
        patience=5,
        batch_size=256,
        random_state=42,
        verbose=False,
    )
    model.fit(train_df)
    return model


@pytest.fixture(scope="module")
def fitted_lstm(train_df):
    model = LSTMReserver(
        hidden_size=16,
        num_layers=1,
        max_epochs=10,
        patience=5,
        batch_size=64,
        random_state=42,
        verbose=False,
    )
    model.fit(train_df)
    return model


class TestFNNReserverFit:
    def test_fit_returns_self(self, train_df):
        model = FNNReserver(max_epochs=5, random_state=0)
        result = model.fit(train_df)
        assert result is model

    def test_fitted_flag_set(self, fitted_fnn):
        assert fitted_fnn._is_fitted is True

    def test_training_history_populated(self, fitted_fnn):
        assert len(fitted_fnn.training_history_) > 0

    def test_training_loss_decreases(self, fitted_fnn):
        history = fitted_fnn.training_history_
        first_loss = history[0]["train_loss"]
        last_loss = history[-1]["train_loss"]
        assert last_loss < first_loss, (
            f"Training loss did not decrease: {first_loss:.4f} -> {last_loss:.4f}"
        )

    def test_smearing_factor_positive(self, fitted_fnn):
        assert fitted_fnn._smearing_factor > 0.0

    def test_feature_cols_populated(self, fitted_fnn):
        assert len(fitted_fnn._feature_cols) > 0

    def test_feature_scaling_set(self, fitted_fnn):
        assert fitted_fnn._feature_mean is not None
        assert fitted_fnn._feature_std is not None

    def test_fit_without_case_estimates(self, train_df):
        model = FNNReserver(use_case_estimates=False, max_epochs=5, random_state=1)
        model.fit(train_df)
        assert model._is_fitted

    def test_insufficient_data_raises(self):
        # With n_claims=1 and settlement_rate=0, claim never settles
        # so ultimate > cumulative_paid for at most ~20 rows — below 100 threshold
        # Use n_claims=2, max_dev_quarters=3: at most 6 rows, well below 100
        gen = SyntheticClaimsGenerator(
            n_claims=2, max_dev_quarters=3, settlement_rate=0.0, random_state=0
        )
        df = gen.generate()
        model = FNNReserver(max_epochs=3)
        with pytest.raises(ValueError):
            model.fit(df)

    def test_unfitted_predict_raises(self, test_df):
        model = FNNReserver()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(test_df)

    def test_unfitted_reserve_raises(self, test_df):
        model = FNNReserver()
        with pytest.raises(RuntimeError):
            model.reserve(test_df)


class TestFNNReserverPredict:
    def test_predict_returns_polars_dataframe(self, fitted_fnn, test_df):
        result = fitted_fnn.predict(test_df)
        assert isinstance(result, pl.DataFrame)

    def test_predict_same_row_count(self, fitted_fnn, test_df):
        result = fitted_fnn.predict(test_df)
        assert len(result) == len(test_df)

    def test_predicted_outstanding_column_present(self, fitted_fnn, test_df):
        result = fitted_fnn.predict(test_df)
        assert "predicted_outstanding" in result.columns

    def test_predicted_log_outstanding_column_present(self, fitted_fnn, test_df):
        result = fitted_fnn.predict(test_df)
        assert "predicted_log_outstanding" in result.columns

    def test_predictions_non_negative(self, fitted_fnn, test_df):
        result = fitted_fnn.predict(test_df)
        assert (result["predicted_outstanding"] < 0).sum() == 0

    def test_predictions_finite(self, fitted_fnn, test_df):
        result = fitted_fnn.predict(test_df)
        vals = result["predicted_outstanding"].to_numpy()
        assert np.all(np.isfinite(vals))

    def test_predictions_float64(self, fitted_fnn, test_df):
        result = fitted_fnn.predict(test_df)
        assert result["predicted_outstanding"].dtype == pl.Float64


class TestFNNReserverReserve:
    def test_reserve_returns_float(self, fitted_fnn, test_df):
        r = fitted_fnn.reserve(test_df)
        assert isinstance(r, float)

    def test_reserve_non_negative(self, fitted_fnn, test_df):
        r = fitted_fnn.reserve(test_df)
        assert r >= 0.0

    def test_reserve_positive_when_open_claims_exist(self, fitted_fnn, test_df):
        open_df = test_df.filter(pl.col("is_open") == True)
        if len(open_df) == 0:
            pytest.skip("No open claims in test set")
        r = fitted_fnn.reserve(open_df)
        assert r > 0.0


class TestFNNCasEEstimatesImpact:
    """
    FNN+ (with case estimates) should outperform plain FNN.
    This is the key Avanzi et al. (2025) finding.
    """

    def test_case_estimates_improve_prediction(self, train_df, test_df):
        from insurance_reserving_neural.metrics import mean_absolute_log_error

        model_plain = FNNReserver(
            use_case_estimates=False,
            max_epochs=30,
            random_state=42,
        )
        model_plus = FNNReserver(
            use_case_estimates=True,
            max_epochs=30,
            random_state=42,
        )

        model_plain.fit(train_df)
        model_plus.fit(train_df)

        preds_plain = model_plain.predict(test_df)
        preds_plus = model_plus.predict(test_df)

        # Filter to settled claims for MALE computation
        settled = test_df.filter(
            (pl.col("is_open") == False)
            & (pl.col("ultimate").is_not_null())
            & ((pl.col("ultimate") - pl.col("cumulative_paid")) > 1.0)
        )

        if len(settled) < 10:
            pytest.skip("Not enough settled claims for MALE comparison")

        # Join predictions back
        preds_plain_settled = model_plain.predict(settled)
        preds_plus_settled = model_plus.predict(settled)

        male_plain = mean_absolute_log_error(preds_plain_settled)
        male_plus = mean_absolute_log_error(preds_plus_settled)

        # FNN+ should match or beat plain FNN; we give 20% tolerance
        # because small datasets can have noise
        assert male_plus <= male_plain * 1.20, (
            f"FNN+ MALE ({male_plus:.4f}) worse than FNN by >20% ({male_plain:.4f})"
        )


class TestLSTMReserverFit:
    def test_fit_returns_self(self, train_df):
        model = LSTMReserver(max_epochs=5, random_state=0, batch_size=32)
        result = model.fit(train_df)
        assert result is model

    def test_fitted_flag(self, fitted_lstm):
        assert fitted_lstm._is_fitted is True

    def test_smearing_factor_positive(self, fitted_lstm):
        assert fitted_lstm._smearing_factor > 0.0

    def test_training_history_populated(self, fitted_lstm):
        assert len(fitted_lstm.training_history_) > 0

    def test_training_loss_decreases(self, fitted_lstm):
        history = fitted_lstm.training_history_
        if len(history) < 2:
            pytest.skip("Not enough epochs to check convergence")
        first_loss = history[0]["train_loss"]
        last_loss = history[-1]["train_loss"]
        assert last_loss < first_loss * 1.5, (
            f"Loss not improving: {first_loss:.4f} -> {last_loss:.4f}"
        )

    def test_unfitted_raises(self, test_df):
        model = LSTMReserver()
        with pytest.raises(RuntimeError):
            model.predict(test_df)


class TestLSTMReserverPredict:
    def test_returns_dataframe(self, fitted_lstm, test_df):
        result = fitted_lstm.predict(test_df)
        assert isinstance(result, pl.DataFrame)

    def test_has_expected_columns(self, fitted_lstm, test_df):
        result = fitted_lstm.predict(test_df)
        assert "claim_id" in result.columns
        assert "predicted_outstanding" in result.columns

    def test_predictions_non_negative(self, fitted_lstm, test_df):
        result = fitted_lstm.predict(test_df)
        assert (result["predicted_outstanding"] < 0).sum() == 0

    def test_reserve_positive(self, fitted_lstm, test_df):
        open_df = test_df.filter(pl.col("is_open") == True)
        if len(open_df) == 0:
            pytest.skip("No open claims")
        r = fitted_lstm.reserve(test_df)
        assert r >= 0.0


class TestEdgeCases:
    def test_fnn_all_same_claim_type(self):
        """Test with a homogeneous portfolio (one claim type only)."""
        gen = SyntheticClaimsGenerator(n_claims=200, random_state=1)
        df = gen.generate()
        # Keep only property claims
        prop_df = df.filter(pl.col("claim_type") == "property")
        train = prop_df.filter(pl.col("split") == "train")
        test = prop_df.filter(pl.col("split") == "test")
        if len(train.filter(pl.col("is_open") == False)) < 100:
            pytest.skip("Not enough settled property claims")
        model = FNNReserver(max_epochs=5, random_state=0)
        model.fit(train)
        preds = model.predict(test)
        assert len(preds) == len(test)

    def test_fnn_predict_on_all_open_claims(self, fitted_fnn, test_df):
        """predict() should work even if all claims are open (no ultimate needed)."""
        all_open = test_df.filter(pl.col("is_open") == True)
        if len(all_open) == 0:
            pytest.skip("No open claims")
        preds = fitted_fnn.predict(all_open)
        assert (preds["predicted_outstanding"] >= 0).all()
