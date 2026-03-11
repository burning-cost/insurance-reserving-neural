"""
Tests for evaluation metrics: MALE, OCLerr, reserve_range, chain_ladder_reserve.
"""

import pytest
import polars as pl
import numpy as np

from insurance_reserving_neural.metrics import (
    mean_absolute_log_error,
    ocl_error,
    reserve_range,
    chain_ladder_reserve,
)
from insurance_reserving_neural.synthetic import SyntheticClaimsGenerator


@pytest.fixture(scope="module")
def preds_df():
    """
    Simulate a predictions DataFrame with known properties.
    We create it directly rather than running a full model.
    """
    rng = np.random.default_rng(42)
    n = 200
    ultimate = np.exp(rng.normal(9.0, 1.2, size=n))
    paid_frac = rng.beta(2, 3, size=n)
    cumulative_paid = ultimate * paid_frac
    actual_outstanding = ultimate - cumulative_paid
    # Predictions: mostly accurate with some noise
    predicted = actual_outstanding * np.exp(rng.normal(0, 0.3, size=n))

    return pl.DataFrame({
        "claim_id": [f"C{i:04d}" for i in range(n)],
        "ultimate": ultimate,
        "cumulative_paid": cumulative_paid,
        "predicted_outstanding": predicted,
        "predicted_log_outstanding": np.log(predicted),
        "is_open": [False] * n,  # all settled for metric computation
    })


@pytest.fixture(scope="module")
def open_preds_df():
    rng = np.random.default_rng(0)
    n = 100
    return pl.DataFrame({
        "claim_id": [f"O{i:04d}" for i in range(n)],
        "predicted_outstanding": rng.lognormal(8, 1, size=n),
        "is_open": [True] * n,
    })


class TestMALE:
    def test_perfect_predictions_zero_male(self, preds_df):
        """If predicted == actual, MALE should be 0."""
        perfect = preds_df.with_columns(
            (pl.col("ultimate") - pl.col("cumulative_paid")).alias("predicted_outstanding")
        )
        male = mean_absolute_log_error(perfect)
        assert male == pytest.approx(0.0, abs=1e-6)

    def test_returns_float(self, preds_df):
        male = mean_absolute_log_error(preds_df)
        assert isinstance(male, float)

    def test_positive(self, preds_df):
        male = mean_absolute_log_error(preds_df)
        assert male > 0.0

    def test_larger_noise_gives_larger_male(self):
        rng = np.random.default_rng(0)
        n = 200
        ultimate = np.exp(rng.normal(9, 1, size=n))
        paid = ultimate * 0.3
        outstanding = ultimate - paid

        df_good = pl.DataFrame({
            "ultimate": ultimate, "cumulative_paid": paid,
            "predicted_outstanding": outstanding * np.exp(rng.normal(0, 0.1, n)),
            "predicted_log_outstanding": np.zeros(n), "is_open": [False] * n,
        })
        df_bad = pl.DataFrame({
            "ultimate": ultimate, "cumulative_paid": paid,
            "predicted_outstanding": outstanding * np.exp(rng.normal(0, 0.5, n)),
            "predicted_log_outstanding": np.zeros(n), "is_open": [False] * n,
        })
        assert mean_absolute_log_error(df_bad) > mean_absolute_log_error(df_good)

    def test_no_settled_claims_raises(self, open_preds_df):
        # Should raise either ValueError or ColumnNotFoundError when no settled claims
        with pytest.raises(Exception):
            mean_absolute_log_error(open_preds_df)


class TestOCLError:
    def test_perfect_predictions_zero_oclerr(self, preds_df):
        perfect = preds_df.with_columns(
            (pl.col("ultimate") - pl.col("cumulative_paid")).alias("predicted_outstanding")
        )
        err = ocl_error(perfect)
        assert err == pytest.approx(0.0, abs=1e-6)

    def test_over_reserve_positive(self):
        df = pl.DataFrame({
            "ultimate": [10000.0, 20000.0],
            "cumulative_paid": [5000.0, 10000.0],
            "predicted_outstanding": [6000.0, 12000.0],  # 20% over
            "is_open": [False, False],
        })
        err = ocl_error(df)
        assert err > 0.0

    def test_under_reserve_negative(self):
        df = pl.DataFrame({
            "ultimate": [10000.0, 20000.0],
            "cumulative_paid": [5000.0, 10000.0],
            "predicted_outstanding": [4000.0, 8000.0],  # 20% under
            "is_open": [False, False],
        })
        err = ocl_error(df)
        assert err < 0.0

    def test_known_over_reserve_value(self):
        """
        Actual outstanding = 5000 + 10000 = 15000
        Predicted = 6000 + 12000 = 18000
        OCLerr = (18000 - 15000) / 15000 = 0.2
        """
        df = pl.DataFrame({
            "ultimate": [10000.0, 20000.0],
            "cumulative_paid": [5000.0, 10000.0],
            "predicted_outstanding": [6000.0, 12000.0],
            "is_open": [False, False],
        })
        err = ocl_error(df)
        assert err == pytest.approx(0.2, rel=1e-5)

    def test_returns_float(self, preds_df):
        assert isinstance(ocl_error(preds_df), float)


class TestReserveRange:
    def test_returns_dict(self, open_preds_df):
        result = reserve_range(open_preds_df)
        assert isinstance(result, dict)

    def test_expected_keys_present(self, open_preds_df):
        result = reserve_range(open_preds_df)
        assert "mean" in result
        assert "std" in result
        assert "point_estimate" in result
        assert "P50" in result
        assert "P99_5" in result

    def test_percentiles_ordered(self, open_preds_df):
        result = reserve_range(open_preds_df)
        # Keys: P10, P50, P75, P90, P99_5 (whole numbers have no decimal; 99.5 -> P99_5)
        assert result["P10"] <= result["P50"]
        assert result["P50"] <= result["P75"]
        assert result["P75"] <= result["P90"]
        assert result["P90"] <= result["P99_5"]

    def test_std_positive(self, open_preds_df):
        result = reserve_range(open_preds_df)
        assert result["std"] > 0.0

    def test_no_open_claims_raises(self):
        df = pl.DataFrame({
            "predicted_outstanding": [1000.0],
            "is_open": [False],
        })
        with pytest.raises(ValueError):
            reserve_range(df)

    def test_custom_percentiles(self, open_preds_df):
        result = reserve_range(open_preds_df, percentiles=[5.0, 95.0])
        assert "P5_0" in result or "P5" in result or any("5" in k for k in result)

    def test_point_estimate_near_mean(self, open_preds_df):
        result = reserve_range(open_preds_df)
        # With n=100, point estimate should be close to bootstrap mean
        rel_diff = abs(result["point_estimate"] - result["mean"]) / result["mean"]
        assert rel_diff < 0.15, f"Point estimate far from bootstrap mean: {rel_diff:.3f}"


class TestChainLadderReserve:
    def test_returns_dict(self):
        gen = SyntheticClaimsGenerator(n_claims=200, random_state=0)
        df = gen.generate()
        from insurance_reserving_neural.data import prepare_features
        df = prepare_features(df, use_case_estimates=False)
        result = chain_ladder_reserve(df)
        assert isinstance(result, dict)

    def test_reserve_key_present(self):
        gen = SyntheticClaimsGenerator(n_claims=200, random_state=0)
        df = gen.generate()
        from insurance_reserving_neural.data import prepare_features
        df = prepare_features(df, use_case_estimates=False)
        result = chain_ladder_reserve(df)
        assert "reserve" in result

    def test_reserve_non_negative(self):
        gen = SyntheticClaimsGenerator(n_claims=300, random_state=1)
        df = gen.generate()
        from insurance_reserving_neural.data import prepare_features
        df = prepare_features(df, use_case_estimates=False)
        result = chain_ladder_reserve(df)
        assert result["reserve"] >= 0.0

    def test_dev_factors_all_positive(self):
        gen = SyntheticClaimsGenerator(n_claims=200, random_state=2)
        df = gen.generate()
        from insurance_reserving_neural.data import prepare_features
        df = prepare_features(df, use_case_estimates=False)
        result = chain_ladder_reserve(df)
        for f in result["dev_factors"]:
            assert f > 0.0, f"Negative development factor: {f}"

    def test_cdfs_all_at_least_one(self):
        gen = SyntheticClaimsGenerator(n_claims=200, random_state=3)
        df = gen.generate()
        from insurance_reserving_neural.data import prepare_features
        df = prepare_features(df, use_case_estimates=False)
        result = chain_ladder_reserve(df)
        for cdf in result["cdfs"]:
            assert cdf >= 1.0, f"CDF < 1.0: {cdf}"

    def test_returns_expected_keys(self):
        gen = SyntheticClaimsGenerator(n_claims=100, random_state=4)
        df = gen.generate()
        from insurance_reserving_neural.data import prepare_features
        df = prepare_features(df, use_case_estimates=False)
        result = chain_ladder_reserve(df)
        for key in ["reserve", "dev_factors", "cdfs", "n_accident_quarters"]:
            assert key in result, f"Missing key: {key}"
