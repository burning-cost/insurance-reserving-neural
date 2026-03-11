"""
Tests for ClaimSchema validation and prepare_features().
"""

import pytest
import polars as pl
import numpy as np
from datetime import date

from insurance_reserving_neural.data import (
    ClaimSchema,
    prepare_features,
    get_feature_columns,
    to_numpy_features,
    to_numpy_targets,
)
from insurance_reserving_neural.synthetic import SyntheticClaimsGenerator


@pytest.fixture
def base_df():
    gen = SyntheticClaimsGenerator(n_claims=100, random_state=42)
    return gen.generate()


class TestClaimSchema:
    def test_valid_df_passes(self, base_df):
        schema = ClaimSchema()
        schema.validate(base_df)  # Should not raise

    def test_missing_required_column_raises(self, base_df):
        schema = ClaimSchema()
        bad_df = base_df.drop("cumulative_paid")
        with pytest.raises(ValueError, match="Missing required columns"):
            schema.validate(bad_df)

    def test_wrong_type_raises(self, base_df):
        schema = ClaimSchema()
        # Cast cumulative_paid to string — should fail
        bad_df = base_df.with_columns(pl.col("cumulative_paid").cast(pl.Utf8))
        with pytest.raises(ValueError):
            schema.validate(bad_df)

    def test_negative_paid_raises(self, base_df):
        schema = ClaimSchema()
        bad_df = base_df.with_columns(pl.lit(-100.0).alias("cumulative_paid"))
        with pytest.raises(ValueError, match="negative"):
            schema.validate(bad_df)

    def test_has_case_history_true_when_present(self, base_df):
        schema = ClaimSchema()
        assert schema.has_case_history(base_df) is True

    def test_has_case_history_false_when_absent(self, base_df):
        schema = ClaimSchema()
        df = base_df.drop(["n_case_revisions", "case_estimate_mean"])
        assert schema.has_case_history(df) is False


class TestPrepareFeatures:
    def test_adds_dev_quarter(self, base_df):
        result = prepare_features(base_df)
        assert "dev_quarter" in result.columns

    def test_adds_log_cumulative_paid(self, base_df):
        result = prepare_features(base_df)
        assert "log_cumulative_paid" in result.columns

    def test_adds_calendar_quarter(self, base_df):
        result = prepare_features(base_df)
        assert "calendar_quarter" in result.columns

    def test_calendar_quarter_range(self, base_df):
        result = prepare_features(base_df)
        q = result["calendar_quarter"].to_numpy()
        assert np.all(q >= 1) and np.all(q <= 4)

    def test_dev_quarter_non_negative(self, base_df):
        result = prepare_features(base_df)
        assert (result["dev_quarter"] < 0).sum() == 0

    def test_log_cumulative_paid_is_finite(self, base_df):
        result = prepare_features(base_df)
        vals = result["log_cumulative_paid"].to_numpy()
        assert np.all(np.isfinite(vals))

    def test_no_case_features_when_disabled(self, base_df):
        result = prepare_features(base_df, use_case_estimates=False)
        # Should not add case_estimate_std etc as extra cols
        # The case columns may exist as passthrough from source data
        # but get_feature_columns should exclude them
        cols = get_feature_columns(result, use_case_estimates=False)
        assert "n_case_revisions" not in cols
        assert "case_estimate_mean" not in cols

    def test_case_features_included_when_enabled(self, base_df):
        result = prepare_features(base_df, use_case_estimates=True)
        cols = get_feature_columns(result, use_case_estimates=True)
        assert "n_case_revisions" in cols
        assert "case_estimate_mean" in cols

    def test_row_count_unchanged(self, base_df):
        result = prepare_features(base_df)
        assert len(result) == len(base_df)


class TestGetFeatureColumns:
    def test_returns_list_of_strings(self, base_df):
        df = prepare_features(base_df)
        cols = get_feature_columns(df)
        assert isinstance(cols, list)
        assert all(isinstance(c, str) for c in cols)

    def test_all_cols_exist_in_df(self, base_df):
        df = prepare_features(base_df)
        cols = get_feature_columns(df)
        for c in cols:
            assert c in df.columns

    def test_feat_columns_included(self, base_df):
        df = prepare_features(base_df)
        cols = get_feature_columns(df)
        # feat_claim_type, feat_litigation, feat_fault should be there
        assert "feat_claim_type" in cols
        assert "feat_litigation" in cols

    def test_more_cols_with_case_estimates(self, base_df):
        df = prepare_features(base_df)
        cols_with = get_feature_columns(df, use_case_estimates=True)
        cols_without = get_feature_columns(df, use_case_estimates=False)
        assert len(cols_with) > len(cols_without)


class TestToNumpyFeatures:
    def test_returns_float32_array(self, base_df):
        df = prepare_features(base_df)
        cols = get_feature_columns(df)
        X = to_numpy_features(df, cols)
        assert X.dtype == np.float32
        assert X.ndim == 2

    def test_shape_matches_rows_and_features(self, base_df):
        df = prepare_features(base_df)
        cols = get_feature_columns(df)
        X = to_numpy_features(df, cols)
        assert X.shape == (len(df), len(cols))

    def test_no_nan_in_output(self, base_df):
        df = prepare_features(base_df)
        cols = get_feature_columns(df)
        X = to_numpy_features(df, cols)
        assert not np.any(np.isnan(X))


class TestToNumpyTargets:
    def test_returns_float32(self, base_df):
        settled = base_df.filter(
            (pl.col("is_open") == False)
            & (pl.col("ultimate").is_not_null())
            & (pl.col("ultimate") > pl.col("cumulative_paid"))
        )
        if len(settled) == 0:
            pytest.skip("No settled claims")
        y = to_numpy_targets(settled)
        assert y.dtype == np.float32

    def test_targets_are_finite(self, base_df):
        settled = base_df.filter(
            (pl.col("is_open") == False)
            & (pl.col("ultimate").is_not_null())
            & (pl.col("ultimate") > pl.col("cumulative_paid"))
        )
        if len(settled) == 0:
            pytest.skip("No settled claims")
        y = to_numpy_targets(settled)
        assert np.all(np.isfinite(y))
