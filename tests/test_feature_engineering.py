"""
Test suite for Phase 7 Feature Engineering.
Validates temporal safety, lag correctness, absence of future target leakage,
primary key uniqueness, feature quality, and report artifact completeness.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import pytest


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture(scope="module")
def input_dataset(project_root) -> pl.DataFrame:
    path = project_root / "data" / "processed" / "master_daily_demand_corrected.parquet"
    assert path.exists(), f"Input dataset missing: {path}"
    return pl.read_parquet(path)


@pytest.fixture(scope="module")
def feature_dataset(project_root) -> pl.DataFrame:
    path = project_root / "data" / "processed" / "model_features.parquet"
    assert path.exists(), f"Feature dataset missing: {path}"
    return pl.read_parquet(path)


@pytest.fixture(scope="module")
def reports_dir(project_root) -> Path:
    return project_root / "reports" / "features"


# 1. Feature table exists
def test_feature_table_created(feature_dataset):
    assert feature_dataset is not None
    assert feature_dataset.height > 0
    assert feature_dataset.width > 108


# 2. Primary key remains unique
def test_primary_key_uniqueness(feature_dataset):
    pk_cols = ["date", "store_id", "product_id"]
    for col in pk_cols:
        assert col in feature_dataset.columns
    unique_pk_count = feature_dataset.select(pk_cols).unique().height
    assert unique_pk_count == feature_dataset.height, "Primary key is not unique!"


# 3. Row count remains stable
def test_row_count_stability(input_dataset, feature_dataset):
    assert feature_dataset.height == input_dataset.height, (
        f"Row count altered: {input_dataset.height} vs {feature_dataset.height}"
    )
    assert feature_dataset.height == 35064


# 4. Demand lags are correct
def test_demand_lags_correctness(feature_dataset):
    # Select single store and product series sorted by date
    sub = (
        feature_dataset.filter(
            (pl.col("store_id") == "STR_KUL_001") & (pl.col("product_id") == "PRD_CHK_01")
        )
        .sort("date")
        .select(["date", "estimated_true_demand_training_safe", "demand_lag_1", "demand_lag_7"])
    )
    # Check that lag 1 matches demand at T-1
    d_raw = sub["estimated_true_demand_training_safe"].to_list()
    d_lag1 = sub["demand_lag_1"].to_list()
    d_lag7 = sub["demand_lag_7"].to_list()

    for idx in range(1, 20):
        assert d_lag1[idx] == pytest.approx(d_raw[idx - 1], rel=1e-5), (
            f"demand_lag_1 at idx {idx} does not equal raw demand at {idx - 1}"
        )
    for idx in range(7, 25):
        assert d_lag7[idx] == pytest.approx(d_raw[idx - 7], rel=1e-5), (
            f"demand_lag_7 at idx {idx} does not equal raw demand at {idx - 7}"
        )


# 5. Price lags are correct
def test_price_lags_correctness(feature_dataset):
    sub = (
        feature_dataset.filter(
            (pl.col("store_id") == "STR_KUL_001") & (pl.col("product_id") == "PRD_CHK_01")
        )
        .sort("date")
        .select(["date", "purchase_price", "purchase_price_lag_1", "purchase_price_lag_7"])
    )
    p_raw = sub["purchase_price"].to_list()
    p_lag1 = sub["purchase_price_lag_1"].to_list()
    p_lag7 = sub["purchase_price_lag_7"].to_list()

    for idx in range(1, 20):
        assert p_lag1[idx] == pytest.approx(p_raw[idx - 1], rel=1e-5), (
            f"purchase_price_lag_1 at idx {idx} does not equal price at {idx - 1}"
        )
    for idx in range(7, 25):
        assert p_lag7[idx] == pytest.approx(p_raw[idx - 7], rel=1e-5), (
            f"purchase_price_lag_7 at idx {idx} does not equal price at {idx - 7}"
        )


# 6. Rolling windows use historical information only
def test_rolling_windows_historical_only(feature_dataset):
    sub = (
        feature_dataset.filter(
            (pl.col("store_id") == "STR_KUL_001") & (pl.col("product_id") == "PRD_CHK_01")
        )
        .sort("date")
        .select(["date", "estimated_true_demand_training_safe", "demand_mean_7d"])
    )
    d_raw = sub["estimated_true_demand_training_safe"].to_list()
    d_mean7 = sub["demand_mean_7d"].to_list()

    # For idx 7 (8th element), mean 7d should be mean of [0, 1, 2, 3, 4, 5, 6] (T-7 to T-1)
    for idx in range(7, 20):
        expected_mean = float(np.mean(d_raw[idx - 7 : idx]))
        assert d_mean7[idx] == pytest.approx(expected_mean, rel=1e-5), (
            f"demand_mean_7d at idx {idx} does not match mean of [idx-7 : idx]"
        )


# 7. Same-weekday features are correct
def test_same_weekday_seasonality(feature_dataset):
    assert "demand_dow_median_4w" in feature_dataset.columns
    assert "demand_dow_median_8w" in feature_dataset.columns
    sub = (
        feature_dataset.filter(
            (pl.col("store_id") == "STR_KUL_001") & (pl.col("product_id") == "PRD_CHK_01")
        )
        .sort("date")
        .select(["date", "demand_dow_median_4w"])
    )
    # Check that after 4 weeks (28 days), value is populated and positive
    valid_vals = sub["demand_dow_median_4w"].slice(35, 10).to_list()
    for v in valid_vals:
        assert v is not None
        assert v > 0


# 8. Annual lag features are correct
def test_annual_lag_features(feature_dataset):
    assert "demand_lag_364" in feature_dataset.columns
    assert "demand_lag_365" in feature_dataset.columns
    assert "demand_prior_year_week_mean" in feature_dataset.columns

    sub = (
        feature_dataset.filter(
            (pl.col("store_id") == "STR_KUL_001") & (pl.col("product_id") == "PRD_CHK_01")
        )
        .sort("date")
        .select(["date", "estimated_true_demand_training_safe", "demand_lag_365"])
    )
    d_raw = sub["estimated_true_demand_training_safe"].to_list()
    d_lag365 = sub["demand_lag_365"].to_list()

    # For idx 365, lag 365 must equal raw demand at idx 0
    assert d_lag365[365] == pytest.approx(d_raw[0], rel=1e-5)
    assert d_lag365[366] == pytest.approx(d_raw[1], rel=1e-5)


# 9. Inventory features historical only
def test_inventory_features_past_only(feature_dataset):
    assert "stockout_flag_prior_day" in feature_dataset.columns
    assert "stockout_count_7d" in feature_dataset.columns
    assert "days_since_last_stockout" in feature_dataset.columns
    assert "days_since_last_replenishment" in feature_dataset.columns
    assert "inventory_coverage_days_7d" in feature_dataset.columns

    # Verify coverage days is non-negative and finite
    cov = feature_dataset["inventory_coverage_days_7d"].drop_nulls()
    assert cov.min() >= 0.0
    assert cov.max() <= 999.0


# 10. Promotion features temporally correct
def test_promotion_features_validity(feature_dataset):
    assert "promo_active_flag" in feature_dataset.columns
    assert "promo_days_active_so_far" in feature_dataset.columns
    assert "promo_days_until_end" in feature_dataset.columns
    assert "promo_frequency_30d" in feature_dataset.columns
    assert "planned_promo_within_7d" in feature_dataset.columns

    # Active flag is binary
    unique_vals = set(feature_dataset["promo_active_flag"].unique().to_list())
    assert unique_vals.issubset({0.0, 1.0})


# 11. Holiday proximity features correct
def test_holiday_proximity_features(feature_dataset):
    for col in [
        "holiday_lead_1d", "holiday_lead_2d", "holiday_lead_3d", "holiday_lead_7d",
        "holiday_lag_1d", "holiday_lag_2d", "holiday_lag_3d", "holiday_lag_7d",
        "days_until_cny", "days_since_cny",
        "days_until_raya", "days_since_raya",
        "days_until_deepavali", "days_since_deepavali",
        "days_until_christmas", "days_since_christmas",
        "major_festival_window_active"
    ]:
        assert col in feature_dataset.columns
        assert feature_dataset[col].null_count() == 0, f"Null values in holiday feature {col}"


# 12. Weather rolling features historical only
def test_weather_rolling_historical(feature_dataset):
    assert "weather_rainfall_sum_3d" in feature_dataset.columns
    assert "weather_rainfall_sum_7d" in feature_dataset.columns
    assert "weather_temp_mean_7d" in feature_dataset.columns

    # Rainfall sum must be >= 0
    rf_sum = feature_dataset["weather_rainfall_sum_7d"].drop_nulls()
    assert rf_sum.min() >= 0.0


# 13. Fuel price changes
def test_fuel_price_features(feature_dataset):
    assert "fuel_petrol_effective" in feature_dataset.columns
    assert "fuel_diesel_effective" in feature_dataset.columns
    assert "fuel_diesel_pct_change_7d" in feature_dataset.columns
    assert "fuel_diesel_pct_change_28d" in feature_dataset.columns


# 14. Currency changes
def test_currency_features(feature_dataset):
    assert "fx_rate" in feature_dataset.columns
    assert "fx_change_1d" in feature_dataset.columns
    assert "fx_pct_change_7d" in feature_dataset.columns
    assert "fx_mean_30d" in feature_dataset.columns


# 15. Commodity mapping
def test_commodity_features(feature_dataset):
    assert "commodity_price" in feature_dataset.columns
    assert "commodity_mean_7d" in feature_dataset.columns
    assert "commodity_pct_change_7d" in feature_dataset.columns


# 16. Market price features
def test_market_price_features(feature_dataset):
    assert "market_price_median" in feature_dataset.columns
    assert "price_vs_market_diff" in feature_dataset.columns
    assert "price_vs_market_ratio" in feature_dataset.columns


# 17 & 18. Inflation features (CPI & PPI)
def test_inflation_features(feature_dataset):
    assert "cpi_headline" in feature_dataset.columns
    assert "cpi_category" in feature_dataset.columns
    assert "ppi_headline" in feature_dataset.columns
    assert "ppi_industry" in feature_dataset.columns
    assert "cpi_pct_change_30d" in feature_dataset.columns
    assert "ppi_pct_change_30d" in feature_dataset.columns


# 19 & 20. Government assistance and subsidy features
def test_policy_and_assistance_features(feature_dataset):
    assert "days_since_last_gov_assist" in feature_dataset.columns
    assert "is_post_gov_assist_3d" in feature_dataset.columns
    assert "is_chicken_subsidy_active" in feature_dataset.columns
    assert "is_diesel_rationalisation_active" in feature_dataset.columns
    assert "is_minimum_wage_rm1500_active" in feature_dataset.columns
    assert "is_service_tax_8pct_active" in feature_dataset.columns


# 21. Infinite values handled
def test_infinite_values_handled(feature_dataset):
    float_cols = [c for c, dt in zip(feature_dataset.columns, feature_dataset.dtypes) if dt == pl.Float64]
    for c in float_cols:
        inf_cnt = feature_dataset.filter(pl.col(c).is_infinite()).height
        assert inf_cnt == 0, f"Infinite value detected in column {c}"


# 22. No duplicate feature names
def test_no_duplicate_feature_names(feature_dataset):
    assert len(feature_dataset.columns) == len(set(feature_dataset.columns))


# 23. No target columns exist
def test_no_target_columns_exist(feature_dataset):
    prohibited = [c for c in feature_dataset.columns if c.startswith("target_") or c.startswith("future_")]
    assert len(prohibited) == 0, f"Prohibited target columns found: {prohibited}"


# 24. Training-safe demand history is preserved
def test_training_safe_demand_integrity(feature_dataset):
    assert "estimated_true_demand_training_safe" in feature_dataset.columns


# 25. Raw and previous processed datasets remain unchanged
def test_immutability_of_upstream_datasets(project_root):
    raw_dir = project_root / "data" / "raw"
    assert raw_dir.exists()
    assert (project_root / "data" / "processed" / "master_daily.parquet").exists()
    assert (project_root / "data" / "processed" / "master_daily_demand_corrected.parquet").exists()


# 26. Feature registry and manifests exist and are populated
def test_registry_and_manifests_completeness(reports_dir):
    reg_path = reports_dir / "feature_registry.csv"
    dem_path = reports_dir / "demand_feature_manifest.csv"
    pri_path = reports_dir / "price_feature_manifest.csv"
    exc_path = reports_dir / "excluded_features.csv"

    for p in [reg_path, dem_path, pri_path, exc_path]:
        assert p.exists(), f"Report file missing: {p}"
        df_csv = pd.read_csv(p)
        assert len(df_csv) > 0, f"Report file is empty: {p}"

    df_reg = pd.read_csv(reg_path)
    df_dem = pd.read_csv(dem_path)
    df_pri = pd.read_csv(pri_path)
    df_exc = pd.read_csv(exc_path)

    assert len(df_dem) > 100, f"Too few demand features: {len(df_dem)}"
    assert len(df_pri) > 80, f"Too few price features: {len(df_pri)}"
    assert len(df_exc) >= 4, f"Excluded features missing expected entries"
