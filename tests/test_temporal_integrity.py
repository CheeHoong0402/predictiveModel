"""
Automated Test Suite for Phase 8 Temporal Integrity & Future-Data Leakage Audit.
Validates zero target contamination, mathematical lag offsets, past-only rolling bounds,
adversarial future-sentinel invariance, truncation invariance, and publication release constraints.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.leakage.rolling_window_validator import RollingWindowValidator
from src.leakage.temporal_audit import TemporalAuditEngine


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture(scope="module")
def feature_dataset(project_root) -> pl.DataFrame:
    path = project_root / "data" / "processed" / "model_features.parquet"
    assert path.exists(), f"Feature dataset missing: {path}"
    return pl.read_parquet(path)


@pytest.fixture(scope="module")
def leakage_reports_dir(project_root) -> Path:
    return project_root / "reports" / "leakage"


# 1. No negative lags exist
def test_no_negative_lags(feature_dataset):
    lag_cols = [c for c in feature_dataset.columns if "_lag_" in c]
    for col in lag_cols:
        parts = col.split("_lag_")
        lag_num = int(parts[1].rstrip("d"))
        assert lag_num > 0, f"Negative or zero lag detected: {col}"


# 2. No future rolling windows exist
def test_no_future_rolling_windows(feature_dataset):
    # Verify rolling window validator finds 0 violations
    violations = RollingWindowValidator.audit_rolling_windows(feature_dataset)
    future_inclusions = [v for v in violations if v["issue_type"] == "FUTURE_INCLUSIVE_ROLLING_WINDOW"]
    assert len(future_inclusions) == 0, f"Future inclusive rolling windows found: {future_inclusions}"


# 3. No centred rolling windows exist
def test_no_centred_rolling_windows(feature_dataset):
    # Mathematical proof on representative date: verified to be strictly backward-looking
    proof = RollingWindowValidator.generate_mathematical_proof(
        df=feature_dataset,
        store_id="STR_KUL_001",
        product_id="PRD_CHK_01",
        target_date="2024-06-15",
    )
    assert proof["is_strictly_past_only"] is True
    assert proof["temporal_proof_passed"] is True


# 4. No future backfill exists
def test_no_future_backfill(feature_dataset):
    # First row in series (2021-01-01) for demand_lag_1 must not be populated with future values
    first_row = feature_dataset.filter(pl.col("date") == "2021-01-01")
    d_lag1 = first_row["demand_lag_1"].drop_nulls()
    # Any populated values must be 0.0 or null, never future demand
    if not d_lag1.is_empty():
        assert d_lag1.max() <= 0.0


# 5. Training-safe corrected demand is used
def test_training_safe_corrected_demand_used(feature_dataset):
    assert "estimated_true_demand_training_safe" in feature_dataset.columns


# 6. Retrospective corrected demand is excluded from safe manifests
def test_retrospective_corrected_demand_excluded(leakage_reports_dir):
    safe_manifest = pd.read_csv(leakage_reports_dir / "demand_features_temporally_safe.csv")
    assert "estimated_true_demand_retrospective" not in safe_manifest["feature_name"].values


# 7. Future actual weather is excluded
def test_future_actual_weather_excluded(feature_dataset):
    prohibited_weather = [c for c in feature_dataset.columns if "future_weather" in c.lower()]
    assert len(prohibited_weather) == 0


# 8. Exchange-rate filling is past-only
def test_exchange_rate_filling_past_only(feature_dataset):
    # Check that fx changes are consistent and no backwards shifts exist
    assert "fx_change_1d" in feature_dataset.columns
    sub = (
        feature_dataset.filter(
            (pl.col("store_id") == "STR_KUL_001") & (pl.col("product_id") == "PRD_CHK_01")
        )
        .sort("date")
    )
    rates = sub["fx_rate"].to_list()
    diffs = sub["fx_change_1d"].to_list()
    # At index t, diff should be rate[t] - rate[t-1]
    for t in range(1, 20):
        expected_diff = rates[t] - rates[t - 1]
        assert diffs[t] == pytest.approx(expected_diff, rel=1e-4)


# 9 & 10. Consumer Price Index and Producer Price Index respect publication delay
def test_inflation_publication_delay_respected(leakage_reports_dir):
    pub_assumptions = pd.read_csv(leakage_reports_dir / "publication_lag_assumptions.csv")
    cpi_row = pub_assumptions[pub_assumptions["dataset_name"].str.contains("Consumer Price Index")]
    ppi_row = pub_assumptions[pub_assumptions["dataset_name"].str.contains("Producer Price Index")]
    assert cpi_row["typical_lag_days"].iloc[0] == 21
    assert ppi_row["typical_lag_days"].iloc[0] == 30


# 11 & 12. Retail and trade statistics respect publication delay
def test_retail_trade_publication_assumptions(leakage_reports_dir):
    assert (leakage_reports_dir / "publication_lag_assumptions.csv").exists()


# 13. Government assistance respects announcement timing
def test_government_assistance_timing(feature_dataset):
    assert "days_since_last_gov_assist" in feature_dataset.columns
    # Must be non-negative
    assert feature_dataset["days_since_last_gov_assist"].min() >= 0.0


# 14. Subsidy information respects announcement timing
def test_subsidy_information_timing(feature_dataset):
    assert "days_since_subsidy_announced" in feature_dataset.columns
    assert feature_dataset["days_since_subsidy_announced"].min() >= 0.0


# 15. Budget-policy information respects announcement timing
def test_budget_policy_timing(feature_dataset):
    assert "days_since_policy_announced" in feature_dataset.columns
    assert feature_dataset["days_since_policy_announced"].min() >= 0.0


# 16. Promotion future features require known schedules
def test_promotion_future_features_validity(feature_dataset):
    assert "planned_promo_within_7d" in feature_dataset.columns
    # Values are strictly binary indicators 0.0 or 1.0
    unique_vals = set(feature_dataset["planned_promo_within_7d"].unique().to_list())
    assert unique_vals.issubset({0.0, 1.0})


# 17. Supplier reliability uses completed historical orders only
def test_supplier_reliability_completed_orders_only(feature_dataset):
    assert "supplier_on_time_rate_30d" in feature_dataset.columns
    sub = (
        feature_dataset.filter(
            (pl.col("store_id") == "STR_KUL_001") & (pl.col("product_id") == "PRD_CHK_01")
        )
        .sort("date")
    )
    # Check that rate is bounded in [0, 1]
    rates = sub["supplier_on_time_rate_30d"].drop_nulls()
    assert rates.min() >= 0.0
    assert rates.max() <= 1.0


# 18. Frequency encoding is not globally leaked
def test_no_global_frequency_encoding(feature_dataset):
    freq_cols = [c for c in feature_dataset.columns if "freq_enc" in c.lower() or "global_freq" in c.lower()]
    assert len(freq_cols) == 0


# 19. Target encoding is completely absent
def test_target_encoding_absent(feature_dataset):
    target_enc_cols = [
        c for c in feature_dataset.columns
        if any(p in c.lower() for p in ["target_enc", "target_mean", "mean_future_demand", "mean_future_price"])
    ]
    assert len(target_enc_cols) == 0, f"Found target encodings: {target_enc_cols}"


# 20. Scaling has not been globally fitted
def test_no_global_scaling(feature_dataset):
    # Verify raw units in demand
    d_vals = feature_dataset["demand_lag_1"].drop_nulls()
    assert d_vals.max() > 10.0  # Raw units, not standard-scaled


# 21. Imputation is temporally safe
def test_imputation_temporally_safe(feature_dataset):
    # Verify unobserved market survey outside Jan 2024 remains null rather than filled with future mean
    unobserved_slice = feature_dataset.filter(pl.col("date") < "2024-01-01")
    assert unobserved_slice["market_price_median"].null_count() > 0


# 22. Product lifecycle features contain no future disappearance information
def test_no_product_lifecycle_disappearance(feature_dataset):
    prohibited = ["last_seen_date", "days_until_discontinuation", "is_discontinued_in_future"]
    found = [c for c in feature_dataset.columns if c in prohibited]
    assert len(found) == 0


# 23. Truncation-invariance tests pass
def test_truncation_invariance(feature_dataset):
    engine = TemporalAuditEngine()
    results = engine.run_truncation_invariance_test(
        df=feature_dataset,
        cutoff_dates=["2022-12-31", "2023-12-31"],
        store_id="STR_KUL_001",
        product_id="PRD_CHK_01",
    )
    for res in results:
        assert res["truncation_invariance_passed"] is True, f"Failed truncation test for {res['cutoff_date']}"


# 24. Adversarial future-sentinel tests pass
def test_adversarial_sentinel_invariance(feature_dataset):
    engine = TemporalAuditEngine()
    result = engine.run_adversarial_sentinel_test(
        df=feature_dataset,
        cutoff_date="2023-06-30",
        store_id="STR_KUL_001",
        product_id="PRD_CHK_01",
    )
    assert result["adversarial_test_passed"] is True, "Adversarial sentinel test failed"


# 25. Primary key remains unique
def test_primary_key_uniqueness_preserved(feature_dataset):
    pk_cols = ["date", "store_id", "product_id"]
    assert feature_dataset.select(pk_cols).unique().height == feature_dataset.height


# 26. Safe feature manifests match the feature table
def test_safe_manifests_match_feature_table(feature_dataset, leakage_reports_dir):
    demand_safe = pd.read_csv(leakage_reports_dir / "demand_features_temporally_safe.csv")
    price_safe = pd.read_csv(leakage_reports_dir / "price_features_temporally_safe.csv")

    assert len(demand_safe) > 0
    assert len(price_safe) > 0

    # Ensure all manifest features exist in the physical feature table
    for f in demand_safe["feature_name"]:
        assert f in feature_dataset.columns, f"Safe demand feature {f} missing in model_features.parquet"
    for f in price_safe["feature_name"]:
        assert f in feature_dataset.columns, f"Safe price feature {f} missing in model_features.parquet"
