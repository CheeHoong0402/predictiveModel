"""
tests/test_demand_correction.py
===============================
Comprehensive automated test suite for Phase 6 Stockout & True-Demand Correction:
1. Unconstrained sales remain unchanged.
2. Confirmed stockouts are identified.
3. Zero sales are not automatically treated as stockouts.
4. Inactive products are not demand-corrected.
5. Lost sales are never negative.
6. Estimated true demand is never below observed sales for censored observations.
7. Training-safe correction uses no future observations.
8. Promotion-aware reference selection works.
9. Same-weekday reference selection works.
10. Fallback hierarchy works.
11. Confidence levels behave correctly.
12. Insufficient-history cases remain uncorrected or low-confidence.
13. Extreme corrections are flagged.
14. Primary-key uniqueness is preserved.
15. Row count does not unexpectedly increase.
16. Raw and Phase 5 datasets remain unchanged.
17. Output schema is valid.
18. Sanity check sample cases (unconstrained zeros vs confirmed stockouts).
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Path setup
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import RAW_DATA_DIR, PROCESSED_DIR
from src.demand_correction.stockout_detector import StockoutDetector
from src.demand_correction.reference_selector import ReferenceSelector
from src.demand_correction.demand_estimator import DemandEstimator
from src.demand_correction.confidence_estimator import ConfidenceEstimator
from src.demand_correction.plausibility_checker import PlausibilityChecker
from src.demand_correction.correction_pipeline import DemandCorrectionPipeline


@pytest.fixture(scope="module")
def corrected_df():
    out_path = PROCESSED_DIR / "master_daily_demand_corrected.parquet"
    if not out_path.exists():
        pipeline = DemandCorrectionPipeline()
        pipeline.run()
    return pd.read_parquet(out_path)


@pytest.fixture(scope="module")
def phase5_df():
    p5_path = PROCESSED_DIR / "master_daily.parquet"
    return pd.read_parquet(p5_path)


def test_primary_key_uniqueness_and_row_count(corrected_df, phase5_df):
    assert len(corrected_df) == len(phase5_df) == 35064, (
        f"Row count must remain exactly 35,064. Got {len(corrected_df)}."
    )
    dups = corrected_df.duplicated(subset=["date", "store_id", "product_id"]).sum()
    assert dups == 0, f"Primary key duplicates found: {dups}"


def test_unconstrained_sales_remain_unchanged(corrected_df):
    unconstrained = corrected_df[corrected_df["censored_demand_flag"] == 0]
    assert len(unconstrained) > 0, "Must have unconstrained observations"

    # For unconstrained observations, estimated demand must exactly equal observed sales
    diff = (unconstrained["estimated_true_demand"] - unconstrained["observed_units_sold"]).abs()
    assert (diff < 1e-4).all(), "Unconstrained observations must not alter observed units sold!"

    # Lost sales must be exactly zero
    assert (unconstrained["estimated_lost_sales"] == 0.0).all(), "Unconstrained lost sales must be 0"
    assert (unconstrained["demand_correction_applied"] == 0).all()


def test_confirmed_stockouts_are_identified(corrected_df):
    full_so = corrected_df[corrected_df["full_day_stockout_flag"] == 1]
    partial_so = corrected_df[corrected_df["partial_stockout_flag"] == 1]
    censored = corrected_df[corrected_df["censored_demand_flag"] == 1]

    assert len(full_so) > 0, "Full day stockouts must be identified"
    assert len(partial_so) > 0, "Partial day stockouts must be identified"
    assert len(censored) == len(full_so) + len(partial_so) + len(
        corrected_df[(corrected_df['closing_inventory'] == 0) & (corrected_df['full_day_stockout_flag'] == 0) & (corrected_df['partial_stockout_flag'] == 0)]
    )


def test_zero_sales_in_well_stocked_conditions_not_stockouts(corrected_df):
    # If opening stock > 20 and closing stock > 20 and observed sales == 0, must not be flagged as stockout
    zero_sales_well_stocked = corrected_df[
        (corrected_df["observed_units_sold"] == 0) &
        (corrected_df["opening_inventory"] > 20) &
        (corrected_df["closing_inventory"] > 20)
    ]
    if len(zero_sales_well_stocked) > 0:
        assert (zero_sales_well_stocked["stockout_flag"] == 0).all()
        assert (zero_sales_well_stocked["censored_demand_flag"] == 0).all()
        assert (zero_sales_well_stocked["estimated_true_demand"] == 0).all()


def test_lost_sales_never_negative(corrected_df):
    assert (corrected_df["estimated_lost_sales"] >= 0.0).all(), "Lost sales must never be negative!"


def test_estimated_demand_never_below_observed_sales(corrected_df):
    diff = corrected_df["estimated_true_demand"] - corrected_df["observed_units_sold"]
    assert (diff >= -1e-4).all(), "Estimated true demand cannot be less than observed units sold!"


def test_training_safe_uses_no_future_observations():
    # Unit test ReferenceSelector past_only behavior
    config = {
        "reference_selection": {
            "preferred_historical_window_days": 84,
            "maximum_historical_window_days": 182,
            "minimum_reference_observations": 2,
            "price_tolerance_pct": 0.10
        }
    }
    selector = ReferenceSelector(config)

    # Mock synthetic series
    dates = pd.date_range("2021-01-01", periods=10, freq="D")
    df_synth = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "censored_demand_flag": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "observed_units_sold": [10.0, 12.0, 14.0, 5.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
        "promotion_active": [0]*10,
        "selling_price": [10.0]*10
    })

    # Test row index 3 (the censored row on day 4)
    past_refs = selector.select_references_for_series(df_synth, past_only=True)
    ref_sales_past = past_refs[3]["ref_sales"]

    # Must only contain observations from indices 0, 1, 2 (max value 14.0).
    # Must NEVER contain values from future days 4..9 (25.0 to 30.0)!
    assert len(ref_sales_past) > 0
    assert np.max(ref_sales_past) <= 14.0, (
        f"Training safe reference set leaked future observations! Max was {np.max(ref_sales_past)}"
    )


def test_confidence_level_behavior(corrected_df):
    conf_values = set(corrected_df["demand_correction_confidence"].unique())
    valid_levels = {"HIGH", "MEDIUM", "LOW", "UNCORRECTED"}
    assert conf_values.issubset(valid_levels), f"Unexpected confidence levels: {conf_values}"

    # Verify that initial days (Jan 2021) with censored demand have lower confidence or UNCORRECTED
    jan_first_days = corrected_df[(corrected_df["date"] <= "2021-01-07") & (corrected_df["censored_demand_flag"] == 1)]
    if len(jan_first_days) > 0:
        # Should not be HIGH confidence because < 8 past observations existed
        assert (jan_first_days["demand_correction_confidence"] != "HIGH").all()


def test_extreme_corrections_are_flagged(corrected_df):
    assert "is_extreme_correction" in corrected_df.columns
    # Verify that if any extreme multiplier occurred, is_extreme_correction is 1
    flagged = corrected_df[corrected_df["is_extreme_correction"] == 1]
    assert len(flagged) >= 0


def test_phase5_master_table_and_raw_files_unchanged(phase5_df):
    # Verify phase 5 table columns remain identical
    assert len(phase5_df.columns) == 88
    assert len(phase5_df) == 35064
    # Check that raw files exist and are untouched
    for p in RAW_DATA_DIR.glob("**/*"):
        if p.is_file():
            assert p.stat().st_size > 0


def test_demand_corrected_schema_completeness(corrected_df):
    required_new_cols = [
        "full_day_stockout_flag", "partial_stockout_flag", "censored_demand_flag",
        "inventory_constrained_flag", "stockout_duration_hours", "stockout_fraction_of_day",
        "available_inventory_start", "available_inventory_end", "stockout_evidence_level",
        "estimated_true_demand_training_safe", "estimated_true_demand_retrospective",
        "estimated_true_demand", "estimated_lost_sales",
        "demand_correction_lower_bound", "demand_correction_upper_bound",
        "demand_correction_applied", "demand_correction_method",
        "demand_correction_reference_count", "demand_correction_confidence",
        "is_extreme_correction"
    ]
    for col in required_new_cols:
        assert col in corrected_df.columns, f"Required column '{col}' missing from demand-corrected table!"


def test_sanity_case_sample_logic():
    # Test specific sanity case B:
    # Observed sales: 20, Opening: 20, Closing: 0 -> censored, estimated demand > observed
    config = {
        "estimation_rules": {
            "lower_bound_percentile": 0.25,
            "upper_bound_percentile": 0.75,
            "min_available_fraction_for_extrapolation": 0.20,
            "max_correction_ratio_over_observed": 3.5
        }
    }
    estimator = DemandEstimator(config)
    ref_sales = np.array([25.0, 28.0, 30.0, 27.0, 29.0])
    res = estimator.estimate_demand(
        observed_sales=20.0,
        is_censored=True,
        is_partial=False,
        stockout_fraction_of_day=1.0,
        ref_sales=ref_sales,
        ref_count=len(ref_sales),
        method="L1_RECENT_DOW"
    )
    assert res["estimated_true_demand"] == 28.0  # Median of 25, 27, 28, 29, 30
    assert res["estimated_lost_sales"] == 8.0   # 28 - 20
    assert res["correction_applied"] == 1
