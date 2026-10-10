"""
test_price_data_sufficiency.py
==============================
Unit tests for the purchase-price training data sufficiency and model sanity audit.
Tests core audit logic, verification functions, temporal integrity, and baseline calculations.
"""

import numpy as np
import pandas as pd
import pytest

from src.validation.price_data_sufficiency import (
    DensityThresholds,
    assert_no_holdout,
    check_no_future_backfill,
    classify_density,
    count_price_changes,
    decide_status,
    find_target_equal_features,
    flag_forward_filled,
    genuine_observation_mask,
    horizon_movement,
    is_available_at_origin,
    last_price_baseline_metrics,
    normalised_value,
    price_variation_stats,
    target_coverage,
    HOLDOUT_START,
)


def test_classify_density():
    th = DensityThresholds(high=300, medium=100, low=30)
    assert classify_density(350, th) == "HIGH"
    assert classify_density(300, th) == "HIGH"
    assert classify_density(150, th) == "MEDIUM"
    assert classify_density(100, th) == "MEDIUM"
    assert classify_density(50, th) == "LOW"
    assert classify_density(30, th) == "LOW"
    assert classify_density(10, th) == "COLD_START"


def test_flag_forward_filled():
    # Run of 3 identical numbers: first is genuine, subsequent 2 are flagged
    prices = pd.Series([10.0, 10.0, 10.0, 11.0, 12.0, 12.0])
    flagged = flag_forward_filled(prices, min_run=3)
    assert not flagged.iloc[0]
    assert flagged.iloc[1]
    assert flagged.iloc[2]
    assert not flagged.iloc[3]
    # Run of 2 with min_run=3 is not flagged as a long forward fill run
    assert not flagged.iloc[4]
    assert not flagged.iloc[5]


def test_genuine_observation_mask():
    df = pd.DataFrame({
        "store_id": ["S1"] * 5,
        "product_id": ["P1"] * 5,
        "date": pd.date_range("2023-01-01", periods=5),
        "current_supplier_purchase_price": [5.0, 5.0, 5.0, 6.0, np.nan],
    })
    mask = genuine_observation_mask(df, min_run=3)
    # 5.0 (gen), 5.0 (ff), 5.0 (ff), 6.0 (gen), nan (not gen)
    assert mask.tolist() == [True, False, False, True, False]


def test_count_price_changes():
    prices = pd.Series([10.0, 10.0, 10.5, 10.5, 10.2])
    assert count_price_changes(prices) == 2


def test_price_variation_stats():
    prices = pd.Series([10.0, 11.0, 10.0, 11.0])
    stats = price_variation_stats(prices)
    assert stats["n_obs"] == 4
    assert stats["mean"] == 10.5
    assert stats["unique_prices"] == 2
    assert stats["change_count"] == 3
    assert stats["unchanged_rate"] == 0.0
    assert stats["mean_abs_change"] == 1.0


def test_horizon_movement():
    cur = pd.Series([10.0, 12.0, 15.0])
    fut = pd.Series([10.0, 13.0, 14.0])
    mv = horizon_movement(cur, fut)
    assert mv["n"] == 3
    assert mv["genuine_change_count"] == 2
    assert abs(mv["mean_abs_move"] - (0.0 + 1.0 + 1.0) / 3) < 1e-6
    assert abs(mv["pct_differing"] - 66.666666) < 1e-3


def test_target_coverage():
    dates = pd.date_range("2024-09-28", periods=5, freq="D")
    # Holdout is 2024-10-01. dates: 09-28, 09-29, 09-30 (dev=3), 10-01, 10-02 (holdout=2)
    valid = pd.Series([True, True, False, True, True])
    cov = target_coverage(valid, dates, pd.Timestamp("2024-09-30"))
    assert cov["candidate_rows"] == 3
    assert cov["valid_rows"] == 2
    assert cov["invalid_rows"] == 1
    assert abs(cov["coverage_pct"] - 66.666666) < 1e-3
    assert cov["holdout_rows_counted_only"] == 2


def test_last_price_baseline_metrics():
    cur = pd.Series([10.0, 20.0])
    act = pd.Series([11.0, 18.0])
    res = last_price_baseline_metrics(cur, act)
    # err: 1.0, 2.0 -> mae = 1.5
    assert res["mae"] == 1.5
    # rmse: sqrt((1 + 4)/2) = sqrt(2.5) = 1.5811
    assert abs(res["rmse"] - np.sqrt(2.5)) < 1e-4
    assert res["n"] == 2


def test_check_no_future_backfill():
    origins = pd.Series(["2023-01-05", "2023-01-05"])
    origin_obs = pd.Series(["2023-01-05", "2023-01-06"])  # 2nd is from future!
    target_obs = pd.Series(["2023-01-07", "2023-01-05"])  # 2nd is before due date (5 + 2 = 7)
    res = check_no_future_backfill(origins, origin_obs, target_obs, horizon=2)
    assert res["origin_price_from_future"] == 1
    assert res["target_before_due_date"] == 1
    assert res["checked_rows"] == 2


def test_assert_no_holdout():
    df_clean = pd.DataFrame({"date": [pd.Timestamp("2024-09-30")]})
    assert_no_holdout(df_clean)  # Passes

    df_contaminated = pd.DataFrame({"date": [pd.Timestamp("2024-10-01")]})
    with pytest.raises(AssertionError):
        assert_no_holdout(df_contaminated)


def test_is_available_at_origin():
    # Obs 2023-01-01, lag 2 days -> available 2023-01-03
    assert is_available_at_origin(pd.Timestamp("2023-01-01"), 2, pd.Timestamp("2023-01-03"))
    assert not is_available_at_origin(pd.Timestamp("2023-01-01"), 2, pd.Timestamp("2023-01-02"))


def test_find_target_equal_features():
    target = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    features = pd.DataFrame({
        "feat_clean": np.random.randn(10),
        "feat_leaked_exact": target,
        "feat_leaked_corr": target * 2.0 + 0.0001,
    })
    hits = find_target_equal_features(features, target)
    assert "feat_leaked_exact" in hits
    assert "feat_leaked_corr" in hits
    assert "feat_clean" not in hits


def test_normalised_value():
    assert normalised_value(0.2, 0.2) == 1.0
    assert normalised_value(0.1, 0.2) == 0.5
    assert np.isnan(normalised_value(0.1, 0.0))


def test_decide_status():
    assert decide_status([{"severity": "CRITICAL"}]) == "FAIL"
    assert decide_status([{"severity": "HIGH"}]) == "PASS_WITH_WARNINGS"
    assert decide_status([{"severity": "MEDIUM"}]) == "PASS_WITH_WARNINGS"
    assert decide_status([{"severity": "LOW"}]) == "PASS_WITH_WARNINGS"
    assert decide_status([]) == "PASS"
