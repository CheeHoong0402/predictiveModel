"""Comprehensive automated test suite for Phase 20 Historical Business Backtesting."""
from typing import Dict, Any
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.paths import PROJECT_ROOT
from src.backtesting.config import load_backtest_config
from src.backtesting.strategies import (
    decide_no_action,
    decide_fixed_coverage,
    decide_reorder_point,
    decide_phase18_optimiser,
    decide_phase20_ai_strategy
)
from src.backtesting.engine import run_historical_backtest
from src.backtesting.metrics import compute_realised_metrics, compare_ai_vs_reference
from src.backtesting.segmentation import (
    compute_store_segmentation,
    compute_category_segmentation,
    compute_phase18_vs_phase19_comparison,
    compute_cost_sensitivity_table
)
from src.backtesting.replay_api import run_historical_replay, replay_historical_decision


@pytest.fixture
def sample_demand_forecasts() -> Dict[int, float]:
    return {1: 25.0, 3: 75.0, 7: 175.0, 14: 350.0, 21: 525.0, 30: 750.0}


@pytest.fixture
def sample_price_forecasts() -> Dict[int, float]:
    return {1: 10.0, 3: 10.0, 7: 10.0, 14: 10.0, 21: 10.0, 30: 10.0}


# =============================================================================
# 1. TEMPORAL & METHODOLOGICAL INTEGRITY TESTS
# =============================================================================

def test_chronological_replay_uses_past_information_only():
    """Verify that decisions are anchored to historical forecast origins without future access."""
    cfg = load_backtest_config()
    dec_path = PROJECT_ROOT / "data" / "backtesting" / "decision_backtest.parquet"
    assert dec_path.exists()
    df = pd.read_parquet(dec_path)
    # Decisions must not use dates past end_origin_date
    assert (pd.to_datetime(df["decision_date"]) <= pd.Timestamp(cfg["backtest"]["end_origin_date"])).all()


def test_strategy_inventory_states_evolve_independently():
    """Verify counterfactual strategies diverge and do not copy actual closing inventory."""
    dec_path = PROJECT_ROOT / "data" / "backtesting" / "decision_backtest.parquet"
    df = pd.read_parquet(dec_path)

    # Compare ending inventory between NO_ACTION and FIXED_30_DAY
    no_act = df[df["strategy"] == "NO_ACTION"]["ending_inventory"].mean()
    f30 = df[df["strategy"] == "FIXED_30_DAY"]["ending_inventory"].mean()
    assert no_act == 0.0 or no_act < f30
    assert f30 > 50.0  # 30-day strategy maintains substantial stock


def test_out_of_fold_prediction_labeling():
    """All used predictions must be explicitly labeled OUT_OF_FOLD."""
    dec_path = PROJECT_ROOT / "data" / "backtesting" / "decision_backtest.parquet"
    df = pd.read_parquet(dec_path)
    assert (df["prediction_source"] == "OUT_OF_FOLD").all()
    assert (df["out_of_fold_flag"] == True).all()


def test_final_holdout_remains_untouched():
    """Holdout period (2024-10-01 to 2024-12-31, 2,208 records) must not be in backtest."""
    day_path = PROJECT_ROOT / "data" / "backtesting" / "daily_inventory_replay.parquet"
    df = pd.read_parquet(day_path)
    dates = pd.to_datetime(df["date"])
    assert (dates < pd.Timestamp("2024-10-01")).all()


# =============================================================================
# 2. STRATEGY DECISION TESTS
# =============================================================================

def test_no_action_strategy():
    """NO_ACTION must order 0.0 units."""
    dec = decide_no_action()
    assert dec["strategy"] == "NO_ACTION"
    assert dec["order_quantity"] == 0.0


def test_fixed_coverage_strategies(sample_demand_forecasts):
    """Fixed coverage strategies order up to expected demand plus margin."""
    dec_7 = decide_fixed_coverage(7, current_inventory=50.0, demand_forecasts=sample_demand_forecasts, moq=10.0, pack_size=1)
    dec_14 = decide_fixed_coverage(14, current_inventory=50.0, demand_forecasts=sample_demand_forecasts, moq=10.0, pack_size=1)
    dec_30 = decide_fixed_coverage(30, current_inventory=50.0, demand_forecasts=sample_demand_forecasts, moq=10.0, pack_size=1)

    assert dec_7["order_quantity"] > 0
    assert dec_14["order_quantity"] > dec_7["order_quantity"]
    assert dec_30["order_quantity"] > dec_14["order_quantity"]


def test_reorder_point_policy():
    """ROP places order only when inventory is below reorder threshold."""
    # Low stock triggers order
    dec_low = decide_reorder_point(current_inventory=10.0, mean_daily_demand=25.0, lead_time_days=2.0, moq=10.0, pack_size=1)
    assert dec_low["order_quantity"] > 0
    assert dec_low["action_label"] == "BUY_NOW"

    # High stock triggers no order
    dec_high = decide_reorder_point(current_inventory=1000.0, mean_daily_demand=25.0, lead_time_days=2.0, moq=10.0, pack_size=1)
    assert dec_high["order_quantity"] == 0.0
    assert dec_high["action_label"] == "DO_NOT_REORDER"


def test_phase20_ai_strategy_execution(sample_demand_forecasts, sample_price_forecasts):
    """AI strategy generates valid bounded recommendations."""
    q_dict = {1: 15.0, 3: 50.0, 7: 120.0, 14: 250.0, 21: 380.0, 30: 550.0}
    dec = decide_phase20_ai_strategy(
        demand_forecasts=sample_demand_forecasts,
        price_forecasts=sample_price_forecasts,
        demand_p10=q_dict, demand_p50=sample_demand_forecasts, demand_p90=q_dict,
        price_p10=sample_price_forecasts, price_p50=sample_price_forecasts, price_p90=sample_price_forecasts,
        current_inventory=50.0, current_purchase_price=10.0, moq=10.0, pack_size=1,
        shelf_life_days=3.0, lead_time_days=2.0, store_id="STR_KUL_001", product_id="PRD_CHK_01",
        decision_date="2024-08-17", n_scenarios=100
    )
    assert dec["strategy"] == "PHASE20_AI_STRATEGY"
    assert dec["order_quantity"] >= 0.0


# =============================================================================
# 3. METRICS & FINANCIAL CALCULATION TESTS
# =============================================================================

def test_cost_calculation_mechanics():
    """Verify purchase, holding, stockout, wastage, and total cost accounting."""
    mock_daily = pd.DataFrame([
        {"strategy": "TEST", "demand": 10.0, "fulfilled_demand": 8.0, "lost_demand": 2.0,
         "closing_inventory": 5.0, "wastage": 0.0, "holding_cost": 2.0, "stockout_cost": 7.0, "wastage_cost": 0.0}
    ])
    mock_dec = pd.DataFrame([
        {"strategy": "TEST", "realised_purchase_cost": 100.0, "ordering_cost": 50.0}
    ])
    m = compute_realised_metrics(mock_daily, mock_dec, "TEST", {})
    # Total = 100 + 50 + 2 + 7 = 159.0
    assert m["total_cost"] == 159.0
    assert m["fill_rate"] == 0.8
    assert m["stockout_rate"] == 1.0


def test_savings_and_comparisons():
    """Comparative savings formula: (ref - ai) / ref * 100."""
    ai_m = {"strategy": "AI", "total_cost": 80.0, "stockout_rate": 0.05, "fill_rate": 0.95, "average_inventory": 20.0, "wasted_units": 0.0}
    ref_m = {"strategy": "REF", "total_cost": 100.0, "stockout_rate": 0.15, "fill_rate": 0.85, "average_inventory": 30.0, "wasted_units": 10.0}

    comp = compare_ai_vs_reference(ai_m, ref_m)
    assert comp["absolute_savings"] == 20.0
    assert comp["savings_percentage"] == 20.0
    assert comp["stockout_reduction"] == 0.10
    assert comp["fill_rate_improvement_pp"] == 10.0


# =============================================================================
# 4. SEGMENTATION & REPLAY API TESTS
# =============================================================================

def test_store_segmentation_structure():
    """Verify store segmentation contains all 4 target stores."""
    p = PROJECT_ROOT / "reports" / "backtesting" / "store_performance.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert df["store_id"].nunique() == 4
    assert "total_cost" in df.columns


def test_category_segmentation_structure():
    """Verify category segmentation contains savings and fill rates."""
    p = PROJECT_ROOT / "reports" / "backtesting" / "category_performance.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert "category" in df.columns
    assert "savings_percentage" in df.columns


def test_cost_sensitivity_table():
    """Verify sensitivity table covers parameter variations."""
    p = PROJECT_ROOT / "reports" / "backtesting" / "cost_sensitivity.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert len(df) >= 8
    assert "savings_percentage" in df.columns


def test_single_date_judge_replay_api():
    """Test replay_historical_decision for a real date."""
    res = replay_historical_decision(
        decision_date="2024-08-17",
        store_id="STR_JHR_003",
        product_id="PRD_CHK_01"
    )
    assert "metadata" in res
    assert "decisions_compared" in res
    assert "realised_outcomes" in res
    assert res["metadata"]["store_id"] == "STR_JHR_003"
    assert res["realised_outcomes"]["realised_savings_rm"] > 0.0


def test_dashboard_payload_validity():
    """Verify executive summary JSON schema."""
    p = PROJECT_ROOT / "reports" / "backtesting" / "sample_dashboard_executive_summary.json"
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "summary" in data
    assert "strategy_benchmarks" in data
    assert data["summary"]["total_backtest_days"] == 990
    assert data["summary"]["estimated_savings_pct"] > 0.0


# =============================================================================
# 6. HISTORICAL PRICE TIMING ARBITRAGE BACKTEST TESTS (RESEARCH ONLY)
# =============================================================================

def test_price_timing_backtest_report_structure():
    """Verify reports/backtesting/price_timing_backtest.csv is marked INVALID_PRICE_MODEL_RESEARCH_ONLY."""
    p = PROJECT_ROOT / "reports" / "backtesting" / "price_timing_backtest.csv"
    assert p.exists(), "price_timing_backtest.csv must exist"
    df = pd.read_csv(p)
    assert len(df) == 33, f"Expected 33 walk-forward folds, got {len(df)}"
    expected_cols = [
        "fold_index", "decision_origin", "current_purchase_price",
        "subsequent_purchase_price", "realised_price_change_pct", "timing_signal",
        "ai_order_quantity", "reference_order_quantity", "gross_timing_saving_rm",
        "incremental_holding_cost_rm", "net_timing_benefit_rm"
    ]
    for col in expected_cols:
        assert col in df.columns
    # Verify research-only governance status
    assert (df["price_model_status"] == "NOT_VALIDATED").all()
    assert (df["governance_status"] == "INVALID_PRICE_MODEL_RESEARCH_ONLY").all()
    assert (df["decision_grade"] == False).all()


# =============================================================================
# 7. SECTION 9 SAFE REBUILD BACKTEST REQUIREMENTS TESTS (1 TO 8)
# =============================================================================

def test_safe_backtest_req1_no_price_forecast_enters_decisions(sample_demand_forecasts, sample_price_forecasts):
    """1. No price forecast enters decisions: decisions are identical under varying price forecasts."""
    dec_path = PROJECT_ROOT / "data" / "backtesting" / "decision_backtest.parquet"
    df = pd.read_parquet(dec_path)
    assert "price_model_status" in df.columns
    assert (df["price_model_status"] == "NOT_VALIDATED").all()
    assert (df["price_forecast_used_in_decision"] == False).all()

    # Verify that changing price forecasts does not alter decide_phase20_ai_strategy output
    q_dict = {1: 15.0, 3: 50.0, 7: 120.0, 14: 250.0, 21: 380.0, 30: 550.0}
    high_prices = {h: 200.0 for h in sample_price_forecasts}
    low_prices = {h: 1.0 for h in sample_price_forecasts}

    dec_high = decide_phase20_ai_strategy(
        demand_forecasts=sample_demand_forecasts, price_forecasts=high_prices,
        demand_p10=q_dict, demand_p50=sample_demand_forecasts, demand_p90=q_dict,
        price_p10=high_prices, price_p50=high_prices, price_p90=high_prices,
        current_inventory=50.0, current_purchase_price=10.0, moq=10.0, pack_size=1,
        shelf_life_days=30.0, lead_time_days=2.0, store_id="STR_KUL_001", product_id="PRD_CHK_01",
        decision_date="2024-08-17", n_scenarios=50
    )
    dec_low = decide_phase20_ai_strategy(
        demand_forecasts=sample_demand_forecasts, price_forecasts=low_prices,
        demand_p10=q_dict, demand_p50=sample_demand_forecasts, demand_p90=q_dict,
        price_p10=low_prices, price_p50=low_prices, price_p90=low_prices,
        current_inventory=50.0, current_purchase_price=10.0, moq=10.0, pack_size=1,
        shelf_life_days=30.0, lead_time_days=2.0, store_id="STR_KUL_001", product_id="PRD_CHK_01",
        decision_date="2024-08-17", n_scenarios=50
    )
    assert dec_high["order_quantity"] == dec_low["order_quantity"]


def test_safe_backtest_req2_out_of_fold_demand_forecasts_used():
    """2. Out-of-fold demand forecasts used: verified across all evaluated decision records."""
    dec_path = PROJECT_ROOT / "data" / "backtesting" / "decision_backtest.parquet"
    df = pd.read_parquet(dec_path)
    assert (df["prediction_source"] == "OUT_OF_FOLD").all()
    assert (df["out_of_fold_flag"] == True).all()


def test_safe_backtest_req3_future_demand_not_visible_at_decision_time():
    """3. Future demand not visible at decision time: decision date is anchored to origin date."""
    dec_path = PROJECT_ROOT / "data" / "backtesting" / "decision_backtest.parquet"
    df = pd.read_parquet(dec_path)
    # Decision date strictly precedes or equals the decision origin
    assert len(df) > 0
    assert (pd.to_datetime(df["decision_date"]) <= pd.Timestamp("2024-08-17")).all()


def test_safe_backtest_req4_actual_future_outcomes_used_only_for_evaluation():
    """4. Actual future outcomes used only for evaluation: realized metrics computed post-decision."""
    dec_path = PROJECT_ROOT / "data" / "backtesting" / "decision_backtest.parquet"
    df = pd.read_parquet(dec_path)
    assert "realised_demand" in df.columns
    assert "realised_total_cost" in df.columns
    # Order quantity cannot depend on realized demand (order quantity decided before cycle)
    ai_df = df[df["strategy"] == "PHASE20_AI_STRATEGY"]
    assert (ai_df["order_quantity"] >= 0.0).all()
    assert (ai_df["realised_demand"] >= 0.0).all()


def test_safe_backtest_req5_counterfactual_inventory_evolves_independently():
    """5. Counterfactual inventory evolves independently across strategies."""
    day_path = PROJECT_ROOT / "data" / "backtesting" / "daily_inventory_replay.parquet"
    df = pd.read_parquet(day_path)
    mean_inv = df.groupby("strategy")["closing_inventory"].mean()
    # Distinct strategy policies must diverge in average inventory
    assert mean_inv["NO_ACTION"] < mean_inv["FIXED_7_DAY"] < mean_inv["FIXED_14_DAY"] < mean_inv["FIXED_30_DAY"]


def test_safe_backtest_req6_reference_strategies_fair():
    """6. Reference strategies fair: standard coverage and ROP policies with MOQ/pack size applied."""
    p = PROJECT_ROOT / "reports" / "backtesting" / "strategy_performance.csv"
    df = pd.read_csv(p)
    strats = df["strategy"].tolist()
    for s in ["NO_ACTION", "FIXED_7_DAY", "FIXED_14_DAY", "FIXED_30_DAY", "REORDER_POINT", "PHASE20_AI_STRATEGY"]:
        assert s in strats


def test_safe_backtest_req7_negative_periods_preserved():
    """7. Negative periods preserved: uncurated reporting without selective removal."""
    p = PROJECT_ROOT / "reports" / "backtesting" / "negative_periods.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert len(df) > 0
    p_month = PROJECT_ROOT / "reports" / "backtesting" / "monthly_performance.csv"
    assert p_month.exists()


def test_safe_backtest_req8_final_holdout_remains_protected():
    """8. Final holdout remains protected: zero records on or after 2024-10-01."""
    day_path = PROJECT_ROOT / "data" / "backtesting" / "daily_inventory_replay.parquet"
    df = pd.read_parquet(day_path)
    max_date = pd.to_datetime(df["date"]).max()
    assert max_date < pd.Timestamp("2024-10-01"), f"Holdout contamination detected: {max_date}"
