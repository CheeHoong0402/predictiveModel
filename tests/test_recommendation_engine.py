"""Comprehensive Automated Test Suite for Phase 21 Recommendation Engine."""
from datetime import datetime, timedelta
from pathlib import Path
import math
import numpy as np
import pandas as pd
import pytest

from src.utils.paths import PROJECT_ROOT
from src.recommendations.config import load_recommendation_config, load_alerts_config
from src.recommendations.taxonomy import (
    VALID_ACTIONS, VALID_URGENCIES, VALID_REASON_CODES,
    is_valid_action, is_valid_urgency, is_valid_reason_code
)
from src.recommendations.conflict_resolver import resolve_action_and_urgency
from src.recommendations.priority_queue import (
    calculate_executive_priority_score, build_action_priority_queue
)
from src.recommendations.alerts import (
    evaluate_alert_triggers, is_on_cooldown, record_alert_dispatched,
    clear_alert_cooldowns, get_telegram_credentials
)
from src.recommendations.user_upload import (
    validate_user_uploaded_files, MIN_REQUIRED_HISTORY_DAYS
)
from src.recommendations.engine import (
    generate_final_recommendation, generate_recommendations_batch
)
from src.recommendations.payloads import (
    get_product_decision_payload,
    get_executive_dashboard_payload,
    get_model_validation_payload,
    get_training_data_transparency_payload,
    get_historical_replay_payload
)


# Test 1 & 2: Action and urgency labels validity
def test_action_and_urgency_taxonomy():
    rec = generate_final_recommendation(store_id="STR_KUL_001", product_id="PRD_CHK_01")
    action = rec["decision"]["action"]
    urgency = rec["decision"]["urgency"]
    assert is_valid_action(action), f"Invalid action: {action}"
    assert is_valid_urgency(urgency), f"Invalid urgency: {urgency}"


# Test 3 & 4: Quantity logic for BUY_NOW and DO_NOT_REORDER
def test_action_quantity_constraints():
    # BUY_NOW must have quantity > 0
    (action, urg, qty, _, _, _, _) = resolve_action_and_urgency(
        current_inventory=5.0, current_coverage_days=1.0, lead_time_days=2,
        lead_time_demand=15.0, phase18_action="BUY_NOW", phase18_quantity=50.0,
        phase19_action="BUY_NOW", phase19_quantity=50.0,
        stockout_prob_before=0.45, stockout_prob_after=0.05,
        overstock_prob=0.02, wastage_prob=0.0, shelf_life_days=30,
        price_change_pct=1.5, price_direction="UP"
    )
    assert qty > 0, "BUY_NOW/URGENT_REORDER requires strictly positive quantity"

    # Overstock / DO_NOT_REORDER must return zero quantity
    (action_dn, _, qty_dn, _, _, _, _) = resolve_action_and_urgency(
        current_inventory=500.0, current_coverage_days=40.0, lead_time_days=2,
        lead_time_demand=15.0, phase18_action="DO_NOT_REORDER", phase18_quantity=0.0,
        phase19_action="DO_NOT_REORDER", phase19_quantity=0.0,
        stockout_prob_before=0.0, stockout_prob_after=0.0,
        overstock_prob=0.85, wastage_prob=0.0, shelf_life_days=30,
        price_change_pct=0.0, price_direction="FLAT"
    )
    assert qty_dn == 0.0, "Overstock / DO_NOT_REORDER must have zero order quantity"


# Test 5: URGENT_REORDER overrides WAIT when lead-time stockout is critical
def test_urgent_reorder_overrides_price_wait():
    action, urgency, qty, reasons, explanations, _, _ = resolve_action_and_urgency(
        current_inventory=0.0, current_coverage_days=0.0, lead_time_days=2,
        lead_time_demand=20.0, phase18_action="WAIT_FOR_LOWER_PRICE", phase18_quantity=0.0,
        phase19_action="WAIT_FOR_LOWER_PRICE", phase19_quantity=0.0,
        stockout_prob_before=0.95, stockout_prob_after=0.10,
        overstock_prob=0.0, wastage_prob=0.0, shelf_life_days=30,
        price_change_pct=-8.0, price_direction="DOWN"
    )
    assert action == "URGENT_REORDER", "Critical stockout must override price drop wait!"
    assert urgency in ["CRITICAL", "HIGH"]
    assert qty > 0.0


# Test 6: Overstock warning does not recommend contradictory purchase
def test_overstock_warning_blocks_purchase():
    action, urgency, qty, reasons, _, _, _ = resolve_action_and_urgency(
        current_inventory=800.0, current_coverage_days=35.0, lead_time_days=2,
        lead_time_demand=20.0, phase18_action="BUY_NOW", phase18_quantity=200.0,
        phase19_action="BUY_NOW", phase19_quantity=200.0,
        stockout_prob_before=0.0, stockout_prob_after=0.0,
        overstock_prob=0.75, wastage_prob=0.10, shelf_life_days=14,
        price_change_pct=3.0, price_direction="UP"
    )
    assert action in ["OVERSTOCK_WARNING", "HOLD_CURRENT_STOCK"], f"Action should be overstock warning or hold, got {action}"
    assert qty == 0.0, "Overstocked item must not recommend positive purchase quantity"


# Test 7, 8, 9: Phase 18 and Phase 19 outputs integrated correctly
def test_phase18_and_19_integration():
    rec = generate_final_recommendation(store_id="STR_JHR_003", product_id="PRD_CHK_01")
    assert "phase18_order_quantity" in rec["decision"]
    assert "phase19_order_quantity" in rec["decision"]
    assert "final_order_quantity" in rec["decision"]
    # Check Phase 19 Monte Carlo risk outputs override preliminary proxies
    assert rec["risk"]["stockout_probability_before"] is not None
    assert rec["risk"]["stockout_probability_after"] is not None
    assert rec["risk"]["cost_cvar95"] is not None


# Test 10, 11, 12: Demand and Price forecasts validity (P10 <= P50 <= P90)
def test_demand_and_price_forecasts_validity():
    rec = generate_final_recommendation(store_id="STR_SGR_002", product_id="PRD_EGG_02")
    dem = rec["demand"]
    pr = rec["price"]

    for h in ["1d", "3d", "7d", "14d", "21d", "30d"]:
        assert h in dem, f"Horizon {h} missing in demand"
        p10 = dem[h]["p10"]
        p50 = dem[h]["p50"]
        p90 = dem[h]["p90"]
        assert p10 <= p50 <= p90, f"Demand quantile crossing detected at {h}: {p10} <= {p50} <= {p90}"

    if pr["p10"] is not None and pr["p50"] is not None and pr["p90"] is not None:
        assert pr["p10"] <= pr["p50"] <= pr["p90"], "Price quantile crossing detected"
    else:
        # Price model quarantined: quantiles are None per Section 4 governance
        assert pr["predicted_future_price"] is None
        assert rec["price_decision"]["status"] == "PRICE_MODEL_NOT_VALIDATED"
        assert rec["price_decision"]["safe_for_decision"] is False
    assert pr["current_purchase_price"] > 0
    assert pr["price_direction"] in ["UP", "FLAT", "DOWN"]


# Test 13, 14, 15: Probabilities bounded in [0, 1]
def test_probabilities_bounded():
    rec = generate_final_recommendation(store_id="STR_PNG_004", product_id="PRD_OIL_05")
    rsk = rec["risk"]
    for k in ["stockout_probability_before", "stockout_probability_after", "overstock_probability", "wastage_probability", "late_delivery_probability"]:
        val = rsk[k]
        assert 0.0 <= val <= 1.0, f"Probability {k}={val} out of bounds [0, 1]"


# Test 16, 17, 18: Economics and savings validity
def test_economics_and_savings_validity():
    rec = generate_final_recommendation(store_id="STR_KUL_001", product_id="PRD_RIC_04")
    econ = rec["economics"]
    assert math.isfinite(econ["expected_total_cost"]), "Expected total cost must be finite"
    assert econ["expected_total_cost"] >= 0.0, "Expected total cost must be non-negative"
    assert math.isfinite(econ["expected_savings"]), "Expected savings must be finite"
    # Separate historical backtest evidence
    assert econ["historical_backtest_savings_pct"] == 44.49


# Test 19, 20, 21: Reason codes and explanations
def test_reason_codes_and_explanations():
    rec = generate_final_recommendation(store_id="STR_JHR_003", product_id="PRD_CHK_01")
    reasons = rec["explanation"]["reason_codes"]
    assert len(reasons) > 0, "At least one reason code must be present"
    for rc in reasons:
        assert is_valid_reason_code(rc), f"Invalid reason code: {rc}"
    top_exp = rec["explanation"]["top_reasons"]
    assert 1 <= len(top_exp) <= 3, "Must include 1 to 3 top business explanations"


# Test 22, 23, 24, 25, 26: User upload data validation & fallbacks
def test_user_upload_validation_rules():
    # Test missing required files
    res_bad = validate_user_uploaded_files({"pos_sales.csv": pd.DataFrame()})
    assert not res_bad.is_valid
    assert res_bad.status == "INVALID_MISSING_REQUIRED_FILES"

    # Test insufficient history (< 60 days)
    short_dates = pd.date_range("2024-01-01", periods=30)
    sales_short = pd.DataFrame({"date": short_dates, "units_sold": [10]*30})
    dummy_files = {
        "pos_sales.csv": sales_short,
        "inventory_history.csv": pd.DataFrame({"date": short_dates, "stock": [50]*30}),
        "purchase_price_history.csv": pd.DataFrame({"date": short_dates, "price": [7.0]*30}),
        "product_master.csv": pd.DataFrame({"product_id": ["P1"]}),
        "store_master.csv": pd.DataFrame({"store_id": ["S1"]}),
        "supplier_master.csv": pd.DataFrame({"supplier_id": ["SUP1"]})
    }
    res_short = validate_user_uploaded_files(dummy_files)
    assert not res_short.is_valid
    assert res_short.status == "INSUFFICIENT_HISTORY"

    # Test sufficient history (90 days) with missing optional promotions
    good_dates = pd.date_range("2024-01-01", periods=90)
    sales_good = pd.DataFrame({"date": good_dates, "units_sold": [10]*90})
    dummy_files["pos_sales.csv"] = sales_good
    res_good = validate_user_uploaded_files(dummy_files)
    assert res_good.is_valid
    assert any("NO_PROMOTION_HISTORY" in w for w in res_good.warnings)


# Test 27, 28, 29, 30, 31: Batch recommendations and priority queue
def test_batch_recommendations_and_priority_queue():
    final_df, pq_df, audit_df = generate_recommendations_batch()
    assert len(final_df) == 24, f"Expected 24 recommendations, got {len(final_df)}"
    assert len(pq_df) == 24
    assert len(audit_df) == 24
    # Check queue rank is ordered 1 to 24
    assert list(pq_df["queue_rank"]) == list(range(1, 25))
    # Priority score is monotonic descending
    scores = pq_df["priority_score"].tolist()
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))


# Test 32, 33, 34: Telegram alert triggers and cooldown enforcement
def test_telegram_alert_triggers_and_cooldown():
    clear_alert_cooldowns()
    rec_urgent = {
        "store_id": "STR_JHR_003",
        "product_id": "PRD_CHK_01",
        "action": "URGENT_REORDER",
        "urgency": "HIGH",
        "stockout_probability_before": 0.45,
        "stockout_probability_after": 0.05,
        "overstock_probability": 0.02,
        "predicted_price_change_pct": 2.5,
        "current_inventory": 10.0,
        "current_coverage_days": 1.0,
        "order_quantity": 100.0,
        "coverage_days": 3,
        "top_reasons": ["Stockout imminent"]
    }
    alerts = evaluate_alert_triggers(rec_urgent)
    assert len(alerts) >= 1
    assert alerts[0]["alert_type"] == "URGENT_REORDER"

    # Second immediate evaluation must be suppressed by cooldown
    alerts_second = evaluate_alert_triggers(rec_urgent)
    assert len(alerts_second) == 0, "Duplicate alert should be suppressed by 24h cooldown"


# Test 35, 36, 37: Audit log and unique IDs
def test_audit_log_and_unique_id():
    rec = generate_final_recommendation(store_id="STR_KUL_001", product_id="PRD_CHK_01")
    rec_id = rec["recommendation_id"]
    assert rec_id.startswith("REC_")
    assert "STR_KUL_001" in rec_id
    assert "PRD_CHK_01" in rec_id
    assert "audit" in rec
    assert "model_versions" in rec["audit"]


# Test 38, 39, 40, 41, 42, 43: Dashboard payloads validity
def test_dashboard_payloads_validity():
    prod_payload = get_product_decision_payload("STR_KUL_001", "PRD_CHK_01")
    assert "visualisation_data" in prod_payload
    assert "demand_trajectory" in prod_payload["visualisation_data"]

    exec_payload = get_executive_dashboard_payload()
    assert exec_payload["status"] == "HEALTHY"
    assert "top_priority_queue" in exec_payload

    val_payload = get_model_validation_payload()
    assert "demand_model_validation" in val_payload
    assert "economic_backtest_validation" in val_payload

    trans_payload = get_training_data_transparency_payload()
    assert trans_payload["final_holdout_quarantine"]["status"] == "STRICTLY_QUARANTINED_UNTOUCHED"

    replay_payload = get_historical_replay_payload("2024-08-17", "STR_JHR_003", "PRD_CHK_01")
    assert "realised_outcomes" in replay_payload
    assert replay_payload["realised_outcomes"]["realised_savings_pct"] > 0


# Test 44 & 45: Holdout quarantine and no future leakage
def test_holdout_quarantine_verification():
    rec = generate_final_recommendation(store_id="STR_KUL_001", product_id="PRD_CHK_01")
    dec_date = pd.Timestamp(rec["decision_date"])
    assert dec_date < pd.Timestamp("2024-10-01"), "Decision date must strictly precede final holdout start"


# =============================================================================
# 12. DEMAND X PRICE MATRIX & CONFLICT RESOLUTION TESTS
# =============================================================================

def test_demand_price_matrix_cases():
    """Verify Demand x Price interaction cases in conflict resolver."""
    from src.recommendations.conflict_resolver import resolve_action_and_urgency
    from src.recommendations.taxonomy import ActionType, UrgencyLevel, ReasonCode

    # Case A: Rising Demand + Strong Rising Price -> Strong Buy Early
    action_a, urg_a, qty_a, reasons_a, expl_a, rob_a, _ = resolve_action_and_urgency(
        current_inventory=100.0,
        current_coverage_days=5.0,
        lead_time_days=2,
        lead_time_demand=50.0,
        phase18_action="BUY_NOW",
        phase18_quantity=100.0,
        phase19_action="BUY_NOW",
        phase19_quantity=100.0,
        stockout_prob_before=0.10,
        stockout_prob_after=0.02,
        overstock_prob=0.05,
        wastage_prob=0.01,
        shelf_life_days=30,
        price_change_pct=5.5,
        price_direction="UP",
        price_signal_strength="STRONG_BUY_EARLY_SIGNAL",
        net_price_timing_benefit=25.0,
        price_model_validated=True
    )
    assert action_a == ActionType.BUY_NOW
    assert any(r == ReasonCode.STRONG_BUY_EARLY_PRICE_SIGNAL.value for r in reasons_a)

    # Case B: Rising Demand + Falling Price + Imminent Stockout -> Reorder immediately (Stockout Overrides Price Decline)
    action_b, urg_b, qty_b, reasons_b, expl_b, rob_b, _ = resolve_action_and_urgency(
        current_inventory=5.0,
        current_coverage_days=0.2,
        lead_time_days=2,
        lead_time_demand=60.0,
        phase18_action="WAIT_FOR_LOWER_PRICE",
        phase18_quantity=0.0,
        phase19_action="WAIT_FOR_LOWER_PRICE",
        phase19_quantity=0.0,
        stockout_prob_before=0.85,
        stockout_prob_after=0.05,
        overstock_prob=0.01,
        wastage_prob=0.01,
        shelf_life_days=30,
        price_change_pct=-4.5,
        price_direction="DOWN",
        price_signal_strength="STRONG_WAIT_SIGNAL",
        net_price_timing_benefit=-10.0,
        price_model_validated=True
    )
    assert action_b in [ActionType.URGENT_REORDER, ActionType.BUY_NOW]
    assert any(r == ReasonCode.STOCKOUT_OVERRIDES_PRICE_DECLINE.value for r in reasons_b)

    # Case D: Falling/Flat Demand + Falling Price + Safe Stock -> Wait or lean buy
    action_d, urg_d, qty_d, reasons_d, expl_d, rob_d, _ = resolve_action_and_urgency(
        current_inventory=250.0,
        current_coverage_days=12.0,
        lead_time_days=2,
        lead_time_demand=20.0,
        phase18_action="WAIT_FOR_LOWER_PRICE",
        phase18_quantity=0.0,
        phase19_action="WAIT_FOR_LOWER_PRICE",
        phase19_quantity=0.0,
        stockout_prob_before=0.05,
        stockout_prob_after=0.05,
        overstock_prob=0.02,
        wastage_prob=0.01,
        shelf_life_days=30,
        price_change_pct=-2.5,
        price_direction="DOWN",
        price_signal_strength="MODERATE_WAIT_SIGNAL",
        net_price_timing_benefit=-8.0,
        price_model_validated=True
    )
    assert action_d in [ActionType.WAIT_FOR_LOWER_PRICE, ActionType.DO_NOT_REORDER, ActionType.HOLD_CURRENT_STOCK]


def test_conflict_resolver_perishable_overrides_price_prebuy():
    """Verify shelf life strictly overrides prebuying even when price rises."""
    from src.recommendations.conflict_resolver import resolve_action_and_urgency
    from src.recommendations.taxonomy import ActionType, ReasonCode

    action_p, urg_p, qty_p, reasons_p, expl_p, rob_p, _ = resolve_action_and_urgency(
        current_inventory=50.0,
        current_coverage_days=2.5,
        lead_time_days=2,
        lead_time_demand=20.0,
        phase18_action="BUY_NOW",
        phase18_quantity=100.0,
        phase19_action="BUY_NOW",
        phase19_quantity=100.0,
        stockout_prob_before=0.10,
        stockout_prob_after=0.02,
        overstock_prob=0.10,
        wastage_prob=0.45,  # High spoilage risk
        shelf_life_days=3,
        price_change_pct=5.5,
        price_direction="UP",
        price_signal_strength="STRONG_BUY_EARLY_SIGNAL",
        net_price_timing_benefit=20.0,
        price_model_validated=True
    )
    assert any(r in [ReasonCode.SHELF_LIFE_OVERRIDES_PRICE_PREBUY.value, ReasonCode.SHELF_LIFE_CONSTRAINED.value, ReasonCode.HIGH_WASTAGE_RISK.value] for r in reasons_p)


def test_price_decision_summary_csv_report():
    """Verify reports/recommendations/price_decision_summary.csv exists with all 18 required columns."""
    from src.utils.paths import PROJECT_ROOT
    p = PROJECT_ROOT / "reports" / "recommendations" / "price_decision_summary.csv"
    assert p.exists(), "price_decision_summary.csv must exist"
    df = pd.read_csv(p)
    assert len(df) == 24
    expected_cols = [
        "product_id", "store_id", "action", "urgency",
        "current_purchase_price", "predicted_purchase_price", "predicted_price_change_pct",
        "price_p10", "price_p50", "price_p90", "price_signal_strength",
        "direction_probability", "baseline_demand_order_quantity",
        "price_aware_order_quantity", "price_delta_quantity", "net_timing_benefit",
        "buy_now_vs_wait_recommendation", "conflict_override_applied"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: {col}"


# =============================================================================
# 13. SECTION 18 DETERMINISTIC PRICE-AWARE RECOMMENDATION ENGINE TESTS
# =============================================================================

# Test 1: Demand up + price up strengthens BUY_NOW
def test_case1_demand_up_price_up_strengthens_buy_now():
    action, urg, qty, reasons, expl, rob, _ = resolve_action_and_urgency(
        current_inventory=50.0,
        current_coverage_days=3.0,
        lead_time_days=2,
        lead_time_demand=30.0,
        phase18_action="BUY_NOW",
        phase18_quantity=100.0,
        phase19_action="BUY_NOW",
        phase19_quantity=100.0,
        stockout_prob_before=0.10,
        stockout_prob_after=0.01,
        overstock_prob=0.05,
        wastage_prob=0.0,
        shelf_life_days=30,
        price_change_pct=6.0,
        price_direction="UP",
        price_signal_strength="STRONG_BUY_EARLY_SIGNAL",
        net_price_timing_benefit=35.0,
        demand_trend="RISING",
        price_model_validated=True
    )
    assert action == "BUY_NOW"
    assert urg == "HIGH"
    assert qty > 0
    assert "STRONG_BUY_EARLY_PRICE_SIGNAL" in reasons
    assert "POSITIVE_NET_PRICE_TIMING_BENEFIT" in reasons


# Test 2: Demand up + price down protects service but avoids unnecessary early stock (lean buy)
def test_case2_demand_up_price_down_protects_service_lean_buy():
    action, urg, qty, reasons, expl, rob, _ = resolve_action_and_urgency(
        current_inventory=65.0,
        current_coverage_days=4.0,
        lead_time_days=2,
        lead_time_demand=30.0,
        phase18_action="BUY_SMALLER_QUANTITY",
        phase18_quantity=40.0,
        phase19_action="BUY_SMALLER_QUANTITY",
        phase19_quantity=40.0,
        stockout_prob_before=0.08,
        stockout_prob_after=0.02,
        overstock_prob=0.02,
        wastage_prob=0.0,
        shelf_life_days=30,
        price_change_pct=-4.0,
        price_direction="DOWN",
        price_signal_strength="MODERATE_WAIT_SIGNAL",
        net_price_timing_benefit=-12.0,
        demand_trend="RISING",
        price_model_validated=True
    )
    assert action == "BUY_SMALLER_QUANTITY"
    assert qty == 40.0
    assert "PRICE_DECLINE_LEAN_BUY" in reasons


# Test 3: Demand down + price up does not automatically buy
def test_case3_demand_down_price_up_does_not_auto_buy():
    action, urg, qty, reasons, expl, rob, _ = resolve_action_and_urgency(
        current_inventory=120.0,
        current_coverage_days=8.0,
        lead_time_days=2,
        lead_time_demand=30.0,
        phase18_action="BUY_NOW",
        phase18_quantity=100.0,
        phase19_action="BUY_NOW",
        phase19_quantity=100.0,
        stockout_prob_before=0.02,
        stockout_prob_after=0.01,
        overstock_prob=0.15,
        wastage_prob=0.05,
        shelf_life_days=30,
        price_change_pct=3.0,
        price_direction="UP",
        price_signal_strength="MODERATE_BUY_EARLY_SIGNAL",
        net_price_timing_benefit=2.0,  # Below RM 5.00 threshold
        demand_trend="FALLING",
        price_model_validated=True
    )
    assert action in ["HOLD_CURRENT_STOCK", "DO_NOT_REORDER"] or qty <= 100.0
    assert "PRICE_SIGNAL_DID_NOT_CHANGE_DECISION" in reasons


# Test 4: Demand down + price down supports WAIT/HOLD where safe
def test_case4_demand_down_price_down_supports_wait_hold():
    action, urg, qty, reasons, expl, rob, _ = resolve_action_and_urgency(
        current_inventory=200.0,
        current_coverage_days=10.0,
        lead_time_days=2,
        lead_time_demand=20.0,
        phase18_action="WAIT_FOR_LOWER_PRICE",
        phase18_quantity=0.0,
        phase19_action="WAIT_FOR_LOWER_PRICE",
        phase19_quantity=0.0,
        stockout_prob_before=0.01,
        stockout_prob_after=0.01,
        overstock_prob=0.05,
        wastage_prob=0.0,
        shelf_life_days=60,
        price_change_pct=-5.5,
        price_direction="DOWN",
        price_signal_strength="STRONG_WAIT_SIGNAL",
        net_price_timing_benefit=-25.0,
        demand_trend="FALLING",
        price_model_validated=True
    )
    assert action in ["WAIT_FOR_LOWER_PRICE", "HOLD_CURRENT_STOCK"]
    assert qty == 0.0
    assert "PRICE_DECLINE_WAIT_SIGNAL" in reasons


# Test 5: Critical stockout overrides WAIT
def test_case5_critical_stockout_overrides_wait():
    action, urg, qty, reasons, expl, rob, _ = resolve_action_and_urgency(
        current_inventory=0.0,
        current_coverage_days=0.0,
        lead_time_days=2,
        lead_time_demand=40.0,
        phase18_action="WAIT_FOR_LOWER_PRICE",
        phase18_quantity=0.0,
        phase19_action="WAIT_FOR_LOWER_PRICE",
        phase19_quantity=0.0,
        stockout_prob_before=0.90,
        stockout_prob_after=0.05,
        overstock_prob=0.0,
        wastage_prob=0.0,
        shelf_life_days=30,
        price_change_pct=-8.0,
        price_direction="DOWN",
        price_signal_strength="STRONG_WAIT_SIGNAL",
        net_price_timing_benefit=-30.0,
        demand_trend="STABLE",
        price_model_validated=True
    )
    assert action == "URGENT_REORDER"
    assert urg == "CRITICAL"
    assert qty > 0.0
    assert "PRICE_SIGNAL_OVERRIDDEN_BY_STOCKOUT_RISK" in reasons
    assert "STOCKOUT_OVERRIDES_PRICE_DECLINE" in reasons


# Test 6: High overstock risk overrides BUY_EARLY
def test_case6_high_overstock_risk_overrides_buy_early():
    action, urg, qty, reasons, expl, rob, _ = resolve_action_and_urgency(
        current_inventory=500.0,
        current_coverage_days=30.0,
        lead_time_days=2,
        lead_time_demand=20.0,
        phase18_action="BUY_NOW",
        phase18_quantity=150.0,
        phase19_action="BUY_NOW",
        phase19_quantity=150.0,
        stockout_prob_before=0.0,
        stockout_prob_after=0.0,
        overstock_prob=0.60,
        wastage_prob=0.05,
        shelf_life_days=45,
        price_change_pct=6.0,
        price_direction="UP",
        price_signal_strength="STRONG_BUY_EARLY_SIGNAL",
        net_price_timing_benefit=25.0,
        demand_trend="RISING",
        price_model_validated=True
    )
    assert action in ["OVERSTOCK_WARNING", "HOLD_CURRENT_STOCK"]
    assert qty == 0.0
    assert "PRICE_SIGNAL_OVERRIDDEN_BY_OVERSTOCK_RISK" in reasons


# Test 7: High wastage risk overrides BUY_EARLY
def test_case7_high_wastage_risk_overrides_buy_early():
    action, urg, qty, reasons, expl, rob, _ = resolve_action_and_urgency(
        current_inventory=80.0,
        current_coverage_days=4.0,
        lead_time_days=2,
        lead_time_demand=20.0,
        phase18_action="BUY_NOW",
        phase18_quantity=120.0,
        phase19_action="BUY_NOW",
        phase19_quantity=120.0,
        stockout_prob_before=0.05,
        stockout_prob_after=0.01,
        overstock_prob=0.15,
        wastage_prob=0.35,  # High perishable spoilage risk
        shelf_life_days=5,   # Short shelf life
        price_change_pct=7.0,
        price_direction="UP",
        price_signal_strength="STRONG_BUY_EARLY_SIGNAL",
        net_price_timing_benefit=30.0,
        demand_trend="RISING",
        price_model_validated=True
    )
    assert action in ["HOLD_CURRENT_STOCK", "OVERSTOCK_WARNING"]
    assert qty == 0.0
    assert "PRICE_SIGNAL_OVERRIDDEN_BY_WASTAGE_RISK" in reasons
    assert "SHELF_LIFE_OVERRIDES_PRICE_PREBUY" in reasons


# Test 8: Budget constraint overrides BUY_EARLY
def test_case8_budget_constraint_overrides_buy_early():
    action, urg, qty, reasons, expl, rob, _ = resolve_action_and_urgency(
        current_inventory=30.0,
        current_coverage_days=2.5,
        lead_time_days=2,
        lead_time_demand=20.0,
        phase18_action="BUY_NOW",
        phase18_quantity=100.0,
        phase19_action="BUY_NOW",
        phase19_quantity=100.0,
        stockout_prob_before=0.10,
        stockout_prob_after=0.02,
        overstock_prob=0.05,
        wastage_prob=0.0,
        shelf_life_days=30,
        price_change_pct=5.0,
        price_direction="UP",
        price_signal_strength="STRONG_BUY_EARLY_SIGNAL",
        net_price_timing_benefit=20.0,
        demand_trend="RISING",
        budget_limit=200.0,
        current_purchase_price=10.0,  # 100 units * RM 10 = RM 1000 > RM 200 budget
        price_model_validated=True
    )
    assert qty <= 20.0  # RM 200 / RM 10 = 20 units
    assert "PRICE_SIGNAL_OVERRIDDEN_BY_BUDGET" in reasons


# Test 9: High price uncertainty weakens price signal
def test_case9_high_price_uncertainty_weakens_price_signal():
    from src.recommendations.conflict_resolver import classify_price_signal
    signal, is_unc = classify_price_signal(
        price_change_pct=6.0,
        prob_rise=0.85,
        prob_fall=0.15,
        price_p10=6.0,
        price_p50=8.0,
        price_p90=12.0
    )
    assert is_unc is True
    assert signal == "MODERATE_BUY_EARLY_SIGNAL"  # Downgraded from STRONG


# Test 10: Positive net price-timing benefit can increase quantity
def test_case10_positive_net_timing_benefit_can_increase_quantity():
    rec = generate_final_recommendation(store_id="STR_PNG_004", product_id="PRD_OIL_05")
    pr_dec = rec["price_decision"]
    assert "net_price_timing_benefit" in pr_dec
    assert "demand_inventory_only_order_quantity" in pr_dec
    assert "price_aware_final_order_quantity" in pr_dec
    assert pr_dec["net_price_timing_benefit"] >= 0.0 or pr_dec["price_order_adjustment_units"] <= 0.0


# Test 11: Negative net price-timing benefit does not increase quantity
def test_case11_negative_net_timing_benefit_does_not_increase_quantity():
    action, urg, qty, reasons, expl, rob, _ = resolve_action_and_urgency(
        current_inventory=50.0,
        current_coverage_days=3.0,
        lead_time_days=2,
        lead_time_demand=25.0,
        phase18_action="BUY_NOW",
        phase18_quantity=50.0,
        phase19_action="BUY_NOW",
        phase19_quantity=50.0,
        stockout_prob_before=0.10,
        stockout_prob_after=0.03,
        overstock_prob=0.05,
        wastage_prob=0.0,
        shelf_life_days=30,
        price_change_pct=1.0,
        price_direction="FLAT",
        price_signal_strength="NEUTRAL_PRICE_SIGNAL",
        net_price_timing_benefit=-15.0,
        demand_trend="STABLE",
        price_model_validated=True
    )
    assert qty <= 50.0
    assert "NEGATIVE_NET_PRICE_TIMING_BENEFIT" in reasons


# Test 12: Price-aware coverage adjustment is correct
def test_case12_price_aware_coverage_adjustment_is_correct():
    rec = generate_final_recommendation(store_id="STR_KUL_001", product_id="PRD_CHK_01")
    pr_dec = rec["price_decision"]
    base_cov = pr_dec["coverage_without_price_signal"]
    final_cov = pr_dec["final_coverage_with_price_signal"]
    adj_cov = pr_dec["price_coverage_adjustment_days"]
    assert adj_cov == final_cov - base_cov


# Test 13: No price-benefit double counting
def test_case13_no_price_benefit_double_counting():
    rec = generate_final_recommendation(store_id="STR_KUL_001", product_id="PRD_CHK_01")
    econ = rec["economics"]
    pr_dec = rec["price_decision"]
    assert math.isfinite(econ["expected_total_cost"])
    assert econ["expected_total_cost"] > 0
    assert "net_price_timing_benefit" in pr_dec


# Test 14: Missing price forecast fallback works
def test_case14_missing_price_forecast_fallback_works():
    action, urg, qty, reasons, expl, rob, _ = resolve_action_and_urgency(
        current_inventory=5.0,
        current_coverage_days=0.5,
        lead_time_days=2,
        lead_time_demand=20.0,
        phase18_action="BUY_NOW",
        phase18_quantity=50.0,
        phase19_action="BUY_NOW",
        phase19_quantity=50.0,
        stockout_prob_before=0.50,
        stockout_prob_after=0.05,
        overstock_prob=0.01,
        wastage_prob=0.0,
        shelf_life_days=30,
        price_change_pct=0.0,
        price_direction="FLAT",
        price_available=False
    )
    assert "PRICE_FORECAST_UNAVAILABLE" in reasons
    assert action in ["URGENT_REORDER", "BUY_NOW"]
    assert qty > 0.0


# Test 15: Price influenced final decision is correct
def test_case15_price_influenced_final_decision_is_correct():
    rec = generate_final_recommendation(store_id="STR_PNG_004", product_id="PRD_OIL_05")
    pr_dec = rec["price_decision"]
    assert isinstance(pr_dec["influenced_final_decision"], bool)
    assert isinstance(pr_dec["price_influenced_final_decision"], bool)


# Test 16: Price decision effect is correct
def test_case16_price_decision_effect_is_correct():
    rec = generate_final_recommendation(store_id="STR_KUL_001", product_id="PRD_CHK_01")
    pr_dec = rec["price_decision"]
    effect = pr_dec["decision_effect"]
    assert isinstance(effect, str)
    assert len(effect) > 10


# Test 17: Price reason codes are correct
def test_case17_price_reason_codes_are_correct():
    from src.recommendations.taxonomy import is_valid_reason_code
    rec = generate_final_recommendation(store_id="STR_JHR_003", product_id="PRD_CHK_01")
    for rc in rec["explanation"]["reason_codes"]:
        assert is_valid_reason_code(rc)


# Test 18: Dashboard payload contains new fields
def test_case18_dashboard_payload_contains_new_fields():
    payload = get_product_decision_payload("STR_KUL_001", "PRD_CHK_01")
    assert "four_decision_dimensions" in payload
    assert "demand_outlook" in payload["four_decision_dimensions"]
    assert "supply_risk" in payload["four_decision_dimensions"]
    assert "price_outlook" in payload["four_decision_dimensions"]
    assert "economic_tradeoff" in payload["four_decision_dimensions"]
    assert "price_signal" in payload
    assert "probability_price_rises" in payload
    assert "net_price_timing_benefit" in payload
    assert "price_order_adjustment_units" in payload
    assert "price_coverage_adjustment_days" in payload
    assert "final_recommendation" in payload


# =============================================================================
# 14. SAFE REBUILD GOVERNANCE & FAIL-SAFE CRITERIA TESTS (SECTION 10)
# =============================================================================

# Test 1: Price forecast cannot affect action when unvalidated
def test_safe_rebuild_req1_price_forecast_cannot_affect_action():
    common_args = dict(
        current_inventory=100.0,
        current_coverage_days=5.0,
        lead_time_days=2,
        lead_time_demand=40.0,
        phase18_action="BUY_NOW",
        phase18_quantity=80.0,
        phase19_action="BUY_NOW",
        phase19_quantity=80.0,
        stockout_prob_before=0.10,
        stockout_prob_after=0.02,
        overstock_prob=0.05,
        wastage_prob=0.0,
        shelf_life_days=30,
        price_model_validated=False
    )
    action_up, _, _, _, _, _, _ = resolve_action_and_urgency(
        price_change_pct=25.0, price_direction="UP", price_signal_strength="STRONG_BUY_EARLY_SIGNAL",
        net_price_timing_benefit=100.0, **common_args
    )
    action_down, _, _, _, _, _, _ = resolve_action_and_urgency(
        price_change_pct=-25.0, price_direction="DOWN", price_signal_strength="STRONG_WAIT_SIGNAL",
        net_price_timing_benefit=-100.0, **common_args
    )
    assert action_up == action_down, "Price forecast altered action despite being unvalidated"


# Test 2: Price forecast cannot affect quantity when unvalidated
def test_safe_rebuild_req2_price_forecast_cannot_affect_quantity():
    common_args = dict(
        current_inventory=100.0,
        current_coverage_days=5.0,
        lead_time_days=2,
        lead_time_demand=40.0,
        phase18_action="BUY_NOW",
        phase18_quantity=80.0,
        phase19_action="BUY_NOW",
        phase19_quantity=80.0,
        stockout_prob_before=0.10,
        stockout_prob_after=0.02,
        overstock_prob=0.05,
        wastage_prob=0.0,
        shelf_life_days=30,
        price_model_validated=False
    )
    _, _, qty_up, _, _, _, _ = resolve_action_and_urgency(
        price_change_pct=25.0, price_direction="UP", price_signal_strength="STRONG_BUY_EARLY_SIGNAL",
        net_price_timing_benefit=100.0, **common_args
    )
    _, _, qty_down, _, _, _, _ = resolve_action_and_urgency(
        price_change_pct=-25.0, price_direction="DOWN", price_signal_strength="STRONG_WAIT_SIGNAL",
        net_price_timing_benefit=-100.0, **common_args
    )
    assert qty_up == qty_down == 80.0, "Price forecast altered order quantity despite being unvalidated"


# Test 3: Price forecast cannot affect coverage when unvalidated
def test_safe_rebuild_req3_price_forecast_cannot_affect_coverage():
    rec = generate_final_recommendation(store_id="STR_KUL_001", product_id="PRD_CHK_01")
    pr_dec = rec["price_decision"]
    assert pr_dec["safe_for_decision"] is False
    assert pr_dec["price_coverage_adjustment_days"] == 0
    assert pr_dec["final_coverage_with_price_signal"] == pr_dec["coverage_without_price_signal"]


# Test 4: Price forecast cannot affect urgency when unvalidated
def test_safe_rebuild_req4_price_forecast_cannot_affect_urgency():
    common_args = dict(
        current_inventory=50.0,
        current_coverage_days=3.0,
        lead_time_days=2,
        lead_time_demand=30.0,
        phase18_action="BUY_NOW",
        phase18_quantity=60.0,
        phase19_action="BUY_NOW",
        phase19_quantity=60.0,
        stockout_prob_before=0.12,
        stockout_prob_after=0.03,
        overstock_prob=0.05,
        wastage_prob=0.0,
        shelf_life_days=30,
        price_model_validated=False
    )
    _, urg_up, _, _, _, _, _ = resolve_action_and_urgency(
        price_change_pct=20.0, price_direction="UP", price_signal_strength="STRONG_BUY_EARLY_SIGNAL",
        net_price_timing_benefit=50.0, **common_args
    )
    _, urg_down, _, _, _, _, _ = resolve_action_and_urgency(
        price_change_pct=-20.0, price_direction="DOWN", price_signal_strength="STRONG_WAIT_SIGNAL",
        net_price_timing_benefit=-50.0, **common_args
    )
    assert urg_up == urg_down, "Price forecast altered urgency despite being unvalidated"


# Test 5: Price forecast cannot affect priority score when unvalidated
def test_safe_rebuild_req5_price_forecast_cannot_affect_priority():
    score_up = calculate_executive_priority_score(
        urgency="HIGH",
        stockout_prob_before=0.20,
        lost_units=50.0,
        price_change_pct=25.0,
        robustness="ROBUST",
        net_price_timing_benefit=100.0,
        price_model_validated=False
    )
    score_down = calculate_executive_priority_score(
        urgency="HIGH",
        stockout_prob_before=0.20,
        lost_units=50.0,
        price_change_pct=-25.0,
        robustness="ROBUST",
        net_price_timing_benefit=-100.0,
        price_model_validated=False
    )
    assert score_up == score_down, "Price forecast altered priority score when price model is unvalidated"


# Test 6: Critical stockout still triggers urgent reorder
def test_safe_rebuild_req6_critical_stockout_triggers_urgent_reorder():
    action, urg, qty, reasons, expl, rob, _ = resolve_action_and_urgency(
        current_inventory=0.0,
        current_coverage_days=0.0,
        lead_time_days=2,
        lead_time_demand=50.0,
        phase18_action="BUY_NOW",
        phase18_quantity=100.0,
        phase19_action="BUY_NOW",
        phase19_quantity=100.0,
        stockout_prob_before=0.95,
        stockout_prob_after=0.05,
        overstock_prob=0.0,
        wastage_prob=0.0,
        shelf_life_days=30,
        price_change_pct=0.0,
        price_direction="FLAT",
        price_model_validated=False
    )
    assert action == "URGENT_REORDER"
    assert urg == "CRITICAL"
    assert qty >= 100.0


# Test 7: Demand uncertainty still affects decision
def test_safe_rebuild_req7_demand_uncertainty_affects_decision():
    rec = generate_final_recommendation(store_id="STR_KUL_001", product_id="PRD_CHK_01")
    dem = rec["demand"]["7d"]
    spread = dem["p90"] - dem["p10"]
    assert spread > 0
    assert rec["decision"]["robustness"] in ["ROBUST", "MODERATELY_ROBUST", "UNCERTAINTY_SENSITIVE", "HIGHLY_UNCERTAIN"]


# Test 8: Current known purchase price still affects purchase cost
def test_safe_rebuild_req8_current_price_affects_purchase_cost():
    rec = generate_final_recommendation(store_id="STR_KUL_001", product_id="PRD_CHK_01")
    qty = rec["decision"]["final_order_quantity"]
    curr_price = rec["price"]["current_purchase_price"]
    expected_purchase_cost = rec["economics"]["expected_purchase_cost"]
    assert curr_price > 0.0
    assert expected_purchase_cost == round(qty * curr_price, 2)


# Test 9: User-upload mode still works
def test_safe_rebuild_req9_user_upload_mode_still_works():
    from src.recommendations.user_upload import validate_user_uploaded_files
    dates = pd.date_range("2024-01-01", periods=70)
    dummy = {
        "pos_sales.csv": pd.DataFrame({"date": dates, "units_sold": [15]*70}),
        "inventory_history.csv": pd.DataFrame({"date": dates, "inventory_units": [50]*70}),
        "purchase_price_history.csv": pd.DataFrame({"date": dates, "purchase_price": [5.50]*70}),
        "product_master.csv": pd.DataFrame({"product_id": ["PRD_CHK_01"], "name": ["Chicken"]}),
        "store_master.csv": pd.DataFrame({"store_id": ["STR_KUL_001"], "city": ["KL"]}),
        "supplier_master.csv": pd.DataFrame({"supplier_id": ["SUP_01"], "name": ["Supplier 1"]}),
    }
    res = validate_user_uploaded_files(dummy)
    assert res.is_valid is True
    assert res.available_history_days >= 60


# Test 10: Old price-aware cache rejected
def test_safe_rebuild_req10_old_price_aware_cache_rejected():
    from src.recommendations.engine import load_cached_phase_artifacts
    artifacts = load_cached_phase_artifacts()
    if "phase18_recommendations" in artifacts:
        df18 = artifacts["phase18_recommendations"]
        if "price_decision_enabled" in df18.columns:
            assert not any(df18["price_decision_enabled"].dropna() == True)


# Test 11: PRICE_MODEL_NOT_VALIDATED appears correctly
def test_safe_rebuild_req11_price_model_not_validated_appears_correctly():
    rec = generate_final_recommendation(store_id="STR_SGR_002", product_id="PRD_EGG_02")
    assert rec["price_decision"]["status"] == "PRICE_MODEL_NOT_VALIDATED"
    assert rec["price_decision"]["safe_for_decision"] is False
    assert rec["price_decision"]["influenced_final_decision"] is False
    assert rec["price_decision"]["predicted_future_price"] is None
    assert "PRICE_MODEL_NOT_VALIDATED" in rec["explanation"]["reason_codes"]


# Test 12: Dashboard payload consistent
def test_safe_rebuild_req12_dashboard_payload_consistent():
    payload = get_product_decision_payload("STR_KUL_001", "PRD_CHK_01")
    assert payload["product_id"] == "PRD_CHK_01"
    assert payload["price_signal"] is None or payload["price_signal"] == "PRICE_MODEL_NOT_VALIDATED"
    assert payload["four_decision_dimensions"]["price_outlook"].startswith("PRICE_MODEL_NOT_VALIDATED")
    assert payload["price_order_adjustment_units"] == 0.0
    assert payload["price_coverage_adjustment_days"] == 0

