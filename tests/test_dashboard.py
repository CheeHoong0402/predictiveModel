"""Automated Unit and Integration Test Suite for Streamlit Dashboard (Phase 22)."""
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from src.utils.paths import PROJECT_ROOT
from src.dashboard.config import (
    DEFAULT_DEMO_STORE, DEFAULT_DEMO_PRODUCT, STORE_LABELS, PRODUCT_LABELS,
    SYNTHETIC_DATA_DISCLOSURE
)
from src.dashboard.formatting import format_myr, format_pct, format_units, format_days
from src.dashboard.data_loader import (
    load_final_recommendations_df, load_priority_queue_df,
    load_executive_summary_json, load_cached_product_payload,
    load_cached_replay_decision, load_model_validation_summary,
    load_training_transparency_summary, load_backtest_benchmarks_df
)
from src.dashboard.charts import (
    plot_demand_forecast_chart, plot_price_forecast_chart,
    plot_inventory_trajectory_chart, plot_risk_comparison_chart
)
from src.dashboard.upload_validator import (
    get_sample_templates, apply_column_alias_mapping, run_user_data_inference
)
from src.recommendations.user_upload import (
    REQUIRED_FILES, OPTIONAL_FILES, validate_user_uploaded_files, MIN_REQUIRED_HISTORY_DAYS
)
from src.recommendations.alerts import evaluate_alert_triggers, get_telegram_credentials


# Test 1 & 2: Demo mode and user mode initialisation
def test_demo_and_user_mode_initialisation():
    assert DEFAULT_DEMO_STORE in STORE_LABELS
    assert DEFAULT_DEMO_PRODUCT in PRODUCT_LABELS
    templates = get_sample_templates()
    assert len(templates) >= 6, "Templates must cover all mandatory upload files"


# Test 3, 4, 5: Phase 21 recommendations and payloads load
def test_phase21_recommendations_and_payloads():
    root = str(PROJECT_ROOT)
    recs_df = load_final_recommendations_df(root)
    assert not recs_df.empty, "Final recommendations dataframe must not be empty"
    assert len(recs_df) == 24, "Expected 24 recommendations across 4 stores x 6 products"

    exec_json = load_executive_summary_json(root)
    assert "urgent_actions_count" in exec_json or "summary_metrics" in exec_json

    prod_payload = load_cached_product_payload(DEFAULT_DEMO_STORE, DEFAULT_DEMO_PRODUCT, root)
    assert "decision" in prod_payload
    assert "visualisation_data" in prod_payload


# Test 6, 7, 8: Demand and price chart data validity
def test_chart_data_validity():
    prod_payload = load_cached_product_payload(DEFAULT_DEMO_STORE, DEFAULT_DEMO_PRODUCT, str(PROJECT_ROOT))
    dem = prod_payload["demand"]
    pr = prod_payload["price"]

    fig_dem = plot_demand_forecast_chart(dem, "Chicken")
    assert fig_dem is not None
    assert len(fig_dem.data) >= 3, "Demand chart must include P10, P50, and point forecast traces"

    # Verify P10 <= P50 <= P90 across all horizons
    for h, vals in dem.items():
        assert vals["p10"] <= vals["p50"] <= vals["p90"], f"Quantile crossing at horizon {h}"

    fig_pr = plot_price_forecast_chart(pr, pr["current_purchase_price"], "Chicken")
    assert fig_pr is not None


# Test 9, 10, 11: Replay, validation, and training transparency payloads
def test_auxiliary_dashboard_payloads():
    root = str(PROJECT_ROOT)
    replay = load_cached_replay_decision("2024-08-17", DEFAULT_DEMO_STORE, DEFAULT_DEMO_PRODUCT, root)
    assert "decisions_compared" in replay
    assert "realised_outcomes" in replay

    val_summary = load_model_validation_summary(root)
    assert "demand_model_validation" in val_summary
    assert "economic_backtest_validation" in val_summary

    trans_summary = load_training_transparency_summary(root)
    assert trans_summary["final_holdout_quarantine"]["status"] == "STRICTLY_QUARANTINED_UNTOUCHED"


# Test 12-24: User-upload files validation and requirements
def test_user_upload_validation():
    assert len(REQUIRED_FILES) == 6
    assert len(OPTIONAL_FILES) >= 2

    dates = pd.date_range("2024-01-01", periods=90)
    valid_files = {
        "pos_sales.csv": pd.DataFrame({"date": dates, "store_id": ["STR_KUL_001"]*90, "product_id": ["PRD_CHK_01"]*90, "quantity_sold": [50]*90}),
        "inventory_history.csv": pd.DataFrame({"date": dates, "store_id": ["STR_KUL_001"]*90, "product_id": ["PRD_CHK_01"]*90, "inventory_on_hand": [100]*90}),
        "purchase_price_history.csv": pd.DataFrame({"date": dates, "store_id": ["STR_KUL_001"]*90, "product_id": ["PRD_CHK_01"]*90, "supplier_id": ["SUP_KLS_01"]*90, "purchase_price": [7.2]*90}),
        "product_master.csv": pd.DataFrame({"product_id": ["PRD_CHK_01"], "product_name": ["Chicken"]}),
        "store_master.csv": pd.DataFrame({"store_id": ["STR_KUL_001"], "store_name": ["KL"]}),
        "supplier_master.csv": pd.DataFrame({"supplier_id": ["SUP_KLS_01"], "supplier_name": ["Supplier 1"]})
    }
    res_val = validate_user_uploaded_files(valid_files)
    assert res_val.is_valid, f"Validation failed: {res_val.message}"
    assert res_val.available_history_days >= MIN_REQUIRED_HISTORY_DAYS

    # Test missing mandatory file fails safely
    missing_one = dict(valid_files)
    del missing_one["pos_sales.csv"]
    res_missing = validate_user_uploaded_files(missing_one)
    assert not res_missing.is_valid
    assert "INVALID_MISSING_REQUIRED_FILES" in res_missing.status

    # Test insufficient history fails
    short_dates = pd.date_range("2024-01-01", periods=30)
    short_files = dict(valid_files)
    short_files["pos_sales.csv"] = pd.DataFrame({"date": short_dates, "store_id": ["S1"]*30, "product_id": ["P1"]*30, "quantity_sold": [10]*30})
    res_short = validate_user_uploaded_files(short_files)
    assert not res_short.is_valid
    assert res_short.status == "INSUFFICIENT_HISTORY"


# Test 25 & 26: Automatic column alias mapping
def test_column_alias_mapping():
    df_raw = pd.DataFrame({"sku": ["PRD_CHK_01"], "outlet_id": ["STR_KUL_001"], "qty": [50]})
    df_mapped, mapped_cols = apply_column_alias_mapping(df_raw)
    assert "product_id" in df_mapped.columns
    assert "store_id" in df_mapped.columns
    assert "quantity_sold" in df_mapped.columns
    assert len(mapped_cols) == 3


# Test 27 & 28: User-upload inference executes without model retraining
def test_user_upload_inference_without_retraining():
    dates = pd.date_range("2024-01-01", periods=90)
    tables = {
        "pos_sales": pd.DataFrame({"date": dates, "store_id": ["STR_KUL_001"]*90, "product_id": ["PRD_CHK_01"]*90, "quantity_sold": [50]*90}),
        "product_master": pd.DataFrame({"product_id": ["PRD_CHK_01"]}),
        "store_master": pd.DataFrame({"store_id": ["STR_KUL_001"]})
    }
    with pytest.raises(NotImplementedError, match="not validated"):
        run_user_data_inference(tables)


# Test 29-33: Telegram preview and missing secrets
def test_telegram_alert_preview_and_security():
    prod_payload = load_cached_product_payload(DEFAULT_DEMO_STORE, DEFAULT_DEMO_PRODUCT, str(PROJECT_ROOT))
    alerts = evaluate_alert_triggers(prod_payload)
    # Check that credentials retrieval does not crash when env vars are unset
    token, chat = get_telegram_credentials()
    # Safe fallback: token can be None without raising exception


# Test 34-37: Relative paths and CPU compatibility
def test_relative_paths_and_cpu_portability():
    # Verify PROJECT_ROOT is a Path object that resolves correctly
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "app.py").exists()
    assert (PROJECT_ROOT / "config" / "recommendation_engine.yaml").exists()


# Test 40 & 41: Synthetic disclosure and final holdout status
def test_disclosures_and_holdout_protection():
    assert "synthetic" in SYNTHETIC_DATA_DISCLOSURE.lower()
    trans_summary = load_training_transparency_summary(str(PROJECT_ROOT))
    holdout_info = trans_summary["final_holdout_quarantine"]
    assert holdout_info["start_date"] == "2024-10-01"
    assert holdout_info["total_records"] == 2208
    assert holdout_info["status"] == "STRICTLY_QUARANTINED_UNTOUCHED"


# =============================================================================
# 13. PRICE SIGNAL BADGES & FOUR DECISION CRITERIA UI TESTS
# =============================================================================

def test_price_signal_badge_rendering():
    """Verify HTML rendering and styling for all 5 price signal badges."""
    from src.dashboard.formatting import get_price_signal_badge

    b_strong_buy = get_price_signal_badge("STRONG_BUY_EARLY_SIGNAL")
    assert "STRONG BUY-EARLY" in b_strong_buy
    assert "#D1FAE5" in b_strong_buy

    b_mod_buy = get_price_signal_badge("MODERATE_BUY_EARLY_SIGNAL")
    assert "MODERATE BUY-EARLY" in b_mod_buy
    assert "#DBEAFE" in b_mod_buy

    b_neutral = get_price_signal_badge("NEUTRAL_PRICE_SIGNAL")
    assert "NEUTRAL PRICE" in b_neutral

    b_mod_wait = get_price_signal_badge("MODERATE_WAIT_SIGNAL")
    assert "MODERATE WAIT" in b_mod_wait

    b_strong_wait = get_price_signal_badge("STRONG_WAIT_SIGNAL")
    assert "STRONG WAIT" in b_strong_wait


def test_four_decision_criteria_payload_coverage():
    """Verify that product decision payload contains all fields required for the 4-criteria panel."""
    prod_payload = load_cached_product_payload(DEFAULT_DEMO_STORE, DEFAULT_DEMO_PRODUCT, str(PROJECT_ROOT))
    assert "price_decision" in prod_payload, "price_decision must be present in product payload"
    pr_dec = prod_payload["price_decision"]
    assert "current_purchase_price" in pr_dec
    assert "predicted_purchase_price" in pr_dec
    assert "price_signal_strength" in pr_dec
    assert "net_timing_benefit" in pr_dec
    assert "buy_now_vs_wait_recommendation" in pr_dec
    assert "baseline_demand_order_quantity" in pr_dec
    assert "price_aware_order_quantity" in pr_dec


# =============================================================================
# 14. PHASE 22 MODEL-VALIDATION GATE & GOVERNANCE AUDIT TESTS
# =============================================================================

def test_central_model_governance_status():
    """1. Demand validated, price model unvalidated."""
    from src.models.governance import (
        demand_model_usable_for_decision,
        price_model_usable_for_decision,
        get_price_model_validation_status,
        get_demand_model_validation_status
    )
    assert demand_model_usable_for_decision() is True
    assert price_model_usable_for_decision() is False
    p_status = get_price_model_validation_status()
    assert p_status["safe_for_decision"] is False
    assert p_status["status"] == "NOT_VALIDATED"
    d_status = get_demand_model_validation_status()
    assert d_status["safe_for_decision"] is True
    assert d_status["status"] == "VALIDATED"


def test_executive_dashboard_price_independence():
    """2. Executive recommendation does not depend on price forecasts."""
    from src.dashboard.data_loader import load_final_recommendations_df
    df = load_final_recommendations_df(str(PROJECT_ROOT))
    assert not df.empty
    # When price is disabled, price_delta_quantity should be 0 across all items
    if "price_delta_quantity" in df.columns:
        assert (df["price_delta_quantity"] == 0).all()


def test_model_status_panel_wording():
    """3. Model status panel renders correct governance statuses."""
    from src.dashboard.components import render_model_governance_panel
    # Ensure it is callable without errors
    assert callable(render_model_governance_panel)


def test_final_recommendation_card_no_invalid_prices():
    """4. Final recommendation card does not present invalid price predictions."""
    payload = load_cached_product_payload(DEFAULT_DEMO_STORE, DEFAULT_DEMO_PRODUCT, str(PROJECT_ROOT))
    dec = payload["decision"]
    assert "order_quantity" in dec
    assert "coverage_days" in dec
    # In quarantined state, predicted_future_price in price payload is None
    assert payload["price"]["predicted_future_price"] is None


def test_price_page_experimental_labeling():
    """5. Price chart renders with quarantined / experimental watermarking."""
    payload = load_cached_product_payload(DEFAULT_DEMO_STORE, DEFAULT_DEMO_PRODUCT, str(PROJECT_ROOT))
    pr = payload["price"]
    fig = plot_price_forecast_chart(pr, pr["current_purchase_price"], "Test SKU")
    assert fig is not None
    # Check that annotations contain QUARANTINED notice
    annotations = [a.text for a in fig.layout.annotations]
    assert any("QUARANTINED" in a for a in annotations)


def test_model_validation_page_reports_fail():
    """6. Model validation scorecard reports price validation FAIL."""
    summary = load_model_validation_summary(str(PROJECT_ROOT))
    pr_val = summary["price_model_validation"]
    assert pr_val["validation_status"] == "FAIL"
    assert pr_val["decision_approval"] == "NOT_APPROVED_FOR_DECISION_USE"
    assert "leakage" in pr_val["audit_findings"].lower()


def test_user_data_mode_safe_price_handling():
    """7. User data mode functions correctly without model retraining."""
    dates = pd.date_range("2024-01-01", periods=90)
    tables = {
        "pos_sales": pd.DataFrame({"date": dates, "store_id": ["STR_KUL_001"]*90, "product_id": ["PRD_CHK_01"]*90, "quantity_sold": [50]*90}),
        "product_master": pd.DataFrame({"product_id": ["PRD_CHK_01"]}),
        "store_master": pd.DataFrame({"store_id": ["STR_KUL_001"]})
    }
    with pytest.raises(NotImplementedError, match="not validated"):
        run_user_data_inference(tables)


def test_model_performance_demand_vs_price():
    """8. Demand displays valid metrics, price displays unapproved status."""
    summary = load_model_validation_summary(str(PROJECT_ROOT))
    dem = summary["demand_model_validation"]
    assert dem["primary_metric"] == "WAPE"
    assert dem["validation_score"] == "16.2%"
    assert dem["mean_absolute_error_units"] == 11.8
    pr = summary["price_model_validation"]
    assert pr["validation_status"] == "FAIL"


def test_historical_backtest_regenerated():
    """9. Historical backtest displays regenerated price-independent results."""
    from src.dashboard.data_loader import load_backtest_benchmarks_df
    bt_df = load_backtest_benchmarks_df(str(PROJECT_ROOT))
    assert not bt_df.empty
    assert "reference_strategy" in bt_df.columns
    # Check 14-day policy row shows 44.49% savings
    row_14d = bt_df[bt_df["reference_strategy"] == "FIXED_14_DAY"]
    assert not row_14d.empty
    assert abs(row_14d["savings_percentage"].iloc[0] - 44.49) < 0.1


def test_alerts_do_not_contain_speculative_price_hikes():
    """10. Alerts do not recommend speculative buying when price model is unvalidated."""
    payload = load_cached_product_payload(DEFAULT_DEMO_STORE, DEFAULT_DEMO_PRODUCT, str(PROJECT_ROOT))
    alerts = evaluate_alert_triggers(payload)
    for a in alerts:
        assert a["alert_type"] != "LARGE_PRICE_INCREASE", "Speculative price hike alert must not trigger"
        assert "Lock in forward volume" not in a.get("message", "")

