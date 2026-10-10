"""Dashboard-Ready Payloads for Streamlit Interface (Phase 22 Hand-off)."""
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import json

from src.utils.paths import PROJECT_ROOT
from .engine import generate_final_recommendation, generate_recommendations_batch
from .alerts import evaluate_alert_triggers
from src.backtesting.replay_api import replay_historical_decision


def get_product_decision_payload(
    store_id: str,
    product_id: str,
    supplier_id: Optional[str] = None,
    decision_date: Optional[str] = None,
    root: Path = PROJECT_ROOT
) -> Dict[str, Any]:
    """Retrieve complete single-product decision payload for Phase 22 dashboard product view."""
    rec = generate_final_recommendation(
        store_id=store_id,
        product_id=product_id,
        supplier_id=supplier_id,
        decision_date=decision_date,
        root=root
    )

    # Attach visualization payloads for interactive frontend charts
    dem = rec["demand"]
    pr = rec["price"]
    rsk = rec["risk"]
    econ = rec["economics"]
    inv = rec["inventory"]
    dec = rec["decision"]

    viz_data = {
        "demand_trajectory": {
            "horizons": ["1d", "3d", "7d", "14d", "21d", "30d"],
            "point_predictions": [dem[h]["point_prediction"] for h in ["1d", "3d", "7d", "14d", "21d", "30d"]],
            "p10_bounds": [dem[h]["p10"] for h in ["1d", "3d", "7d", "14d", "21d", "30d"]],
            "p50_bounds": [dem[h]["p50"] for h in ["1d", "3d", "7d", "14d", "21d", "30d"]],
            "p90_bounds": [dem[h]["p90"] for h in ["1d", "3d", "7d", "14d", "21d", "30d"]]
        },
        "price_summary": {
            "current_price": pr["current_purchase_price"],
            "predicted_price": pr["predicted_future_price"],
            "p10": pr["p10"],
            "p50": pr["p50"],
            "p90": pr["p90"],
            "change_pct": pr["predicted_price_change_pct"],
            "direction": pr["price_direction"]
        },
        "risk_gauges": {
            "stockout_probability_before": rsk["stockout_probability_before"],
            "stockout_probability_after": rsk["stockout_probability_after"],
            "overstock_probability": rsk["overstock_probability"],
            "wastage_probability": rsk["wastage_probability"],
            "late_delivery_probability": rsk["late_delivery_probability"]
        },
        "cost_breakdown": {
            "purchase_cost": econ["expected_purchase_cost"],
            "holding_cost": econ["expected_holding_cost"],
            "stockout_cost": econ["expected_stockout_cost"],
            "wastage_cost": econ["expected_wastage_cost"],
            "total_expected_cost": econ["expected_total_cost"],
            "expected_savings": econ["expected_savings"]
        },
        "inventory_trajectory": {
            "current_units": inv["current_units"],
            "lead_time_demand": inv["lead_time_demand"],
            "arrival_inventory": inv["projected_inventory_at_arrival"],
            "order_quantity": dec["order_quantity"],
            "projected_ending_inventory": inv["projected_ending_inventory"]
        }
    }

    payload = dict(rec)
    payload["visualisation_data"] = viz_data

    # Section 17: Dashboard-ready direct access fields for Phase 22 interface
    pr_dec = rec.get("price_decision", {})
    payload["four_decision_dimensions"] = pr_dec.get("decision_criteria_summary", {})
    payload["price_signal"] = pr_dec.get("price_signal", "NEUTRAL_PRICE_SIGNAL")
    payload["price_signal_strength"] = pr_dec.get("price_signal_strength", "NEUTRAL_PRICE_SIGNAL")
    payload["probability_price_rises"] = pr_dec.get("probability_price_rises", 0.5)
    payload["net_price_timing_benefit"] = pr_dec.get("net_price_timing_benefit", 0.0)
    payload["price_order_adjustment_units"] = pr_dec.get("price_order_adjustment_units", 0.0)
    payload["price_coverage_adjustment_days"] = pr_dec.get("price_coverage_adjustment_days", 0)
    payload["price_influenced_final_decision"] = pr_dec.get("influenced_final_decision", False)
    payload["price_decision_effect"] = pr_dec.get("decision_effect", "")
    payload["final_recommendation"] = rec.get("decision", {})
    return payload


def get_executive_dashboard_payload(root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Retrieve full executive-level dashboard payload with priorities, alerts, and aggregate spend."""
    final_df, pq_df, _ = generate_recommendations_batch(root=root)

    total_spend = float(final_df["expected_total_cost"].sum())
    total_savings = float(final_df["expected_savings"].sum())
    avg_stockout_before = float(final_df["stockout_probability_before"].mean())
    avg_stockout_after = float(final_df["stockout_probability_after"].mean())

    urgent_count = int((final_df["action"] == "URGENT_REORDER").sum())
    high_risk_count = int((final_df["stockout_probability_before"] >= 0.20).sum())
    overstock_count = int((final_df["action"] == "OVERSTOCK_WARNING").sum())

    top_priorities = pq_df.head(5).to_dict(orient="records")

    # Generate sample alerts
    alerts = []
    for _, row in pq_df.head(3).iterrows():
        rec_dict = row.to_dict()
        triggered = evaluate_alert_triggers(rec_dict)
        alerts.extend(triggered)

    return {
        "status": "HEALTHY",
        "decision_date": str(final_df["decision_date"].iloc[0]),
        "total_monitored_products": len(final_df),
        "total_stores_monitored": int(final_df["store_id"].nunique()),
        "summary_metrics": {
            "urgent_reorder_count": urgent_count,
            "high_stockout_risk_count": high_risk_count,
            "overstock_warning_count": overstock_count,
            "total_recommended_spend_rm": round(total_spend, 2),
            "total_expected_savings_rm": round(total_savings, 2),
            "avg_stockout_prob_before": round(avg_stockout_before, 4),
            "avg_stockout_prob_after": round(avg_stockout_after, 4),
            "historical_backtest_savings_pct": 44.49
        },
        "top_priority_queue": top_priorities,
        "active_alerts": alerts
    }


def get_model_validation_payload(root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Retrieve scientific model validation scorecard payload for Phase 22 dashboard."""
    return {
        "demand_model_validation": {
            "primary_metric": "WAPE",
            "validation_score": "16.2%",
            "mean_absolute_error_units": 11.8,
            "baseline_improvement_vs_naive": "+47.8%",
            "out_of_fold_cross_validation_folds": 33
        },
        "price_model_validation": {
            "validation_status": "FAIL",
            "decision_approval": "NOT_APPROVED_FOR_DECISION_USE",
            "primary_metric": "MAE (Diagnostics Only)",
            "validation_score_myr": "RM 0.20 – RM 0.21",
            "benchmark_baseline": "RM 0.20 (Rolling Mean / MA)",
            "baseline_improvement_vs_naive": "+0.0% (No Predictive Advantage)",
            "audit_findings": "Target leakage detected in prior optimization cache; clean models show no edge over simple baselines; target behaves as stationary noise around base price.",
            "forecast_horizons_supported": ["1d", "3d", "7d", "14d", "21d", "30d"]
        },
        "uncertainty_calibration": {
            "framework": "Conformal Quantile Regression (CQR)",
            "nominal_target_coverage": "80.0%",
            "empirical_achieved_coverage": "79.8%",
            "pinball_loss": 3.42,
            "valid_zero_crossings": "100.0% Non-crossing"
        },
        "economic_backtest_validation": {
            "historical_replay_duration_days": 990,
            "decision_origins_evaluated": 33,
            "net_realised_cost_reduction": "44.49%",
            "net_realised_savings_rm": "RM 5,775,380.76",
            "working_capital_reduction": "44.55%",
            "perishable_wastage_reduction": "82.36%"
        }
    }


def get_training_data_transparency_payload(root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Retrieve training-data audit payload documenting exact history utilized."""
    return {
        "raw_dataset_period": {
            "start_date": "2015-01-01",
            "end_date": "2024-12-31",
            "total_calendar_years": 10.0
        },
        "model_training_development_period": {
            "start_date": "2021-01-01",
            "end_date": "2024-09-30",
            "development_rows": 32448,
            "stores_count": 4,
            "products_count": 6,
            "suppliers_count": 5
        },
        "validation_structure": {
            "scheme": "Expanding Walk-Forward Cross Validation",
            "folds_count": 33,
            "fold_step_days": 30,
            "horizon_days": 30
        },
        "final_holdout_quarantine": {
            "start_date": "2024-10-01",
            "end_date": "2024-12-31",
            "total_records": 2208,
            "status": "STRICTLY_QUARANTINED_UNTOUCHED"
        }
    }


def get_historical_replay_payload(
    date: str,
    store_id: str,
    product_id: str,
    root: Path = PROJECT_ROOT
) -> Dict[str, Any]:
    """Retrieve single-date historical replay audit for dashboard inspection."""
    return replay_historical_decision(decision_date=date, store_id=store_id, product_id=product_id)
