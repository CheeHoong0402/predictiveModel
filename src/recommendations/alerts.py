"""Telegram and System Alert Orchestration with Cooldown and Token Security."""
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from src.utils.paths import PROJECT_ROOT
from .config import load_alerts_config

# In-memory cooldown tracking dictionary: (store_id, product_id, reason_code) -> datetime
_ALERT_COOLDOWN_REGISTRY: Dict[Tuple[str, str, str], datetime] = {}


def get_telegram_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Retrieve Telegram credentials strictly from environment variables without hardcoding."""
    config = load_alerts_config()
    token_var = config.get("security", {}).get("bot_token_env_var", "TELEGRAM_BOT_TOKEN")
    chat_var = config.get("security", {}).get("chat_id_env_var", "TELEGRAM_CHAT_ID")
    bot_token = os.environ.get(token_var)
    chat_id = os.environ.get(chat_var)
    return bot_token, chat_id


def is_on_cooldown(
    store_id: str,
    product_id: str,
    reason_code: str,
    current_time: Optional[datetime] = None,
    cooldown_hours: float = 24.0
) -> bool:
    """Check if an alert for (store, product, reason) is currently in cooldown."""
    if current_time is None:
        current_time = datetime.now()
    key = (store_id, product_id, reason_code)
    last_sent = _ALERT_COOLDOWN_REGISTRY.get(key)
    if last_sent is None:
        return False
    return (current_time - last_sent) < timedelta(hours=cooldown_hours)


def record_alert_dispatched(
    store_id: str,
    product_id: str,
    reason_code: str,
    timestamp: Optional[datetime] = None
) -> None:
    """Record an alert dispatch timestamp for cooldown enforcement."""
    if timestamp is None:
        timestamp = datetime.now()
    _ALERT_COOLDOWN_REGISTRY[(store_id, product_id, reason_code)] = timestamp


def clear_alert_cooldowns() -> None:
    """Reset the cooldown registry (useful for testing)."""
    _ALERT_COOLDOWN_REGISTRY.clear()


def evaluate_alert_triggers(
    recommendation: Dict[str, Any],
    product_catalog: Optional[Dict[str, str]] = None,
    current_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """Evaluate whether a recommendation triggers any Telegram alerts under current policy and cooldown."""
    config = load_alerts_config()
    triggers = config.get("triggers", {})
    cooldown_hours = float(config.get("cooldown", {}).get("cooldown_window_hours", 24.0))
    if current_time is None:
        current_time = datetime.now()

    store_id = recommendation.get("store_id", "UNKNOWN_STORE")
    product_id = recommendation.get("product_id", "UNKNOWN_PRODUCT")
    product_name = (product_catalog or {}).get(product_id, product_id)
    action = recommendation.get("action", "")
    urgency = recommendation.get("urgency", "")
    stockout_before = float(recommendation.get("stockout_probability_before", 0.0))
    stockout_after = float(recommendation.get("stockout_probability_after", 0.0))
    overstock_prob = float(recommendation.get("overstock_probability", 0.0))
    wastage_prob = float(recommendation.get("wastage_probability", 0.0))
    late_delivery_prob = float(recommendation.get("late_delivery_probability", 0.0))
    raw_pr_pct = recommendation.get("predicted_price_change_pct")
    price_change_pct = float(raw_pr_pct) if raw_pr_pct is not None else 0.0
    price_direction = recommendation.get("price_direction", "FLAT")
    current_inv = float(recommendation.get("current_inventory", 0.0))
    current_cov = float(recommendation.get("current_coverage_days", 0.0))
    final_qty = float(recommendation.get("order_quantity", 0.0))
    final_cov = float(recommendation.get("coverage_days", 0.0))
    top_reasons = recommendation.get("top_reasons", [])
    top_reason = top_reasons[0] if top_reasons else "Inventory action recommended."

    alerts: List[Dict[str, Any]] = []

    # Check 1: Urgent reorder
    if triggers.get("urgent_reorder", {}).get("enabled", True):
        if action == "URGENT_REORDER" and urgency in ["CRITICAL", "HIGH"]:
            reason_key = "URGENT_REORDER"
            if not is_on_cooldown(store_id, product_id, reason_key, current_time, cooldown_hours):
                pr_str = f"{price_change_pct:+.1f}% ({price_direction})" if raw_pr_pct is not None else "N/A (NOT_VALIDATED)"
                msg = (
                    f"🚨 *URGENT REORDER ALERT*\n"
                    f"🏬 *Store:* {store_id}\n"
                    f"📦 *Product:* {product_name} ({product_id})\n"
                    f"📊 *Current Stock:* {current_inv:.0f} units ({current_cov:.1f} days)\n"
                    f"🛒 *Recommended Order:* {final_qty:.0f} units ({final_cov:.0f} days)\n"
                    f"📉 *Stockout Risk:* {stockout_before:.1%} → {stockout_after:.1%}\n"
                    f"💵 *Price Outlook:* {pr_str}\n"
                    f"💡 *Main Reason:* {top_reason}"
                )
                alerts.append({
                    "alert_type": "URGENT_REORDER",
                    "store_id": store_id,
                    "product_id": product_id,
                    "reason_code": reason_key,
                    "urgency": urgency,
                    "message": msg
                })
                record_alert_dispatched(store_id, product_id, reason_key, current_time)

    # Check 2: High stockout risk
    if triggers.get("high_stockout_risk", {}).get("enabled", True):
        thresh = float(triggers.get("high_stockout_risk", {}).get("stockout_probability_threshold", 0.20))
        if stockout_before >= thresh and action != "URGENT_REORDER":
            reason_key = "HIGH_STOCKOUT_RISK"
            if not is_on_cooldown(store_id, product_id, reason_key, current_time, cooldown_hours):
                msg = (
                    f"⚠️ *HIGH STOCKOUT RISK*\n"
                    f"🏬 *Store:* {store_id}\n"
                    f"📦 *Product:* {product_name} ({product_id})\n"
                    f"📉 *Stockout Risk:* {stockout_before:.1%}\n"
                    f"📊 *Current Stock:* {current_inv:.0f} units ({current_cov:.1f} days)\n"
                    f"💡 *Action:* {action} ({final_qty:.0f} units)"
                )
                alerts.append({
                    "alert_type": "HIGH_STOCKOUT_RISK",
                    "store_id": store_id,
                    "product_id": product_id,
                    "reason_code": reason_key,
                    "urgency": urgency,
                    "message": msg
                })
                record_alert_dispatched(store_id, product_id, reason_key, current_time)

    # Check 3: Large price increase (only triggered if price model is validated and raw_pr_pct is present)
    from src.models.governance import price_model_usable_for_decision
    if price_model_usable_for_decision() and raw_pr_pct is not None and triggers.get("large_price_increase", {}).get("enabled", True):
        thresh_price = float(triggers.get("large_price_increase", {}).get("price_increase_threshold_pct", 3.0))
        if price_change_pct >= thresh_price:
            reason_key = "LARGE_PRICE_INCREASE"
            if not is_on_cooldown(store_id, product_id, reason_key, current_time, cooldown_hours):
                msg = (
                    f"📈 *SUPPLIER PRICE HIKE ALERT*\n"
                    f"🏬 *Store:* {store_id}\n"
                    f"📦 *Product:* {product_name} ({product_id})\n"
                    f"💵 *Expected Increase:* {price_change_pct:+.1f}%\n"
                    f"💡 *Recommendation:* Lock in forward volume ({final_qty:.0f} units) before cost increases."
                )
                alerts.append({
                    "alert_type": "LARGE_PRICE_INCREASE",
                    "store_id": store_id,
                    "product_id": product_id,
                    "reason_code": reason_key,
                    "urgency": urgency,
                    "message": msg
                })
                record_alert_dispatched(store_id, product_id, reason_key, current_time)

    # Check 4: Overstock warning
    if triggers.get("overstock_warning", {}).get("enabled", True):
        thresh_over = float(triggers.get("overstock_warning", {}).get("overstock_probability_threshold", 0.35))
        if action == "OVERSTOCK_WARNING" or overstock_prob >= thresh_over:
            reason_key = "OVERSTOCK_WARNING"
            if not is_on_cooldown(store_id, product_id, reason_key, current_time, cooldown_hours):
                msg = (
                    f"📦 *OVERSTOCK WARNING*\n"
                    f"🏬 *Store:* {store_id}\n"
                    f"📦 *Product:* {product_name} ({product_id})\n"
                    f"📊 *Current Coverage:* {current_cov:.1f} days (Excess Risk: {overstock_prob:.1%})\n"
                    f"🚫 *Action:* Halting orders to protect working capital."
                )
                alerts.append({
                    "alert_type": "OVERSTOCK_WARNING",
                    "store_id": store_id,
                    "product_id": product_id,
                    "reason_code": reason_key,
                    "urgency": urgency,
                    "message": msg
                })
                record_alert_dispatched(store_id, product_id, reason_key, current_time)

    return alerts


def build_daily_priority_summary_message(
    decision_date: str,
    recommendations_list: List[Dict[str, Any]]
) -> str:
    """Format daily executive summary message for Telegram or executive broadcast."""
    urgent_count = sum(1 for r in recommendations_list if r.get("action") == "URGENT_REORDER")
    high_risk_count = sum(1 for r in recommendations_list if float(r.get("stockout_probability_before", 0.0)) >= 0.20)
    overstock_count = sum(1 for r in recommendations_list if r.get("action") == "OVERSTOCK_WARNING" or float(r.get("overstock_probability", 0.0)) >= 0.35)
    total_spend = sum(float(r.get("expected_purchase_cost", 0.0)) for r in recommendations_list)
    total_savings = sum(float(r.get("expected_savings", 0.0)) for r in recommendations_list)

    top_rec = recommendations_list[0] if recommendations_list else {}
    top_action = f"{top_rec.get('action', 'N/A')} on {top_rec.get('product_id', 'N/A')} ({top_rec.get('store_id', 'N/A')})"

    return (
        f"📋 *TODAY'S INVENTORY PRIORITIES*\n"
        f"📅 *Date:* {decision_date}\n"
        f"🚨 *Urgent Reorders:* {urgent_count}\n"
        f"⚠️ *High-Risk Products:* {high_risk_count}\n"
        f"📦 *Overstock Warnings:* {overstock_count}\n"
        f"💰 *Recommended Purchasing:* RM {total_spend:,.2f}\n"
        f"✨ *Expected Savings:* RM {total_savings:,.2f}\n"
        f"🏆 *Top Priority Action:* {top_action}"
    )
