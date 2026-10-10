"""Performance and economic metric calculation for Phase 20 Historical Backtesting."""
from typing import Dict, Any, List
import pandas as pd
import numpy as np


def compute_realised_metrics(
    daily_replay_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    strategy_name: str,
    cost_params: Dict[str, float]
) -> Dict[str, Any]:
    """Calculate aggregated business and operational metrics for a strategy."""
    strat_daily = daily_replay_df[daily_replay_df["strategy"] == strategy_name].copy()
    strat_dec = decision_df[decision_df["strategy"] == strategy_name].copy()

    total_days = len(strat_daily)
    total_demand = float(strat_daily["demand"].sum())
    fulfilled_demand = float(strat_daily["fulfilled_demand"].sum())
    lost_demand = float(strat_daily["lost_demand"].sum())

    # Stockout metrics
    stockout_days = int((strat_daily["lost_demand"] > 0).sum())
    stockout_rate = float(stockout_days / max(1, total_days))
    fill_rate = float(fulfilled_demand / max(1e-6, total_demand))

    # Inventory metrics
    avg_inventory = float(strat_daily["closing_inventory"].mean())
    median_inventory = float(strat_daily["closing_inventory"].median())
    peak_inventory = float(strat_daily["closing_inventory"].max()) if total_days > 0 else 0.0
    ending_inventory = float(strat_daily["closing_inventory"].iloc[-1]) if total_days > 0 else 0.0

    avg_daily_demand = float(total_demand / max(1, total_days))
    avg_days_stock = float(avg_inventory / max(0.01, avg_daily_demand))

    # Wastage metrics
    wasted_units = float(strat_daily["wastage"].sum())

    # Financial costs
    purch_cost = float(strat_dec["realised_purchase_cost"].sum())
    order_cost = float(strat_dec["ordering_cost"].sum())
    hold_cost = float(strat_daily["holding_cost"].sum())
    stock_cost = float(strat_daily["stockout_cost"].sum())
    waste_cost = float(strat_daily["wastage_cost"].sum())
    total_cost = purch_cost + order_cost + hold_cost + stock_cost + waste_cost

    return {
        "strategy": strategy_name,
        "total_purchase_cost": round(purch_cost, 2),
        "ordering_cost": round(order_cost, 2),
        "holding_cost": round(hold_cost, 2),
        "stockout_cost": round(stock_cost, 2),
        "wastage_cost": round(waste_cost, 2),
        "total_cost": round(total_cost, 2),
        "stockout_days": stockout_days,
        "lost_units": round(lost_demand, 1),
        "stockout_rate": round(stockout_rate, 4),
        "fill_rate": round(fill_rate, 4),
        "average_inventory": round(avg_inventory, 1),
        "average_days_stock": round(avg_days_stock, 1),
        "wasted_units": round(wasted_units, 1)
    }


def compare_ai_vs_reference(
    ai_metrics: Dict[str, Any],
    ref_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate comparative business value and savings of AI versus a reference policy."""
    ref_cost = float(ref_metrics["total_cost"])
    ai_cost = float(ai_metrics["total_cost"])
    abs_savings = ref_cost - ai_cost
    pct_savings = (abs_savings / ref_cost * 100.0) if ref_cost > 0 else 0.0

    ref_stk = float(ref_metrics["stockout_rate"])
    ai_stk = float(ai_metrics["stockout_rate"])
    stk_reduction = ref_stk - ai_stk

    ref_fill = float(ref_metrics["fill_rate"])
    ai_fill = float(ai_metrics["fill_rate"])
    fill_diff = (ai_fill - ref_fill) * 100.0

    ref_inv = float(ref_metrics["average_inventory"])
    ai_inv = float(ai_metrics["average_inventory"])
    inv_diff_pct = ((ref_inv - ai_inv) / ref_inv * 100.0) if ref_inv > 0 else 0.0

    return {
        "reference_strategy": ref_metrics["strategy"],
        "ai_total_cost": round(ai_cost, 2),
        "reference_total_cost": round(ref_cost, 2),
        "absolute_savings": round(abs_savings, 2),
        "savings_percentage": round(pct_savings, 2),
        "stockout_rate_ai": round(ai_stk, 4),
        "stockout_rate_reference": round(ref_stk, 4),
        "stockout_reduction": round(stk_reduction, 4),
        "fill_rate_ai": round(ai_fill, 4),
        "fill_rate_reference": round(ref_fill, 4),
        "fill_rate_improvement_pp": round(fill_diff, 2),
        "average_inventory_ai": round(ai_inv, 1),
        "average_inventory_reference": round(ref_inv, 1),
        "inventory_reduction_pct": round(inv_diff_pct, 2),
        "wastage_ai": round(float(ai_metrics["wasted_units"]), 1),
        "wastage_reference": round(float(ref_metrics["wasted_units"]), 1)
    }
