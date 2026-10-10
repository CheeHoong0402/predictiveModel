"""Public and Judge-Replay APIs for Phase 20 Historical Backtesting."""
from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path

from src.utils.paths import PROJECT_ROOT
from src.backtesting.engine import run_historical_backtest
from src.backtesting.config import load_backtest_config


def run_historical_replay(
    start_date: str,
    end_date: str,
    strategy: str,
    stores: Optional[List[str]] = None,
    products: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Execute on-demand historical replay for dashboard and exploratory analysis."""
    dec_df, day_df = run_historical_backtest(
        stores=stores,
        products=products,
        strategies_to_run=[strategy, "FIXED_14_DAY"]
    )
    # Filter between start_date and end_date
    day_df["date"] = pd.to_datetime(day_df["date"])
    sub_day = day_df[(day_df["date"] >= pd.Timestamp(start_date)) & (day_df["date"] <= pd.Timestamp(end_date))]
    strat_day = sub_day[sub_day["strategy"] == strategy]

    tot_d = float(strat_day["demand"].sum())
    ful_d = float(strat_day["fulfilled_demand"].sum())
    lost_d = float(strat_day["lost_demand"].sum())
    stk_days = int((strat_day["lost_demand"] > 0).sum())
    tot_days = len(strat_day)

    return {
        "strategy": strategy,
        "start_date": start_date,
        "end_date": end_date,
        "total_days_evaluated": tot_days,
        "total_demand": round(tot_d, 1),
        "fulfilled_demand": round(ful_d, 1),
        "lost_demand": round(lost_d, 1),
        "unit_fill_rate": round(ful_d / max(1e-6, tot_d), 4),
        "stockout_days": stk_days,
        "stockout_rate": round(stk_days / max(1, tot_days), 4),
        "average_inventory": round(float(strat_day["closing_inventory"].mean()), 1),
        "total_wasted_units": round(float(strat_day["wastage"].sum()), 1)
    }


def replay_historical_decision(
    decision_date: str,
    store_id: str,
    product_id: str,
    decision_backtest_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Retrieve full judge-facing historical audit comparison for a single decision date."""
    if decision_backtest_df is None:
        dec_path = PROJECT_ROOT / "data" / "backtesting" / "decision_backtest.parquet"
        if dec_path.exists():
            decision_backtest_df = pd.read_parquet(dec_path)
        else:
            dec_df, _ = run_historical_backtest(
                stores=[store_id],
                products=[product_id],
                strategies_to_run=["PHASE20_AI_STRATEGY", "FIXED_14_DAY"]
            )
            decision_backtest_df = dec_df

    # Match row for AI and reference
    sub = decision_backtest_df[
        (decision_backtest_df["decision_date"] == str(decision_date)[:10]) &
        (decision_backtest_df["store_id"] == store_id) &
        (decision_backtest_df["product_id"] == product_id)
    ]

    ai_row = sub[sub["strategy"] == "PHASE20_AI_STRATEGY"]
    ref_row = sub[sub["strategy"] == "FIXED_14_DAY"]

    if len(ai_row) == 0 or len(ref_row) == 0:
        return {"error": f"Decision point not found for {decision_date}, {store_id}, {product_id}"}

    ai_res = ai_row.iloc[0]
    ref_res = ref_row.iloc[0]

    ai_cost = float(ai_res["realised_total_cost"])
    ref_cost = float(ref_res["realised_total_cost"])
    savings = ref_cost - ai_cost
    savings_pct = (savings / ref_cost * 100.0) if ref_cost > 0 else 0.0

    return {
        "metadata": {
            "decision_date": str(decision_date)[:10],
            "store_id": store_id,
            "product_id": product_id,
            "supplier_id": ai_res.get("supplier_id", "SUP_UNKNOWN"),
            "price_model_status": "NOT_VALIDATED",
            "price_forecast_used_in_decision": False
        },
        "information_known_at_T": {
            "current_purchase_price": float(ai_res["realised_purchase_price"]),
            "prediction_source": str(ai_res["prediction_source"]),
            "out_of_fold": bool(ai_res["out_of_fold_flag"]),
            "price_model_status": "NOT_VALIDATED",
            "price_forecast_used_in_decision": False
        },
        "decisions_compared": {
            "ai_recommended_action": "BUY_NOW" if float(ai_res["order_quantity"]) > 0 else "DO_NOT_REORDER",
            "ai_order_quantity": float(ai_res["order_quantity"]),
            "ai_coverage_days": int(ai_res["order_coverage_days"]),
            "reference_action": "BUY_NOW" if float(ref_res["order_quantity"]) > 0 else "DO_NOT_REORDER",
            "reference_order_quantity": float(ref_res["order_quantity"]),
            "reference_coverage_days": int(ref_res["order_coverage_days"])
        },
        "realised_outcomes": {
            "realised_cycle_demand": float(ai_res["realised_demand"]),
            "realised_purchase_price": float(ai_res["realised_purchase_price"]),
            "ai_realised_total_cost": round(ai_cost, 2),
            "reference_realised_total_cost": round(ref_cost, 2),
            "realised_savings_rm": round(savings, 2),
            "realised_savings_pct": round(savings_pct, 2),
            "ai_stockout_occurred": bool(ai_res["stockout_occurred"]),
            "reference_stockout_occurred": bool(ref_res["stockout_occurred"]),
            "ai_lost_units": float(ai_res["lost_units"]),
            "reference_lost_units": float(ref_res["lost_units"]),
            "ai_ending_inventory": float(ai_res["ending_inventory"]),
            "reference_ending_inventory": float(ref_res["ending_inventory"])
        }
    }
