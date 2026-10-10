"""Historical Business Replay and Simulation Engine for Phase 20."""
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import math
import numpy as np
import pandas as pd

from src.utils.paths import PROJECT_ROOT
from src.backtesting.config import load_backtest_config
from src.backtesting.strategies import (
    decide_no_action,
    decide_fixed_coverage,
    decide_reorder_point,
    decide_phase18_optimiser,
    decide_phase20_ai_strategy
)


def run_historical_backtest(
    config: Optional[Dict[str, Any]] = None,
    stores: Optional[List[str]] = None,
    products: Optional[List[str]] = None,
    strategies_to_run: Optional[List[str]] = None,
    checkpoint_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete chronological historical replay across development walk-forward folds.

    Returns:
        Tuple of (decision_backtest_df, daily_inventory_replay_df).
    """
    cfg = config or load_backtest_config()
    cost_cfg = cfg.get("costs", {})
    c_hold = float(cost_cfg.get("inventory_holding_cost_per_unit_per_day", 0.25)) + float(cost_cfg.get("storage_cost_per_unit_per_day", 0.15))
    c_stock = float(cost_cfg.get("stockout_cost_per_unit", 3.50))
    c_order = float(cost_cfg.get("ordering_cost_per_po", 50.00))
    c_waste_default = float(cost_cfg.get("wastage_cost_per_unit_default", 7.20))

    strat_list = strategies_to_run or cfg.get("strategies", {}).get("evaluated", [
        "NO_ACTION", "FIXED_7_DAY", "FIXED_14_DAY", "FIXED_30_DAY",
        "REORDER_POINT", "PHASE18_OPTIMISER", "PHASE20_AI_STRATEGY"
    ])

    # 1. Load Master Table and Predictions
    master_path = PROJECT_ROOT / "data" / "processed" / "master_daily_demand_corrected.parquet"
    master_df = pd.read_parquet(master_path)
    master_df["date"] = pd.to_datetime(master_df["date"])

    dp_path = PROJECT_ROOT / "data" / "predictions" / "final_calibrated_demand_predictions.parquet"
    pp_path = PROJECT_ROOT / "data" / "predictions" / "final_calibrated_price_predictions.parquet"
    dp = pd.read_parquet(dp_path)
    pp = pd.read_parquet(pp_path)

    dp["forecast_origin"] = pd.to_datetime(dp["forecast_origin"]).dt.strftime("%Y-%m-%d")
    pp["forecast_origin"] = pd.to_datetime(pp["forecast_origin"]).dt.strftime("%Y-%m-%d")

    # Filter origins strictly before holdout quarantine (2024-10-01)
    holdout_start = pd.Timestamp(cfg.get("backtest", {}).get("holdout_quarantine_start", "2024-10-01"))
    all_origins = sorted([o for o in dp["forecast_origin"].unique() if pd.to_datetime(o) < holdout_start])

    # Filter target stores and products
    target_stores = stores or sorted(master_df["store_id"].unique())
    target_products = products or sorted(master_df["product_id"].unique())

    decision_records: List[Dict[str, Any]] = []
    daily_records: List[Dict[str, Any]] = []

    # Map product metadata
    prod_meta = master_df.groupby("product_id").first().reset_index()
    prod_meta_dict = {
        row["product_id"]: {
            "moq": float(row.get("minimum_order_quantity", 10.0)),
            "pack_size": int(row.get("supplier_pack_size", 1)),
            "shelf_life": float(row["shelf_life_days"]) if pd.notnull(row.get("shelf_life_days")) and row.get("shelf_life_days") > 0 else None,
            "category": str(row.get("category", "General")),
            "lead_time": float(row.get("expected_lead_time_days", 2.0))
        }
        for _, row in prod_meta.iterrows()
    }

    # 2. Iterate each Store-Product Series
    for s_id in target_stores:
        for p_id in target_products:
            p_info = prod_meta_dict[p_id]
            moq = p_info["moq"]
            pack_size = p_info["pack_size"]
            shelf_life = p_info["shelf_life"]
            exp_lt = p_info["lead_time"]

            # Sub-slice master for this series
            s_mask = (master_df["store_id"] == s_id) & (master_df["product_id"] == p_id)
            series_master = master_df[s_mask].sort_values("date").set_index("date")

            # Initial starting inventory at first origin
            first_origin_dt = pd.to_datetime(all_origins[0])
            init_inv = float(series_master.loc[first_origin_dt, "opening_inventory"]) if first_origin_dt in series_master.index else 50.0

            # State tracking per strategy
            strat_states = {}
            for strat in strat_list:
                strat_states[strat] = {
                    "current_stock": init_inv,
                    "batches": [[init_inv, int(round(shelf_life))] if shelf_life else [init_inv, 999]],
                    "pending_orders": []  # List of dict: {arrival_date, quantity, unit_price}
                }

            # 3. Simulate through the 33 Walk-Forward Cycles
            for k, origin_str in enumerate(all_origins):
                origin_dt = pd.to_datetime(origin_str)
                # Next origin determines cycle length (approx 30 days)
                if k + 1 < len(all_origins):
                    next_origin_dt = pd.to_datetime(all_origins[k + 1])
                else:
                    next_origin_dt = origin_dt + pd.Timedelta(days=30)

                # Extract forecasts at this origin
                d_sub = dp[(dp["forecast_origin"] == origin_str) & (dp["store_id"] == s_id) & (dp["product_id"] == p_id)]
                p_sub = pp[(pp["forecast_origin"] == origin_str) & (pp["store_id"] == s_id) & (pp["product_id"] == p_id)]

                if len(d_sub) == 0 or len(p_sub) == 0:
                    continue

                d_pts = dict(zip(d_sub["horizon"], d_sub["point_prediction"]))
                d_p10 = dict(zip(d_sub["horizon"], d_sub["calibrated_p10"]))
                d_p50 = dict(zip(d_sub["horizon"], d_sub["calibrated_p50"]))
                d_p90 = dict(zip(d_sub["horizon"], d_sub["calibrated_p90"]))

                p_pts = dict(zip(p_sub["horizon"], p_sub["point_prediction"]))
                p_p10 = dict(zip(p_sub["horizon"], p_sub["calibrated_p10"]))
                p_p50 = dict(zip(p_sub["horizon"], p_sub["calibrated_p50"]))
                p_p90 = dict(zip(p_sub["horizon"], p_sub["calibrated_p90"]))

                curr_purchase_price = float(p_sub["current_purchase_price"].iloc[0])
                supplier_id = str(p_sub["supplier_id"].iloc[0])

                # Historical rolling mean demand prior to origin
                hist_slice = series_master.loc[:origin_dt]
                mean_daily_d = float(hist_slice["estimated_true_demand_training_safe"].tail(30).mean()) if len(hist_slice) > 0 else 25.0
                mean_daily_d = max(1.0, mean_daily_d)

                # Realised supplier lead time for this specific cycle
                actual_lt_days = int(round(float(series_master.loc[origin_dt, "actual_lead_time_days"]))) if (origin_dt in series_master.index and pd.notnull(series_master.loc[origin_dt, "actual_lead_time_days"])) else int(round(exp_lt))
                actual_lt_days = max(0, min(10, actual_lt_days))

                # Realised purchase price on order date
                actual_price = float(series_master.loc[origin_dt, "purchase_price"]) if origin_dt in series_master.index else curr_purchase_price

                # A. Make Decisions for Each Strategy at Origin T
                cycle_decisions = {}
                for strat in strat_list:
                    curr_stock = strat_states[strat]["current_stock"]

                    if strat == "NO_ACTION":
                        dec = decide_no_action()
                    elif strat == "FIXED_7_DAY":
                        dec = decide_fixed_coverage(7, curr_stock, d_pts, moq, pack_size)
                    elif strat == "FIXED_14_DAY":
                        dec = decide_fixed_coverage(14, curr_stock, d_pts, moq, pack_size)
                    elif strat == "FIXED_30_DAY":
                        dec = decide_fixed_coverage(30, curr_stock, d_pts, moq, pack_size)
                    elif strat == "REORDER_POINT":
                        dec = decide_reorder_point(curr_stock, mean_daily_d, exp_lt, moq, pack_size)
                    elif strat == "PHASE18_OPTIMISER":
                        dec = decide_phase18_optimiser(
                            d_pts, p_pts, d_p10, d_p50, d_p90, p_p10, p_p50, p_p90,
                            curr_stock, curr_purchase_price, moq, pack_size, shelf_life,
                            exp_lt, s_id, p_id, origin_str
                        )
                    elif strat == "PHASE20_AI_STRATEGY":
                        dec = decide_phase20_ai_strategy(
                            d_pts, p_pts, d_p10, d_p50, d_p90, p_p10, p_p50, p_p90,
                            curr_stock, curr_purchase_price, moq, pack_size, shelf_life,
                            exp_lt, s_id, p_id, origin_str, n_scenarios=200
                        )
                    else:
                        dec = decide_no_action()

                    order_q = float(dec["order_quantity"])
                    cov_days = int(dec.get("coverage_days", 0))

                    # If order placed, register pending arrival at origin_dt + actual_lt_days
                    if order_q > 0:
                        arr_dt = origin_dt + pd.Timedelta(days=actual_lt_days)
                        strat_states[strat]["pending_orders"].append({
                            "arrival_date": arr_dt,
                            "quantity": order_q,
                            "unit_price": actual_price
                        })
                        po_order_cost = c_order
                        purch_cost = order_q * actual_price
                    else:
                        po_order_cost = 0.0
                        purch_cost = 0.0

                    cycle_decisions[strat] = {
                        "decision_date": origin_str,
                        "store_id": s_id,
                        "product_id": p_id,
                        "supplier_id": supplier_id,
                        "strategy": strat,
                        "order_quantity": order_q,
                        "order_coverage_days": cov_days,
                        "realised_purchase_price": round(actual_price, 2),
                        "realised_purchase_cost": round(purch_cost, 2),
                        "ordering_cost": round(po_order_cost, 2),
                        "prediction_source": "OUT_OF_FOLD",
                        "out_of_fold_flag": True,
                        "price_model_status": "NOT_VALIDATED",
                        "price_forecast_used_in_decision": False
                    }

                # B. Advance Daily Time Step-by-Step across Cycle [origin_dt, next_origin_dt)
                curr_date = origin_dt
                cycle_realised_demand = 0.0

                while curr_date < next_origin_dt:
                    # True demand on this day
                    if curr_date in series_master.index:
                        d_true = float(series_master.loc[curr_date, "estimated_true_demand_training_safe"])
                    else:
                        d_true = mean_daily_d

                    d_true = max(0.0, d_true)
                    cycle_realised_demand += d_true

                    # Advance each strategy for this day
                    for strat in strat_list:
                        st = strat_states[strat]
                        day_open_inv = st["current_stock"]

                        # Check arrivals today
                        arrivals_today = 0.0
                        surv_orders = []
                        for po in st["pending_orders"]:
                            if po["arrival_date"] <= curr_date:
                                arrivals_today += po["quantity"]
                                # Add to FIFO batches
                                st["batches"].append([po["quantity"], int(round(shelf_life)) if shelf_life else 999])
                            else:
                                surv_orders.append(po)
                        st["pending_orders"] = surv_orders

                        avail_stock = day_open_inv + arrivals_today

                        # Consume demand FIFO
                        rem_demand = d_true
                        for b in st["batches"]:
                            take = min(b[0], rem_demand)
                            b[0] -= take
                            rem_demand -= take
                            if rem_demand <= 0:
                                break

                        fulfilled_d = d_true - rem_demand
                        lost_d = rem_demand

                        # Age batches and calculate perishable spoilage
                        expired_units = 0.0
                        surv_batches = []
                        for b in st["batches"]:
                            b[1] -= 1  # Age 1 day
                            if b[1] <= 0:
                                expired_units += b[0]
                            elif b[0] > 0:
                                surv_batches.append(b)
                        st["batches"] = surv_batches

                        # Closing inventory
                        closing_stock = sum(b[0] for b in st["batches"])
                        st["current_stock"] = closing_stock

                        # Daily financial costs
                        day_holding = closing_stock * c_hold
                        day_stockout = lost_d * c_stock
                        unit_waste_cost = actual_price if actual_price > 0 else c_waste_default
                        day_wastage = expired_units * unit_waste_cost

                        daily_records.append({
                            "date": curr_date.strftime("%Y-%m-%d"),
                            "store_id": s_id,
                            "product_id": p_id,
                            "strategy": strat,
                            "opening_inventory": round(day_open_inv, 2),
                            "deliveries": round(arrivals_today, 2),
                            "demand": round(d_true, 2),
                            "fulfilled_demand": round(fulfilled_d, 2),
                            "lost_demand": round(lost_d, 2),
                            "closing_inventory": round(closing_stock, 2),
                            "wastage": round(expired_units, 2),
                            "holding_cost": round(day_holding, 2),
                            "stockout_cost": round(day_stockout, 2),
                            "wastage_cost": round(day_wastage, 2)
                        })

                    curr_date += pd.Timedelta(days=1)

                # C. Finalize Decision Records for this Cycle
                for strat in strat_list:
                    d_rec = cycle_decisions[strat]
                    d_rec["realised_demand"] = round(cycle_realised_demand, 1)
                    # Accumulate cycle holding, stockout, wastage for decision record
                    recent_daily = [r for r in daily_records[-len(strat_list)*int((next_origin_dt - origin_dt).days):] if r["strategy"] == strat]
                    cycle_hold = sum(r["holding_cost"] for r in recent_daily)
                    cycle_stockout = sum(r["stockout_cost"] for r in recent_daily)
                    cycle_waste = sum(r["wastage_cost"] for r in recent_daily)
                    cycle_lost = sum(r["lost_demand"] for r in recent_daily)

                    d_rec["realised_holding_cost"] = round(cycle_hold, 2)
                    d_rec["realised_stockout_cost"] = round(cycle_stockout, 2)
                    d_rec["realised_wastage_cost"] = round(cycle_waste, 2)
                    d_rec["realised_total_cost"] = round(d_rec["realised_purchase_cost"] + d_rec["ordering_cost"] + cycle_hold + cycle_stockout + cycle_waste, 2)
                    d_rec["stockout_occurred"] = bool(cycle_lost > 0)
                    d_rec["lost_units"] = round(cycle_lost, 1)
                    d_rec["ending_inventory"] = round(strat_states[strat]["current_stock"], 2)

                    decision_records.append(d_rec)

    decision_df = pd.DataFrame(decision_records)
    daily_df = pd.DataFrame(daily_records)

    return decision_df, daily_df
