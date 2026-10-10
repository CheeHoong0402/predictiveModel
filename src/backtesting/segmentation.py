"""Performance segmentation and diagnostic analysis for Phase 20."""
from typing import Dict, Any, List
import pandas as pd
import numpy as np


def compute_store_segmentation(
    daily_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    ref_strategy: str = "FIXED_14_DAY"
) -> pd.DataFrame:
    """Calculate performance metrics broken down by store."""
    records = []
    stores = sorted(daily_df["store_id"].unique())

    for s_id in stores:
        d_sub = daily_df[daily_df["store_id"] == s_id]
        dec_sub = decision_df[decision_df["store_id"] == s_id]

        for strat in ["PHASE20_AI_STRATEGY", ref_strategy]:
            d_strat = d_sub[d_sub["strategy"] == strat]
            dec_strat = dec_sub[dec_sub["strategy"] == strat]

            tot_days = len(d_strat)
            tot_d = float(d_strat["demand"].sum())
            ful_d = float(d_strat["fulfilled_demand"].sum())
            lost_d = float(d_strat["lost_demand"].sum())
            stk_days = int((d_strat["lost_demand"] > 0).sum())

            p_cost = float(dec_strat["realised_purchase_cost"].sum())
            ord_cost = float(dec_strat["ordering_cost"].sum())
            h_cost = float(d_strat["holding_cost"].sum())
            s_cost = float(d_strat["stockout_cost"].sum())
            w_cost = float(d_strat["wastage_cost"].sum())
            tot_cost = p_cost + ord_cost + h_cost + s_cost + w_cost

            records.append({
                "store_id": s_id,
                "strategy": strat,
                "total_cost": round(tot_cost, 2),
                "purchase_cost": round(p_cost, 2),
                "holding_cost": round(h_cost, 2),
                "stockout_cost": round(s_cost, 2),
                "wastage_cost": round(w_cost, 2),
                "stockout_rate": round(stk_days / max(1, tot_days), 4),
                "fill_rate": round(ful_d / max(1e-6, tot_d), 4),
                "average_inventory": round(float(d_strat["closing_inventory"].mean()), 1),
                "lost_units": round(lost_d, 1)
            })

    df = pd.DataFrame(records)
    return df


def compute_category_segmentation(
    daily_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    master_df: pd.DataFrame,
    ref_strategy: str = "FIXED_14_DAY"
) -> pd.DataFrame:
    """Calculate performance metrics broken down by product category."""
    cat_map = master_df.groupby("product_id")["category"].first().to_dict()
    d_df = daily_df.copy()
    d_df["category"] = d_df["product_id"].map(cat_map)

    dec_df = decision_df.copy()
    dec_df["category"] = dec_df["product_id"].map(cat_map)

    categories = sorted(d_df["category"].unique())
    records = []

    for cat in categories:
        d_cat = d_df[d_df["category"] == cat]
        dec_cat = dec_df[dec_df["category"] == cat]

        ai_d = d_cat[d_cat["strategy"] == "PHASE20_AI_STRATEGY"]
        ref_d = d_cat[d_cat["strategy"] == ref_strategy]

        ai_dec = dec_cat[dec_cat["strategy"] == "PHASE20_AI_STRATEGY"]
        ref_dec = dec_cat[dec_cat["strategy"] == ref_strategy]

        ai_cost = float(ai_dec["realised_purchase_cost"].sum() + ai_dec["ordering_cost"].sum() + ai_d["holding_cost"].sum() + ai_d["stockout_cost"].sum() + ai_d["wastage_cost"].sum())
        ref_cost = float(ref_dec["realised_purchase_cost"].sum() + ref_dec["ordering_cost"].sum() + ref_d["holding_cost"].sum() + ref_d["stockout_cost"].sum() + ref_d["wastage_cost"].sum())

        savings_pct = ((ref_cost - ai_cost) / ref_cost * 100.0) if ref_cost > 0 else 0.0

        ai_fill = float(ai_d["fulfilled_demand"].sum() / max(1e-6, ai_d["demand"].sum()))
        ref_fill = float(ref_d["fulfilled_demand"].sum() / max(1e-6, ref_d["demand"].sum()))

        records.append({
            "category": cat,
            "ai_total_cost": round(ai_cost, 2),
            "reference_total_cost": round(ref_cost, 2),
            "absolute_savings": round(ref_cost - ai_cost, 2),
            "savings_percentage": round(savings_pct, 2),
            "ai_fill_rate": round(ai_fill, 4),
            "reference_fill_rate": round(ref_fill, 4),
            "fill_rate_improvement_pp": round((ai_fill - ref_fill) * 100.0, 2),
            "ai_wasted_units": round(float(ai_d["wastage"].sum()), 1),
            "reference_wasted_units": round(float(ref_d["wastage"].sum()), 1)
        })

    return pd.DataFrame(records)


def compute_environmental_segmentation(
    daily_df: pd.DataFrame,
    master_df: pd.DataFrame,
    env_col: str,
    env_name: str,
    ref_strategy: str = "FIXED_14_DAY"
) -> pd.DataFrame:
    """Segment performance during active environmental events (promotions, festivals, etc.)."""
    # Merge event flag
    m_sub = master_df[["date", "store_id", "product_id", env_col]].drop_duplicates()
    m_sub["date"] = pd.to_datetime(m_sub["date"]).dt.strftime("%Y-%m-%d")

    merged = pd.merge(daily_df, m_sub, on=["date", "store_id", "product_id"], how="left")
    merged[env_col] = merged[env_col].fillna(0).astype(int)

    records = []
    for flag_val in [1, 0]:
        period_label = f"{env_name}_ACTIVE" if flag_val == 1 else "NORMAL_PERIOD"
        sub = merged[merged[env_col] == flag_val]

        ai_d = sub[sub["strategy"] == "PHASE20_AI_STRATEGY"]
        ref_d = sub[sub["strategy"] == ref_strategy]

        ai_stk = float((ai_d["lost_demand"] > 0).mean()) if len(ai_d) > 0 else 0.0
        ref_stk = float((ref_d["lost_demand"] > 0).mean()) if len(ref_d) > 0 else 0.0

        ai_fill = float(ai_d["fulfilled_demand"].sum() / max(1e-6, ai_d["demand"].sum())) if len(ai_d) > 0 else 0.0
        ref_fill = float(ref_d["fulfilled_demand"].sum() / max(1e-6, ref_d["demand"].sum())) if len(ref_d) > 0 else 0.0

        ai_cost = float(ai_d["holding_cost"].sum() + ai_d["stockout_cost"].sum() + ai_d["wastage_cost"].sum())
        ref_cost = float(ref_d["holding_cost"].sum() + ref_d["stockout_cost"].sum() + ref_d["wastage_cost"].sum())
        savings_pct = ((ref_cost - ai_cost) / ref_cost * 100.0) if ref_cost > 0 else 0.0

        records.append({
            "segment": period_label,
            "days_evaluated": len(ai_d),
            "ai_stockout_rate": round(ai_stk, 4),
            "reference_stockout_rate": round(ref_stk, 4),
            "stockout_reduction_pp": round((ref_stk - ai_stk) * 100.0, 2),
            "ai_fill_rate": round(ai_fill, 4),
            "reference_fill_rate": round(ref_fill, 4),
            "savings_percentage": round(savings_pct, 2)
        })

    return pd.DataFrame(records)


def compute_phase18_vs_phase19_comparison(
    daily_df: pd.DataFrame,
    decision_df: pd.DataFrame
) -> pd.DataFrame:
    """Compare Phase 18 expected-cost minimizer alone vs Phase 19 risk-adjusted strategy."""
    records = []
    for strat in ["PHASE18_OPTIMISER", "PHASE20_AI_STRATEGY"]:
        d_sub = daily_df[daily_df["strategy"] == strat]
        dec_sub = decision_df[decision_df["strategy"] == strat]

        tot_cost = float(dec_sub["realised_purchase_cost"].sum() + dec_sub["ordering_cost"].sum() + d_sub["holding_cost"].sum() + d_sub["stockout_cost"].sum() + d_sub["wastage_cost"].sum())
        stk_rate = float((d_sub["lost_demand"] > 0).mean())
        fill_rate = float(d_sub["fulfilled_demand"].sum() / max(1e-6, d_sub["demand"].sum()))
        lost_u = float(d_sub["lost_demand"].sum())
        waste_u = float(d_sub["wastage"].sum())

        records.append({
            "strategy": strat,
            "total_realised_cost": round(tot_cost, 2),
            "stockout_rate": round(stk_rate, 4),
            "unit_fill_rate": round(fill_rate, 4),
            "lost_units": round(lost_u, 1),
            "wasted_units": round(waste_u, 1),
            "average_inventory": round(float(d_sub["closing_inventory"].mean()), 1)
        })

    return pd.DataFrame(records)


def compute_cost_sensitivity_table(
    daily_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    ref_strategy: str = "FIXED_14_DAY"
) -> pd.DataFrame:
    """Evaluate headline savings under +/- 20% variations in cost parameters."""
    variations = [
        ("Baseline (Standard Parameters)", 1.0, 1.0, 1.0),
        ("Holding Cost -20%", 0.8, 1.0, 1.0),
        ("Holding Cost +20%", 1.2, 1.0, 1.0),
        ("Stockout Cost -20%", 1.0, 0.8, 1.0),
        ("Stockout Cost +20%", 1.0, 1.2, 1.0),
        ("Wastage Cost -20%", 1.0, 1.0, 0.8),
        ("Wastage Cost +20%", 1.0, 1.0, 1.2),
        ("Severe Shock (Hold +20%, Stockout +20%)", 1.2, 1.2, 1.0)
    ]
    records = []

    ai_d = daily_df[daily_df["strategy"] == "PHASE20_AI_STRATEGY"]
    ref_d = daily_df[daily_df["strategy"] == ref_strategy]
    ai_dec = decision_df[decision_df["strategy"] == "PHASE20_AI_STRATEGY"]
    ref_dec = decision_df[decision_df["strategy"] == ref_strategy]

    base_ai_purch = float(ai_dec["realised_purchase_cost"].sum() + ai_dec["ordering_cost"].sum())
    base_ref_purch = float(ref_dec["realised_purchase_cost"].sum() + ref_dec["ordering_cost"].sum())

    base_ai_hold = float(ai_d["holding_cost"].sum())
    base_ref_hold = float(ref_d["holding_cost"].sum())

    base_ai_stock = float(ai_d["stockout_cost"].sum())
    base_ref_stock = float(ref_d["stockout_cost"].sum())

    base_ai_waste = float(ai_d["wastage_cost"].sum())
    base_ref_waste = float(ref_d["wastage_cost"].sum())

    for label, h_mult, s_mult, w_mult in variations:
        ai_tot = base_ai_purch + (base_ai_hold * h_mult) + (base_ai_stock * s_mult) + (base_ai_waste * w_mult)
        ref_tot = base_ref_purch + (base_ref_hold * h_mult) + (base_ref_stock * s_mult) + (base_ref_waste * w_mult)
        sav_pct = ((ref_tot - ai_tot) / ref_tot * 100.0) if ref_tot > 0 else 0.0

        records.append({
            "scenario": label,
            "ai_total_cost": round(ai_tot, 2),
            "reference_total_cost": round(ref_tot, 2),
            "absolute_savings": round(ref_tot - ai_tot, 2),
            "savings_percentage": round(sav_pct, 2)
        })

    return pd.DataFrame(records)
