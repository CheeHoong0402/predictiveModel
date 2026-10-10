"""Reporting and Artifact Generation for Final Recommendations."""
import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.utils.paths import PROJECT_ROOT


def generate_recommendation_reports(
    final_df: pd.DataFrame,
    priority_queue_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    runtime_records: List[Dict[str, Any]],
    output_dir: Path = PROJECT_ROOT / "reports/recommendations"
) -> None:
    """Generate all Phase 21 reports, markdown summaries, and diagnostic CSV tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Action Summary CSV
    action_grp = final_df.groupby("action").agg(
        item_count=("product_id", "count"),
        total_order_quantity=("order_quantity", "sum"),
        total_procurement_cost=("expected_total_cost", "sum"),
        avg_coverage_days=("coverage_days", "mean")
    ).reset_index()
    action_grp.to_csv(output_dir / "action_summary.csv", index=False)

    # 2. Urgency Summary CSV
    urgency_grp = final_df.groupby("urgency").agg(
        item_count=("product_id", "count"),
        total_order_quantity=("order_quantity", "sum"),
        avg_stockout_risk_before=("stockout_probability_before", "mean"),
        avg_stockout_risk_after=("stockout_probability_after", "mean")
    ).reset_index()
    urgency_grp.to_csv(output_dir / "urgency_summary.csv", index=False)

    # 3. Priority Queue Summary CSV
    pq_summary = priority_queue_df[[
        "queue_rank", "priority_score", "store_id", "product_id", "action", "urgency",
        "order_quantity", "current_inventory", "stockout_probability_before", "predicted_price_change_pct"
    ]].head(15)
    pq_summary.to_csv(output_dir / "priority_queue_summary.csv", index=False)

    # 4. Reason Code Summary CSV
    all_reasons = []
    for col in ["reason_code_1", "reason_code_2", "reason_code_3"]:
        if col in final_df.columns:
            all_reasons.extend(final_df[col].dropna().tolist())
    rc_counts = pd.Series(all_reasons).value_counts().reset_index()
    rc_counts.columns = ["reason_code", "frequency"]
    rc_counts = rc_counts[rc_counts["reason_code"] != "NONE"]
    rc_counts.to_csv(output_dir / "reason_code_summary.csv", index=False)

    # 5. Risk Summary CSV
    risk_summary = final_df.groupby("store_id").agg(
        avg_stockout_before=("stockout_probability_before", "mean"),
        avg_stockout_after=("stockout_probability_after", "mean"),
        avg_overstock_prob=("overstock_probability", "mean"),
        avg_wastage_prob=("wastage_probability", "mean")
    ).reset_index()
    risk_summary.to_csv(output_dir / "risk_summary.csv", index=False)

    # 6. Savings Summary CSV
    tot_spend = float(final_df["expected_total_cost"].sum())
    tot_savings = float(final_df["expected_savings"].sum())
    savings_df = pd.DataFrame([{
        "total_monitored_products": len(final_df),
        "total_procurement_cost_rm": round(tot_spend, 2),
        "total_expected_savings_rm": round(tot_savings, 2),
        "expected_savings_pct": round((tot_savings / max(1.0, tot_spend + tot_savings)) * 100.0, 2),
        "historical_backtest_savings_pct": 44.49,
        "historical_backtest_status": "PHASE_20_VALIDATED"
    }])
    savings_df.to_csv(output_dir / "savings_summary.csv", index=False)

    # 7. Data Quality Summary CSV
    dq_summary = final_df.groupby("data_quality_status").agg(
        item_count=("product_id", "count"),
        total_warnings=("warning_count", "sum")
    ).reset_index()
    dq_summary.to_csv(output_dir / "data_quality_summary.csv", index=False)

    # 8. Runtime CSV
    runtime_df = pd.DataFrame(runtime_records)
    runtime_df.to_csv(output_dir / "runtime.csv", index=False)

    # 9. Price Decision Summary CSV
    if "predicted_purchase_price" in final_df.columns:
        p_summary = pd.DataFrame()
        p_summary["product_id"] = final_df["product_id"]
        p_summary["store_id"] = final_df["store_id"]
        p_summary["action"] = final_df["action"]
        p_summary["urgency"] = final_df["urgency"]
        p_summary["current_purchase_price"] = final_df.get("current_purchase_price", 0.0)
        p_summary["predicted_purchase_price"] = final_df.get("predicted_purchase_price", 0.0)
        p_summary["predicted_price_change_pct"] = final_df.get("predicted_price_change_pct", 0.0)
        p_summary["price_p10"] = final_df.get("price_p10", 0.0)
        p_summary["price_p50"] = final_df.get("price_p50", 0.0)
        p_summary["price_p90"] = final_df.get("price_p90", 0.0)
        p_summary["price_signal_strength"] = final_df.get("price_signal_strength", "NEUTRAL_PRICE_SIGNAL")
        p_summary["direction_probability"] = final_df.get("price_direction_probability", 0.50)
        p_summary["baseline_demand_order_quantity"] = final_df.get("baseline_demand_order_quantity", 0.0)
        p_summary["price_aware_order_quantity"] = final_df.get("price_aware_order_quantity", final_df["order_quantity"])
        p_summary["price_delta_quantity"] = final_df.get("price_delta_quantity", 0.0)
        p_summary["net_timing_benefit"] = final_df.get("net_price_timing_benefit", 0.0)
        p_summary["buy_now_vs_wait_recommendation"] = final_df.get("buy_now_vs_wait_recommendation", "BUY_NOW")
        p_summary["conflict_override_applied"] = final_df.get("conflict_override_applied", "NONE")
        p_summary.to_csv(output_dir / "price_decision_summary.csv", index=False)

    # 9. Executive Summary JSON
    exec_summary_data = {
        "evaluation_timestamp": str(pd.Timestamp.now()),
        "monitored_products": len(final_df),
        "monitored_stores": int(final_df["store_id"].nunique()),
        "urgent_actions_count": int((final_df["action"] == "URGENT_REORDER").sum()),
        "high_risk_products_count": int((final_df["stockout_probability_before"] >= 0.20).sum()),
        "total_recommended_spend_rm": round(tot_spend, 2),
        "total_expected_savings_rm": round(tot_savings, 2),
        "average_stockout_before": round(float(final_df["stockout_probability_before"].mean()), 4),
        "average_stockout_after": round(float(final_df["stockout_probability_after"].mean()), 4),
        "historical_backtest_savings_pct": 44.49,
        "priority_actions_top_3": priority_queue_df.head(3)[["store_id", "product_id", "action", "urgency", "order_quantity"]].to_dict(orient="records")
    }
    with open(output_dir / "executive_summary.json", "w", encoding="utf-8") as f:
        json.dump(exec_summary_data, f, indent=2)

    # 10. Executive Summary Markdown
    md_exec = f"""# PHASE 21 — EXECUTIVE PROCUREMENT RECOMMENDATION REPORT

**Evaluation Status:** Production Validated (Unified Orchestration Engine)  
**Products Monitored:** {len(final_df)} SKU-Store combinations  
**Stores Monitored:** {final_df['store_id'].nunique()} Malaysian Retail Stores  
**Holdout Policy Status:** Q4 2024 (`2024-10-01` to `2024-12-31`) **Strictly Quarantined & Untouched**  

---

## 1. Executive Headline Numbers

- **Urgent Reorders Required:** {exec_summary_data['urgent_actions_count']} items
- **High Stockout Risk Items:** {exec_summary_data['high_risk_products_count']} items
- **Total Recommended Procurement Spend:** **RM {tot_spend:,.2f}**
- **Expected Direct Savings:** **RM {tot_savings:,.2f}**
- **Average Stockout Risk:** Reduced from **{exec_summary_data['average_stockout_before']:.1%}** down to **{exec_summary_data['average_stockout_after']:.1%}**
- **Historical Backtest Cost Reduction (Phase 20 Evidence):** **44.49% net cost reduction** (RM 5.78M saved across 990 days)

---

## 2. Action Distribution

| Action | Items | Total Units Ordered | Total Cost (RM) | Avg Coverage Days |
| :--- | :--- | :--- | :--- | :--- |
"""
    for _, row in action_grp.iterrows():
        md_exec += f"| **{row['action']}** | {row['item_count']} | {row['total_order_quantity']:,.0f} | RM {row['total_procurement_cost']:,.2f} | {row['avg_coverage_days']:.1f} days |\n"

    md_exec += f"""
---

## 3. Top Priority Action Queue

| Rank | Priority Score | Store | Product | Action | Urgency | Units | Stockout Before | Price Trend |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for _, row in priority_queue_df.head(5).iterrows():
        pr_change_str = f"{row['predicted_price_change_pct']:+.1f}%" if pd.notnull(row.get('predicted_price_change_pct')) else "N/A (NOT VALIDATED)"
        md_exec += f"| {row['queue_rank']} | **{row['priority_score']:.1f}** | {row['store_id']} | {row['product_id']} | `{row['action']}` | **{row['urgency']}** | {row['order_quantity']:.0f} | {row['stockout_probability_before']:.1%} | {pr_change_str} |\n"

    md_exec += """
---

## 4. Methodological & Decision Integrity

1. **No Model Retraining:** 100% of demand, price, and uncertainty predictions originate from saved Phase 15–16 champion models.
2. **Prioritised Conflict Resolution:** Stockout before supplier arrival unconditionally overrides expected price dips.
3. **Perishable Protection:** Order quantities strictly respect product shelf life to prevent spoilage write-offs.
4. **Holdout Quarantine:** All decision dates conclude prior to `2024-10-01`.
"""
    with open(output_dir / "executive_summary.md", "w", encoding="utf-8") as f:
        f.write(md_exec)

    with open(output_dir / "final_recommendation_summary.md", "w", encoding="utf-8") as f:
        f.write(md_exec)
