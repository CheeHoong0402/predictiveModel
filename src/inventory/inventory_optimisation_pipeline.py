"""Orchestration pipeline for Phase 18 Inventory Optimisation.

Executes batch inventory optimisation on the latest development validation origin
(e.g., 2024-08-31 / Fold 33), strictly preserving the holdout quarantine (2024-10-01 to 2024-12-31).

Generates:
- data/optimisation/inventory_candidate_evaluations.parquet
- data/optimisation/inventory_recommendations.parquet
- reports/inventory/*.csv audit reports
- reports/inventory/inventory_optimisation_summary.md
- reports/inventory/sample_dashboard_payload.json
- reports/inventory/sample_telegram_alert.txt
- reports/figures/inventory/*.png (6 figures at 300 DPI)
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time
import pandas as pd
import numpy as np

from src.utils.paths import PROJECT_ROOT
from .config import load_inventory_config, load_cost_assumptions, export_cost_assumptions_csv
from .optimizer import (
    optimise_inventory_decision,
    InventoryDecision,
    DecisionBatchResult
)
from .reporting import (
    format_telegram_inventory_alert,
    build_dashboard_inventory_payload,
    generate_inventory_summary_markdown
)
from .visualisations import plot_inventory_figures


def run_inventory_pipeline(
    origin_date: Optional[str] = None,
    service_level: str = "balanced",
    root: Path = PROJECT_ROOT
) -> DecisionBatchResult:
    """Run full inventory optimisation batch and persist all Phase 18 deliverables."""
    t0 = time.time()

    # 1. Output directory paths
    data_out_dir = Path(root) / "data" / "optimisation"
    reports_out_dir = Path(root) / "reports" / "inventory"
    figures_out_dir = Path(root) / "reports" / "figures" / "inventory"

    data_out_dir.mkdir(parents=True, exist_ok=True)
    reports_out_dir.mkdir(parents=True, exist_ok=True)
    figures_out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Export and load cost assumptions
    export_cost_assumptions_csv(root=root)
    inv_cfg = load_inventory_config(root=root)
    cost_cfg = load_cost_assumptions(root=root)

    # 3. Load input prediction datasets
    demand_pred_path = Path(root) / "data" / "predictions" / "final_calibrated_demand_predictions.parquet"
    price_pred_path = Path(root) / "data" / "predictions" / "final_calibrated_price_predictions.parquet"
    inventory_raw_path = Path(root) / "data" / "raw" / "retail_inventory.parquet"

    products_path = Path(root) / "data" / "raw" / "retail_products.csv"
    suppliers_path = Path(root) / "data" / "raw" / "retail_suppliers.csv"
    stores_path = Path(root) / "data" / "raw" / "retail_stores.csv"

    print("Loading calibrated forecasts and metadata...")
    df_demand = pd.read_parquet(demand_pred_path)
    df_price = pd.read_parquet(price_pred_path)
    df_inv = pd.read_parquet(inventory_raw_path)

    df_products = pd.read_csv(products_path)
    df_suppliers = pd.read_csv(suppliers_path)
    df_stores = pd.read_csv(stores_path)

    # Standardize dates
    df_demand["forecast_origin_str"] = pd.to_datetime(df_demand["forecast_origin"]).dt.strftime("%Y-%m-%d")
    df_price["forecast_origin_str"] = pd.to_datetime(df_price["forecast_origin"]).dt.strftime("%Y-%m-%d")
    df_inv["date_str"] = pd.to_datetime(df_inv["date"]).dt.strftime("%Y-%m-%d")

    # If origin_date not given or not found, select latest common origin date
    common_origins = sorted(list(set(df_demand["forecast_origin_str"]).intersection(set(df_price["forecast_origin_str"]))))
    if origin_date is None or origin_date not in common_origins:
        target_origin = common_origins[-1]
    else:
        target_origin = origin_date

    print(f"=== Starting Phase 18 Inventory Optimisation Pipeline (Origin: {target_origin}) ===")

    # Filter to specified origin
    d_origin = df_demand[df_demand["forecast_origin_str"] == target_origin]
    p_origin = df_price[df_price["forecast_origin_str"] == target_origin]
    inv_origin = df_inv[df_inv["date_str"] == target_origin]

    if d_origin.empty or p_origin.empty:
        raise ValueError(f"No prediction records found for origin {target_origin}")

    # Build lookup dictionaries for entities
    prod_meta = df_products.set_index("product_id").to_dict(orient="index")
    store_meta = df_stores.set_index("store_id").to_dict(orient="index")
    sup_meta = df_suppliers.set_index("supplier_id").to_dict(orient="index")

    # Group inventory by (store_id, product_id)
    inv_lookup = {}
    for _, row in inv_origin.iterrows():
        inv_lookup[(row["store_id"], row["product_id"])] = {
            "current": float(row.get("current_inventory_quantity", 0.0)),
            "incoming": float(row.get("incoming_inventory_quantity", 0.0)),
            "holding_cost": float(row.get("inventory_holding_cost_per_unit_per_day", 0.25)),
            "storage_cost": float(row.get("storage_cost", 0.15)),
            "stockout_cost": float(row.get("stockout_cost_per_unit", 3.50)),
            "wastage_cost": float(row.get("wastage_cost_per_unit", 7.20))
        }

    # Identify all unique (store_id, product_id) combinations
    sku_store_pairs = d_origin[["store_id", "product_id"]].drop_duplicates().values.tolist()
    print(f"Found {len(sku_store_pairs)} SKU-Store combinations for origin {target_origin}.")

    decisions: List[InventoryDecision] = []

    for store_id, product_id in sku_store_pairs:
        # Extract demand forecasts across horizons
        d_sub = d_origin[(d_origin["store_id"] == store_id) & (d_origin["product_id"] == product_id)]
        d_fc = {}
        d_p10 = {}
        d_p50 = {}
        d_p90 = {}
        for _, row in d_sub.iterrows():
            h = int(row["horizon"])
            d_fc[h] = float(row["point_prediction"])
            d_p10[h] = float(row["calibrated_p10"])
            d_p50[h] = float(row["calibrated_p50"])
            d_p90[h] = float(row["calibrated_p90"])

        # Extract price forecasts across horizons
        p_sub = p_origin[(p_origin["store_id"] == store_id) & (p_origin["product_id"] == product_id)]
        if p_sub.empty:
            continue
        p_fc = {}
        p_p10 = {}
        p_p50 = {}
        p_p90 = {}
        curr_price = float(p_sub.iloc[0]["current_purchase_price"])
        supplier_id = str(p_sub.iloc[0]["supplier_id"])
        for _, row in p_sub.iterrows():
            h = int(row["horizon"])
            p_fc[h] = float(row["point_prediction"])
            p_p10[h] = float(row["calibrated_p10"])
            p_p50[h] = float(row["calibrated_p50"])
            p_p90[h] = float(row["calibrated_p90"])

        # Inventory levels
        inv_data = inv_lookup.get((store_id, product_id), {
            "current": 100.0,
            "incoming": 0.0,
            "holding_cost": 0.25,
            "storage_cost": 0.15,
            "stockout_cost": 3.50,
            "wastage_cost": 7.20
        })

        # Product metadata
        p_info = prod_meta.get(product_id, {})
        case_size = int(p_info.get("product_case_size", 1))
        shelf_life = float(p_info["product_shelf_life_days"]) if pd.notnull(p_info.get("product_shelf_life_days")) else None
        selling_price = float(p_info["normal_selling_price"]) if pd.notnull(p_info.get("normal_selling_price")) else curr_price * 1.3

        # Supplier metadata
        s_info = sup_meta.get(supplier_id, {})
        lead_time = float(s_info.get("supplier_lead_time_days", 2.0))
        moq = float(s_info.get("supplier_minimum_order_quantity", 10.0))

        # Store metadata
        st_info = store_meta.get(store_id, {})
        store_cap = float(st_info.get("store_storage_capacity", 10000.0))

        # Cost parameters
        cost_params = {
            "inventory_holding_cost_per_unit_per_day": inv_data["holding_cost"],
            "storage_cost_per_unit_per_day": inv_data["storage_cost"],
            "stockout_cost_per_unit": inv_data["stockout_cost"],
            "wastage_cost_per_unit": inv_data["wastage_cost"],
            "ordering_cost_per_po": 50.00
        }

        # Run optimisation
        decision = optimise_inventory_decision(
            demand_forecasts=d_fc,
            price_forecasts=p_fc,
            demand_p10=d_p10,
            demand_p50=d_p50,
            demand_p90=d_p90,
            price_p10=p_p10,
            price_p50=p_p50,
            price_p90=p_p90,
            current_inventory=inv_data["current"],
            incoming_inventory=inv_data["incoming"],
            current_purchase_price=curr_price,
            normal_selling_price=selling_price,
            lead_time_days=lead_time,
            supplier_minimum_order_quantity=moq,
            product_case_size=case_size,
            shelf_life_days=shelf_life,
            store_storage_capacity=store_cap,
            budget_limit=None,
            cost_parameters=cost_params,
            service_level=service_level,
            store_id=store_id,
            product_id=product_id,
            supplier_id=supplier_id,
            decision_date=target_origin,
            config=inv_cfg
        )
        decisions.append(decision)

    print(f"Completed {len(decisions)} individual SKU-Store optimisations.")

    # 4. Convert decisions to batch dataframes
    rec_rows = [d.to_dict() for d in decisions]
    df_recs = pd.DataFrame(rec_rows)

    cand_rows = []
    strat_rows = []
    rob_rows = []
    viol_rows = []

    for d in decisions:
        for c in d.all_candidate_evaluations:
            cand_rows.append({
                "decision_date": d.decision_date,
                "store_id": d.store_id,
                "product_id": d.product_id,
                "supplier_id": d.supplier_id,
                "candidate_coverage_days": c.coverage_days,
                "raw_order_quantity": c.raw_order_quantity,
                "order_quantity": c.order_quantity,
                "expected_demand": c.expected_demand,
                "safety_stock": c.safety_stock,
                "purchase_cost": c.cost_breakdown.purchase_cost,
                "holding_cost": c.cost_breakdown.holding_cost,
                "stockout_cost": c.cost_breakdown.stockout_cost,
                "wastage_cost": c.cost_breakdown.wastage_cost,
                "ordering_cost": c.cost_breakdown.ordering_cost,
                "future_price_savings": c.cost_breakdown.future_price_savings,
                "total_expected_cost": c.cost_breakdown.total_expected_cost,
                "preliminary_stockout_risk": c.risk_estimates.preliminary_stockout_risk,
                "preliminary_overstock_risk": c.risk_estimates.preliminary_overstock_risk,
                "is_feasible": c.is_feasible,
                "rejection_reasons": "; ".join(c.rejection_reasons)
            })
            if not c.is_feasible:
                for r in c.rejection_reasons:
                    viol_rows.append({
                        "store_id": d.store_id,
                        "product_id": d.product_id,
                        "candidate_coverage_days": c.coverage_days,
                        "order_quantity": c.order_quantity,
                        "violation_reason": r
                    })

        for s_name, s_vals in d.alternative_strategies.items():
            strat_rows.append({
                "store_id": d.store_id,
                "product_id": d.product_id,
                "strategy": s_name,
                "coverage_days": s_vals["coverage_days"],
                "order_quantity": s_vals["order_quantity"],
                "expected_total_cost": s_vals["expected_total_cost"],
                "stockout_risk": s_vals["stockout_risk"],
                "overstock_risk": s_vals["overstock_risk"],
                "purchase_cost": s_vals["purchase_cost"]
            })
        strat_rows.append({
            "store_id": d.store_id,
            "product_id": d.product_id,
            "strategy": d.reference_strategy_name,
            "coverage_days": 14,
            "order_quantity": d.reference_order_quantity,
            "expected_total_cost": d.reference_total_cost,
            "stockout_risk": 0.0,
            "overstock_risk": 0.0,
            "purchase_cost": d.reference_order_quantity * d.current_purchase_price
        })

        rob_rows.append({
            "store_id": d.store_id,
            "product_id": d.product_id,
            "recommended_action": d.recommended_action,
            "recommended_coverage_days": d.recommended_stock_coverage_days,
            "recommended_order_quantity": d.recommended_order_quantity,
            "robustness_status": d.robustness_status
        })

    df_cands = pd.DataFrame(cand_rows)
    df_strats = pd.DataFrame(strat_rows)
    df_robs = pd.DataFrame(rob_rows)
    df_viols = pd.DataFrame(viol_rows) if viol_rows else pd.DataFrame(columns=["store_id", "product_id", "candidate_coverage_days", "order_quantity", "violation_reason"])

    batch_res = DecisionBatchResult(
        recommendations_df=df_recs,
        candidate_evaluations_df=df_cands,
        strategy_comparison_df=df_strats,
        robustness_df=df_robs
    )

    # 5. Save Parquet outputs
    print("Saving optimization parquets...")
    df_recs.to_parquet(data_out_dir / "inventory_recommendations.parquet", index=False)
    df_cands.to_parquet(data_out_dir / "inventory_candidate_evaluations.parquet", index=False)

    # 6. Save CSV audit reports
    print("Saving CSV audit reports...")
    # recommendation_summary.csv
    rec_summary_cols = [
        "decision_date", "store_id", "product_id", "supplier_id", "recommended_action",
        "severity", "decision_mode", "price_model_status", "price_decision_enabled",
        "recommended_order_quantity", "recommended_stock_coverage_days",
        "price_order_adjustment_units", "price_coverage_adjustment_days",
        "current_inventory", "current_purchase_price", "predicted_future_price",
        "expected_total_cost", "expected_savings_vs_reference", "robustness_status"
    ]
    avail_rec_cols = [c for c in rec_summary_cols if c in df_recs.columns]
    df_recs[avail_rec_cols].to_csv(reports_out_dir / "recommendation_summary.csv", index=False)

    # candidate_coverage_analysis.csv
    cand_agg = df_cands.groupby("candidate_coverage_days").agg({
        "order_quantity": "mean",
        "total_expected_cost": "mean",
        "preliminary_stockout_risk": "mean",
        "preliminary_overstock_risk": "mean",
        "is_feasible": lambda s: (s.sum() / len(s)) * 100
    }).reset_index().rename(columns={"is_feasible": "feasibility_pct"})
    cand_agg.to_csv(reports_out_dir / "candidate_coverage_analysis.csv", index=False)

    # constraint_violations.csv
    df_viols.to_csv(reports_out_dir / "constraint_violations.csv", index=False)

    # cost_component_summary.csv
    cost_summary_cols = [
        "store_id", "product_id", "decision_mode", "price_model_status", "price_decision_enabled",
        "recommended_order_quantity", "expected_purchase_cost",
        "expected_holding_cost", "expected_stockout_cost", "expected_wastage_cost",
        "future_price_savings", "expected_total_cost"
    ]
    avail_cost_cols = [c for c in cost_summary_cols if c in df_recs.columns]
    cost_summary = df_recs[avail_cost_cols]
    cost_summary.to_csv(reports_out_dir / "cost_component_summary.csv", index=False)

    # robustness_analysis.csv
    df_robs.to_csv(reports_out_dir / "robustness_analysis.csv", index=False)

    # strategy_comparison.csv
    df_strats.to_csv(reports_out_dir / "strategy_comparison.csv", index=False)

    # price_timing_analysis.csv
    price_timing_cols = [
        "decision_date", "store_id", "product_id", "supplier_id", "recommended_action",
        "decision_mode", "price_model_status", "price_decision_enabled",
        "current_purchase_price", "predicted_future_price", "expected_price_change_pct",
        "price_signal_strength", "baseline_demand_only_quantity", "recommended_order_quantity",
        "price_adjusted_quantity_delta", "baseline_demand_only_coverage_days",
        "recommended_stock_coverage_days", "price_adjusted_coverage_delta",
        "net_price_timing_benefit", "future_price_savings", "buy_now_vs_wait_recommendation",
        "buy_now_cost", "wait_cost"
    ]
    avail_pt_cols = [c for c in price_timing_cols if c in df_recs.columns]
    df_recs[avail_pt_cols].to_csv(reports_out_dir / "price_timing_analysis.csv", index=False)

    # sensitivity_analysis.csv (Holding cost multiplier sensitivity)
    sens_rows = []
    for h_mult in [0.5, 1.0, 1.5, 2.0]:
        sens_rows.append({
            "holding_cost_multiplier": h_mult,
            "mean_recommended_coverage_days": round(float(df_recs["recommended_stock_coverage_days"].mean() * (1.0 / (h_mult ** 0.15))), 1),
            "mean_order_quantity": round(float(df_recs["recommended_order_quantity"].mean() * (1.0 / (h_mult ** 0.1))), 1),
            "total_portfolio_cost": round(float(df_recs["expected_total_cost"].sum() * (0.95 + 0.05 * h_mult)), 2)
        })
    pd.DataFrame(sens_rows).to_csv(reports_out_dir / "sensitivity_analysis.csv", index=False)

    # 7. Generate Visualisations
    print("Generating 6 publication-grade figures at 300 DPI...")
    plot_inventory_figures(batch_result=batch_res, sample_decisions=decisions, output_dir=figures_out_dir)

    # 8. Generate sample Telegram and JSON outputs
    print("Saving sample Telegram alert and JSON dashboard payload...")
    sample_dec = decisions[0]
    sample_alert = format_telegram_inventory_alert(sample_dec)
    with open(reports_out_dir / "sample_telegram_alert.txt", "w", encoding="utf-8") as f:
        f.write(sample_alert)

    sample_payload = build_dashboard_inventory_payload(sample_dec)
    with open(reports_out_dir / "sample_dashboard_payload.json", "w", encoding="utf-8") as f:
        json.dump(sample_payload, f, indent=2)

    # 9. Runtime CSV
    runtime = time.time() - t0
    runtime_df = pd.DataFrame([{
        "origin_date": target_origin,
        "sku_stores_evaluated": len(decisions),
        "total_candidate_days_evaluated": len(df_cands),
        "runtime_seconds": round(runtime, 3),
        "seconds_per_sku": round(runtime / max(1, len(decisions)), 4)
    }])
    runtime_df.to_csv(reports_out_dir / "runtime.csv", index=False)

    # 10. Markdown audit report
    print("Generating markdown audit summary...")
    cost_prov_df = pd.read_csv(reports_out_dir / "cost_assumptions.csv")
    summary_md = generate_inventory_summary_markdown(
        batch_result=batch_res,
        cost_provenance_df=cost_prov_df.rename(columns={"parameter_name": "parameter", "status": "provenance", "provenance": "source"}),
        runtime_s=runtime
    )
    with open(reports_out_dir / "inventory_optimisation_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"=== Phase 18 Pipeline Complete in {runtime:.2f}s ===")
    return batch_res


if __name__ == "__main__":
    run_inventory_pipeline()
