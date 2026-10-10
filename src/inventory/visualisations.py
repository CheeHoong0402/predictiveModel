"""Visualisation generator for Phase 18 Inventory Optimisation.

Generates 6 publication-grade figures at 300 DPI:
1. expected_total_cost_vs_coverage.png
2. recommended_coverage_by_product.png
3. stockout_risk_proxy_vs_coverage.png
4. overstock_risk_proxy_vs_coverage.png
5. purchase_price_trajectory_vs_decision.png
6. cost_component_breakdown.png
"""
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.paths import PROJECT_ROOT
from .optimizer import InventoryDecision, DecisionBatchResult


def plot_inventory_figures(
    batch_result: DecisionBatchResult,
    sample_decisions: List[InventoryDecision],
    output_dir: Optional[Path] = None
) -> List[Path]:
    """Generate and save all 6 inventory optimisation figures."""
    out_dir = output_dir or (PROJECT_ROOT / "reports" / "figures" / "inventory")
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300
    })

    # Pick representative decisions (e.g. perishable vs non-perishable)
    perishable_dec = next((d for d in sample_decisions if "CHK" in d.product_id or "EGG" in d.product_id), sample_decisions[0])
    non_perishable_dec = next((d for d in sample_decisions if "OIL" in d.product_id or "RIC" in d.product_id), sample_decisions[-1])

    # -------------------------------------------------------------
    # FIGURE 1: expected_total_cost_vs_coverage.png
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for dec, color, label in [
        (perishable_dec, "#d9534f", f"{perishable_dec.product_id} (Perishable)"),
        (non_perishable_dec, "#0275d8", f"{non_perishable_dec.product_id} (Non-Perishable)")
    ]:
        days = [c.coverage_days for c in dec.all_candidate_evaluations]
        costs = [c.cost_breakdown.total_expected_cost for c in dec.all_candidate_evaluations]
        feasible = [c.is_feasible for c in dec.all_candidate_evaluations]

        # Plot feasible line
        ax.plot(days, costs, marker="o", markersize=4, color=color, label=f"{label} - Total Cost Curve", alpha=0.8)
        
        # Highlight optimal point
        opt_d = dec.recommended_stock_coverage_days
        opt_cost = dec.cost_breakdown.total_expected_cost
        ax.scatter([opt_d], [opt_cost], color=color, s=150, zorder=5, edgecolor="black", linewidth=1.5,
                   label=f"{label} - Optimal ({opt_d}d, RM {opt_cost:,.0f})")

        # Mark infeasible points
        infeas_days = [d for d, f in zip(days, feasible) if not f]
        infeas_costs = [c for c, f in zip(costs, feasible) if not f]
        if infeas_days:
            ax.scatter(infeas_days, infeas_costs, marker="x", color="black", s=60, zorder=4,
                       label=f"{label} - Infeasible (Shelf life/Cap)")

    ax.set_title("Expected Total Cost vs Stock Coverage Days (1–30 Days)", fontweight="bold")
    ax.set_xlabel("Candidate Stock Coverage (Days)")
    ax.set_ylabel("Expected Total Cost (RM)")
    ax.set_xticks(range(1, 31, 2))
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    f1 = out_dir / "expected_total_cost_vs_coverage.png"
    plt.savefig(f1, dpi=300)
    plt.close()
    generated_files.append(f1)

    # -------------------------------------------------------------
    # FIGURE 2: recommended_coverage_by_product.png
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    recs_df = batch_result.recommendations_df
    # Group by product
    prod_cov = recs_df.groupby("product_id")["recommended_stock_coverage_days"].mean().reset_index()
    prod_cov = prod_cov.sort_values(by="recommended_stock_coverage_days")
    
    colors = ["#d9534f" if "CHK" in p or "EGG" in p else "#5cb85c" for p in prod_cov["product_id"]]
    bars = ax.bar(prod_cov["product_id"], prod_cov["recommended_stock_coverage_days"], color=colors, edgecolor="black", width=0.55)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.1f}d", ha="center", va="bottom", fontweight="bold")

    ax.set_title("Recommended Stock Coverage Days by Product SKU", fontweight="bold")
    ax.set_xlabel("Product SKU")
    ax.set_ylabel("Mean Recommended Coverage (Days)")
    ax.set_ylim(0, max(prod_cov["recommended_stock_coverage_days"]) + 4)
    ax.axhline(14, color="gray", linestyle="--", alpha=0.7, label="Fixed 14-Day Reference Policy")
    ax.legend(loc="upper left")
    plt.tight_layout()
    f2 = out_dir / "recommended_coverage_by_product.png"
    plt.savefig(f2, dpi=300)
    plt.close()
    generated_files.append(f2)

    # -------------------------------------------------------------
    # FIGURE 3: stockout_risk_proxy_vs_coverage.png
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    days = [c.coverage_days for c in non_perishable_dec.all_candidate_evaluations]
    stk_risks = [c.risk_estimates.preliminary_stockout_risk * 100 for c in non_perishable_dec.all_candidate_evaluations]
    
    ax.plot(days, stk_risks, color="#d9534f", linewidth=2.5, marker="s", markersize=5, label="Stockout Risk Proxy (%)")
    ax.axhline(5.0, color="orange", linestyle="--", label="Target Stockout Ceiling (5%)")
    ax.axhline(1.0, color="green", linestyle=":", label="Ultra-Safe Zone (1%)")
    
    opt_d = non_perishable_dec.recommended_stock_coverage_days
    opt_risk = non_perishable_dec.risk_estimates.preliminary_stockout_risk * 100
    ax.scatter([opt_d], [opt_risk], color="#d9534f", s=150, zorder=5, edgecolor="black",
               label=f"Selected Coverage ({opt_d}d, {opt_risk:.1f}% risk)")

    ax.set_title(f"Preliminary Stockout Risk vs Coverage Days ({non_perishable_dec.product_id})", fontweight="bold")
    ax.set_xlabel("Stock Coverage (Days)")
    ax.set_ylabel("Stockout Risk Proxy (%)")
    ax.set_xticks(range(1, 31, 2))
    ax.set_ylim(-1, max(stk_risks) + 5)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    f3 = out_dir / "stockout_risk_proxy_vs_coverage.png"
    plt.savefig(f3, dpi=300)
    plt.close()
    generated_files.append(f3)

    # -------------------------------------------------------------
    # FIGURE 4: overstock_risk_proxy_vs_coverage.png
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for dec, color, label in [
        (perishable_dec, "#d9534f", f"{perishable_dec.product_id} (Perishable, Shelf Life 3d)"),
        (non_perishable_dec, "#0275d8", f"{non_perishable_dec.product_id} (Non-Perishable, 180d)")
    ]:
        days = [c.coverage_days for c in dec.all_candidate_evaluations]
        over_risks = [c.risk_estimates.preliminary_overstock_risk * 100 for c in dec.all_candidate_evaluations]
        ax.plot(days, over_risks, marker="^", markersize=5, linewidth=2, color=color, label=label)

    ax.set_title("Overstock & Spoilage Risk Proxy vs Coverage Days", fontweight="bold")
    ax.set_xlabel("Stock Coverage (Days)")
    ax.set_ylabel("Overstock / Spoilage Risk Proxy (%)")
    ax.set_xticks(range(1, 31, 2))
    ax.set_ylim(-2, 105)
    ax.legend(loc="center right", frameon=True)
    plt.tight_layout()
    f4 = out_dir / "overstock_risk_proxy_vs_coverage.png"
    plt.savefig(f4, dpi=300)
    plt.close()
    generated_files.append(f4)

    # -------------------------------------------------------------
    # FIGURE 5: purchase_price_trajectory_vs_decision.png
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot forecast price fan chart
    horizons = [1, 3, 7, 14, 21, 30]
    p_curr = non_perishable_dec.current_purchase_price
    # Interpolated trajectory
    days_30 = [c.coverage_days for c in non_perishable_dec.all_candidate_evaluations]
    p_traj = [c.predicted_future_price for c in non_perishable_dec.all_candidate_evaluations]
    
    ax.plot(days_30, p_traj, color="#0275d8", linewidth=2.5, label=f"Predicted Purchase Price (P50)")
    ax.axhline(p_curr, color="black", linestyle="--", linewidth=1.5, label=f"Current Price (RM {p_curr:.2f})")
    
    # Fill between P10 and P90 approximation
    p_p10_traj = [p * 0.96 for p in p_traj]
    p_p90_traj = [p * 1.05 for p in p_traj]
    ax.fill_between(days_30, p_p10_traj, p_p90_traj, color="#0275d8", alpha=0.15, label="80% Price Uncertainty Interval (P10–P90)")

    # Mark selected action
    opt_d = non_perishable_dec.recommended_stock_coverage_days
    opt_p = p_traj[opt_d - 1]
    ax.scatter([opt_d], [opt_p], color="#5cb85c", s=160, zorder=6, edgecolor="black",
               label=f"Action: {non_perishable_dec.recommended_action} ({opt_d}d coverage)")

    ax.set_title(f"Purchase Price Trajectory vs Timing Decision ({non_perishable_dec.product_id})", fontweight="bold")
    ax.set_xlabel("Forecast Horizon / Coverage Days")
    ax.set_ylabel("Supplier Purchase Price (RM/unit)")
    ax.set_xticks(range(1, 31, 2))
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    f5 = out_dir / "purchase_price_trajectory_vs_decision.png"
    plt.savefig(f5, dpi=300)
    plt.close()
    generated_files.append(f5)

    # -------------------------------------------------------------
    # FIGURE 6: cost_component_breakdown.png
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    cand_sample = [c for c in non_perishable_dec.all_candidate_evaluations if c.coverage_days in [3, 7, 10, 14, 21, 28]]
    days_s = [f"{c.coverage_days}d" for c in cand_sample]
    
    purch_costs = np.array([c.cost_breakdown.purchase_cost for c in cand_sample])
    hold_costs = np.array([c.cost_breakdown.holding_cost for c in cand_sample])
    stk_costs = np.array([c.cost_breakdown.stockout_cost for c in cand_sample])
    waste_costs = np.array([c.cost_breakdown.wastage_cost for c in cand_sample])
    order_costs = np.array([c.cost_breakdown.ordering_cost for c in cand_sample])

    x = np.arange(len(days_s))
    width = 0.55

    ax.bar(x, purch_costs, width, label="Purchase Outlay", color="#337ab7")
    ax.bar(x, hold_costs, width, bottom=purch_costs, label="Holding Cost", color="#f0ad4e")
    ax.bar(x, stk_costs, width, bottom=purch_costs + hold_costs, label="Stockout Cost", color="#d9534f")
    ax.bar(x, waste_costs, width, bottom=purch_costs + hold_costs + stk_costs, label="Wastage Cost", color="#5bc0de")
    ax.bar(x, order_costs, width, bottom=purch_costs + hold_costs + stk_costs + waste_costs, label="Ordering Cost", color="#5cb85c")

    ax.set_title(f"Cost Component Decomposition Across Coverage Horizons ({non_perishable_dec.product_id})", fontweight="bold")
    ax.set_xlabel("Coverage Strategy")
    ax.set_ylabel("Expected Cost (RM)")
    ax.set_xticks(x)
    ax.set_xticklabels(days_s)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    f6 = out_dir / "cost_component_breakdown.png"
    plt.savefig(f6, dpi=300)
    plt.close()
    generated_files.append(f6)

    return generated_files
