"""Publication-grade visualisations module for Explainability and Model Coverage Audit.

Produces 9 publication-grade figures at 300 DPI saved to reports/figures/explainability/:
1. demand_top15_features.png
2. demand_feature_families.png
3. demand_family_contribution_by_horizon.png
4. price_top15_features.png
5. price_feature_families.png
6. price_family_contribution_by_horizon.png
7. training_timeline.png
8. local_demand_waterfall_example.png
9. local_price_waterfall_example.png
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.utils.paths import PROJECT_ROOT


def plot_top_features(
    importance_df: pd.DataFrame,
    task: str,
    output_path: Path,
    top_n: int = 15
) -> None:
    """Plot horizontal bar chart of top N features by mean absolute SHAP importance."""
    plt.figure(figsize=(10, 7), dpi=300)
    
    sub_df = importance_df.head(top_n).iloc[::-1]  # Highest on top
    y_pos = np.arange(len(sub_df))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(sub_df)))
    bars = plt.barh(y_pos, sub_df["mean_abs_shap"], color=colors, edgecolor="black", linewidth=0.6, height=0.7)
    
    plt.yticks(y_pos, sub_df["display_name"], fontsize=10)
    plt.xlabel("Mean |SHAP Value| (Impact on Model Output)", fontsize=11, fontweight="bold")
    task_title = "Product Demand" if task == "demand" else "Supplier Purchase Price"
    plt.title(f"Top {top_n} Most Influential Drivers for {task_title} Forecasting", fontsize=12, fontweight="bold", pad=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    
    # Add value annotations
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.01 * sub_df["mean_abs_shap"].max(), bar.get_y() + bar.get_height() / 2,
                 f"{w:.3f}", va="center", ha="left", fontsize=9)
        
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_families(
    family_df: pd.DataFrame,
    task: str,
    output_path: Path
) -> None:
    """Plot bar chart of aggregate feature family importance."""
    plt.figure(figsize=(10, 6), dpi=300)
    
    sub_df = family_df.sort_values("mean_abs_shap", ascending=True)
    y_pos = np.arange(len(sub_df))
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(sub_df)))
    bars = plt.barh(y_pos, sub_df["mean_abs_shap"], color=colors, edgecolor="black", linewidth=0.6, height=0.65)
    
    plt.yticks(y_pos, sub_df["business_family"], fontsize=10)
    plt.xlabel("Aggregate Mean |SHAP Value|", fontsize=11, fontweight="bold")
    task_title = "Product Demand" if task == "demand" else "Supplier Purchase Price"
    plt.title(f"Predictive Importance by Business Family for {task_title}", fontsize=12, fontweight="bold", pad=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.01 * sub_df["mean_abs_shap"].max(), bar.get_y() + bar.get_height() / 2,
                 f"{w:.3f}", va="center", ha="left", fontsize=9)
        
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_family_contribution_by_horizon(
    horizon_family_df: pd.DataFrame,
    task: str,
    output_path: Path
) -> None:
    """Plot stacked bar chart of feature family importance by forecast horizon."""
    # Pivot data: index=horizon, columns=family, values=share
    pivot_df = horizon_family_df.pivot(index="horizon", columns="business_family", values="importance_share_pct")
    # Sort horizons naturally
    h_order = ["1d", "3d", "7d", "14d", "21d", "30d"]
    pivot_df = pivot_df.reindex([h for h in h_order if h in pivot_df.index])
    
    plt.figure(figsize=(11, 7), dpi=300)
    ax = plt.subplot(111)
    
    colormap = plt.cm.tab20
    colors = [colormap(i) for i in np.linspace(0, 1, len(pivot_df.columns))]
    
    pivot_df.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="black", linewidth=0.5, width=0.65)
    
    task_title = "Demand" if task == "demand" else "Purchase Price"
    plt.title(f"{task_title} Feature Family Importance Shift across Forecast Horizons", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Forecast Horizon", fontsize=11, fontweight="bold")
    plt.ylabel("Relative Importance Share (%)", fontsize=11, fontweight="bold")
    plt.xticks(rotation=0, fontsize=10)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=9, title="Business Family")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_timeline(
    output_path: Path,
    wf_folds_df: Optional[pd.DataFrame] = None
) -> None:
    """Plot Gantt timeline showing full dataset span, 33 walk-forward folds, and holdout reservation."""
    plt.figure(figsize=(12, 6), dpi=300)
    ax = plt.subplot(111)
    
    # Timeline milestones
    t_start = pd.Timestamp("2021-01-01")
    t_dev_end = pd.Timestamp("2024-09-30")
    t_holdout_start = pd.Timestamp("2024-10-01")
    t_holdout_end = pd.Timestamp("2024-12-31")
    
    # 1. Full Dataset Bar
    ax.barh(4, (t_holdout_end - t_start).days, left=t_start, height=0.4, color="#34495e", label="Total Dataset (4.0 Years, 35,064 rows)")
    
    # 2. Pre-Holdout Development Bar
    ax.barh(3, (t_dev_end - t_start).days, left=t_start, height=0.4, color="#2980b9", label="Pre-Holdout Development (3.75 Years, 32,856 rows)")
    
    # 3. Final Production Refit Window (1d vs 30d)
    ax.barh(2, (t_dev_end - t_start).days, left=t_start, height=0.4, color="#27ae60", label="Full Development Refit Window")
    
    # 4. Walk-Forward Expanding Folds Span
    f1_start = pd.Timestamp("2021-01-01")
    f33_end = pd.Timestamp("2024-09-16")
    ax.barh(1, (f33_end - f1_start).days, left=f1_start, height=0.4, color="#e67e22", label="33 Walk-Forward Expanding Folds Evaluation")
    
    # 5. Strict Final Holdout Bar
    ax.barh(0, (t_holdout_end - t_holdout_start).days, left=t_holdout_start, height=0.4, color="#c0392b", label="Final Holdout (Strictly Untouched, 92 days / 2,208 rows)")
    
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels([
        "Strict Final Holdout",
        "33 Walk-Forward Folds",
        "Full Dev Refit Span",
        "Pre-Holdout Development",
        "Total Dataset Span"
    ], fontsize=10, fontweight="bold")
    
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    
    plt.axvline(t_holdout_start, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
    plt.text(t_holdout_start + pd.Timedelta(days=5), 2.5, "Strict Holdout Cutoff\n(2024-10-01)",
             color="#c0392b", fontweight="bold", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#c0392b", alpha=0.9))
    
    plt.title("Model Training Coverage, Temporal Horizon & Final Holdout Reservation Timeline", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Calendar Year", fontsize=11, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", bbox_to_anchor=(0.0, -0.15), ncol=2, frameon=True, fontsize=9)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_waterfall(
    explanation: Dict[str, Any],
    task: str,
    output_path: Path
) -> None:
    """Plot local prediction waterfall showing baseline, top drivers, and final forecast."""
    plt.figure(figsize=(10, 6), dpi=300)
    
    if task == "demand":
        base_val = explanation["base_value"]
        point_pred = explanation["point_prediction"]
        top_pos = explanation["top_positive_drivers"][:4]
        top_neg = explanation["top_negative_drivers"][:4]
        unit = "units"
        p10 = explanation["p10"]
        p90 = explanation["p90"]
        
        labels = ["Baseline Demand"]
        deltas = [base_val]
        
        for d in top_pos:
            labels.append(f"+ {d['display_name'][:25]}")
            deltas.append(d["contribution"])
            
        for d in top_neg:
            labels.append(f"- {d['display_name'][:25]}")
            deltas.append(d["contribution"])
            
        # Other residual features
        explained_sum = base_val + sum(d["contribution"] for d in top_pos) + sum(d["contribution"] for d in top_neg)
        other_delta = point_pred - explained_sum
        labels.append("Other Features")
        deltas.append(other_delta)
        
        labels.append(f"Forecast ({point_pred:.1f} {unit})")
        deltas.append(point_pred)
    else:
        # Price
        curr_price = explanation["current_price"]
        base_val = explanation["base_value_myr"]
        point_pred = explanation["point_prediction"]
        top_pos = explanation["top_positive_drivers"][:4]
        top_neg = explanation["top_negative_drivers"][:4]
        unit = "MYR"
        p10 = explanation["p10"]
        p90 = explanation["p90"]
        
        labels = [f"Current Price (RM {curr_price:.2f})"]
        deltas = [curr_price]
        
        # Base shift
        base_shift = base_val - curr_price
        labels.append("Historical Trend")
        deltas.append(base_shift)
        
        for d in top_pos:
            labels.append(f"+ {d['display_name'][:25]}")
            deltas.append(d["contribution_myr"])
            
        for d in top_neg:
            labels.append(f"- {d['display_name'][:25]}")
            deltas.append(d["contribution_myr"])
            
        explained_sum = base_val + sum(d["contribution_myr"] for d in top_pos) + sum(d["contribution_myr"] for d in top_neg)
        other_delta = point_pred - explained_sum
        labels.append("Other Drivers")
        deltas.append(other_delta)
        
        labels.append(f"Forecast (RM {point_pred:.2f})")
        deltas.append(point_pred)

    # Compute step coordinates for waterfall
    n_steps = len(labels)
    bottoms = [0.0] * n_steps
    heights = [0.0] * n_steps
    colors = []
    
    current_level = 0.0
    for idx in range(n_steps):
        if idx == 0:
            # First bar
            bottoms[idx] = 0.0
            heights[idx] = deltas[0]
            current_level = deltas[0]
            colors.append("#34495e")
        elif idx == n_steps - 1:
            # Final bar
            bottoms[idx] = 0.0
            heights[idx] = deltas[-1]
            colors.append("#2980b9")
        else:
            # Incremental deltas
            delta = deltas[idx]
            if delta >= 0:
                bottoms[idx] = current_level
                heights[idx] = delta
                colors.append("#27ae60")  # Green positive
            else:
                bottoms[idx] = current_level + delta
                heights[idx] = abs(delta)
                colors.append("#e74c3c")  # Red negative
            current_level += delta

    x_pos = np.arange(n_steps)
    bars = plt.bar(x_pos, heights, bottom=bottoms, color=colors, edgecolor="black", linewidth=0.6, width=0.6)
    
    plt.xticks(x_pos, labels, rotation=35, ha="right", fontsize=9)
    plt.ylabel(f"Forecast Value ({unit})", fontsize=11, fontweight="bold")
    task_name = "Demand" if task == "demand" else "Purchase Price"
    store_prod = f"{explanation.get('store_id', '')} - {explanation.get('product_id', '')}"
    plt.title(f"Local {task_name} Forecast Explanation Waterfall ({store_prod}, {explanation['horizon']}d Horizon)",
              fontsize=12, fontweight="bold", pad=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Annotate uncertainty interval on final bar
    plt.errorbar(
        [x_pos[-1]], [point_pred],
        yerr=[[point_pred - p10], [p90 - point_pred]],
        fmt="o", color="black", ecolor="#2c3e50", elinewidth=2, capsize=5,
        label=f"80% Uncertainty Interval [{p10:.1f}, {p90:.1f}]"
    )
    plt.legend(loc="upper left", frameon=True, fontsize=9)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
