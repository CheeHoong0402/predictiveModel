"""Publication-grade visualisations for Phase 20 Historical Business Backtesting."""
from pathlib import Path
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.utils.paths import PROJECT_ROOT


def plot_cumulative_cost(daily_df: pd.DataFrame, out_path: Path):
    """1. Cumulative total costs by strategy over historical replay."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Compute daily total cost per strategy
    d_df = daily_df.copy()
    d_df["date"] = pd.to_datetime(d_df["date"])
    d_df["day_cost"] = d_df["holding_cost"] + d_df["stockout_cost"] + d_df["wastage_cost"]

    # Sum across all stores/products for each date and strategy
    daily_agg = d_df.groupby(["date", "strategy"])["day_cost"].sum().reset_index()

    strategies = sorted(daily_agg["strategy"].unique())
    colors = {
        "NO_ACTION": "#7f7f7f",
        "FIXED_7_DAY": "#ff7f0e",
        "FIXED_14_DAY": "#d62728",
        "FIXED_30_DAY": "#9467bd",
        "REORDER_POINT": "#8c564b",
        "PHASE18_OPTIMISER": "#1f77b4",
        "PHASE20_AI_STRATEGY": "#2ca02c"
    }

    for strat in strategies:
        sub = daily_agg[daily_agg["strategy"] == strat].sort_values("date")
        sub["cum_cost"] = sub["day_cost"].cumsum()
        ax.plot(
            sub["date"],
            sub["cum_cost"] / 1000.0,
            label=strat,
            color=colors.get(strat, "#333333"),
            linewidth=2.5 if strat == "PHASE20_AI_STRATEGY" else 1.5,
            linestyle="-" if "AI" in strat or "14" in strat else "--"
        )

    ax.set_title("Cumulative Operating Inventory Cost Over Historical Replay (990 Days)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Operational Cost (k RM)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_cumulative_savings(daily_df: pd.DataFrame, decision_df: pd.DataFrame, out_path: Path):
    """2. Cumulative net savings of AI Strategy vs Fixed 14-Day benchmark."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    d_df = daily_df.copy()
    d_df["date"] = pd.to_datetime(d_df["date"])
    d_df["day_cost"] = d_df["holding_cost"] + d_df["stockout_cost"] + d_df["wastage_cost"]

    daily_agg = d_df.groupby(["date", "strategy"])["day_cost"].sum().unstack()

    if "PHASE20_AI_STRATEGY" in daily_agg.columns and "FIXED_14_DAY" in daily_agg.columns:
        # Daily operational savings
        diff_day = daily_agg["FIXED_14_DAY"] - daily_agg["PHASE20_AI_STRATEGY"]
        cum_savings = diff_day.cumsum() / 1000.0

        ax.plot(cum_savings.index, cum_savings.values, color="#2ca02c", linewidth=2.5, label="Cumulative Operational Savings")
        ax.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax.fill_between(cum_savings.index, 0, cum_savings.values, where=(cum_savings.values >= 0), color="#2ca02c", alpha=0.2)
        ax.fill_between(cum_savings.index, 0, cum_savings.values, where=(cum_savings.values < 0), color="#d62728", alpha=0.2)

    ax.set_title("Cumulative Realised Net Savings: AI vs Fixed 14-Day Baseline", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Net Savings (k RM)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_stockout_rate(perf_df: pd.DataFrame, out_path: Path):
    """3. Stockout rate by inventory strategy."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    df_sorted = perf_df.sort_values("stockout_rate", ascending=False)
    colors = ["#2ca02c" if "AI" in s else "#1f77b4" for s in df_sorted["strategy"]]

    bars = ax.barh(df_sorted["strategy"], df_sorted["stockout_rate"] * 100.0, color=colors, edgecolor="black", alpha=0.85)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")

    ax.set_title("Stockout Rate Across Evaluated Inventory Strategies", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Stockout Rate (%)", fontsize=11)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_service_level(perf_df: pd.DataFrame, out_path: Path):
    """4. Unit fill rate (service level) by strategy."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    df_sorted = perf_df.sort_values("fill_rate", ascending=True)
    colors = ["#2ca02c" if "AI" in s else "#3274a1" for s in df_sorted["strategy"]]

    bars = ax.barh(df_sorted["strategy"], df_sorted["fill_rate"] * 100.0, color=colors, edgecolor="black", alpha=0.85)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")

    ax.set_title("Unit Fill Rate (Demand Service Level) by Strategy", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Unit Fill Rate (%)", fontsize=11)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_average_inventory(perf_df: pd.DataFrame, out_path: Path):
    """5. Average inventory units by strategy."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    df_sorted = perf_df.sort_values("average_inventory", ascending=False)
    colors = ["#2ca02c" if "AI" in s else "#e377c2" for s in df_sorted["strategy"]]

    bars = ax.barh(df_sorted["strategy"], df_sorted["average_inventory"], color=colors, edgecolor="black", alpha=0.85)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 5, bar.get_y() + bar.get_height()/2, f"{w:.1f} u", va="center", fontsize=9, fontweight="bold")

    ax.set_title("Average On-Hand Inventory Units by Strategy", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Average Inventory Units", fontsize=11)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_wastage_by_strategy(perf_df: pd.DataFrame, out_path: Path):
    """6. Perishable wastage units across strategies."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    df_sorted = perf_df.sort_values("wasted_units", ascending=False)
    colors = ["#2ca02c" if "AI" in s else "#d62728" for s in df_sorted["strategy"]]

    bars = ax.barh(df_sorted["strategy"], df_sorted["wasted_units"], color=colors, edgecolor="black", alpha=0.85)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 10, bar.get_y() + bar.get_height()/2, f"{w:.1f} u", va="center", fontsize=9, fontweight="bold")

    ax.set_title("Total Perishable Wastage Units Across Strategies", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Expired / Wasted Units", fontsize=11)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_monthly_savings_timeline(daily_df: pd.DataFrame, out_path: Path):
    """7. Monthly net savings timeline."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    d_df = daily_df.copy()
    d_df["month"] = pd.to_datetime(d_df["date"]).dt.to_period("M").astype(str)
    d_df["day_cost"] = d_df["holding_cost"] + d_df["stockout_cost"] + d_df["wastage_cost"]

    monthly_agg = d_df.groupby(["month", "strategy"])["day_cost"].sum().unstack()

    if "PHASE20_AI_STRATEGY" in monthly_agg.columns and "FIXED_14_DAY" in monthly_agg.columns:
        diff = (monthly_agg["FIXED_14_DAY"] - monthly_agg["PHASE20_AI_STRATEGY"]) / 1000.0
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in diff.values]

        ax.bar(diff.index, diff.values, color=colors, edgecolor="black", alpha=0.85)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xticklabels(diff.index, rotation=45, ha="right", fontsize=8)

    ax.set_title("Monthly Net Operating Cost Savings: AI vs Fixed 14-Day", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Monthly Savings (k RM)", fontsize=11)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_savings_by_store(store_df: pd.DataFrame, out_path: Path):
    """8. Savings by Store bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Compute savings per store
    piv = store_df.pivot(index="store_id", columns="strategy", values="total_cost")
    if "PHASE20_AI_STRATEGY" in piv.columns and "FIXED_14_DAY" in piv.columns:
        savings = (piv["FIXED_14_DAY"] - piv["PHASE20_AI_STRATEGY"]) / 1000.0
        bars = ax.bar(savings.index, savings.values, color="#1f77b4", edgecolor="black", alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"RM {h:.1f}k", ha="center", fontsize=9, fontweight="bold")

    ax.set_title("Total Realised Cost Savings by Retail Store", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Savings (k RM)", fontsize=11)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_savings_by_category(cat_df: pd.DataFrame, out_path: Path):
    """9. Savings by Product Category."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    df_sorted = cat_df.sort_values("absolute_savings", ascending=True)
    bars = ax.barh(df_sorted["category"], df_sorted["absolute_savings"] / 1000.0, color="#2ca02c", edgecolor="black", alpha=0.85)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height()/2, f"RM {w:.1f}k", va="center", fontsize=9, fontweight="bold")

    ax.set_title("Total Realised Cost Savings by Product Category", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Net Savings (k RM)", fontsize=11)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_cost_component_breakdown(perf_df: pd.DataFrame, out_path: Path):
    """10. Cost-component breakdown stacked bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    df = perf_df.set_index("strategy")
    cols = ["total_purchase_cost", "ordering_cost", "holding_cost", "stockout_cost", "wastage_cost"]
    labels = ["Purchase Cost", "Ordering Cost", "Holding Cost", "Stockout Cost", "Wastage Cost"]
    colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#d62728", "#9467bd"]

    df_k = df[cols] / 1000.0
    bottom = np.zeros(len(df))

    for col, label, color in zip(cols, labels, colors):
        vals = df_k[col].values
        ax.bar(df.index, vals, bottom=bottom, label=label, color=color, edgecolor="black", alpha=0.85)
        bottom += vals

    ax.set_title("Total Realised Cost Component Breakdown Across Strategies", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Total Realised Cost (k RM)", fontsize=11)
    ax.set_xticklabels(df.index, rotation=35, ha="right", fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def generate_all_backtest_figures(
    daily_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    perf_df: pd.DataFrame,
    store_df: pd.DataFrame,
    cat_df: pd.DataFrame,
    output_dir: Path
):
    """Generate all 10 required diagnostic figures at 300 DPI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_cumulative_cost(daily_df, output_dir / "cumulative_cost_by_strategy.png")
    plot_cumulative_savings(daily_df, decision_df, output_dir / "cumulative_savings_over_time.png")
    plot_stockout_rate(perf_df, output_dir / "stockout_rate_by_strategy.png")
    plot_service_level(perf_df, output_dir / "service_level_by_strategy.png")
    plot_average_inventory(perf_df, output_dir / "average_inventory_by_strategy.png")
    plot_wastage_by_strategy(perf_df, output_dir / "wastage_by_strategy.png")
    plot_monthly_savings_timeline(daily_df, output_dir / "monthly_savings_timeline.png")
    plot_savings_by_store(store_df, output_dir / "savings_by_store.png")
    plot_savings_by_category(cat_df, output_dir / "savings_by_product_category.png")
    plot_cost_component_breakdown(perf_df, output_dir / "cost_component_breakdown.png")
