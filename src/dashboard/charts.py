"""Interactive Plotly visualisations for Streamlit Dashboard."""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def plot_demand_forecast_chart(demand_dict: Dict[str, Dict[str, float]], product_name: str) -> go.Figure:
    """Plot cumulative demand forecasts across 6 horizons with calibrated 80% P10-P90 prediction interval."""
    horizons = ["1d", "3d", "7d", "14d", "21d", "30d"]
    points = [demand_dict[h]["point_prediction"] for h in horizons if h in demand_dict]
    p10s = [demand_dict[h]["p10"] for h in horizons if h in demand_dict]
    p50s = [demand_dict[h]["p50"] for h in horizons if h in demand_dict]
    p90s = [demand_dict[h]["p90"] for h in horizons if h in demand_dict]
    h_labels = [h for h in horizons if h in demand_dict]

    fig = go.Figure()

    # P90 upper bound line
    fig.add_trace(go.Scatter(
        x=h_labels,
        y=p90s,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="Upper P90 Bound"
    ))

    # P10 lower bound line with filled shaded area
    fig.add_trace(go.Scatter(
        x=h_labels,
        y=p10s,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(37, 99, 235, 0.15)",
        name="80% Calibrated Interval (P10–P90)",
        hoverinfo="x+y"
    ))

    # P50 Median Line
    fig.add_trace(go.Scatter(
        x=h_labels,
        y=p50s,
        mode="lines+markers",
        line=dict(color="#2563EB", width=3, dash="dot"),
        marker=dict(size=7, color="#2563EB"),
        name="Calibrated Median (P50)"
    ))

    # Point Forecast
    fig.add_trace(go.Scatter(
        x=h_labels,
        y=points,
        mode="lines+markers",
        line=dict(color="#1E40AF", width=3),
        marker=dict(size=8, color="#1E40AF"),
        name="Ensemble Point Forecast"
    ))

    fig.update_layout(
        title="Cumulative demand and uncertainty",
        height=440,
        xaxis_title="Forecast Horizon",
        yaxis_title="Cumulative Demand (Units)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=25, t=65, b=110),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0)
    )
    return fig


def plot_price_forecast_chart(price_dict: Dict[str, Any], current_price: float, product_name: str) -> go.Figure:
    """Plot supplier purchase-price trajectory and forecast bounds.
    
    Safe against quarantined models where predicted values are None.
    """
    horizons = ["Current", "1d", "3d", "7d", "14d", "21d", "30d"]
    
    if price_dict.get("predicted_future_price") is None:
        fig = go.Figure(go.Bar(x=["Current known price"], y=[current_price],
                              name="Historical invoice price", marker_color="#64748B"))
        fig.update_layout(title="Purchase price — NOT VALIDATED", height=340,
                          yaxis_title="Current purchase price (RM)", template="plotly_white",
                          showlegend=False, margin=dict(l=40, r=25, t=65, b=50))
        fig.add_annotation(text="Future forecast QUARANTINED.<br>No future prices or intervals shown.",
                           xref="paper", yref="paper", x=0.98, y=0.95, showarrow=False,
                           bgcolor="white", font=dict(color="#991B1B", size=12))
        return fig

    # Handle None safely when price model is quarantined/unvalidated
    raw_pred = price_dict.get("predicted_future_price")
    pred_val = float(raw_pred) if raw_pred is not None else float(current_price)
    
    raw_p10 = price_dict.get("p10")
    p10_val = float(raw_p10) if raw_p10 is not None else pred_val * 0.96
    
    raw_p90 = price_dict.get("p90")
    p90_val = float(raw_p90) if raw_p90 is not None else pred_val * 1.05

    is_quarantined = raw_pred is None

    # Trajectory points
    y_vals = [current_price] + [pred_val] * 6
    y_p10 = [current_price] + [p10_val] * 6
    y_p90 = [current_price] + [p90_val] * 6

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=horizons,
        y=y_p90,
        mode="lines",
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=horizons,
        y=y_p10,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(239, 68, 68, 0.15)" if is_quarantined else "rgba(16, 185, 129, 0.15)",
        name="Quarantined Bounds (Ref Only)" if is_quarantined else "80% Price Interval (P10–P90)"
    ))

    fig.add_trace(go.Scatter(
        x=horizons,
        y=y_vals,
        mode="lines+markers",
        line=dict(color="#EF4444" if is_quarantined else "#059669", width=3, dash="dash" if is_quarantined else "solid"),
        marker=dict(size=8, color="#EF4444" if is_quarantined else "#059669"),
        name="Flat Invoice Price (Quarantined)" if is_quarantined else "Predicted Purchase Price"
    ))

    title_suffix = " [EXPERIMENTAL / NOT VALIDATED FOR DECISION USE]" if is_quarantined else ""
    fig.update_layout(
        title=f"Supplier Purchase Price (MYR / Unit) — {product_name}{title_suffix}",
        xaxis_title="Horizon",
        yaxis_title="Purchase Price (RM)",
        template="plotly_white",
        margin=dict(l=40, r=25, t=65, b=110),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0)
    )

    if is_quarantined:
        fig.add_annotation(
            text="⚠️ QUARANTINED: EXCLUDED FROM DECISION LOGIC",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="rgba(220, 38, 38, 0.6)"),
            bordercolor="rgba(220, 38, 38, 0.4)",
            borderwidth=1,
            borderpad=6,
            bgcolor="rgba(254, 242, 242, 0.8)"
        )

    return fig


def plot_inventory_trajectory_chart(inv_dict: Dict[str, Any], order_qty: float) -> go.Figure:
    """Plot inventory evolution trajectory under the recommended decision."""
    stages = ["Current Stock", "At Supplier Arrival", "Post-Replenishment"]
    current_inv = inv_dict.get("current_units", 0.0)
    arr_inv = inv_dict.get("projected_inventory_at_arrival", 0.0)
    end_inv = arr_inv + order_qty

    colors = ["#6B7280", "#DC2626" if arr_inv < 10 else "#F59E0B", "#10B981"]

    fig = go.Figure(go.Bar(
        x=stages,
        y=[current_inv, arr_inv, end_inv],
        text=[f"{int(current_inv):,} units", f"{int(arr_inv):,} units", f"{int(end_inv):,} units"],
        textposition="auto",
        marker_color=colors
    ))

    fig.update_layout(
        title="Projected Inventory Depletion & Replenishment Trajectory",
        yaxis_title="Inventory on Hand (Units)",
        template="plotly_white",
        margin=dict(l=40, r=25, t=65, b=110)
    )
    return fig


def plot_risk_comparison_chart(rsk_dict: Dict[str, Any]) -> go.Figure:
    """Plot risk probability gauge bars before vs after recommended replenishment."""
    metrics = ["Stockout Risk Before", "Stockout Risk After", "Overstock Risk", "Wastage Spoilage Risk"]
    vals = [
        rsk_dict.get("stockout_probability_before", 0.0) * 100.0,
        rsk_dict.get("stockout_probability_after", 0.0) * 100.0,
        rsk_dict.get("overstock_probability", 0.0) * 100.0,
        rsk_dict.get("wastage_probability", 0.0) * 100.0
    ]
    colors = ["#EF4444", "#10B981", "#F59E0B", "#EC4899"]

    fig = go.Figure(go.Bar(
        x=metrics,
        y=vals,
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        marker_color=colors
    ))

    fig.update_layout(
        title="Probabilistic Risk Profile Before vs After Recommended Action",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, max(100, max(vals) + 15)]),
        template="plotly_white",
        margin=dict(l=40, r=25, t=65, b=110)
    )
    return fig


def plot_cost_breakdown_chart(cost_dict: Dict[str, Any]) -> go.Figure:
    """Plot detailed breakdown of expected operational costs."""
    categories = ["Purchase Cost", "Holding Cost", "Stockout Penalty", "Wastage Cost"]
    costs = [
        cost_dict.get("purchase_cost", 0.0),
        cost_dict.get("holding_cost", 0.0),
        cost_dict.get("stockout_cost", 0.0),
        cost_dict.get("wastage_cost", 0.0)
    ]
    colors = ["#3B82F6", "#F59E0B", "#EF4444", "#EC4899"]

    fig = go.Figure(go.Bar(
        x=categories,
        y=costs,
        text=[f"RM {c:,.2f}" for c in costs],
        textposition="auto",
        marker_color=colors
    ))

    fig.update_layout(
        title="Expected Cost Components Under Recommended Policy (MYR)",
        yaxis_title="Expected Cost (RM)",
        template="plotly_white",
        margin=dict(l=40, r=25, t=65, b=110)
    )
    return fig


def plot_historical_backtest_cumulative_savings(monthly_df: pd.DataFrame) -> go.Figure:
    """Plot historical cumulative net savings across 33 walk-forward origins."""
    if monthly_df.empty or "net_savings_rm" not in monthly_df.columns:
        return go.Figure()

    df = monthly_df.copy()
    df["cumulative_savings_rm"] = df["net_savings_rm"].cumsum()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["month"],
        y=df["cumulative_savings_rm"],
        mode="lines+markers",
        line=dict(color="#10B981", width=3),
        marker=dict(size=6, color="#059669"),
        fill="tozeroy",
        fillcolor="rgba(16, 185, 129, 0.15)",
        name="Cumulative Net Savings (RM)"
    ))

    fig.update_layout(
        title="Cumulative Realised Net Savings Across 990 Historical Days (AI vs 14-Day Baseline)",
        xaxis_title="Replay Month",
        yaxis_title="Cumulative Net Savings (RM)",
        template="plotly_white",
        margin=dict(l=40, r=25, t=65, b=110)
    )
    return fig
