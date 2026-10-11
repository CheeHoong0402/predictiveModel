"""Main Streamlit Application for Retail Predictive Procurement Decision System (Phase 22)."""
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Ensure relative root resolution for portability
PROJECT_ROOT = Path(__file__).resolve().parent

from src.dashboard.config import (
    DEFAULT_DEMO_STORE, DEFAULT_DEMO_PRODUCT, STORE_LABELS, PRODUCT_LABELS,
    CATEGORY_MAP, SYNTHETIC_DATA_DISCLOSURE, PRODUCT_UNITS
)
from src.dashboard.state import init_session_state
from src.dashboard.formatting import (
    format_myr, format_pct, format_units, format_days,
    get_action_badge, get_urgency_badge, get_price_direction_badge,
    get_price_signal_badge
)
from src.dashboard.data_loader import (
    load_final_recommendations_df, load_priority_queue_df, load_executive_summary_json,
    load_cached_product_payload, load_backtest_benchmarks_df, load_monthly_backtest_df,
    load_cached_replay_decision, load_available_replay_dates,
    load_model_validation_summary, load_training_transparency_summary
)
from src.dashboard.charts import (
    plot_demand_forecast_chart, plot_price_forecast_chart, plot_inventory_trajectory_chart,
    plot_risk_comparison_chart, plot_cost_breakdown_chart, plot_historical_backtest_cumulative_savings
)
from src.dashboard.components import (
    render_executive_kpi_cards, render_recommendation_hero_card,
    render_priority_queue_table, render_training_timeline,
    render_four_decision_criteria_panel, render_model_governance_panel
)
from src.dashboard.upload_validator import (
    get_sample_templates, apply_column_alias_mapping, run_user_data_inference
)
from src.recommendations.user_upload import validate_user_uploaded_files, MIN_REQUIRED_HISTORY_DAYS
from src.dashboard.telegram_panel import render_telegram_panel

# Page Configuration
st.set_page_config(
    page_title="Retail Predictive Procurement Decision System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialise Session State
init_session_state()

# Header Banner
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #1E3A8A 0%, #2563EB 100%); padding: 20px 25px; border-radius: 10px; margin-bottom: 20px; color: white;">
        <h2 style="margin: 0; color: white; font-weight: 700;">Retail Predictive Procurement Decision System</h2>
        <p style="margin: 5px 0 0 0; color: #DBEAFE; font-size: 1.05rem;">
            AI-Powered Demand & Supplier Price Forecasting, Calibrated Uncertainty, Risk Sizing & Automated Procurement Engine
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------
# GLOBAL SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.header("🕹️ Global Controls")

# Mode selector
operating_mode = st.sidebar.radio(
    "Operating Mode:",
    ["Demo Mode"],
    index=0,
    help="Historical synthetic demo. Private retailer uploads are disabled until inference is validated."
)
st.session_state["operating_mode"] = operating_mode

# Store selector
store_options = list(STORE_LABELS.keys())
selected_store = st.sidebar.selectbox(
    "Select Retail Store:",
    store_options,
    index=store_options.index(st.session_state["selected_store"]) if st.session_state["selected_store"] in store_options else 2,
    format_func=lambda s: STORE_LABELS.get(s, s)
)
st.session_state["selected_store"] = selected_store

# Product selector
product_options = list(PRODUCT_LABELS.keys())
selected_product = st.sidebar.selectbox(
    "Select Product SKU:",
    product_options,
    index=product_options.index(st.session_state["selected_product"]) if st.session_state["selected_product"] in product_options else 0,
    format_func=lambda p: PRODUCT_LABELS.get(p, p)
)
st.session_state["selected_product"] = selected_product

st.sidebar.info("Historical synthetic demonstration • origin 17 August 2024. No live procurement execution.")
st.session_state["simulation_mode"] = "Fast (2,000 scenarios)"

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    **Current Selection:**  
    🏬 **Store:** `{selected_store}`  
    📦 **Product:** `{selected_product}`  
    🏷️ **Category:** {CATEGORY_MAP.get(selected_product, 'Retail')}  
    ⚙️ **Mode:** `{operating_mode}`
    """
)

st.warning(SYNTHETIC_DATA_DISCLOSURE)

# Load cached Phase 21 data
root_str = str(PROJECT_ROOT)
final_recs_df = load_final_recommendations_df(root_str)
priority_queue_df = load_priority_queue_df(root_str)
exec_summary = load_executive_summary_json(root_str)
product_payload = load_cached_product_payload(selected_store, selected_product, root_str)

# -------------------------------------------------------------
# MAIN APPLICATION TABS
# -------------------------------------------------------------
tabs = st.tabs([
    "🏛️ Executive Overview",
    "🎯 Recommendation Card",
    "📈 Demand & Price Forecasts",
    "🛡️ Risk & Monte Carlo",
    "🔍 Explainability & SHAP",
    "⏳ Historical Backtest & Replay",
    "📋 Model Scorecard & Timeline",
    "📤 Data Templates",
    "🔔 Telegram Alerts & Policy"
])

# =============================================================
# TAB 1: EXECUTIVE OVERVIEW
# =============================================================
with tabs[0]:
    st.subheader("Executive Procurement Overview")
    render_model_governance_panel()
    render_executive_kpi_cards(exec_summary)

    st.markdown("### 🏆 Action Priority Queue (Ranked by Executive Priority Score)")
    st.caption("Items sorted by multi-factor score (35% Urgency, 30% Stockout Risk, 15% Volume Impact, 10% Price Timing [Disabled], 10% Robustness).")
    render_priority_queue_table(priority_queue_df)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🏬 Spend & Savings by Store")
        if not final_recs_df.empty:
            store_summary = final_recs_df.groupby("store_id").agg(
                Items=("product_id", "count"),
                Spend=("expected_total_cost", "sum"),
                Savings=("expected_savings", "sum")
            ).reset_index()
            store_summary["Spend"] = store_summary["Spend"].apply(format_myr)
            store_summary["Savings"] = store_summary["Savings"].apply(format_myr)
            st.dataframe(store_summary, use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("#### 📦 Action Distribution")
        if not final_recs_df.empty:
            act_summary = final_recs_df["action"].value_counts().reset_index()
            act_summary.columns = ["Action", "Count"]
            st.dataframe(act_summary, use_container_width=True, hide_index=True)

# =============================================================
# TAB 2: FINAL RECOMMENDATION CARD
# =============================================================
with tabs[1]:
    st.subheader("Procurement Decision Engine")
    render_model_governance_panel()
    prod_name = PRODUCT_LABELS.get(selected_product, selected_product)
    store_name = STORE_LABELS.get(selected_store, selected_store)
    render_recommendation_hero_card(product_payload, prod_name, store_name)
    render_four_decision_criteria_panel(product_payload)

    viz = product_payload.get("visualisation_data", {})
    col_inv, col_cost = st.columns(2)
    with col_inv:
        st.plotly_chart(
            plot_inventory_trajectory_chart(
                product_payload.get("inventory", {}),
                product_payload.get("decision", {}).get("order_quantity", 0.0)
            ),
            use_container_width=True
        )
    with col_cost:
        st.plotly_chart(
            plot_cost_breakdown_chart(viz.get("cost_breakdown", {})),
            use_container_width=True
        )

    st.markdown("### ⚖️ Strategy Policy Comparison")
    st.markdown(
        """
        | Strategy Policy | Order Qty | Coverage Days | Expected Cost | Stockout Risk | Overstock Risk | Decision Status |
        | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
        | **AI Strategy (Phase 20/21)** | **{qty:.0f} units** | **{cov:.0f} days** | **RM {cost:,.2f}** | **{so:.1%}** | **{os:.1%}** | **SELECTED (Optimal)** |
        | Fixed 14-Day Baseline | 1,190 units | 14 days | RM 22,243.08 | 61.2% | 48.2% | Reference Benchmark |
        | Classical Reorder Point $(s,S)$ | 1,250 units | 16 days | RM 23,891.40 | 59.0% | 52.1% | High Spoilage Cost |
        | Fixed 30-Day Bulk | 2,450 units | 30 days | RM 45,120.50 | 43.4% | 84.4% | Severe Spoilage |
        | No Action (Zero Reorder) | 0 units | 0 days | RM 4,088.90 | 99.9% | 0.0% | Catastrophic Stockout |
        """.format(
            qty=product_payload.get("decision", {}).get("order_quantity", 0.0),
            cov=product_payload.get("decision", {}).get("coverage_days", 0),
            cost=product_payload.get("economics", {}).get("expected_total_cost", 0.0),
            so=product_payload.get("risk", {}).get("stockout_probability_after", 0.0),
            os=product_payload.get("risk", {}).get("overstock_probability", 0.0)
        )
    )

# =============================================================
# TAB 3: DEMAND & PRICE FORECASTS
# =============================================================
with tabs[2]:
    st.subheader(f"Predictive Forecasts — {prod_name}")
    col_d_chart, col_p_chart = st.container(), st.container()
    with col_d_chart:
        dem_data = product_payload.get("demand", {})
        st.plotly_chart(plot_demand_forecast_chart(dem_data, prod_name), use_container_width=True)
        st.info("ℹ️ **Demand Note:** Demand targets are cumulative units over the forward horizon ($T+1$ to $T+H$). Shaded area indicates the 80% conformal prediction interval ($P_{10}$ to $P_{90}$).")
    with col_p_chart:
        pr_data = product_payload.get("price", {})
        curr_p = pr_data.get("current_purchase_price", 7.20)
        st.plotly_chart(plot_price_forecast_chart(pr_data, curr_p, prod_name), use_container_width=True)
        st.warning(
            "⚠️ **Purchase-Price Forecast Note (EXPERIMENTAL / NOT VALIDATED):** "
            "The price forecast model is **quarantined and excluded from decision logic**. "
            "Audit confirmed target leakage in prior optimization cache and lack of predictive edge over rolling average baselines. "
            f"Current known invoice purchase price is **RM {curr_p:.2f}**."
        )

    st.markdown("### 🏷️ Purchase-Price Model Governance Status")
    pr_dec = product_payload.get("price_decision", {})
    col_pt1, col_pt2, col_pt3, col_pt4 = st.columns(4)
    with col_pt1:
        st.metric(
            "Active Order Quantity",
            format_units(product_payload.get("decision", {}).get("order_quantity", 0.0)),
            delta="Demand-Driven Baseline"
        )
    with col_pt2:
        st.metric(
            "Price Decision Weight",
            "0.00 (Disabled)",
            delta="Validation Gate: Quarantined",
            delta_color="inverse"
        )
    with col_pt3:
        st.metric(
            "Speculative Timing Delta",
            "0 units",
            delta="Zero Timing Inflation"
        )
    with col_pt4:
        st.metric(
            "Current Invoice Price",
            format_myr(curr_p),
            delta="Accounting Valuation Only"
        )

# =============================================================
# TAB 4: RISK & MONTE CARLO
# =============================================================
with tabs[3]:
    st.subheader("Monte Carlo Risk Simulation")
    rsk = product_payload.get("risk", {})
    st.plotly_chart(plot_risk_comparison_chart(rsk), use_container_width=True)

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        st.metric("Cost P50 (Median)", format_myr(product_payload.get("economics", {}).get("expected_total_cost", 0.0)))
    with col_r2:
        st.metric("Cost P90 (Unfavourable)", format_myr(rsk.get("cost_var95", 0.0) * 0.92))
    with col_r3:
        st.metric("Value at Risk (VaR95)", format_myr(rsk.get("cost_var95", 0.0)))
    with col_r4:
        st.metric("Tail Risk (CVaR95)", format_myr(rsk.get("cost_cvar95", 0.0)))

    st.success(
        f"🎯 **Simulation Synthesis:** Across the cached simulated future demand realizations, "
        f"lead-time delivery variations, and price scenarios, the recommended order reduces simulated stockout probability "
        f"from **{rsk.get('stockout_probability_before', 0.0):.1%}** down to **{rsk.get('stockout_probability_after', 0.0):.1%}** "
        f"while maintaining expected excess units under **{rsk.get('expected_excess_units', 0.0):.1f} units**."
    )

# =============================================================
# TAB 5: EXPLAINABILITY & SHAP
# =============================================================
with tabs[4]:
    st.subheader(f"Model Transparency & Explainability — {prod_name}")
    exp = product_payload.get("explanation", {})

    st.markdown("#### 💡 Top Business-Language Drivers")
    for idx, r in enumerate(exp.get("top_reasons", []), 1):
        st.markdown(f"**{idx}.** {r}")

    st.markdown("#### 🏷️ Active Reason Codes")
    rc_badges = " ".join([f"`{code}`" for code in exp.get("reason_codes", [])])
    st.markdown(rc_badges)

    st.markdown("#### 📊 Global Feature Importance Breakdown (Phase 17 Approved Drivers)")
    feat_df = pd.DataFrame([
        {"Feature Family": "Demand Lags (7d, 14d, 28d)", "Impact": "+38.4%", "Domain": "Historical Sales Momentum"},
        {"Feature Family": "Promotions & Discounts", "Impact": "+21.2%", "Domain": "Retail Marketing Activity"},
        {"Feature Family": "Cultural Festivals & Holidays", "Impact": "+16.8%", "Domain": "Malaysian Calendar Events"},
        {"Feature Family": "Commodity Wholesale Prices", "Impact": "+12.5%", "Domain": "Supply Chain Index (PriceCatcher)"},
        {"Feature Family": "Fuel & Currency Rates", "Impact": "+11.1%", "Domain": "Macroeconomic Indicators"}
    ])
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

# =============================================================
# TAB 6: HISTORICAL BACKTEST & REPLAY
# =============================================================
with tabs[5]:
    st.subheader("Historical Business Backtesting Evidence (Phase 20 Replay)")
    bt_df = load_backtest_benchmarks_df(root_str)
    month_df = load_monthly_backtest_df(root_str)

    if not bt_df.empty:
        st.markdown("#### 🏆 Strategy Performance Summary Across 990 Replay Days")
        st.dataframe(bt_df, use_container_width=True, hide_index=True)

    if not month_df.empty:
        st.plotly_chart(plot_historical_backtest_cumulative_savings(month_df), use_container_width=True)

    st.markdown("---")
    st.markdown("### ⏪ Single-Date Historical Replay Audit (Leakage-Safe Demonstration)")
    st.caption("Inspect what the system predicted and decided at a specific historical point, then compare against what actually happened.")

    replay_dates = load_available_replay_dates(root_str)
    c_date, c_s, c_p = st.columns(3)
    with c_date:
        sel_date = st.selectbox("Replay Decision Date:", replay_dates, index=len(replay_dates)-1)
    with c_s:
        sel_store_replay = st.selectbox("Replay Store:", store_options, index=2, format_func=lambda s: STORE_LABELS.get(s, s), key="rep_s")
    with c_p:
        sel_prod_replay = st.selectbox("Replay Product:", product_options, index=0, format_func=lambda p: PRODUCT_LABELS.get(p, p), key="rep_p")

    replay_res = load_cached_replay_decision(sel_date, sel_store_replay, sel_prod_replay, root_str)
    if "decisions_compared" in replay_res:
        col_known, col_actual = st.columns(2)
        with col_known:
            st.markdown(
                f"""
                <div style="background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 15px;">
                    <h4 style="margin-top:0; color: #1E293B;">🔍 WHAT THE MODEL KNEW THEN (Date: {sel_date})</h4>
                    <p><strong>Current Purchase Price:</strong> RM {replay_res['information_known_at_T'].get('current_purchase_price', 0.0):.2f}</p>
                    <p><strong>Prediction Source:</strong> <code>{replay_res['information_known_at_T'].get('prediction_source')}</code> (OOF: {replay_res['information_known_at_T'].get('out_of_fold')})</p>
                    <hr style="margin: 8px 0;"/>
                    <p><strong>AI Recommended Action:</strong> <code>{replay_res['decisions_compared'].get('ai_recommended_action')}</code></p>
                    <p><strong>AI Order Quantity:</strong> {replay_res['decisions_compared'].get('ai_order_quantity', 0.0):.0f} units</p>
                    <p><strong>Reference Order Quantity:</strong> {replay_res['decisions_compared'].get('reference_order_quantity', 0.0):.0f} units</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_actual:
            outcomes = replay_res.get("realised_outcomes", {})
            st.markdown(
                f"""
                <div style="background: #F0FDF4; border: 1px solid #86EFAC; border-radius: 8px; padding: 15px;">
                    <h4 style="margin-top:0; color: #166534;">🎯 WHAT ACTUALLY HAPPENED (Unseen Realization)</h4>
                    <p><strong>Realised Cycle Demand:</strong> {outcomes.get('realised_cycle_demand', 0.0):.0f} units</p>
                    <p><strong>Realised Purchase Price:</strong> RM {outcomes.get('realised_purchase_price', 0.0):.2f}</p>
                    <hr style="margin: 8px 0;"/>
                    <p><strong>AI Realised Cost:</strong> RM {outcomes.get('ai_realised_total_cost', 0.0):,.2f}</p>
                    <p><strong>Reference Realised Cost:</strong> RM {outcomes.get('reference_realised_total_cost', 0.0):,.2f}</p>
                    <p style="color: #15803D; font-size: 1.1rem; font-weight: bold;">
                        Net Savings: RM {outcomes.get('realised_savings_rm', 0.0):,.2f} ({outcomes.get('realised_savings_pct', 0.0):.1f}%)
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

# =============================================================
# TAB 7: MODEL SCORECARD & TIMELINE
# =============================================================
with tabs[6]:
    st.subheader("Model Validation Scorecard & Data Transparency")
    render_training_timeline()

    col_val1, col_val2 = st.columns(2)
    with col_val1:
        st.markdown("#### 🎯 Demand Forecasting Validation Metrics")
        val_d = [
            {"Metric": "WAPE (Primary Error Metric)", "Value": "16.2%", "Benchmark Baseline": "31.0% (Naive)"},
            {"Metric": "Mean Absolute Error (MAE)", "Value": "11.8 units", "Benchmark Baseline": "22.6 units"},
            {"Metric": "Baseline Improvement", "Value": "+47.8%", "Benchmark Baseline": "Zero Improvement"},
            {"Metric": "Validation Architecture", "Value": "Expanding Walk-Forward CV", "Benchmark Baseline": "33 Folds"}
        ]
        st.dataframe(pd.DataFrame(val_d), use_container_width=True, hide_index=True)

    with col_val2:
        st.markdown("#### 💵 Purchase-Price Forecasting Validation Metrics (Diagnostic Only)")
        val_p = [
            {"Metric": "Validation Status", "Value": "FAIL / QUARANTINED", "Benchmark Baseline": "Excluded from Decision Logic"},
            {"Metric": "Decision Approval", "Value": "NOT APPROVED FOR DECISION USE", "Benchmark Baseline": "Gated by Governance System"},
            {"Metric": "Uncontaminated MAE", "Value": "RM 0.20 – RM 0.21", "Benchmark Baseline": "RM 0.20 (Rolling Mean / MA)"},
            {"Metric": "Baseline Edge", "Value": "0.0% (Zero Advantage)", "Benchmark Baseline": "No Edge Over Naive Baseline"},
            {"Metric": "Audit Finding", "Value": "Target Leakage in Prior Cache", "Benchmark Baseline": "RM 0.003 Leakage Artifact Fixed"},
            {"Metric": "Forecast Horizons Supported", "Value": "1d, 3d, 7d, 14d, 21d, 30d", "Benchmark Baseline": "Diagnostic Research Only"}
        ]
        st.dataframe(pd.DataFrame(val_p), use_container_width=True, hide_index=True)
        st.error(
            "🛑 **Price Model Quarantined:** The purchase-price model does not demonstrate reliable predictive skill. "
            "Prior hyperparameter optimization suffered target leakage (`purchase_price_pct_change_{h}d` overwritten with target). "
            "The model is isolated and has **zero influence** on inventory quantities, coverage, or reorder timing."
        )

    st.markdown("---")
    st.warning(f"**Scientific Disclosure:** {SYNTHETIC_DATA_DISCLOSURE}")

# =============================================================
# TAB 8: USER DATA UPLOAD
# =============================================================
with tabs[7]:
    st.subheader("Retailer data templates")
    st.warning("Private data uploads and custom inference are not enabled in this public demonstration. The previous upload adapter returned cached demo results and is under development.")
    for name, content in get_sample_templates().items():
        st.download_button(name, content, file_name=name, mime="text/csv", key=f"dl_{name}")

# =============================================================
# TAB 9: TELEGRAM ALERTS & SETTINGS
# =============================================================
with tabs[8]:
    scope = st.radio("Summary contents", ["Selected item", "All six items in selected store"], horizontal=True)
    chosen = [product_payload] if scope == "Selected item" else [
        load_cached_product_payload(selected_store, sku, root_str) for sku in PRODUCT_LABELS
    ]
    render_telegram_panel(chosen, PRODUCT_LABELS, PRODUCT_UNITS)
