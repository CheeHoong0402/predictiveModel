"""Streamlit Dashboard Support Package (Phase 22)."""
from .config import (
    DEFAULT_DEMO_STORE, DEFAULT_DEMO_PRODUCT, STORE_LABELS, PRODUCT_LABELS,
    CATEGORY_MAP, HORIZONS, SYNTHETIC_DATA_DISCLOSURE
)
from .formatting import format_myr, format_pct, format_units, format_days, get_action_badge, get_urgency_badge
from .data_loader import (
    load_final_recommendations_df, load_priority_queue_df, load_executive_summary_json,
    load_cached_product_payload, load_backtest_benchmarks_df, load_monthly_backtest_df,
    load_cached_replay_decision, load_available_replay_dates,
    load_model_validation_summary, load_training_transparency_summary
)
from .charts import (
    plot_demand_forecast_chart, plot_price_forecast_chart, plot_inventory_trajectory_chart,
    plot_risk_comparison_chart, plot_cost_breakdown_chart, plot_historical_backtest_cumulative_savings
)
from .components import render_executive_kpi_cards, render_recommendation_hero_card, render_priority_queue_table, render_training_timeline
from .upload_validator import get_sample_templates, apply_column_alias_mapping, run_user_data_inference
from .state import init_session_state

__all__ = [
    "DEFAULT_DEMO_STORE",
    "DEFAULT_DEMO_PRODUCT",
    "STORE_LABELS",
    "PRODUCT_LABELS",
    "CATEGORY_MAP",
    "HORIZONS",
    "SYNTHETIC_DATA_DISCLOSURE",
    "format_myr",
    "format_pct",
    "format_units",
    "format_days",
    "get_action_badge",
    "get_urgency_badge",
    "load_final_recommendations_df",
    "load_priority_queue_df",
    "load_executive_summary_json",
    "load_cached_product_payload",
    "load_backtest_benchmarks_df",
    "load_monthly_backtest_df",
    "load_cached_replay_decision",
    "load_available_replay_dates",
    "load_model_validation_summary",
    "load_training_transparency_summary",
    "plot_demand_forecast_chart",
    "plot_price_forecast_chart",
    "plot_inventory_trajectory_chart",
    "plot_risk_comparison_chart",
    "plot_cost_breakdown_chart",
    "plot_historical_backtest_cumulative_savings",
    "render_executive_kpi_cards",
    "render_recommendation_hero_card",
    "render_priority_queue_table",
    "render_training_timeline",
    "get_sample_templates",
    "apply_column_alias_mapping",
    "run_user_data_inference",
    "init_session_state"
]
