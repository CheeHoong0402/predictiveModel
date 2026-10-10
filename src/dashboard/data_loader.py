"""Cached data loaders and state adapters for Streamlit application."""
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import pandas as pd
import streamlit as st

from src.utils.paths import PROJECT_ROOT
from src.recommendations.payloads import (
    get_product_decision_payload,
    get_executive_dashboard_payload,
    get_model_validation_payload,
    get_training_data_transparency_payload,
    get_historical_replay_payload
)


@st.cache_data(show_spinner=False)
def load_final_recommendations_df(root_dir: str = str(PROJECT_ROOT)) -> pd.DataFrame:
    """Load final recommendations parquet."""
    path = Path(root_dir) / "data/recommendations/final_recommendations.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_priority_queue_df(root_dir: str = str(PROJECT_ROOT)) -> pd.DataFrame:
    """Load ranked action priority queue."""
    path = Path(root_dir) / "data/recommendations/action_priority_queue.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_executive_summary_json(root_dir: str = str(PROJECT_ROOT)) -> Dict[str, Any]:
    """Load precomputed executive summary JSON."""
    path = Path(root_dir) / "reports/recommendations/executive_summary.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback to direct payload generator
    return get_executive_dashboard_payload(root=Path(root_dir))


@st.cache_data(show_spinner=False)
def load_cached_product_payload(
    store_id: str,
    product_id: str,
    root_dir: str = str(PROJECT_ROOT)
) -> Dict[str, Any]:
    """Retrieve full product decision and chart datasets."""
    return get_product_decision_payload(store_id=store_id, product_id=product_id, root=Path(root_dir))


@st.cache_data(show_spinner=False)
def load_backtest_benchmarks_df(root_dir: str = str(PROJECT_ROOT)) -> pd.DataFrame:
    """Load Phase 20 strategy comparison benchmarks."""
    path = Path(root_dir) / "reports/backtesting/ai_vs_reference.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_monthly_backtest_df(root_dir: str = str(PROJECT_ROOT)) -> pd.DataFrame:
    """Load monthly backtest timeline data."""
    path = Path(root_dir) / "reports/backtesting/monthly_performance.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_cached_replay_decision(
    date: str,
    store_id: str,
    product_id: str,
    root_dir: str = str(PROJECT_ROOT)
) -> Dict[str, Any]:
    """Load single-date historical replay audit."""
    return get_historical_replay_payload(date=date, store_id=store_id, product_id=product_id, root=Path(root_dir))


@st.cache_data(show_spinner=False)
def load_available_replay_dates(root_dir: str = str(PROJECT_ROOT)) -> List[str]:
    """Retrieve list of historical walk-forward origin dates available for replay."""
    path = Path(root_dir) / "data/backtesting/decision_backtest.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        dates = sorted(df["decision_date"].astype(str).unique().tolist())
        return dates
    return ["2024-08-17", "2024-07-18", "2024-06-18", "2024-05-19"]


@st.cache_data(show_spinner=False)
def load_model_validation_summary(root_dir: str = str(PROJECT_ROOT)) -> Dict[str, Any]:
    return get_model_validation_payload(root=Path(root_dir))


@st.cache_data(show_spinner=False)
def load_training_transparency_summary(root_dir: str = str(PROJECT_ROOT)) -> Dict[str, Any]:
    return get_training_data_transparency_payload(root=Path(root_dir))
