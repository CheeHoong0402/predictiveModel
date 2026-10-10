"""Session State initialisation and management."""
import streamlit as st
from .config import DEFAULT_DEMO_STORE, DEFAULT_DEMO_PRODUCT


def init_session_state() -> None:
    """Initialize default session state keys if not already present."""
    if "operating_mode" not in st.session_state:
        st.session_state["operating_mode"] = "Demo Mode"
    if "selected_store" not in st.session_state:
        st.session_state["selected_store"] = DEFAULT_DEMO_STORE
    if "selected_product" not in st.session_state:
        st.session_state["selected_product"] = DEFAULT_DEMO_PRODUCT
    if "simulation_mode" not in st.session_state:
        st.session_state["simulation_mode"] = "Fast (2,000 scenarios)"
    if "selected_replay_date" not in st.session_state:
        st.session_state["selected_replay_date"] = "2024-08-17"
    if "user_uploaded_tables" not in st.session_state:
        st.session_state["user_uploaded_tables"] = {}
    if "user_analysis_results" not in st.session_state:
        st.session_state["user_analysis_results"] = None
    if "user_validation_status" not in st.session_state:
        st.session_state["user_validation_status"] = None
