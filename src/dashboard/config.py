"""Configuration, labels, and metadata for the Streamlit dashboard."""
from typing import Dict, List

DEFAULT_DEMO_STORE = "STR_JHR_003"
DEFAULT_DEMO_PRODUCT = "PRD_CHK_01"

STORE_LABELS: Dict[str, str] = {
    "STR_KUL_001": "STR_KUL_001 — Kuala Lumpur Flagship Hypermarket",
    "STR_SGR_002": "STR_SGR_002 — Selangor Subang Supercenter",
    "STR_JHR_003": "STR_JHR_003 — Johor Bahru Commercial Mall",
    "STR_PNG_004": "STR_PNG_004 — Penang George Town Store"
}

PRODUCT_LABELS: Dict[str, str] = {
    "PRD_CHK_01": "Fresh whole chicken",
    "PRD_EGG_02": "Grade A eggs (tray of 30)",
    "PRD_VCH_03": "Cameron cabbage",
    "PRD_RIC_04": "Local white rice (10kg bag)",
    "PRD_OIL_05": "Pure palm cooking oil (5kg bottle)",
    "PRD_ONN_06": "Imported red onions",
}
PRODUCT_UNITS = {"PRD_CHK_01": "kg", "PRD_EGG_02": "trays", "PRD_VCH_03": "kg",
                 "PRD_RIC_04": "bags", "PRD_OIL_05": "bottles", "PRD_ONN_06": "kg"}
CATEGORY_MAP = {"PRD_CHK_01": "Poultry & Meat", "PRD_EGG_02": "Dairy & Eggs",
                "PRD_VCH_03": "Fresh Vegetables", "PRD_RIC_04": "Dry Staples",
                "PRD_OIL_05": "Cooking Oils & Fats", "PRD_ONN_06": "Fresh Vegetables"}

HORIZONS: List[str] = ["1d", "3d", "7d", "14d", "21d", "30d"]

SYNTHETIC_DATA_DISCLOSURE = (
    "Demo-mode retailer transaction records are synthetic and are used to demonstrate the "
    "end-to-end predictive procurement system workflow. Historical backtest performance shown "
    "for synthetic data reflects walk-forward validation on unseen simulated distributions and "
    "should not be interpreted as independent real-world commercial performance."
)
