"""User-Upload Standardisation, Validation, and History Sufficiency Checker."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from .taxonomy import DataQualityStatus

REQUIRED_FILES = [
    "pos_sales.csv",
    "inventory_history.csv",
    "purchase_price_history.csv",
    "product_master.csv",
    "store_master.csv",
    "supplier_master.csv"
]

OPTIONAL_FILES = [
    "purchase_orders.csv",
    "promotion_history.csv"
]

MIN_REQUIRED_HISTORY_DAYS = 60


@dataclass
class UploadValidationResult:
    """Result of user-uploaded dataset validation."""
    is_valid: bool
    status: str
    message: str
    warnings: List[str] = field(default_factory=list)
    available_history_days: int = 0
    min_required_history_days: int = MIN_REQUIRED_HISTORY_DAYS
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)


def validate_user_uploaded_files(
    uploaded_files: Dict[str, Union[str, Path, pd.DataFrame]]
) -> UploadValidationResult:
    """Validate user-uploaded tables without triggering model retraining.
    
    Checks:
    1. Presence of required files
    2. Required column existence
    3. Date parseability
    4. Sufficient history length (>= 60 days)
    5. Non-empty inventory and price data
    """
    warnings: List[str] = []
    loaded_tables: Dict[str, pd.DataFrame] = {}

    # Check required files
    missing_files = []
    for req in REQUIRED_FILES:
        base_name = req.replace(".csv", "")
        if req not in uploaded_files and base_name not in uploaded_files:
            missing_files.append(req)

    if missing_files:
        return UploadValidationResult(
            is_valid=False,
            status="INVALID_MISSING_REQUIRED_FILES",
            message=f"Missing mandatory upload files: {', '.join(missing_files)}",
            warnings=warnings
        )

    # Load tables into memory
    for key, val in uploaded_files.items():
        table_name = Path(key).stem
        if isinstance(val, pd.DataFrame):
            loaded_tables[table_name] = val
        elif isinstance(val, (str, Path)):
            loaded_tables[table_name] = pd.read_csv(val)

    # Check optional files
    if "promotion_history" not in loaded_tables:
        warnings.append("NO_PROMOTION_HISTORY: Defaulting promotional flags to inactive.")
    if "purchase_orders" not in loaded_tables:
        warnings.append("NO_PURCHASE_ORDERS: Defaulting incoming pipeline orders to 0 units.")

    # Validate pos_sales dates and history length
    sales_df = loaded_tables.get("pos_sales")
    if sales_df is None or len(sales_df) == 0:
        return UploadValidationResult(
            is_valid=False,
            status=DataQualityStatus.INSUFFICIENT_DATA.value,
            message="POS sales table is empty.",
            warnings=warnings
        )

    date_col = next((c for c in ["date", "transaction_date", "sales_date"] if c in sales_df.columns), None)
    if not date_col:
        return UploadValidationResult(
            is_valid=False,
            status="INVALID_MISSING_REQUIRED_COLUMNS",
            message="POS sales table is missing a 'date' column.",
            warnings=warnings
        )

    try:
        sales_dates = pd.to_datetime(sales_df[date_col])
        min_date = sales_dates.min()
        max_date = sales_dates.max()
        history_days = (max_date - min_date).days + 1
    except Exception as e:
        return UploadValidationResult(
            is_valid=False,
            status="INVALID_DATE_FORMAT",
            message=f"Failed to parse dates in pos_sales: {str(e)}",
            warnings=warnings
        )

    if history_days < MIN_REQUIRED_HISTORY_DAYS:
        return UploadValidationResult(
            is_valid=False,
            status="INSUFFICIENT_HISTORY",
            message=(
                f"Uploaded history is {history_days} days, but minimum {MIN_REQUIRED_HISTORY_DAYS} days "
                "is strictly required to construct temporal rolling lag features without lookahead leakage."
            ),
            warnings=warnings,
            available_history_days=history_days
        )

    # Validate inventory positions
    inv_df = loaded_tables.get("inventory_history")
    if inv_df is None or len(inv_df) == 0:
        return UploadValidationResult(
            is_valid=False,
            status="NO_CURRENT_INVENTORY",
            message="Inventory history table is empty; cannot establish current stock levels.",
            warnings=warnings,
            available_history_days=history_days
        )

    # Validate price history
    price_df = loaded_tables.get("purchase_price_history")
    if price_df is None or len(price_df) == 0:
        warnings.append("PRICE_FORECAST_UNAVAILABLE: Operating in demand-only inventory replenishment mode.")

    final_status = DataQualityStatus.ACCEPTABLE_WITH_WARNINGS.value if warnings else DataQualityStatus.GOOD.value
    return UploadValidationResult(
        is_valid=True,
        status=final_status,
        message="User upload validated successfully for model inference without retraining.",
        warnings=warnings,
        available_history_days=history_days,
        tables=loaded_tables
    )
