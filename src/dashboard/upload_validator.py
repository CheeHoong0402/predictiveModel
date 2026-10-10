"""User Data Upload Validator, Template Generator, and Inference Connector."""
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from src.recommendations.user_upload import (
    REQUIRED_FILES, OPTIONAL_FILES, MIN_REQUIRED_HISTORY_DAYS,
    validate_user_uploaded_files, UploadValidationResult
)
from src.recommendations.engine import generate_final_recommendation


def get_sample_templates() -> Dict[str, str]:
    """Generate downloadable CSV template strings for all supported upload types."""
    templates = {
        "pos_sales.csv": (
            "date,store_id,product_id,quantity_sold,selling_price\n"
            "2024-01-01,STR_KUL_001,PRD_CHK_01,85,9.40\n"
            "2024-01-02,STR_KUL_001,PRD_CHK_01,92,9.40\n"
        ),
        "inventory_history.csv": (
            "date,store_id,product_id,inventory_on_hand,spoilage_units\n"
            "2024-01-01,STR_KUL_001,PRD_CHK_01,150,0\n"
            "2024-01-02,STR_KUL_001,PRD_CHK_01,65,0\n"
        ),
        "purchase_price_history.csv": (
            "date,store_id,product_id,supplier_id,purchase_price\n"
            "2024-01-01,STR_KUL_001,PRD_CHK_01,SUP_KLS_01,7.20\n"
            "2024-01-02,STR_KUL_001,PRD_CHK_01,SUP_KLS_01,7.20\n"
        ),
        "product_master.csv": (
            "product_id,product_name,category,shelf_life_days,pack_size\n"
            "PRD_CHK_01,Fresh Whole Chicken,Poultry & Meat,3,1\n"
            "PRD_EGG_02,Grade A Fresh Eggs,Dairy & Eggs,14,30\n"
        ),
        "store_master.csv": (
            "store_id,store_name,state,region\n"
            "STR_KUL_001,Kuala Lumpur Hypermarket,Kuala Lumpur,Central\n"
            "STR_SGR_002,Selangor Supercenter,Selangor,Central\n"
        ),
        "supplier_master.csv": (
            "supplier_id,supplier_name,lead_time_days,minimum_order_quantity,case_size\n"
            "SUP_KLS_01,Klang Valley Livestock,2,50,10\n"
            "SUP_SLG_02,Selangor Poultry,2,100,30\n"
        ),
        "purchase_orders.csv": (
            "order_id,store_id,product_id,supplier_id,order_date,expected_delivery_date,order_quantity\n"
            "PO_20240101_01,STR_KUL_001,PRD_CHK_01,SUP_KLS_01,2024-01-01,2024-01-03,50\n"
        ),
        "promotion_history.csv": (
            "start_date,end_date,store_id,product_id,discount_pct,promo_type\n"
            "2024-01-15,2024-01-20,STR_KUL_001,PRD_CHK_01,0.10,FESTIVAL\n"
        )
    }
    return templates


def apply_column_alias_mapping(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Map common uploaded column aliases to canonical column names."""
    df_mapped = df.copy()
    alias_map = {
        "sku": "product_id",
        "sku_id": "product_id",
        "item_code": "product_id",
        "product_code": "product_id",
        "outlet_id": "store_id",
        "branch_id": "store_id",
        "shop_id": "store_id",
        "qty": "quantity_sold",
        "units": "quantity_sold",
        "sales_qty": "quantity_sold",
        "price": "selling_price",
        "unit_price": "selling_price",
        "stock": "inventory_on_hand",
        "closing_stock": "inventory_on_hand",
        "cost": "purchase_price",
        "cost_price": "purchase_price"
    }
    applied = []
    for col in df_mapped.columns:
        c_lower = col.strip().lower()
        if c_lower in alias_map and alias_map[c_lower] not in df_mapped.columns:
            canonical = alias_map[c_lower]
            df_mapped.rename(columns={col: canonical}, inplace=True)
            applied.append(f"Mapped column '{col}' &rarr; '{canonical}'")
    return df_mapped, applied


def run_user_data_inference(
    validated_tables: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Run model inference on validated user tables using cached model artifacts without retraining.
    
    Generates recommendations across unique store-product pairs discovered in user data.
    """
    raise NotImplementedError(
        "Custom upload inference is not validated. Cached demo recommendations must not be presented as analysis of uploaded data."
    )
