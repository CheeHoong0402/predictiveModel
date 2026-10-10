"""
test_ingestion.py
=================
Automated test suite verifying the raw data ingestion layer:
- Retailer internal data loaders
- Public external downloaders
- Validation report integrity
"""

import pytest
import sys
import pandas as pd
from pathlib import Path

# Path setup
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.utils.paths import RAW_DATA_DIR, REPORTS_DIR
from src.ingestion.retailer_ingestion import (
    ProductInformationLoader,
    SupplierDataLoader,
    PromotionHistoryLoader,
    POSSalesLoader,
    InventoryLoader,
    PurchasePriceLoader,
    RetailerIngestionManager,
)
from src.ingestion.validate_raw_data import inspect_dataset, run_validation


def test_product_information_loader():
    loader = ProductInformationLoader()
    df = loader.get_default_product_catalog()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 5
    assert "product_id" in df.columns
    assert "product_shelf_life_days" in df.columns
    assert "controlled_goods_status" in df.columns


def test_supplier_data_loader():
    loader = SupplierDataLoader()
    df = loader.get_default_suppliers()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 5
    assert "supplier_id" in df.columns
    assert "supplier_lead_time_days" in df.columns
    assert "supplier_reliability_rate" in df.columns


def test_promotion_history_loader():
    loader = PromotionHistoryLoader()
    df = loader.generate_promotions(start_year=2022, end_year=2023)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "promotion_id" in df.columns
    assert "discount_percentage" in df.columns


def test_retailer_ingestion_manager_outputs():
    mgr = RetailerIngestionManager()
    res = mgr.run_full_ingestion()
    assert "products" in res
    assert "suppliers" in res
    assert "promotions" in res
    assert "sales" in res
    assert "inventory" in res
    assert "purchase_prices" in res

    assert (RAW_DATA_DIR / "retail_pos_sales.parquet").exists()
    assert (RAW_DATA_DIR / "retail_inventory.parquet").exists()
    assert (RAW_DATA_DIR / "retail_purchase_prices.parquet").exists()


def test_raw_dataset_inspection():
    pos_file = RAW_DATA_DIR / "retail_pos_sales.parquet"
    res = inspect_dataset(pos_file)
    assert res["status"] == "VALID"
    assert res["rows"] > 0
    assert res["columns"] > 5


def test_full_validation_run():
    audit_results = run_validation()
    assert len(audit_results) > 10
    # Check that critical files are valid
    valid_files = [r["filename"] for r in audit_results if r["status"] == "VALID"]
    assert "fuel_prices.csv" in valid_files
    assert "cpi_headline.csv" in valid_files
    assert "malaysia_weather_10y.csv" in valid_files
    assert "commodities_10y.csv" in valid_files
