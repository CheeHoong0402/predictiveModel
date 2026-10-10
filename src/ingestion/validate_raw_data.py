"""
validate_raw_data.py
====================
Inspects and validates all raw datasets extracted in data/raw/.
Reports row counts, column counts, date ranges, null metrics, and data integrity
across both internal retailer data and public external datasets.
"""

import sys
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any

# Path setup
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.utils.paths import RAW_DATA_DIR, REPORTS_DIR
from src.utils.logging import setup_logger

logger = setup_logger("validate_raw_data")


def inspect_dataset(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        return {"status": "MISSING", "filename": file_path.name, "path": str(file_path)}
    
    try:
        if file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path, low_memory=False)

        info = {
            "status": "VALID",
            "filename": file_path.name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": str(list(df.columns[:8])),
            "null_cells": int(df.isnull().sum().sum()),
        }
        
        # Check date range if present
        date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "day"])]
        if date_cols:
            d_series = pd.to_datetime(df[date_cols[0]], errors="coerce").dropna()
            if not d_series.empty:
                info["min_date"] = str(d_series.min())[:10]
                info["max_date"] = str(d_series.max())[:10]
                info["date_column_used"] = date_cols[0]

        return info
    except Exception as e:
        return {"status": "ERROR", "filename": file_path.name, "path": str(file_path), "error": str(e)}


def run_validation():
    logger.info("=== STARTING FULL RAW DATA INGESTION AUDIT ===")
    
    # Internal Retailer datasets
    retailer_files = [
        "retail_products.csv",
        "retail_suppliers.csv",
        "retail_promotions.csv",
        "retail_pos_sales.parquet",
        "retail_inventory.parquet",
        "retail_purchase_prices.parquet",
        "retailer_operational_data.parquet"
    ]

    # Public External datasets
    public_files = [
        "pricecatcher_lookup_item.parquet",
        "pricecatcher_lookup_premise.parquet",
        "pricecatcher_2024-01.parquet",
        "fuel_prices.csv",
        "cpi_headline.csv",
        "cpi_core.csv",
        "cpi_state.csv",
        "ppi.csv",
        "exchange_rates_10y.csv",
        "commodities_10y.csv",
        "malaysia_weather_10y.csv",
        "malaysia_public_holidays_10y.csv",
        "malaysia_school_holidays_10y.csv",
        "iowrt_distributive_trade.csv",
        "trade_headline.csv",
        "trade_sitc_1d.csv",
        "trade_enduse_bec.csv",
        "malaysia_government_salary_dates_10y.csv",
        "malaysia_government_assistance_10y.csv",
        "malaysia_subsidies_and_policies.csv",
        "malaysia_national_budget_policies.csv"
    ]

    results = []

    logger.info("--- 1. Retailer Internal Domain Datasets ---")
    for fn in retailer_files:
        path = RAW_DATA_DIR / fn
        res = inspect_dataset(path)
        res["category"] = "Retailer Internal"
        results.append(res)
        if res.get("status") == "VALID":
            logger.info(f"[PASS] {fn:<40} | Rows: {res['rows']:>8,} | Cols: {res['columns']:>3} | Date: {res.get('min_date', 'N/A')} to {res.get('max_date', 'N/A')}")
        else:
            logger.warning(f"[FAIL] {fn:<40} | Status: {res.get('status')} | Details: {res.get('error', 'Missing')}")

    logger.info("--- 2. Public External Macro & Micro Datasets ---")
    for fn in public_files:
        path = RAW_DATA_DIR / fn
        res = inspect_dataset(path)
        res["category"] = "Public External"
        results.append(res)
        if res.get("status") == "VALID":
            logger.info(f"[PASS] {fn:<40} | Rows: {res['rows']:>8,} | Cols: {res['columns']:>3} | Date: {res.get('min_date', 'N/A')} to {res.get('max_date', 'N/A')}")
        else:
            logger.warning(f"[FAIL] {fn:<40} | Status: {res.get('status')} | Details: {res.get('error', 'Missing')}")

    # Output CSV Report
    report_df = pd.DataFrame(results)
    validation_dir = REPORTS_DIR / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    report_path = validation_dir / "raw_data_validation_report.csv"
    report_df.to_csv(report_path, index=False)
    
    valid_count = sum(1 for r in results if r["status"] == "VALID")
    logger.info(f"Summary: {valid_count}/{len(results)} datasets VALID. Full report saved to {report_path}")
    logger.info("=== INGESTION AUDIT COMPLETED ===")
    return results


if __name__ == "__main__":
    run_validation()
