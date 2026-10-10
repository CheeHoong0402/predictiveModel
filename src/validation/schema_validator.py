"""
schema_validator.py
===================
Deep inspection and schema validation for raw structured datasets (CSV, Parquet, JSON, Excel).
Uses Polars lazy scanning and PyArrow for memory-efficient handling of large multi-million row datasets.
Classifies datasets into the 25 official domain categories.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import polars as pl
import pandas as pd
import numpy as np

# Path setup
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.utils.logging import setup_logger

logger = setup_logger("schema_validator")

# Official 25 dataset classifications
DATASET_CATEGORIES = [
    "Point-of-Sale sales",
    "Inventory",
    "Purchase-price history",
    "Procurement",
    "Product master",
    "Supplier master",
    "Store master",
    "Promotion history",
    "Malaysian market prices",
    "Consumer Price Index",
    "Producer Price Index",
    "Fuel prices",
    "Currency exchange rates",
    "Commodity prices",
    "Weather",
    "Public holidays",
    "School holidays",
    "Wholesale and retail statistics",
    "Import statistics",
    "Export statistics",
    "Government assistance",
    "Government subsidies",
    "National budget policies",
    "Regional economic data",
    "Other",
]


def classify_dataset(filename: str, columns: List[str]) -> str:
    """Classifies a dataset into one of the 25 canonical categories based on name and column heuristics."""
    fn = filename.lower()
    cols = [c.lower() for c in columns]

    if "pos" in fn or ("quantity_sold" in cols and "selling_price" in cols and "opening_inventory" not in cols):
        return "Point-of-Sale sales"
    elif "inventory" in fn or ("opening_inventory_quantity" in cols or "closing_inventory_quantity" in cols):
        return "Inventory"
    elif "purchase_price" in fn or ("supplier_purchase_price" in cols and "opening_inventory" not in cols):
        return "Purchase-price history"
    elif "supplier" in fn or ("supplier_id" in cols and "supplier_lead_time_days" in cols):
        return "Supplier master"
    elif "store" in fn or ("store_id" in cols and "store_storage_capacity" in cols and "date" not in cols):
        return "Store master"
    elif "product" in fn or ("product_id" in cols and "product_shelf_life_days" in cols and "date" not in cols):
        return "Product master"
    elif "promotion" in fn or ("promotion_id" in cols and "discount_percentage" in cols):
        return "Promotion history"
    elif "pricecatcher" in fn or "lookup_item" in fn or "lookup_premise" in fn:
        return "Malaysian market prices"
    elif "fuel" in fn or ("ron95" in cols and "diesel" in cols):
        return "Fuel prices"
    elif "cpi" in fn:
        return "Consumer Price Index"
    elif "ppi" in fn:
        return "Producer Price Index"
    elif "exchange" in fn or ("usdyr" in fn or "usdmyr" in cols or "sgdmyr" in cols):
        return "Currency exchange rates"
    elif "commodit" in fn or ("brent_crude_usd" in cols or "palm_oil_myr_per_tonne" in cols):
        return "Commodity prices"
    elif "weather" in fn or ("max_daily_temperature" in cols or "daily_rainfall" in cols):
        return "Weather"
    elif "school_holiday" in fn:
        return "School holidays"
    elif "public_holiday" in fn or "holiday" in fn:
        return "Public holidays"
    elif "distributive" in fn or "iowrt" in fn or "wholesale_retail" in fn:
        return "Wholesale and retail statistics"
    elif "trade_enduse" in fn or "import" in fn:
        return "Import statistics"
    elif "trade_headline" in fn or "trade_sitc" in fn or "export" in fn:
        return "Export statistics"
    elif "assistance" in fn or "salary" in fn:
        return "Government assistance"
    elif "subsid" in fn:
        return "Government subsidies"
    elif "budget" in fn:
        return "National budget policies"
    elif "operational" in fn:
        return "Point-of-Sale sales" # Master operational contains POS sales
    return "Other"


class SchemaValidator:
    """Extracts schema, column semantics, and structural properties of structured datasets."""

    def __init__(self, sample_size: int = 10000):
        self.sample_size = sample_size

    def inspect_file(self, file_path: Path) -> Dict[str, Any]:
        result = {
            "file_path": str(file_path),
            "filename": file_path.name,
            "format": file_path.suffix.lstrip(".").lower(),
            "file_size_bytes": file_path.stat().st_size if file_path.exists() else 0,
            "rows": 0,
            "columns": 0,
            "column_names": [],
            "numeric_columns": [],
            "categorical_columns": [],
            "date_columns": [],
            "earliest_date": None,
            "latest_date": None,
            "is_empty_file": False,
            "is_template_only": False,
            "has_missing_identifiers": False,
            "missing_identifier_details": [],
            "classification": "Other",
            "inconsistent_types": [],
            "error": None
        }

        if not file_path.exists():
            result["error"] = "File not found"
            result["is_empty_file"] = True
            return result

        if result["file_size_bytes"] == 0:
            result["is_empty_file"] = True
            return result

        try:
            # Memory-efficient scanning via Polars
            if result["format"] == "parquet":
                lf = pl.scan_parquet(file_path)
                schema = lf.collect_schema()
                result["column_names"] = list(schema.names())
                result["columns"] = len(result["column_names"])
                
                # Fast row count from Parquet metadata
                import pyarrow.parquet as pq
                meta = pq.read_metadata(file_path)
                result["rows"] = meta.num_rows

                # Analyze sample if rows > 0
                if result["rows"] > 0:
                    sample_df = lf.limit(self.sample_size).collect()
                else:
                    sample_df = pl.DataFrame(schema=schema)

            elif result["format"] == "csv":
                try:
                    lf = pl.scan_csv(file_path, ignore_errors=True)
                    schema = lf.collect_schema()
                    result["column_names"] = list(schema.names())
                    result["columns"] = len(result["column_names"])
                    # Count rows without loading entire file in memory
                    with open(file_path, "rb") as f:
                        result["rows"] = sum(1 for _ in f) - 1 # minus header
                    result["rows"] = max(0, result["rows"])
                    
                    if result["rows"] > 0:
                        sample_df = lf.limit(self.sample_size).collect()
                    else:
                        sample_df = pl.DataFrame(schema=schema)
                except Exception:
                    # Fallback pandas chunk
                    pdf = pd.read_csv(file_path, nrows=self.sample_size, low_memory=False)
                    result["column_names"] = list(pdf.columns)
                    result["columns"] = len(result["column_names"])
                    with open(file_path, "rb") as f:
                        result["rows"] = sum(1 for _ in f) - 1
                    result["rows"] = max(0, result["rows"])
                    sample_df = pl.from_pandas(pdf)
            else:
                result["error"] = f"Unsupported format: {result['format']}"
                return result

            # Detect template-only files (headers present but 0 rows)
            if result["rows"] == 0 and result["columns"] > 0:
                result["is_template_only"] = True

            # Detect Column Types
            for col_name, dtype in sample_df.schema.items():
                if dtype.is_numeric():
                    result["numeric_columns"].append(col_name)
                elif dtype.is_temporal():
                    result["date_columns"].append(col_name)
                else:
                    # Check if string contains dates
                    if any(k in col_name.lower() for k in ["date", "time", "day", "month", "year"]):
                        result["date_columns"].append(col_name)
                    else:
                        result["categorical_columns"].append(col_name)

            # Determine date ranges using scan/aggregation
            if result["date_columns"] and result["rows"] > 0:
                primary_date_col = result["date_columns"][0]
                try:
                    # Compute min/max directly on lazy frame
                    date_stats = (
                        lf.select([
                            pl.col(primary_date_col).cast(pl.Utf8).min().alias("min_d"),
                            pl.col(primary_date_col).cast(pl.Utf8).max().alias("max_d")
                        ])
                        .collect()
                    )
                    min_val = date_stats["min_d"][0]
                    max_val = date_stats["max_d"][0]
                    if min_val and str(min_val).strip() != "":
                        result["earliest_date"] = str(min_val)[:10]
                    if max_val and str(max_val).strip() != "":
                        result["latest_date"] = str(max_val)[:10]
                except Exception as e:
                    logger.debug(f"Date range extraction error on {col_name}: {e}")

            # Classification
            result["classification"] = classify_dataset(result["filename"], result["column_names"])

            # Check identifier completeness for retailer datasets
            if result["classification"] in ["Point-of-Sale sales", "Inventory", "Purchase-price history"]:
                if "product_id" not in result["column_names"]:
                    result["has_missing_identifiers"] = True
                    result["missing_identifier_details"].append("Missing product_id")
                if "store_id" not in result["column_names"] and result["classification"] != "Purchase-price history":
                    result["has_missing_identifiers"] = True
                    result["missing_identifier_details"].append("Missing store_id")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Failed to inspect schema of {file_path.name}: {e}")

        return result
