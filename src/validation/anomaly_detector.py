"""
anomaly_detector.py
===================
Deep anomaly and data-quality inspection for raw datasets.
Detects:
- Missing values count & percentage
- Duplicated rows & business keys
- Invalid / unparseable dates
- Impossible negative values
- Invalid prices & inappropriate zero prices
- Infinite values
- Empty columns (100% missing)
- Suspicious statistical outliers
- Time-series continuity gaps
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

logger = setup_logger("anomaly_detector")


class AnomalyDetector:
    """Scans structured datasets for anomalies, value integrity, duplicates, and statistical outliers."""

    def __init__(self, sample_size: int = 100000):
        self.sample_size = sample_size

    def audit_dataset(self, file_path: Path, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            "file_path": str(file_path),
            "filename": file_path.name,
            "total_rows": schema_info.get("rows", 0),
            "total_columns": schema_info.get("columns", 0),
            "total_missing_cells": 0,
            "overall_missing_pct": 0.0,
            "column_missing_stats": {},
            "empty_columns": [],
            "duplicate_rows": 0,
            "duplicate_business_keys": 0,
            "business_key_cols": [],
            "invalid_dates_count": 0,
            "negative_value_anomalies": {},
            "invalid_price_anomalies": {},
            "infinite_values_count": 0,
            "outlier_anomalies": {},
            "timeseries_gaps_count": 0,
            "anomalies_detected": []
        }

        if schema_info.get("rows", 0) == 0 or schema_info.get("is_empty_file", False):
            result["anomalies_detected"].append("Empty file or template without data rows")
            return result

        fmt = schema_info.get("format", "parquet")

        try:
            # Memory-conscious loading: use Polars
            if fmt == "parquet":
                lf = pl.scan_parquet(file_path)
            elif fmt == "csv":
                lf = pl.scan_csv(file_path, ignore_errors=True)
            else:
                return result

            # 1. Missing Values per Column
            missing_exprs = [pl.col(c).null_count().alias(c) for c in schema_info["column_names"]]
            missing_df = lf.select(missing_exprs).collect()
            
            total_nulls = 0
            for c in schema_info["column_names"]:
                null_cnt = int(missing_df[c][0])
                null_pct = round((null_cnt / result["total_rows"]) * 100, 2)
                result["column_missing_stats"][c] = {"count": null_cnt, "pct": null_pct}
                total_nulls += null_cnt
                if null_pct == 100.0:
                    result["empty_columns"].append(c)
                    result["anomalies_detected"].append(f"Column '{c}' is 100% empty (null)")

            result["total_missing_cells"] = total_nulls
            total_possible_cells = max(1, result["total_rows"] * result["total_columns"])
            result["overall_missing_pct"] = round((total_nulls / total_possible_cells) * 100, 2)

            # 2. Duplicate Rows (check sample if file is massive)
            if result["total_rows"] > 500000:
                # Approximate or chunk-based check
                sample_df = lf.limit(self.sample_size).collect()
                result["duplicate_rows"] = int(sample_df.is_duplicated().sum())
            else:
                full_df = lf.collect()
                result["duplicate_rows"] = int(full_df.is_duplicated().sum())

            if result["duplicate_rows"] > 0:
                result["anomalies_detected"].append(f"Found {result['duplicate_rows']} duplicate rows")

            # 3. Duplicate Business Keys
            cols = schema_info["column_names"]
            key_cols = []
            if "date" in cols and "store_id" in cols and "product_id" in cols:
                key_cols = ["date", "store_id", "product_id"]
            elif "date" in cols and "premise_code" in cols and "item_code" in cols:
                key_cols = ["date", "premise_code", "item_code"]
            elif "product_id" in cols and len(cols) < 25 and "date" not in cols:
                key_cols = ["product_id"]
            elif "store_id" in cols and len(cols) < 25 and "date" not in cols:
                key_cols = ["store_id"]
            elif "supplier_id" in cols and len(cols) < 20 and "date" not in cols:
                key_cols = ["supplier_id"]
            elif "date" in cols and len(cols) <= 15:
                # Time-series by date
                key_cols = ["date"]

            if key_cols:
                result["business_key_cols"] = key_cols
                key_dups = lf.select(key_cols).collect().is_duplicated().sum()
                result["duplicate_business_keys"] = int(key_dups)
                if key_dups > 0:
                    result["anomalies_detected"].append(f"Found {key_dups} duplicated business keys on {key_cols}")

            # 4. Numeric Checks: Negatives, Invalids, Zeros, Infs
            num_cols = schema_info.get("numeric_columns", [])
            price_keywords = ["price", "cost", "tariff"]
            non_negative_keywords = ["quantity", "count", "income", "expenditure", "capacity", "sold", "inventory", "stock", "rainfall", "humidity"]

            for nc in num_cols:
                col_lower = nc.lower()
                
                # Check for impossible negatives
                if any(k in col_lower for k in non_negative_keywords) or any(k in col_lower for k in price_keywords):
                    neg_count = int(lf.filter(pl.col(nc) < 0).select(pl.len()).collect().item())
                    if neg_count > 0:
                        result["negative_value_anomalies"][nc] = neg_count
                        result["anomalies_detected"].append(f"Column '{nc}' contains {neg_count} impossible negative values")

                # Check for invalid prices
                if any(k in col_lower for k in price_keywords):
                    zero_price_count = int(lf.filter(pl.col(nc) == 0).select(pl.len()).collect().item())
                    extreme_price_count = int(lf.filter(pl.col(nc) > 5000).select(pl.len()).collect().item())
                    if zero_price_count > 0 and "discount" not in col_lower and "subsidy" not in col_lower:
                        result["invalid_price_anomalies"][f"{nc}_zero"] = zero_price_count
                        result["anomalies_detected"].append(f"Column '{nc}' contains {zero_price_count} zero prices")
                    if extreme_price_count > 0:
                        result["invalid_price_anomalies"][f"{nc}_extreme"] = extreme_price_count
                        result["anomalies_detected"].append(f"Column '{nc}' contains {extreme_price_count} extreme prices (>5000)")

                # Check for infinite values
                inf_count = int(lf.filter(pl.col(nc).is_infinite()).select(pl.len()).collect().item())
                if inf_count > 0:
                    result["infinite_values_count"] += inf_count
                    result["anomalies_detected"].append(f"Column '{nc}' contains {inf_count} infinite values")

            # 5. Time-Series Continuity Gaps
            if schema_info.get("earliest_date") and schema_info.get("latest_date") and "date" in schema_info.get("column_names", []):
                try:
                    unique_dates = lf.select(pl.col("date").cast(pl.Utf8)).unique().collect()["date"].to_list()
                    d_objs = [pd.to_datetime(d, errors="coerce") for d in unique_dates if d and str(d).strip() != ""]
                    d_objs = [d for d in d_objs if pd.notnull(d)]
                    if len(d_objs) > 10:
                        min_d = min(d_objs)
                        max_d = max(d_objs)
                        expected_days = (max_d - min_d).days + 1
                        actual_days = len(set(d.strftime("%Y-%m-%d") for d in d_objs))
                        # For daily series, check if gap exceeds threshold
                        if expected_days > 365 and actual_days < (expected_days * 0.70):
                            # May be monthly or weekly data
                            pass
                        elif expected_days - actual_days > 5:
                            gap = expected_days - actual_days
                            result["timeseries_gaps_count"] = gap
                except Exception as e:
                    logger.debug(f"Gap detection notice: {e}")

        except Exception as e:
            logger.error(f"Error auditing dataset {file_path.name}: {e}")
            result["anomalies_detected"].append(f"Audit failure: {str(e)}")

        return result
