"""
test_raw_data_audit.py
======================
Unit and integration tests for Phase 3 raw data audit and validation:
- SchemaValidator tests
- AnomalyDetector tests
- DataReadinessEvaluator tests
- End-to-end RawDataAuditor audit run
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
from src.validation.schema_validator import SchemaValidator, classify_dataset
from src.validation.anomaly_detector import AnomalyDetector
from src.validation.data_readiness import DataReadinessEvaluator, REPORTS_DATA_VALIDATION_DIR, ESSENTIAL_RETAILER_DOMAINS
from src.validation.raw_data_audit import RawDataAuditor


def test_classify_dataset_heuristics():
    assert classify_dataset("fuel_prices.csv", ["date", "ron95", "diesel"]) == "Fuel prices"
    assert classify_dataset("retail_pos_sales.parquet", ["date", "store_id", "product_id", "quantity_sold", "selling_price"]) == "Point-of-Sale sales"
    assert classify_dataset("retail_inventory.parquet", ["date", "store_id", "product_id", "opening_inventory_quantity"]) == "Inventory"
    assert classify_dataset("retail_products.csv", ["product_id", "product_name", "product_shelf_life_days"]) == "Product master"
    assert classify_dataset("retail_suppliers.csv", ["supplier_id", "supplier_name", "supplier_lead_time_days"]) == "Supplier master"
    assert classify_dataset("retail_stores.csv", ["store_id", "store_name", "store_storage_capacity"]) == "Store master"
    assert classify_dataset("retail_promotions.csv", ["promotion_id", "discount_percentage"]) == "Promotion history"


def test_schema_validator_inspect_file():
    validator = SchemaValidator()
    # Test on fuel_prices.csv
    fuel_file = RAW_DATA_DIR / "fuel_prices.csv"
    assert fuel_file.exists()
    res = validator.inspect_file(fuel_file)
    assert res["format"] == "csv"
    assert res["rows"] > 500
    assert "date" in res["column_names"]
    assert res["classification"] == "Fuel prices"
    assert res["earliest_date"] is not None
    assert res["latest_date"] is not None


def test_anomaly_detector_basic():
    detector = AnomalyDetector()
    validator = SchemaValidator()
    fuel_file = RAW_DATA_DIR / "fuel_prices.csv"
    schema = validator.inspect_file(fuel_file)
    anomaly = detector.audit_dataset(fuel_file, schema)
    assert anomaly["total_rows"] == schema["rows"]
    assert "column_missing_stats" in anomaly
    assert anomaly["duplicate_rows"] == 0


def test_data_readiness_evaluator():
    evaluator = DataReadinessEvaluator()
    # Mock clean dataset
    clean_schema = {"rows": 100, "columns": 5, "is_empty_file": False, "is_template_only": False, "error": None, "has_missing_identifiers": False}
    clean_anomaly = {"infinite_values_count": 0, "duplicate_rows": 0, "duplicate_business_keys": 0, "empty_columns": [], "overall_missing_pct": 0.0}
    res_ready = evaluator.evaluate_status(clean_schema, clean_anomaly)
    assert res_ready["status"] == "READY"

    # Mock blocked dataset (empty)
    blocked_schema = {"rows": 0, "columns": 5, "is_empty_file": True}
    res_blocked = evaluator.evaluate_status(blocked_schema, clean_anomaly)
    assert res_blocked["status"] == "BLOCKED"


def test_essential_retailer_domains_presence():
    auditor = RawDataAuditor()
    files = auditor.discover_files()
    assert len(files) >= 7

    classifications = set()
    for f in files:
        schema = auditor.schema_validator.inspect_file(f)
        classifications.add(schema["classification"])

    for dom in ESSENTIAL_RETAILER_DOMAINS:
        assert dom in classifications, f"Essential domain '{dom}' missing from repository"


def test_end_to_end_raw_data_audit():
    auditor = RawDataAuditor()
    audit_res = auditor.run_full_audit()
    assert audit_res["total_files"] >= 15
    assert len(audit_res["audit_results"]) >= 15

    # Verify that all 8 report files were created
    rep_dir = REPORTS_DATA_VALIDATION_DIR
    assert (rep_dir / "data_inventory.csv").exists()
    assert (rep_dir / "dataset_quality_report.csv").exists()
    assert (rep_dir / "date_coverage_report.csv").exists()
    assert (rep_dir / "missing_values_report.csv").exists()
    assert (rep_dir / "duplicate_report.csv").exists()
    assert (rep_dir / "anomaly_report.csv").exists()
    assert (rep_dir / "data_readiness_report.csv").exists()
    assert (rep_dir / "data_validation_summary.md").exists()
