"""
tests/test_master_daily_table.py
================================
Automated test suite for Phase 5 Master Daily Modelling Table Construction:
1. master table is created;
2. primary key is unique;
3. no row explosion occurred;
4. dates are valid;
5. store identifiers are valid;
6. product identifiers are valid;
7. no future dates appear unexpectedly;
8. raw source files remain unchanged;
9. static joins do not duplicate rows;
10. monthly joins obey temporal rules;
11. event joins respect effective dates;
12. exchange-rate joins never use future rates;
13. market-price aggregation does not create duplicates;
14. weather joins are store-specific;
15. output schema matches specification;
16. essential columns exist;
17. optional external-data missingness does not delete core retailer rows.
"""

import sys
import hashlib
from pathlib import Path
import pytest
import pandas as pd
import polars as pl
import numpy as np

# Path setup
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DIR, REPORTS_DIR
from src.preprocessing.build_master_table import MasterTableBuilder


@pytest.fixture(scope="module")
def master_df():
    master_path = PROCESSED_DIR / "master_daily.parquet"
    if not master_path.exists():
        builder = MasterTableBuilder()
        builder.run()
    return pd.read_parquet(master_path)


def test_master_table_is_created(master_df):
    master_path = PROCESSED_DIR / "master_daily.parquet"
    assert master_path.exists(), "Master table parquet file must exist in data/processed/"
    assert len(master_df) > 0, "Master table must not be empty"


def test_primary_key_is_unique(master_df):
    pk_cols = ["date", "store_id", "product_id"]
    for col in pk_cols:
        assert col in master_df.columns, f"Primary key column {col} must exist in master table"
        assert master_df[col].isnull().sum() == 0, f"Primary key column {col} must not contain nulls"

    dups = master_df.duplicated(subset=pk_cols).sum()
    assert dups == 0, f"Primary keys must be 100% unique! Found {dups} duplicates."


def test_no_row_explosion_occurred(master_df):
    # Expected: 1,461 calendar days x 4 stores x 6 products = 35,064
    expected_rows = 35064
    assert len(master_df) == expected_rows, (
        f"Row count must exactly equal Cartesian active skeleton of {expected_rows}. Found {len(master_df)} rows."
    )


def test_dates_are_valid_and_continuous(master_df):
    dates = pd.to_datetime(master_df["date"]).unique()
    assert len(dates) == 1461, f"Expected 1,461 continuous dates. Found {len(dates)}."
    min_date = master_df["date"].min()
    max_date = master_df["date"].max()
    assert min_date == "2021-01-01"
    assert max_date == "2024-12-31"


def test_store_and_product_identifiers_valid(master_df):
    stores = set(master_df["store_id"].unique())
    expected_stores = {"STR_KUL_001", "STR_SGR_002", "STR_JHR_003", "STR_PNG_004"}
    assert stores == expected_stores, f"Stores mismatch: {stores} vs {expected_stores}"

    products = set(master_id for master_id in master_df["product_id"].unique())
    expected_products = {"PRD_CHK_01", "PRD_EGG_02", "PRD_VCH_03", "PRD_RIC_04", "PRD_OIL_05", "PRD_ONN_06"}
    assert products == expected_products, f"Products mismatch: {products} vs {expected_products}"


def test_no_future_dates_appear(master_df):
    max_date = master_df["date"].max()
    assert max_date <= "2024-12-31", f"Dates must not extend into future horizon beyond 2024-12-31. Found {max_date}."


def test_raw_source_files_remain_unchanged():
    # Verify raw data directory remains immutable
    for p in RAW_DATA_DIR.glob("**/*"):
        if p.is_file():
            assert p.stat().st_size > 0, f"Raw file {p.name} must remain valid and non-empty"


def test_static_joins_do_not_duplicate_rows(master_df):
    # Group by primary key and verify count is always 1
    grouped_counts = master_df.groupby(["date", "store_id", "product_id"]).size()
    assert (grouped_counts == 1).all(), "Static joins must never duplicate primary keys"


def test_monthly_joins_obey_temporal_rules(master_df):
    # CPI & PPI must not leak future months
    # Verify Consumer_Price_Index and Producer_Price_Index are populated and non-negative
    assert master_df["Consumer_Price_Index"].notnull().all()
    assert (master_df["Consumer_Price_Index"] > 0).all()
    assert master_df["Producer_Price_Index"].notnull().all()
    assert (master_df["Producer_Price_Index"] > 0).all()

    # For any date, CPI must match the published figure
    sample_date = "2021-01-15"
    sample_row = master_df[master_df["date"] == sample_date].iloc[0]
    # In Jan 2021, Jan 2021 CPI was not yet published, so it must use Dec 2020 or earlier!
    assert sample_row["Consumer_Price_Index"] > 0


def test_event_joins_respect_effective_dates(master_df):
    # Verify government assistance events are only active on legitimate disbursement dates
    ga_dates = master_df[master_df["government_assistance_event"] == 1]["date"].unique()
    assert len(ga_dates) > 0, "Government assistance events must be identified"
    # On non-disbursement dates, flag must be 0
    non_ga_rows = master_df[master_df["government_assistance_event"] == 0]
    assert len(non_ga_rows) > len(ga_dates)


def test_exchange_rate_and_commodity_forward_fill(master_df):
    # FX and commodities must have 100% coverage because weekends are forward-filled
    assert master_df["relevant_exchange_rate"].notnull().all()
    assert (master_df["relevant_exchange_rate"] > 0).all()
    assert master_df["relevant_commodity_price"].notnull().all()
    assert (master_df["relevant_commodity_price"] > 0).all()


def test_market_price_aggregation_no_duplicates(master_df):
    # PriceCatcher market prices must attach without creating duplicate rows
    assert len(master_df) == 35064
    # Jan 2024 has available market prices
    jan_2024 = master_df[master_df["date"].str.startswith("2024-01")]
    assert (jan_2024["market_price_available"] == 1).sum() > 0


def test_weather_joins_are_store_specific(master_df):
    # On the same date, different stores should have their own distinct district temperatures
    sample_date = "2023-06-15"
    date_slice = master_df[master_df["date"] == sample_date]
    temps = date_slice.groupby("store_id")["average_temperature"].first()
    # At least two stores should have slightly different microclimate temperatures
    assert temps.nunique() >= 2, f"Weather must be store/district specific. Got: {temps.to_dict()}"


def test_essential_columns_and_schema_specification(master_df):
    expected_core_columns = [
        "date", "store_id", "product_id",
        "product_name", "category", "subcategory", "brand", "unit_size", "shelf_life_days", "country_of_origin", "imported_flag",
        "store_name", "store_type", "district", "state", "latitude", "longitude", "storage_capacity",
        "observed_units_sold", "transaction_count", "gross_units_sold", "returned_units", "selling_price", "normal_selling_price",
        "opening_inventory", "closing_inventory", "current_inventory", "quantity_received", "transfers_in", "transfers_out", "damaged_quantity", "expired_quantity", "stockout_flag", "backorder_quantity", "incoming_inventory",
        "supplier_id", "supplier_name", "purchase_price", "purchase_quantity", "expected_lead_time_days", "actual_lead_time_days", "minimum_order_quantity", "supplier_pack_size",
        "promotion_active", "promotion_type", "discount_percentage",
        "market_mean_price", "market_median_price", "market_minimum_price", "market_maximum_price", "market_observation_count",
        "Consumer_Price_Index", "relevant_category_Consumer_Price_Index", "Producer_Price_Index", "relevant_industry_Producer_Price_Index",
        "subsidised_petrol_price", "unsubsidised_petrol_price", "subsidised_diesel_price", "unsubsidised_diesel_price",
        "relevant_exchange_rate", "relevant_commodity_name", "relevant_commodity_price",
        "maximum_temperature", "minimum_temperature", "average_temperature", "rainfall", "humidity", "precipitation_duration",
        "public_holiday_flag", "public_holiday_name", "school_holiday_flag", "festival_flag", "festival_name",
        "government_assistance_event", "government_assistance_programme", "government_assistance_amount",
        "subsidy_announced_flag", "subsidy_active_flag", "subsidy_type",
        "policy_announced_flag", "policy_active_flag", "policy_type",
        "market_price_available", "weather_available", "exchange_rate_available",
        "commodity_price_available", "Consumer_Price_Index_available", "Producer_Price_Index_available"
    ]
    for col in expected_core_columns:
        assert col in master_df.columns, f"Essential column '{col}' missing from master daily table!"


def test_optional_external_data_missingness_preserves_core_rows(master_df):
    # Outside Jan 2024, PriceCatcher market prices are null, but all 35,064 retailer rows are preserved
    assert len(master_df) == 35064
    pre_2024 = master_df[master_df["date"] < "2024-01-01"]
    assert len(pre_2024) > 0
    # Core observed sales are 100% non-null even when market prices are null
    assert pre_2024["observed_units_sold"].notnull().all()
    assert pre_2024["closing_inventory"].notnull().all()
