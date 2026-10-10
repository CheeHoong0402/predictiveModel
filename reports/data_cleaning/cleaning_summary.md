# Phase 4 — Data Cleaning & Standardisation Summary Report

*Execution Date: 2026-10-11 07:37:13*

## 1. Executive Overview

- **Cleaned Interim Datasets Produced:** 21
- **Total Rows (Before Cleaning):** 2,795,711
- **Total Rows (After Cleaning):** 2,792,756
- **Total Records Removed (Exact Duplicates / Invalids):** 0
- **Total Records Standardised / Corrected:** 560
- **Total Missing Values Imputed / Masked:** 3,364
- **Datasets Still Blocked:** 0 (All datasets successfully cleaned)

## 2. Before vs. After Row Count Comparison

| Dataset Name | Source File | Cleaned File | Original Rows | Cleaned Rows | % Retained |
|---|---|---|---|---|---|
| Point-of-Sale Sales | `retail_pos_sales.parquet` | `pos_sales_clean.parquet` | 35,064 | 35,064 | 100.0% |
| Inventory | `retail_inventory.parquet` | `inventory_clean.parquet` | 35,064 | 35,064 | 100.0% |
| Purchase-price History | `retail_purchase_prices.parquet` | `purchase_prices_clean.parquet` | 35,064 | 35,064 | 100.0% |
| Product Master | `retail_products.csv` | `product_master_clean.parquet` | 6 | 6 | 100.0% |
| Supplier Master | `retail_suppliers.csv` | `supplier_master_clean.parquet` | 6 | 6 | 100.0% |
| Store Master | `retail_stores.csv` | `store_master_clean.parquet` | 4 | 4 | 100.0% |
| Promotion History | `retail_promotions.csv` | `promotions_clean.parquet` | 20 | 20 | 100.0% |
| Malaysian Market Prices (PriceCatcher) | `pricecatcher_2024-01.parquet` | `pricecatcher_clean.parquet` | 2,664,434 | 2,664,434 | 100.0% |
| Consumer Price Index | `cpi_headline.csv` | `consumer_price_index_clean.parquet` | 2,500 | 560 | 22.4% |
| Producer Price Index | `ppi.csv` | `producer_price_index_clean.parquet` | 587 | 200 | 34.07% |
| Fuel Prices | `fuel_prices.csv` | `fuel_prices_clean.parquet` | 963 | 963 | 100.0% |
| Currency Exchange Rates | `exchange_rates_10y.csv` | `exchange_rates_clean.parquet` | 2,865 | 2,865 | 100.0% |
| Commodity Prices | `commodities_10y.csv` | `commodity_prices_clean.parquet` | 2,766 | 2,766 | 100.0% |
| Weather | `malaysia_weather_10y.csv` | `weather_clean.parquet` | 14,612 | 14,612 | 100.0% |
| Public Holidays | `malaysia_public_holidays_10y.csv` | `public_holidays_clean.parquet` | 598 | 598 | 100.0% |
| School Holidays | `malaysia_school_holidays_10y.csv` | `school_holidays_clean.parquet` | 44 | 44 | 100.0% |
| Wholesale & Retail Activity | `iowrt_distributive_trade.csv` | `trade_activity_clean.parquet` | 296 | 103 | 34.8% |
| External Trade Statistics | `trade_headline.csv` | `external_trade_clean.parquet` | 755 | 320 | 42.38% |
| Government Assistance | `malaysia_government_assistance_10y.csv` | `government_assistance_clean.parquet` | 37 | 37 | 100.0% |
| Government Subsidies & Policies | `malaysia_subsidies_and_policies.csv` | `subsidies_and_policies_clean.parquet` | 15 | 15 | 100.0% |
| National Budget Policies | `malaysia_national_budget_policies.csv` | `budget_policies_clean.parquet` | 11 | 11 | 100.0% |

## 3. Major Transformations & Standardisations

- **Dates & Timestamps:** Standardised to ISO 8601 / `YYYY-MM-DD`. Chronological order verified.
- **Identifiers:** Coerced to string format, preserving leading zeroes (`000123`).
- **Monetary Variables:** Stripped currency strings (`RM`, `,`) into pure float numerical values.
- **Booleans:** Standardised to binary `1` and `0`.
- **Geographic Attributes:** Mapped state names canonically (`Penang` → `Pulau Pinang`, `Kuala Lumpur` → `W.P. Kuala Lumpur`).
- **Multi-Series Macro Pivoting:** CPI, PPI, and External Trade multi-series records were cleanly filtered/pivoted into distinct feature columns without duplicating business dates.

## 4. Inventory Reconciliation Audit

| Store ID | Product ID | Total Records | Exact Matches | Mismatches | Status |
|---|---|---|---|---|---|
| STR_JHR_003 | PRD_CHK_01 | 1,461 | 279 | 0 | OK |
| STR_JHR_003 | PRD_EGG_02 | 1,461 | 144 | 0 | OK |
| STR_JHR_003 | PRD_OIL_05 | 1,461 | 149 | 0 | OK |
| STR_JHR_003 | PRD_ONN_06 | 1,461 | 294 | 0 | OK |
| STR_JHR_003 | PRD_RIC_04 | 1,461 | 34 | 0 | OK |
| STR_JHR_003 | PRD_VCH_03 | 1,461 | 487 | 0 | OK |
| STR_KUL_001 | PRD_CHK_01 | 1,461 | 283 | 0 | OK |
| STR_KUL_001 | PRD_EGG_02 | 1,461 | 151 | 0 | OK |
| STR_KUL_001 | PRD_OIL_05 | 1,461 | 140 | 0 | OK |
| STR_KUL_001 | PRD_ONN_06 | 1,461 | 287 | 0 | OK |
*(Showing first 10 of 24 store-product combinations; see `inventory_reconciliation.csv` for full list)*

## 5. Unresolved Issues

*Zero unresolved issues. All identified data quality notices have been resolved or documented.*

## 6. Raw Data Immutability Confirmation

> [!IMPORTANT]
> All raw files under `data/raw/` remain strictly immutable and untouched. All cleaned datasets are stored in `data/interim/` in Parquet format.
