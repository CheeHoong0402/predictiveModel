# Raw Data Audit & Readiness Summary Report

*Audit Date: 2026-10-11 07:37:49*

**Total Structured Files Inspected:** 32

**Readiness Breakdown:**
- **READY:** 22 datasets
- **WARNING:** 10 datasets
- **BLOCKED:** 0 datasets

## 1. Essential Retailer Domains Status
| Essential Domain | Present in Repository | Status |
|---|---|---|
| Point-of-Sale sales | YES | VALIDATED |
| Inventory | YES | VALIDATED |
| Purchase-price history | YES | VALIDATED |
| Product master | YES | VALIDATED |
| Supplier master | YES | VALIDATED |
| Store master | YES | VALIDATED |
| Promotion history | YES | VALIDATED |

> [!NOTE]
> All 7 essential retailer domains are present with full records (no empty templates).

## 2. READY Datasets (Approved for Preprocessing)
| Filename | Classification | Rows | Verification Note |
|---|---|---|---|
| `commodities_10y.csv` | Commodity prices | 2,766 | Dataset passed all baseline checks cleanly |
| `exchange_rates_10y.csv` | Currency exchange rates | 2,865 | Dataset passed all baseline checks cleanly |
| `malaysia_government_assistance_10y.csv` | Government assistance | 37 | Dataset passed all baseline checks cleanly |
| `malaysia_government_salary_dates_10y.csv` | Government assistance | 132 | Dataset passed all baseline checks cleanly |
| `malaysia_national_budget_policies.csv` | National budget policies | 11 | Dataset passed all baseline checks cleanly |
| `malaysia_school_holidays_10y.csv` | School holidays | 44 | Dataset passed all baseline checks cleanly |
| `malaysia_subsidies_and_policies.csv` | Government subsidies | 15 | Dataset passed all baseline checks cleanly |
| `malaysia_weather_10y.csv` | Weather | 14,612 | Dataset passed all baseline checks cleanly |
| `pricecatcher_2024-01.parquet` | Malaysian market prices | 2,664,434 | Dataset passed all baseline checks cleanly |
| `pricecatcher_lookup_item.csv` | Malaysian market prices | 797 | Dataset passed all baseline checks cleanly |
| `pricecatcher_lookup_item.parquet` | Malaysian market prices | 797 | Dataset passed all baseline checks cleanly |
| `pricecatcher_lookup_premise.csv` | Malaysian market prices | 3,921 | Dataset passed all baseline checks cleanly |
| `pricecatcher_lookup_premise.parquet` | Malaysian market prices | 3,916 | Dataset passed all baseline checks cleanly |
| `retail_inventory.parquet` | Inventory | 35,064 | Dataset passed all baseline checks cleanly |
| `retail_pos_sales.parquet` | Point-of-Sale sales | 35,064 | Dataset passed all baseline checks cleanly |
| `retail_products.csv` | Product master | 6 | Dataset passed all baseline checks cleanly |
| `retail_promotions.csv` | Promotion history | 20 | Dataset passed all baseline checks cleanly |
| `retail_purchase_prices.parquet` | Purchase-price history | 35,064 | Dataset passed all baseline checks cleanly |
| `retail_stores.csv` | Store master | 4 | Dataset passed all baseline checks cleanly |
| `retail_suppliers.csv` | Supplier master | 6 | Dataset passed all baseline checks cleanly |
| `retailer_operational_data.csv` | Inventory | 35,064 | Dataset passed all baseline checks cleanly |
| `retailer_operational_data.parquet` | Inventory | 35,064 | Dataset passed all baseline checks cleanly |

## 3. WARNING Datasets (Requiring Preprocessing Treatment)
| Filename | Classification | Rows | Issues Detected |
|---|---|---|---|
| `cpi_core.csv` | Consumer Price Index | 1,456 | Contains 1456 duplicate business keys |
| `cpi_headline.csv` | Consumer Price Index | 2,500 | Contains 2500 duplicate business keys |
| `cpi_state.csv` | Consumer Price Index | 2,500 | Contains 2500 duplicate business keys |
| `fuel_prices.csv` | Fuel prices | 963 | Contains 962 duplicate business keys; Overall missing cell percentage is high (34.93%) |
| `iowrt_distributive_trade.csv` | Wholesale and retail statistics | 296 | Contains 295 duplicate business keys |
| `malaysia_public_holidays_10y.csv` | Public holidays | 598 | Contains 248 duplicate business keys |
| `ppi.csv` | Producer Price Index | 587 | Contains 586 duplicate business keys; Overall missing cell percentage is high (15.16%) |
| `trade_enduse_bec.csv` | Import statistics | 2,500 | Contains 2500 duplicate business keys |
| `trade_headline.csv` | Export statistics | 755 | Contains 658 duplicate business keys |
| `trade_sitc_1d.csv` | Export statistics | 2,500 | Contains 2500 duplicate business keys |

## 4. BLOCKED Datasets (Unusable)
*No datasets BLOCKED. All files contain valid records and valid schemas.*

## 5. Major External Datasets Date Coverage
| Dataset | Earliest Date | Latest Date | Coverage Span | Timeseries Gaps |
|---|---|---|---|---|
| `commodities_10y.csv` | 2015-01-02 | 2025-12-30 | Commodity prices | 0 |
| `cpi_core.csv` | 2018-01-01 | 2026-08-01 | Consumer Price Index | 0 |
| `cpi_headline.csv` | 1980-01-01 | 2026-08-01 | Consumer Price Index | 0 |
| `cpi_state.csv` | 2010-01-01 | 2026-08-01 | Consumer Price Index | 0 |
| `exchange_rates_10y.csv` | 2015-01-01 | 2025-12-30 | Currency exchange rates | 1152 |
| `fuel_prices.csv` | 2017-03-30 | 2026-10-08 | Fuel prices | 0 |
| `iowrt_distributive_trade.csv` | 2018-01-01 | 2026-07-01 | Wholesale and retail statistics | 0 |
| `malaysia_government_assistance_10y.csv` | 2015-01-15 | 2025-11-05 | Government assistance | 0 |
| `malaysia_government_salary_dates_10y.csv` | 2015-01-23 | 2025-12-19 | Government assistance | 0 |
| `malaysia_national_budget_policies.csv` | 2014-10-10 | 2024-10-18 | National budget policies | 0 |
| `malaysia_public_holidays_10y.csv` | 2015-01-01 | 2025-12-25 | Public holidays | 0 |
| `malaysia_school_holidays_10y.csv` | 2015-03-20 | 2025-12-15 | School holidays | 0 |
| `malaysia_subsidies_and_policies.csv` | 2013-10-25 | 2024-10-18 | Government subsidies | 0 |
| `malaysia_weather_10y.csv` | 2015-01-01 | 2024-12-31 | Weather | 0 |
| `ppi.csv` | 2010-01-01 | 2026-08-01 | Producer Price Index | 0 |
| `pricecatcher_2024-01.parquet` | 2024-01-01 | 2024-01-31 | Malaysian market prices | 0 |
| `retail_inventory.parquet` | 2021-01-01 | 2024-12-31 | Inventory | 0 |
| `retail_pos_sales.parquet` | 2021-01-01 | 2024-12-31 | Point-of-Sale sales | 0 |
| `retail_promotions.csv` | 2021-01-20 | 2024-12-18 | Promotion history | 0 |
| `retail_purchase_prices.parquet` | 2021-01-01 | 2024-12-31 | Purchase-price history | 0 |
| `retailer_operational_data.csv` | 2021-01-01 | 2024-12-31 | Inventory | 0 |
| `retailer_operational_data.parquet` | 2021-01-01 | 2024-12-31 | Inventory | 0 |
| `trade_enduse_bec.csv` | 2010-01-01 | 2026-08-01 | Import statistics | 0 |
| `trade_headline.csv` | 2000-01-01 | 2026-08-01 | Export statistics | 0 |
| `trade_sitc_1d.csv` | 2000-01-01 | 2026-07-01 | Export statistics | 0 |