# Phase 5 — Master Daily Modelling Table Summary Report

*Execution Timestamp: 2026-10-10 20:57:34*

## 1. Executive Table Overview

- **Primary Modelling Grain:** `one row = one product × one store × one calendar date` (`date + store_id + product_id`)
- **Total Master Rows:** **35,064**
- **Total Columns:** **88**
- **Unique Stores:** **4** (Bangsar KL, Shah Alam Selangor, Skudai Johor, Georgetown Penang)
- **Unique Products:** **6** (Whole Chicken, Grade A Eggs, Cameron Cabbage, Tempatan Rice, Palm Cooking Oil, Shallots)
- **Temporal Horizon:** **2021-01-01 to 2024-12-31** (1,461 calendar days)
- **Primary-Key Uniqueness:** **100.0% Unique (0 duplicates)**
- **Output Artifact:** [`data/processed/master_daily.parquet`](data/processed/master_daily.parquet) (0.55 MB)
- **Pipeline Execution Time:** **6.10 seconds**

---

## 2. External Dataset Join Matrix & Temporal Rules

| Dataset Name | Source Path | Frequency | Join Keys | Match Rate % | Temporal Safety Rule | Future Leakage Protection |
|---|---|---|---|---|---|---|
| **Point-of-Sale Sales** | `data/interim/pos_sales_clean.parquet` | Daily | `date, store_id, product_id` | 100.0% | Exact calendar date | Zero leakage (historical sales) |
| **Inventory** | `data/interim/inventory_clean.parquet` | Daily | `date, store_id, product_id` | 100.0% | Exact calendar date | Zero leakage (physical daily audits) |
| **Purchase Prices** | `data/interim/purchase_prices_clean.parquet` | Daily | `date, store_id, product_id` | 100.0% | Exact calendar date | Zero leakage (supplier contract logs) |
| **Product Master** | `data/interim/product_master_clean.parquet` | Static | `product_id` | 100.0% | Invariant | Static catalogue |
| **Store Master** | `data/interim/store_master_clean.parquet` | Static | `store_id` | 100.0% | Invariant | Static premises |
| **Supplier Master** | `data/interim/supplier_master_clean.parquet` | Static | `supplier_id` | 100.0% | Invariant | Static vendor directory |
| **Promotion History** | `data/interim/promotions_clean.parquet` | Event-interval | `date interval, category` | 100.0% | Interval containment | Pre-announced campaign windows |
| **PriceCatcher** | `data/interim/pricecatcher_clean.parquet` | Daily | `product_id, district/state, date` | 2.1% (Jan 2024 slice) | Exact date + Hierarchical fallback | No future leakage |
| **Consumer Price Index** | `data/interim/consumer_price_index_clean.parquet` | Monthly | `date` | 100.0% | Strict Publication Lag (`ref_month + 1M + 24D`) | **Guaranteed zero future lookahead** |
| **Producer Price Index** | `data/interim/producer_price_index_clean.parquet` | Monthly | `date` | 100.0% | Strict Publication Lag (`ref_month + 1M + 28D`) | **Guaranteed zero future lookahead** |
| **Fuel Prices** | `data/interim/fuel_prices_clean.parquet` | Weekly | `date` | 100.0% | Stepwise weekly level forward-fill | No future announcements applied retrospectively |
| **Exchange Rates** | `data/interim/exchange_rates_clean.parquet` | Daily | `date` | 100.0% | Weekend forward-fill (max 4 days) | **Strictly no backfilling from future business days** |
| **Commodity Prices** | `data/interim/commodity_prices_clean.parquet` | Daily | `date, product_id` | 100.0% | Product-commodity mapping + forward-fill | **Strictly no backfilling from future trading days** |
| **Weather** | `data/interim/weather_clean.parquet` | Daily | `date, state, district` | 100.0% | Exact store coordinates & date | Historical actual observations only |
| **Public Holidays** | `data/interim/public_holidays_clean.parquet` | Calendar | `date, state` | 100.0% | National and state gazetted holidays | Pre-scheduled statutory calendar |
| **School Holidays** | `data/interim/school_holidays_clean.parquet` | Calendar | `date interval` | 100.0% | MOE academic calendar containment | Pre-scheduled term breaks |
| **Government Assistance** | `data/interim/government_assistance_clean.parquet` | Event-based | `date` | 100.0% | Historical disbursement dates | Only events known by prediction date |
| **Subsidies & Policies** | `data/interim/subsidies_and_policies_clean.parquet` | Regime-based | `date` | 100.0% | Announcement vs Effective dates | Dual-state tracking (announced vs implemented) |

---

## 3. Product-Market Controlled Mapping

All 6 active retailer products were mapped to PriceCatcher items using exact domain classification (no uncontrolled fuzzy heuristics):

| Product ID | Product Name | PriceCatcher Item Code | PriceCatcher Item Name | Unit | Primary Commodity Benchmark |
|---|---|---|---|---|---|
| `PRD_CHK_01` | Ayam Bulat Segar (Standard Whole Chicken) | 1 | AYAM BERSIH - STANDARD | 1kg | `feed_grain_usd` |
| `PRD_EGG_02` | Telur Ayam Gred A (30 biji) | 1109 | TELUR AYAM GRED A | 30 biji | `feed_grain_usd` |
| `PRD_VCH_03` | Kobis Bulat Cameron (Cameron Cabbage) | 105 | KUBIS BULAT (TEMPATAN) | 1kg | `brent_crude_usd` |
| `PRD_RIC_04` | Beras Putih Super Tempatan (SST 5%) 10kg | 992 | BERAS CAP JATI (SST5%) | 10 kg | `rice_usd` |
| `PRD_OIL_05` | Minyak Masak Sawit Tulen 5kg | 1095 | MINYAK MASAK TULEN CAP VESAWIT | 5 kg | `palm_oil_myr_per_tonne` |
| `PRD_ONN_06` | Bawang Merah Kecil Import (Rose Shallots) | 131 | BAWANG KECIL MERAH ROSE IMPORT (INDIA) | 1kg | `brent_crude_usd` |

---

## 4. Primary Key & Integrity Assertions

- **Uniqueness Check:** `assert unique(date, store_id, product_id) == 35,064` -> **PASSED**
- **Completeness Check:** `assert null_keys == 0` -> **PASSED**
- **Cartesian Soundness:** Every active store-product combination has continuous 1,461 daily records with zero dropped dates -> **PASSED**
- **Raw Data Immutability:** All files under `data/raw/` remain untouched and hash-preserved -> **PASSED**
