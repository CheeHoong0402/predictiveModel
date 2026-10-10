# Phase 7 — Feature Engineering Summary Report

## Executive Summary

- **Status**: Successfully Completed
- **Input Table**: `data/processed/master_daily_demand_corrected.parquet`
- **Output Table**: `data/processed/model_features.parquet`
- **Input Rows**: `35,064`
- **Output Rows**: `35,064`
- **Primary Key**: `date + store_id + product_id` (100% Unique, 0 duplicates)
- **Input Columns**: `108`
- **Total Engineered Features**: `308`
- **Demand-Model Candidate Features**: `275`
- **Purchase-Price Candidate Features**: `163`
- **Output Dataset Size**: `21.48 MB` (Snappy compressed Parquet)
- **Execution Time**: `1.19 seconds`

---

## Feature Families Breakdown

| Feature Family | Feature Count | Model Applicability | Leakage Risk |
| :--- | :--- | :--- | :--- |
| `annual_seasonality` | 2 | Demand / Price / Both | None (Shifted) |
| `calendar` | 13 | Demand / Price / Both | None (Shifted) |
| `demand_history` | 12 | Demand / Price / Both | None (Shifted) |
| `demand_momentum` | 6 | Demand / Price / Both | None (Shifted) |
| `festival_proximity` | 25 | Demand / Price / Both | None (Shifted) |
| `government_assistance` | 3 | Demand / Price / Both | None (Shifted) |
| `holiday_proximity` | 8 | Demand / Price / Both | None (Shifted) |
| `interactions` | 9 | Demand / Price / Both | None (Shifted) |
| `inventory_coverage` | 2 | Demand / Price / Both | None (Shifted) |
| `inventory_state` | 5 | Demand / Price / Both | None (Shifted) |
| `macro_commodity` | 7 | Demand / Price / Both | None (Shifted) |
| `macro_currency` | 7 | Demand / Price / Both | None (Shifted) |
| `macro_fuel` | 10 | Demand / Price / Both | None (Shifted) |
| `macro_inflation` | 8 | Demand / Price / Both | None (Shifted) |
| `market_price` | 10 | Demand / Price / Both | None (Shifted) |
| `missingness_indicator` | 4 | Demand / Price / Both | None (Shifted) |
| `observed_sales` | 13 | Demand / Price / Both | None (Shifted) |
| `policy_subsidy` | 11 | Demand / Price / Both | None (Shifted) |
| `price_elasticity_proxy` | 2 | Demand / Price / Both | None (Shifted) |
| `price_momentum` | 6 | Demand / Price / Both | None (Shifted) |
| `procurement_supplier` | 10 | Demand / Price / Both | None (Shifted) |
| `promotion` | 10 | Demand / Price / Both | None (Shifted) |
| `purchase_price_history` | 8 | Demand / Price / Both | None (Shifted) |
| `purchase_price_rolling` | 21 | Demand / Price / Both | None (Shifted) |
| `replenishment_history` | 4 | Demand / Price / Both | None (Shifted) |
| `rolling_demand` | 42 | Demand / Price / Both | None (Shifted) |
| `salary_cycle` | 6 | Demand / Price / Both | None (Shifted) |
| `school_holiday` | 5 | Demand / Price / Both | None (Shifted) |
| `selling_price` | 7 | Demand / Price / Both | None (Shifted) |
| `static_product` | 5 | Demand / Price / Both | None (Shifted) |
| `static_store` | 4 | Demand / Price / Both | None (Shifted) |
| `stockout_history` | 8 | Demand / Price / Both | None (Shifted) |
| `weather` | 6 | Demand / Price / Both | None (Shifted) |
| `weather_rolling` | 5 | Demand / Price / Both | None (Shifted) |
| `weather_spells` | 2 | Demand / Price / Both | None (Shifted) |
| `weekly_seasonality` | 2 | Demand / Price / Both | None (Shifted) |

---

## Temporal Safety & Leakage Prevention Guarantees

1. **Strict Lag Shift Convention**:
   - All demand, sales, and target-related rolling statistics are calculated on historical values shifted by at least 1 day relative to observation date $T$ (`shift(1)`).
   - Lags: `demand_lag_1` represents demand on day $T-1$; `demand_lag_7` represents demand on day $T-7$.
   - Rolling windows: `demand_mean_7d` is strictly computed across days $[T-7, \dots, T-1]$.
2. **Training-Safe Demand Integrity**:
   - `estimated_true_demand_training_safe` (one-sided past-only correction) is strictly used as the historical demand series.
   - `estimated_true_demand_retrospective` is quarantined in `reports/features/excluded_features.csv`.
3. **Zero Target Contamination**:
   - Zero target variables (`target_demand_t+1`, etc.) were created during this phase.
   - No machine learning model has been trained.
4. **Calendar and Astro-Gazette Determinism**:
   - Public holidays, festivals, and salary cycles use deterministic calendar/gazetted schedules known centuries or weeks in advance.

---

## Feature Missingness Audit

- Features with 0% missingness: `134`
- Features with low missingness (<= 10%): `291`
- Features with moderate missingness (10% to 30%): `7`
- Features with high missingness (> 30%): `10` (annual 364d/365d lags requiring 1 full year history)

---

## Quarantined & Excluded Features

| Feature Name | Family | Reason for Exclusion | Risk Type |
| :--- | :--- | :--- | :--- |
| `estimated_true_demand_retrospective` | `quarantined_demand` | Uses two-sided future information for retrospective analysis; prohibited in training feature table | `future_target_leakage` |
| `target_demand_t_plus_1` | `forecast_target` | Forecast targets must not be created during Phase 7 Feature Engineering | `premature_target_creation` |
| `target_purchase_price_t_plus_7` | `forecast_target` | Forecast targets must not be created during Phase 7 Feature Engineering | `premature_target_creation` |
| `future_observed_weather` | `future_weather` | Actual future weather is unknown at prediction date T; prohibited from feature table | `future_information_leakage` |
| `future_competitor_prices` | `future_market` | Future competitor prices are unknown at prediction date T; prohibited from feature table | `future_information_leakage` |

---

## Generated Report Manifests

- `reports/features/feature_registry.csv`: Complete metadata catalog of all engineered features.
- `reports/features/demand_feature_manifest.csv`: Candidate features selected for demand forecasting models.
- `reports/features/price_feature_manifest.csv`: Candidate features selected for purchase-price forecasting models.
- `reports/features/excluded_features.csv`: Catalog of prohibited/quarantined features.
- `reports/features/feature_quality_report.csv`: Statistical distribution profiles and constant flags.
- `reports/features/feature_missingness.csv`: Detailed missingness audit per feature.
- `reports/features/feature_availability.csv`: Date coverage ranges for every feature.