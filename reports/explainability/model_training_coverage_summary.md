# Model Training Coverage and Data Duration Audit

## 1. Executive Summary

This audit definitively documents the exact temporal span, row counts, entity coverage, and exclusions
used to train all final predictive models for Demand Forecasting and Supplier Purchase-Price Forecasting.

### Key Findings:
- **Total Historical Data Available**: 4.0 calendar years (`2021-01-01` to `2024-12-31`, 1,461 days, 35,064 rows across 24 product-store series).
- **Pre-Holdout Development Span**: ~3.75 calendar years / 45.0 months (`2021-01-01` to `2024-09-30`, 1,369 days, 32,856 rows).
- **Effective Training Span**: Ranging from **1,369 days / 32,856 rows** for 1-day horizon models to **1,340 days / 32,160 rows** for 30-day horizon models.
- **Entity Coverage**: 100% complete coverage across **4 stores**, **6 products**, and **6 suppliers**.
- **Final Holdout Strict Reservation**: Exactly **92 calendar days / 2,208 rows** (`2024-10-01` to `2024-12-31`) remained **strictly untouched and unseen** during model training, tuning, ensemble selection, and uncertainty calibration.
- **Longest Lookback Duration**: 365 days (annual seasonality / 365d rolling demand). Lookback-to-training ratio is **26.66%**, confirming over 1,000 days of fully populated historical context.

## 2. Model Training Coverage by Horizon

| Task | Horizon | Selected Architecture | Selected Model | Train Start | Train End | Days | Months | Years | Training Rows | Excluded Rows | Holdout Reserved |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Demand | 1d | ENSEMBLE | `weighted_ensemble` | 2021-01-01 | 2024-09-30 | 1369 | 45.0 | 3.75 | 32,856 | 0 | True |
| Demand | 3d | SINGLE_MODEL | `xgboost` | 2021-01-01 | 2024-09-28 | 1367 | 44.9 | 3.74 | 32,808 | 48 | True |
| Demand | 7d | SINGLE_MODEL | `catboost` | 2021-01-01 | 2024-09-24 | 1363 | 44.8 | 3.73 | 32,712 | 144 | True |
| Demand | 14d | SINGLE_MODEL | `catboost` | 2021-01-01 | 2024-09-17 | 1356 | 44.6 | 3.71 | 32,544 | 312 | True |
| Demand | 21d | SINGLE_MODEL | `catboost` | 2021-01-01 | 2024-09-10 | 1349 | 44.3 | 3.69 | 32,376 | 480 | True |
| Demand | 30d | SINGLE_MODEL | `catboost` | 2021-01-01 | 2024-09-01 | 1340 | 44.0 | 3.67 | 32,160 | 696 | True |
| Price | 1d | SINGLE_MODEL | `lightgbm_pct_change` | 2021-01-01 | 2024-09-30 | 1369 | 45.0 | 3.75 | 32,856 | 0 | True |
| Price | 3d | SINGLE_MODEL | `catboost_pct_change` | 2021-01-01 | 2024-09-28 | 1367 | 44.9 | 3.74 | 32,808 | 48 | True |
| Price | 7d | SINGLE_MODEL | `catboost_pct_change` | 2021-01-01 | 2024-09-24 | 1363 | 44.8 | 3.73 | 32,712 | 144 | True |
| Price | 14d | ENSEMBLE | `optimised_weighted_ensemble` | 2021-01-01 | 2024-09-17 | 1356 | 44.6 | 3.71 | 32,544 | 312 | True |
| Price | 21d | SINGLE_MODEL | `lightgbm_pct_change` | 2021-01-01 | 2024-09-10 | 1349 | 44.3 | 3.69 | 32,376 | 480 | True |
| Price | 30d | SINGLE_MODEL | `catboost_pct_change` | 2021-01-01 | 2024-09-01 | 1340 | 44.0 | 3.67 | 32,160 | 696 | True |

## 3. Walk-Forward Validation Coverage (33 Expanding Folds)

- **Number of Folds**: 33 sequential expanding-window folds.
- **Earliest Training Window**: Fold 1 (`2021-01-01` to `2021-12-31`, 365 days, 8,760 rows).
- **Latest Training Window**: Fold 33 (`2021-01-01` to `2024-08-17`, 1325 days, 31,800 rows).
- **Validation Horizon**: 30-day non-overlapping validation blocks per fold, evaluating both short-term (1d, 3d, 7d) and medium-term (14d, 21d, 30d) performance.

## 4. Training Exclusions & Temporal Boundary Integrity

| Category | Description | Affected Horizons | Max Excluded Rows | Leakage Prevention Rationale |
| :--- | :--- | :--- | :---: | :--- |
| Target Horizon Boundary Window | Rows at the end of development data where t + h > 2024-09-30 (would look into holdout) | 3d (48 rows), 7d (144 rows), 14d (312 rows), 21d (480 rows), 30d (696 rows) | 696 | Prevents targets from peeking into unseen future holdout dates |
| Final Holdout Reservation | Entire Q4 2024 test period (2024-10-01 to 2024-12-31, 92 days x 24 series) | All horizons (1d, 3d, 7d, 14d, 21d, 30d) | 2208 | Strictly reserved for final blind competition evaluation; zero rows used in training |
| Warm-up Feature Lookback (365d) | Initial 365-day period in 2021 where annual rolling features backfill from zero/expanding state | All horizons | 0 | Handled with expanding window aggregations and missingness indicator flags without row drop |
| Lead-target Invalidation | Rows flagged with target_valid == False in forecast targets dataset | Target boundary edge rows | 696 | Ensures models are only trained on complete, verified future actuals |

## 5. Verification & Final Refit Status

- **Refit Scope**: All final production models were refitted on the full pre-holdout development dataset (incorporating all 33 folds).
- **Holdout Protection**: Certified clean. No data points on or after `2024-10-01` were used in feature aggregation, imputation, hyperparameter search, ensemble weights, or calibration scaling.
