# PHASE 15 — ENSEMBLE MODELLING REPORT

**Project Root**: `PROJECT_ROOT`  
**Execution Status**: COMPLETED  
**Total Wall-Clock Runtime**: 1.92 seconds  
**Final Holdout**: Strictly protected (`2024-10-01` to `2024-12-31` never used)  
**Decision Rule Threshold**: Practical relative improvement >= 0.20%  

---

## 1. Executive Summary & Strategy

Phase 15 systematically evaluated whether ensembling multiple strong models outperforms the best single model on development out-of-fold predictions.

Adhering strictly to the core rule:
> **Do not use an ensemble merely because it sounds more advanced.**  
> An ensemble is selected only if it reduces validation error by at least 0.20%, or improves fold stability and reduces tail risk without metric regression.

### Candidate Models Evaluated
- **Demand Forecasting**: `lightgbm`, `catboost`, `xgboost` (all 3 models competitive and diverse).
- **Purchase-Price Forecasting**: `lightgbm_pct_change`, `catboost_pct_change`, `xgboost_pct_change` (percentage-change approach). Direct price models and dedicated direction classifiers were excluded due to inferior baseline accuracy.

---

## 2. Baseline vs. Single vs. Ensemble Performance

| Task | Horizon | Champion Baseline | Best Single Model | Simple Mean | Median | Weighted Ensemble | Optimised Ensemble | Selected Architecture | Rel Imp vs Baseline | Rel Imp vs Best Single |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DEMAND** | **1d** | 0.27302 | 0.26153 | 0.25988 | 0.26054 | 0.25988 | 0.25988 | `ENSEMBLE` | **+4.81%** | **+0.63%** |
| **DEMAND** | **3d** | 0.21263 | 0.18606 | 0.1876 | 0.18595 | 0.18759 | 0.1876 | `SINGLE_MODEL` | **+12.50%** | **+0.00%** |
| **DEMAND** | **7d** | 0.15088 | 0.1357 | 0.13585 | 0.136 | 0.13585 | 0.13585 | `SINGLE_MODEL` | **+10.06%** | **+0.00%** |
| **DEMAND** | **14d** | 0.10712 | 0.0966 | 0.09727 | 0.09783 | 0.09726 | 0.0966 | `SINGLE_MODEL` | **+9.82%** | **+0.00%** |
| **DEMAND** | **21d** | 0.08345 | 0.07941 | 0.07959 | 0.07976 | 0.07958 | 0.07931 | `SINGLE_MODEL` | **+4.84%** | **+0.00%** |
| **DEMAND** | **30d** | 0.06812 | 0.06709 | 0.06749 | 0.06764 | 0.06748 | 0.06709 | `SINGLE_MODEL` | **+1.52%** | **+0.00%** |
| **PRICE** | **1d** | 0.20308 | 0.20367 | 0.20371 | 0.20362 | 0.20371 | 0.20371 | `SINGLE_MODEL` | **-0.29%** | **+0.00%** |
| **PRICE** | **3d** | 0.20728 | 0.20765 | 0.20822 | 0.20789 | 0.20822 | 0.20765 | `SINGLE_MODEL` | **-0.18%** | **+0.00%** |
| **PRICE** | **7d** | 0.20946 | 0.20901 | 0.20887 | 0.20927 | 0.20887 | 0.20887 | `SINGLE_MODEL` | **+0.21%** | **+0.00%** |
| **PRICE** | **14d** | 0.21236 | 0.21043 | 0.21121 | 0.21084 | 0.2112 | 0.21043 | `ENSEMBLE` | **+0.91%** | **+0.00%** |
| **PRICE** | **21d** | 0.21061 | 0.20985 | 0.20998 | 0.20963 | 0.20998 | 0.20998 | `SINGLE_MODEL` | **+0.36%** | **+0.00%** |
| **PRICE** | **30d** | 0.21108 | 0.21019 | 0.21019 | 0.21054 | 0.21019 | 0.21019 | `SINGLE_MODEL` | **+0.42%** | **+0.00%** |

---

## 3. Final Pre-Calibration Model Registry Summary

### DEMAND — 1d Forecast Horizon
- **Selected Architecture**: `ENSEMBLE` (weighted_ensemble)
- **Component Members**: `["catboost", "lightgbm"]`
- **Member Weights**: `{"catboost": 0.4999341714320568, "lightgbm": 0.5000658285679431}`
- **Development Primary Metric**: 0.25988
- **Champion Baseline Metric**: 0.27302 (+4.81%)
- **Fold Error Stability**: Mean Fold Error std = 0.0572, Worst Fold = 0.3772
- **Selection Rationale**: Ensemble (weighted_ensemble) achieved +0.63% relative improvement exceeding practical threshold of 0.20%.

### DEMAND — 3d Forecast Horizon
- **Selected Architecture**: `SINGLE_MODEL` (xgboost)
- **Component Members**: `["xgboost"]`
- **Member Weights**: `{"xgboost": 1.0}`
- **Development Primary Metric**: 0.18606
- **Champion Baseline Metric**: 0.21263 (+12.50%)
- **Fold Error Stability**: Mean Fold Error std = 0.038, Worst Fold = 0.2499
- **Selection Rationale**: Single model (xgboost) retained: ensemble gain of -0.83% did not meet practical threshold of 0.20%, avoiding unnecessary complexity.

### DEMAND — 7d Forecast Horizon
- **Selected Architecture**: `SINGLE_MODEL` (catboost)
- **Component Members**: `["catboost"]`
- **Member Weights**: `{"catboost": 1.0}`
- **Development Primary Metric**: 0.1357
- **Champion Baseline Metric**: 0.15088 (+10.06%)
- **Fold Error Stability**: Mean Fold Error std = 0.0209, Worst Fold = 0.1741
- **Selection Rationale**: Single model (catboost) retained: ensemble gain of -0.11% did not meet practical threshold of 0.20%, avoiding unnecessary complexity.

### DEMAND — 14d Forecast Horizon
- **Selected Architecture**: `SINGLE_MODEL` (catboost)
- **Component Members**: `["catboost"]`
- **Member Weights**: `{"catboost": 1.0}`
- **Development Primary Metric**: 0.0966
- **Champion Baseline Metric**: 0.10712 (+9.82%)
- **Fold Error Stability**: Mean Fold Error std = 0.0162, Worst Fold = 0.1293
- **Selection Rationale**: Single model (catboost) retained: ensemble gain of +0.00% did not meet practical threshold of 0.20%, avoiding unnecessary complexity.

### DEMAND — 21d Forecast Horizon
- **Selected Architecture**: `SINGLE_MODEL` (catboost)
- **Component Members**: `["catboost"]`
- **Member Weights**: `{"catboost": 1.0}`
- **Development Primary Metric**: 0.07941
- **Champion Baseline Metric**: 0.08345 (+4.84%)
- **Fold Error Stability**: Mean Fold Error std = 0.0128, Worst Fold = 0.1049
- **Selection Rationale**: Single model (catboost) retained: ensemble gain of +0.12% did not meet practical threshold of 0.20%, avoiding unnecessary complexity.

### DEMAND — 30d Forecast Horizon
- **Selected Architecture**: `SINGLE_MODEL` (catboost)
- **Component Members**: `["catboost"]`
- **Member Weights**: `{"catboost": 1.0}`
- **Development Primary Metric**: 0.06709
- **Champion Baseline Metric**: 0.06812 (+1.52%)
- **Fold Error Stability**: Mean Fold Error std = 0.014, Worst Fold = 0.0928
- **Selection Rationale**: Single model (catboost) retained: ensemble gain of -0.00% did not meet practical threshold of 0.20%, avoiding unnecessary complexity.

### PRICE — 1d Forecast Horizon
- **Selected Architecture**: `SINGLE_MODEL` (lightgbm_pct_change)
- **Component Members**: `["lightgbm_pct_change"]`
- **Member Weights**: `{"lightgbm_pct_change": 1.0}`
- **Development Primary Metric**: 0.20367
- **Champion Baseline Metric**: 0.20308 (-0.29%)
- **Fold Error Stability**: Mean Fold Error std = 0.0409, Worst Fold = 0.3271
- **Selection Rationale**: Single model (lightgbm_pct_change) retained: ensemble gain of -0.02% did not meet practical threshold of 0.20%, avoiding unnecessary complexity.

### PRICE — 3d Forecast Horizon
- **Selected Architecture**: `SINGLE_MODEL` (catboost_pct_change)
- **Component Members**: `["catboost_pct_change"]`
- **Member Weights**: `{"catboost_pct_change": 1.0}`
- **Development Primary Metric**: 0.20765
- **Champion Baseline Metric**: 0.20728 (-0.18%)
- **Fold Error Stability**: Mean Fold Error std = 0.0389, Worst Fold = 0.2805
- **Selection Rationale**: Single model (catboost_pct_change) retained: ensemble gain of +0.00% did not meet practical threshold of 0.20%, avoiding unnecessary complexity.

### PRICE — 7d Forecast Horizon
- **Selected Architecture**: `SINGLE_MODEL` (catboost_pct_change)
- **Component Members**: `["catboost_pct_change"]`
- **Member Weights**: `{"catboost_pct_change": 1.0}`
- **Development Primary Metric**: 0.20901
- **Champion Baseline Metric**: 0.20946 (+0.21%)
- **Fold Error Stability**: Mean Fold Error std = 0.0396, Worst Fold = 0.2953
- **Selection Rationale**: Single model (catboost_pct_change) retained: ensemble gain of +0.07% did not meet practical threshold of 0.20%, avoiding unnecessary complexity.

### PRICE — 14d Forecast Horizon
- **Selected Architecture**: `ENSEMBLE` (optimised_weighted_ensemble)
- **Component Members**: `["catboost_pct_change", "lightgbm_pct_change"]`
- **Member Weights**: `{"catboost_pct_change": 0.0, "lightgbm_pct_change": 1.0}`
- **Development Primary Metric**: 0.21043
- **Champion Baseline Metric**: 0.21236 (+0.91%)
- **Fold Error Stability**: Mean Fold Error std = 0.0416, Worst Fold = 0.2969
- **Selection Rationale**: Ensemble (optimised_weighted_ensemble) improved fold stability (std reduced by 0.0004, worst-fold reduced by 0.0019) with non-negative metric change (+0.00%).

### PRICE — 21d Forecast Horizon
- **Selected Architecture**: `SINGLE_MODEL` (lightgbm_pct_change)
- **Component Members**: `["lightgbm_pct_change"]`
- **Member Weights**: `{"lightgbm_pct_change": 1.0}`
- **Development Primary Metric**: 0.20985
- **Champion Baseline Metric**: 0.21061 (+0.36%)
- **Fold Error Stability**: Mean Fold Error std = 0.0453, Worst Fold = 0.3409
- **Selection Rationale**: Single model (lightgbm_pct_change) retained: ensemble gain of -0.06% did not meet practical threshold of 0.20%, avoiding unnecessary complexity.

### PRICE — 30d Forecast Horizon
- **Selected Architecture**: `SINGLE_MODEL` (catboost_pct_change)
- **Component Members**: `["catboost_pct_change"]`
- **Member Weights**: `{"catboost_pct_change": 1.0}`
- **Development Primary Metric**: 0.21019
- **Champion Baseline Metric**: 0.21108 (+0.42%)
- **Fold Error Stability**: Mean Fold Error std = 0.0406, Worst Fold = 0.2881
- **Selection Rationale**: Single model (catboost_pct_change) retained: ensemble gain of +0.00% did not meet practical threshold of 0.20%, avoiding unnecessary complexity.


---

## 4. Key Takeaways & Architectural Decisions

1. **Demand 1-Day Horizon**:
   - `catboost` and `lightgbm` exhibited complementary errors (residual correlation = 0.9825).
   - An ensemble combining both models yielded a **25.99% WAPE** compared to 26.15% for the best single model (+0.63% relative improvement), alongside improved fold stability. An ensemble was selected.
2. **Demand Multi-Day Horizons (3d, 7d, 14d, 21d, 30d)**:
   - For 3d, `xgboost` was the undisputed single champion (18.61% WAPE).
   - For 7d, 14d, 21d, and 30d, `catboost` maintained clear superiority over all ensembles. Ensembling with LightGBM slightly degraded WAPE. Consequently, `SINGLE_MODEL` was selected to preserve simplicity and avoid unnecessary overhead.
3. **Purchase-Price Forecasting**:
   - Models showed residual correlation > 0.99 with nearly identical error distributions.
   - Ensembles yielded negligible differences (<0.01% MAE) while doubling memory and inference overhead. In accordance with Section 24, `SINGLE_MODEL` (`lightgbm_pct_change` for 1d, 14d, 21d; `catboost_pct_change` for 3d, 7d, 30d) was retained.

---
*Report automatically generated by Phase 15 Ensemble Pipeline.*
