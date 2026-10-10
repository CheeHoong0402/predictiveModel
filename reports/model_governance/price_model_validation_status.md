# Model Governance Audit: Price Model Validation Status

**Report Generated:** 2026-10-11  
**Project Root:** `PROJECT_ROOT`  
**Governing Standard:** Section 33 Model Risk & Validation Protocol  

---

## 1. Executive Summary

| Model Component | Validation Status | Safe for Decision? | Safe for Speculative Timing? |
| :--- | :--- | :--- | :--- |
| **Demand Forecasting Model** | **VALIDATED** | **YES** | **N/A** |
| **Purchase-Price Model** | **NOT_VALIDATED_FOR_DECISION_USE** | **NO** | **NO** |

Following the comprehensive purchase-price training data sufficiency audit, the purchase-price models have been formally **quarantined and barred from downstream decision-making**.

The entire downstream procurement decision stack has been placed under a strict **Model Governance Gate** (`src/models/governance.py`). If the price model is not validated, the system automatically falls back to **pure baseline demand-and-inventory-driven optimization**.

---

## 2. Root Cause Analysis

### A. Critical Target Leakage in Phase 13/14 Cache
- **Bug Location:** `src/ablation/cache.py` (lines 112–113).
- **Flaw:** During the model-ready matrix cache construction, the backward-looking feature `purchase_price_pct_change_{h}d` was unconditionally overwritten by the forward target column `purchase_price_pct_change_{h}d__targ` for horizons **1-day, 7-day, and 30-day**.
- **Impact:** LightGBM models trained on these cached matrices learned to read the future target directly, resulting in artificial, fraudulent validation MAEs of **RM 0.00275 (1d)**, **RM 0.00361 (7d)**, and **RM 0.00487 (30d)**.
- **Remediation:** Removed the overwriting logic in `src/ablation/cache.py` and instituted mandatory `assert_no_target_leakage()` assertions across all cache and feature ingestion pipelines.

### B. Lack of Demonstrable Predictive Skill (Clean Models)
- Uncontaminated Phase 12 / 15 price models exhibit an MAE of **~RM 0.20 to RM 0.21**.
- When benchmarked against a trivial **historical expanding mean or 30-day moving average**, the machine learning models achieve **-1.33% to +0.53% relative improvement**.
- The models exhibit no statistically defensible edge over constant or moving-average baselines.

### C. Synthetic / Unverified Data Provenance
- Analysis of the raw supplier purchase price series reveals that daily prices behave as stationary Gaussian noise centered around hard-coded catalogue base prices.
- Prices change on 98.1% of days with negligible autocorrelation in daily differences (-0.50, the theoretical signature of white noise difference).
- The series lacks real-world procurement dynamics (e.g. quarterly contracts, sticky wholesale price tiers).

---

## 3. Global Decision Gate Implementation

A central gate was implemented in `src/models/governance.py` and connected to `config/model_governance.yaml`:
```python
if price_model_usable_for_decision():
    enable_price_decision_logic = True
else:
    enable_price_decision_logic = False
```

### Fail-Safe Rules:
1. **Default State:** If `config/model_governance.yaml` is missing, unreadable, or corrupted, `price_model_usable_for_decision()` **strictly returns `False`**.
2. **Order Adjustment Disablement:** When the price model is invalid, price-based forward buying and speculative price-drop delays are completely disabled. Procurement quantities are governed solely by safety stock, demand forecasts, lead times, and storage constraints.
3. **Ensemble Inference Protection:** `src/ensemble/inference.py` prohibits loading models from `models/optimised/price/` and raises a `PermissionError` if an unvalidated price model is requested for decision use.

---

## 4. Quarantined Artefacts

- **6 Optimised Price Models** (`models/optimised/price/*.joblib`): Marked `INVALID_TARGET_LEAKAGE`.
- **54 Clean Price Models** (`models/price/**/*.joblib`): Marked `NOT_VALIDATED_FOR_DECISION_USE`.
- **11 Downstream Artefacts** (predictions, calibration tables, timing arbitrage reports): Marked `STALE_UPSTREAM_PRICE_MODEL_INVALID`.

See [price_model_registry.csv](reports/model_governance/price_model_registry.csv) and [invalidated_price_artifacts.csv](reports/model_governance/invalidated_price_artifacts.csv) for complete inventories.

---

## 5. Verification Checklist

- [x] Target leakage bug fixed in `src/ablation/cache.py`
- [x] Strict `assert_no_target_leakage()` gate implemented
- [x] Feature manifest validation enforces `TargetLeakageError`
- [x] Contaminated price models quarantined (not deleted)
- [x] Ensemble inference blocks loading of quarantined price models
- [x] Recommendation engine explicitly gates price decision logic
- [x] Demand model remains 100% active, validated, and unaffected
- [x] Protected holdout remains strictly untouched (no training or re-tuning)
