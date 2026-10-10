# GOVERNANCE NOTICE: STALE_INVALID_PRICE_MODEL

**Date:** 2026-10-11  
**Project:** Retail Predictive Procurement System  
**Component:** Phase 20 Historical Business Backtesting  
**Model Under Quarantine:** Purchase-Price Prediction Engine (Phase 12, 13, 14, 15)  
**Governance Ruling:** `PRICE MODEL SAFE FOR DECISION = FALSE`

---

## 1. Reason for Invalidation

1. **Target Leakage in Feature Cache:** The Phase 13/14 price feature pipeline contained target contamination in `src/ablation/cache.py`, where `purchase_price_pct_change_{h}d` was overwritten by forward targets (`...__targ`), producing unphysical sub-cent MAEs (~RM 0.003).
2. **Lack of Proven Predictive Edge:** Uncontaminated baseline models exhibited ~RM 0.20–0.21 MAE with zero statistically significant advantage over rolling moving-average baselines.
3. **Quarantine Action:** All price-timing arbitrage, speculative pre-buying, and price forecast influence have been strictly removed from decision-grade procurement recommendations and historical backtest evaluations.

---

## 2. Decision-Grade Backtest Status

- **Approved Strategy:** Final AI Demand + Inventory Optimisation Strategy.
- **Decision Dimensions Active:**
  - Calibrated Demand Forecasting (LightGBM/TFT with conformal $P_{10}/P_{50}/P_{90}$)
  - Inventory Replenishment & Safety Stock Optimisation
  - Lead-time and supplier fill-rate uncertainty
  - Perishable shelf-life constraints and wastage minimization
  - Realised operational costs (Holding, Stockout, Wastage, Purchase, Ordering)
- **Realised Economic Savings:**
  - **44.49% net cost reduction** vs. Fixed 14-Day benchmark (RM 5,775,380.76 saved across 990 days).
  - **47.59% net cost reduction** vs. Reorder Point policy (RM 6,543,130.77 saved).
  - **82.4% reduction in perishable spoilage** (33,806 units vs. 191,697 units).
- **Price Forecasting Contribution:** **0.0%**. Zero savings are claimed or advertised from purchase-price timing arbitrage.

---

## 3. Artefacts Status

- `reports/backtesting/price_timing_backtest.csv`: Marked as `INVALID_PRICE_MODEL_RESEARCH_ONLY` with `decision_grade = false`.
- `reports/backtesting/strategy_performance.csv`: Decision-grade operational comparison.
- `reports/backtesting/ai_vs_reference.csv`: Verified operational comparisons.
- `data/backtesting/decision_backtest.parquet`: Tagged with `price_model_status = "NOT_VALIDATED"` and `price_forecast_used_in_decision = false`.
