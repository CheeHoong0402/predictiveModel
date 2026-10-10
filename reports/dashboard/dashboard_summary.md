# PHASE 22 — STREAMLIT DASHBOARD SUMMARY REPORT

**Application Entry Point:** `app.py`  
**Execution Command:** `streamlit run app.py`  
**Deployment Platform:** Streamlit Community Cloud / Linux Container / Local Windows  
**Operating Modes:** Dual-mode (Mode A: Demo Mode & Mode B: User Data Mode)  

---

## 1. Executive Summary & Architecture

The Phase 22 Streamlit application converts the entire end-to-end predictive procurement pipeline into an interactive, judge-ready decision dashboard.

```
Retailer Operational Data
         ↓
Demand Forecasting (Phases 11–15 Champion Models)
         ↓
Purchase-Price Forecasting (Phases 12–15 Models)
         ↓
Conformal Uncertainty Calibration (Phase 16 CQR P10/P50/P90)
         ↓
Explainability & Transparency (Phase 17 TreeSHAP & Reason Codes)
         ↓
Cost-Minimising Inventory Optimisation (Phase 18)
         ↓
Monte Carlo Risk Simulation (Phase 19 Fast/Detailed CVaR95 Sizing)
         ↓
Controlled Action Orchestrator (Phase 21 Priorities & Urgency)
         ↓
Streamlit Interactive Dashboard (Phase 22 Multi-Tab Visual Interface)
```

---

## 2. Supported Dashboard Sections

1. **🏛️ Executive Overview:** Headline KPI cards (Spend, Savings, Risk Reduction), Ranked Priority Queue table with badges, Store-level spend breakdowns.
2. **🎯 Recommendation Card:** Visual hero decision card with Action, Quantity, Urgency, Coverage, Price trend, Risk reduction, and plain-language reasoning.
3. **📈 Demand & Price Forecasts:** 6-horizon cumulative demand curves with 80% conformal uncertainty intervals ($P_{10}$ to $P_{90}$) and supplier wholesale price trajectories.
4. **🛡️ Risk & Monte Carlo:** Probabilistic stockout reduction gauges, cost distributions ($P_{10}, P_{50}, P_{90}$, VaR95, CVaR95), and delivery delay reliability.
5. **🔍 Explainability & SHAP:** Plain-language business justification, active machine-readable reason codes, and global feature family contributions.
6. **⏳ Historical Backtest & Replay:** 990-day walk-forward cumulative savings chart (RM 5.78M saved / 44.5%), strategy comparisons, and single-date judge replay audit with strict leakage isolation.
7. **📋 Model Scorecard & Timeline:** Scientific error metrics (WAPE 16.2%, Price MAE RM 0.18, CQR coverage 79.8%), 10-year training timeline, and holdout quarantine verification.
8. **📤 User Data Upload:** Guided 8-file upload interface, downloadable template CSVs, automatic alias mapping, and fast inference execution without retraining.
9. **🔔 Telegram Alerts & Policy:** Real-time alert message preview, trigger policy summaries, 24-hour cooldown status, and credential security.

---

## 3. Runtime & Latency Benchmarks

| Operation | Latency (ms) | Latency (sec) | Status |
| :--- | :--- | :--- | :--- |
| **Initial Load Preparation** | 34.99 ms | 0.035 s | Instant |
| **Executive Payload Retrieval** | 0.08 ms | 0.0001 s | Instant |
| **Selected Product Payload** | 22.45 ms | 0.022 s | Interactive |
| **Historical Replay Payload** | 4.89 ms | 0.005 s | Interactive |
| **User Upload Validation** | 1.03 ms | 0.001 s | Instant |
| **User Upload Batch Inference** | 15.84 ms | 0.016 s | Sub-second |

All operations execute in sub-second timeframes with zero GPU dependencies and zero model retraining on user clicks.
