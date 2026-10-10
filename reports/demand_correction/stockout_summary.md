# Phase 6 — Stockout and True-Demand Correction Summary Report

*Execution Timestamp: 2026-10-10 21:10:21*

## 1. Executive Summary

- **Total Observations Analysed:** **35,064**
- **Total Censored / Constrained Observations:** **11,609** (33.14%)
- **Confirmed Full-Day Stockouts:** **3,949** (11.26%)
- **Confirmed Partial-Day Stockouts:** **6,959** (19.85%)
- **Total Corrected Observations:** **10,497**
- **Total Observed Units Sold:** **1,374,675.0 units**
- **Total Estimated True Demand:** **1,720,109.0 units**
- **Total Estimated Lost Sales:** **345,434.0 units**
- **Aggregate Demand Uplift from Correction:** **+25.13%**
- **High-Risk / Extreme Corrections Flagged:** **523**
- **Execution Time:** **3.89 seconds**
- **Output Artifact:** [`data/processed/master_daily_demand_corrected.parquet`](data/processed/master_daily_demand_corrected.parquet)

---

## 2. Confidence Level Distribution

| Confidence Level | Observations | % of Total | Operational Interpretation |
|---|---|---|---|
| **HIGH** | 28,891 | 82.40% | Unconstrained observations or confirmed stockouts with $\ge 8$ strong past non-stockout matches |
| **MEDIUM** | 6,109 | 17.42% | Censored observations with 4 to 7 valid comparable past non-stockout matches |
| **LOW** | 53 | 0.15% | Censored observations with 1 to 3 valid comparable past non-stockout matches |
| **UNCORRECTED** | 11 | 0.03% | Initial observation period with zero preceding history (conservatively uncorrected) |

---

## 3. Stockout Rate & Lost Sales by Product

| Product ID | Product Name | Stockout Days | Stockout Rate % | Observed Sales | Estimated Demand | Estimated Lost Sales | Uplift % |
|---|---|---|---|---|---|---|---|
| `PRD_CHK_01` | Ayam Bulat Segar (Standard Whole Chicken) | 1,249 | 21.4% | 342,449 | 402,641 | 60,192 | +17.6% |
| `PRD_EGG_02` | Telur Ayam Gred A (30 biji) | 2,350 | 40.2% | 235,314 | 310,414 | 75,100 | +31.9% |
| `PRD_OIL_05` | Minyak Masak Sawit Tulen 5kg | 2,465 | 42.2% | 194,467 | 262,188 | 67,721 | +34.8% |
| `PRD_ONN_06` | Bawang Merah Kecil Import (Rose Shallots) | 1,416 | 24.2% | 209,989 | 251,335 | 41,346 | +19.7% |
| `PRD_RIC_04` | Beras Putih Super Tempatan (SST 5%) 10kg | 3,736 | 63.9% | 168,422 | 256,648 | 88,226 | +52.4% |
| `PRD_VCH_03` | Kobis Bulat Cameron (Cameron Cabbage) | 393 | 6.7% | 224,034 | 236,883 | 12,849 | +5.7% |

---

## 4. Methodological Invariants & Leakage Protection

1. **Past-Only Training Safety:** All default target estimates (`estimated_true_demand_training_safe`) strictly query reference observations from calendar dates before date $T$ (`ref_date < T`). Zero future lookahead.
2. **Non-Negativity Invariant:** $\text{estimated\_lost\_sales} = \max(\text{estimated\_demand} - \text{observed\_sales}, 0.0) \ge 0.0$.
3. **Monotonicity Invariant:** $\text{estimated\_true\_demand} \ge \text{observed\_units\_sold}$.
4. **Preservation of Raw & Master Datasets:** Original `master_daily.parquet` and all files under `data/raw/` remain 100% immutable.
