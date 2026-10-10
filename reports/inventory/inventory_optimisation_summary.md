# PHASE 18 — INVENTORY OPTIMISATION & PURCHASE DECISION ENGINE AUDIT REPORT

**Engine Version:** 1.0.0 (Production Candidate)  
**Execution Runtime:** 1.75 seconds  
**Total Product-Store Decisions Evaluated:** 24  

---

## 1. Executive Summary & Procurement Totals

| Metric | Evaluated Value | Notes |
| :--- | :--- | :--- |
| **Total Recommended Units Ordered** | 281 units | Enforces MOQ & Pack Multiples |
| **Total Expected Purchase Outlay** | RM 4,670.36 | Based on Current Supplier Prices |
| **Total Expected Cost (with Risks)** | RM 5,546.77 | Purchase + Holding + Stockout + Wastage |
| **Expected Total Savings vs 14d Fixed** | RM 292,297.07 | Direct Optimisation Surplus |
| **Mean Recommended Coverage** | 1.0 days | Candidate Range 1–30 Days |
| **Perishable Stock Safety Enforced** | 100% | Max coverage capped at shelf-life |

---

## 2. Action Distribution & Operational Severity

### Recommended Actions Breakdown
- **`URGENT_REORDER`**: 21 SKU-Stores (87.5%)
- **`DO_NOT_REORDER`**: 3 SKU-Stores (12.5%)

### Priority / Severity Tiers
- **`CRITICAL`**: 21 SKU-Stores (87.5%)
- **`NONE`**: 3 SKU-Stores (12.5%)

---

## 3. Cost Assumption Provenance & Audit Status

All cost components used in candidate evaluation maintain strict provenance tagging:

| Parameter Key | Unit Cost (RM) | Provenance Status | Business Source |
| :--- | :--- | :--- | :--- |
| `inventory_holding_cost_per_unit_per_day` | RM 0.25 | **`VERIFIED`** | retailer_provided |
| `storage_cost_per_unit_per_day` | RM 0.15 | **`VERIFIED`** | retailer_provided |
| `stockout_cost_per_unit` | RM 3.50 | **`VERIFIED`** | retailer_provided |
| `unit_lost_gross_margin` | DYNAMIC | **`DYNAMIC`** | derived |
| `wastage_cost_per_unit` | RM 7.20 | **`VERIFIED`** | retailer_provided |
| `ordering_cost_per_po` | RM 50.00 | **`ASSUMED_COST`** | default_assumption |
| `annual_capital_holding_rate` | RM 0.15 | **`ASSUMED_COST`** | default_assumption |
| `goodwill_penalty_multiplier` | RM 1.00 | **`ASSUMED_COST`** | default_assumption |
| `emergency_replenishment_surcharge` | RM 0.00 | **`UNAVAILABLE`** | unavailable |

---

## 4. Policy & Strategy Comparison

Comparison against standard procurement benchmarks across all evaluated items:

| Strategy | Mean Coverage Days | Mean Order Quantity | Total Portfolio Cost | Mean Stockout Risk |
| :--- | :--- | :--- | :--- | :--- |
| **BALANCED_STRATEGY** | 1.0 days | 11.7 units | RM 5,546.77 | 0.8% |
| **FIXED_14_DAY_STOCK** | 14.0 days | 749.9 units | RM 297,843.86 | 0.0% |
| **HIGH_SAFETY_STRATEGY** | 14.2 days | 694.9 units | RM 314,284.74 | 12.4% |
| **LOW_STOCK_STRATEGY** | 6.0 days | 272.5 units | RM 106,092.46 | 9.0% |

---

## 5. Model Governance & Price Validation Gate Status

- **Price Model Status:** `NOT_VALIDATED`
- **Price Influence on Decisions:** `DISABLED`
- **Operating Decision Mode:** `DEMAND_INVENTORY_ONLY`
- **Price Influenced Order Quantities:** 0 / 24
- **Price Influenced Coverage Days:** 0 / 24
- **Net Price Timing Savings Subtracted:** RM 0.00 (strictly RM 0.00 when gated)

---

## 6. Decision Engine Robustness Profile

- **Robust Decisions (Consistent across P10 to P90):** 12 / 24
- **Uncertainty-Sensitive Decisions:** 12 / 24

*Audit verification confirmed: All order quantities non-negative, MOQ/pack-size aligned, price quarantine gate respected, and zero future leakage.*
