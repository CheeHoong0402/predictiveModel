# Phase 6 — Stockout and True-Demand Correction Methodology

## 1. Principles of Demand Censorship in Food Retail

In fast-moving food retail, customer demand is latent. When a store runs out of stock:
$$\text{Observed Sales} = \min(\text{True Customer Demand}, \text{Available Stock})$$
Using observed sales directly to train predictive inventory models introduces severe negative bias: the algorithm learns that demand is zero whenever stock is zero, perpetuating chronic stockouts.

## 2. Multi-Tier Evidence Hierarchy

Observations are classified using an empirical evidence hierarchy:
1. **Full-Day Stockouts:** Recorded when `number_of_hours_out_of_stock == 12.0` or opening, closing, and sales were zero with active stockout flag.
2. **Partial-Day Stockouts:** Recorded when `0 < number_of_hours_out_of_stock < 12.0`. Sales are partially observed during operating hours before depletion.
3. **Unconstrained Observations:** Adequate closing inventory (`closing_inventory > 0`) and zero stockout flags.

## 3. Reference-Set Construction (Past-Only Training-Safe)

To prevent lookahead leakage into ML feature pipelines, reference sets for date $T$ only draw from unconstrained observations strictly occurring before $T$ (`ref_date < T`):
- **Level 1 (Highest Quality):** Same store, same SKU, same day of week, within past 12 weeks, matching promotion regime, matching price band ($\pm 10\%$).
- **Level 2 (Extended DOW):** Same store, same SKU, same day of week, within past 26 weeks, matching promotion regime.
- **Level 3 (All-Time DOW):** Same store, same SKU, same day of week across all past history.
- **Level 4 (All Weekdays):** All days of week within past 12 weeks, matching promotion regime.
- **Level 5 (Fallback):** All unconstrained past observations for the store-SKU.

## 4. Mathematical Estimation Formulas

For Full-Day Stockouts:
$$\hat{D}_T = \max(\text{Median}(R_T), S_T)$$

For Partial-Day Stockouts:
$$\hat{D}_T = \max\left(S_T / (1 - f_T), \text{Median}(R_T), S_T\right)$$
where $f_T = \text{stockout\_hours} / 12.0$, and bounded by the historical 75th percentile and maximum correction ratio (3.5x).

Lost Sales:
$$\text{Lost Sales}_T = \max(\hat{D}_T - S_T, 0)$$
