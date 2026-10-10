# Purchase-Price Training Data Sufficiency and Model Sanity Audit

**PRICE TRAINING DATA SUFFICIENCY: FAIL**

Read-only audit. No model was retrained; no hyperparameter was changed; the protected holdout (from 2024-10-01) was only row-counted.

## Synthetic-data disclosure

- **Target (supplier purchase price): SYNTHETIC / unverified provenance.** The repository never defines `generate_retailer_dataset`; base prices are hard-coded; prices change on every day; one supplier per product.
- **External predictors: MIXED.** Real: FX, 7 commodity series, fuel, CPI/PPI, PriceCatcher (1 month only). Synthetic: commodity columns equal to a seeded random walk: `{'palm_oil_myr_per_tonne': True, 'steel_usd_index': True, 'feed_grain_usd': True}`.
- Any accuracy on this target is therefore *not* independent real-world validation.

## Data volume

- Raw: 2021-01-01 to 2024-12-31 (1461 days, 4.00 years), 35,064 rows; development (pre-holdout) 1369 days, 32,856 rows.
- 24 series (4 stores x 6 products; 6 suppliers). Genuine observations 32,796; forward-filled 60 (0.183%).
- Median genuine observations per series: 1369; density classes {'HIGH': 24}.
- Price-change events: 32,199 (98.1% of days).

## Last-price baseline vs final model (validation, pre-holdout)

| h | mean abs move | last price MAE | MA30 MAE | expanding-mean MAE | final MAE | vs last | vs best simple |
|---|---|---|---|---|---|---|---|
| 1 | 0.2913 | 0.2913 | 0.2031 | 0.2014 | 0.2037 | +30.07% | -1.11% |
| 3 | 0.2834 | 0.2834 | 0.2073 | 0.2062 | 0.2077 | +26.72% | -0.70% |
| 7 | 0.2895 | 0.2895 | 0.2095 | 0.2080 | 0.2090 | +27.81% | -0.49% |
| 14 | 0.2883 | 0.2883 | 0.2124 | 0.2116 | 0.2104 | +27.02% | +0.53% |
| 21 | 0.3044 | 0.3044 | 0.2106 | 0.2071 | 0.2099 | +31.06% | -1.33% |
| 30 | 0.2977 | 0.2977 | 0.2111 | 0.2109 | 0.2102 | +29.40% | -0.14% |

## MAE sanity (A-G)

- **A. Predictable dynamics:** none beyond mean reversion. Lag-1 autocorrelation of daily changes -0.50 (iid noise = -0.50); variance ratio 30d/1d = 1.01 (iid = 2, random walk = 30). Price is a stationary band around a base.
- **B. Barely-moving prices:** No - prices change daily; typical |move| is ~2.0% of price, so a small RM MAE is a property of low noise amplitude, not of skill.
- **C. Forward fill:** 0.183% of rows; all target selections are EXACT (tolerance 0); no carry-forward.
- **D. Target alignment:** target observation date == origin + h for all valid rows; zero origin-from-future and zero target-before-due violations.
- **E. Leakage:** Phase 12 / Phase 15 features are clean (max |corr| with future % change 0.71, backward-looking mean-reversion). **CONFIRMED CRITICAL leakage in the Phase 13/14 cache** on horizons [1, 7, 30]: the target is copied into feature `purchase_price_pct_change_{h}d`. This explains Phase 14 MAE 0.0027 / 0.0036 / 0.0049.
- **F. Validation contamination:** training uses `date <= origin - h` (purged); evaluation targets all fall before the holdout. No contamination found in Phase 12/15.
- **G. Learnable synthetic generator:** the final models cannot even beat last-price by >1%, so they are not exploiting generator structure; the very low MAE reported by Phase 14 is leakage, not learned structure.

## Warnings

| Severity | ID | Finding | Evidence | Recommended action |
|---|---|---|---|---|
| CRITICAL | W01_TARGET_IN_FEATURE_CACHE | Phase 13/14 model-ready price cache contains the future target as feature 'purchase_price_pct_change_{h}d' for horizons [1, 7, 30]. | cache/model_ready/price_hd.parquet; src/ablation/cache.py lines 112-113 overwrite the backward-looking feature with the target; Phase 14 'after' MAE [0.00275, 0.00361, 0.00487] vs Phase 12 ~0.20; improvement exactly on leaking horizons only. | Fix cache builder to never overwrite features with target columns, rebuild cache, re-run Phase 13/14 (not done in this audit). |
| HIGH | W02_LEAKY_MODELS_IN_INFERENCE_PATH | src/ensemble/inference.py resolves models/optimised/price/lightgbm_pct_{h}d.joblib BEFORE the Phase 12 models; the 1d/7d/30d optimised models were trained with the target as a feature. | optimised artefacts present: ['lightgbm_pct_14d.joblib', 'lightgbm_pct_1d.joblib', 'lightgbm_pct_21d.joblib', 'lightgbm_pct_30d.joblib', 'lightgbm_pct_3d.joblib', 'lightgbm_pct_7d.joblib'] | Remove/quarantine leaky optimised price artefacts or point inference to Phase 12 models until Phase 14 is re-run on a fixed cache. |
| CRITICAL | W03_NO_DEMONSTRABLE_SKILL | The final price models do not beat the best trivial baseline by a meaningful margin at any horizon. | max improvement vs last price 31.06%; vs best simple baseline 0.53% (values <= 0 mean the model is worse). | Do not present price MAE as evidence of predictive skill; present it as a calibrated noise band around a stable base price. |
| HIGH | W04_PRICE_IS_STATIONARY_NOISE | Purchase prices are mean-reverting noise around a fixed per-series base, not a trending/regime process. | lag-1 autocorr of daily changes -0.50 (iid noise = -0.50); variance ratio 30d/1d = 1.01 (random walk = 30, iid = 2); price noise ~2.0% of level. | Document that supplier price 'direction' signals have little economic content in this data. |
| HIGH | W05_TARGET_PROVENANCE_SYNTHETIC | Supplier purchase price (target) originates from a retailer table with undocumented provenance (generate_retailer_dataset is referenced but not defined in the repository); profile is synthetic-like. | Prices change on 100% of days; historical == current on 100% of rows; base prices hard-coded in retailer_ingestion.get_default_product_catalog; one supplier per product. | Disclose as synthetic / unverified; replace with real procurement history before any real-world claim. |
| MEDIUM | W07_SYNTHETIC_COMMODITY_COLUMNS | Commodity file contains seeded random-walk columns presented as market data. | matches np.random.seed(123) generator: {'palm_oil_myr_per_tonne': True, 'steel_usd_index': True, 'feed_grain_usd': True} | Label palm_oil / steel / feed_grain as synthetic or replace with real MPOB / index data. |
| MEDIUM | W08_SPARSE_VALIDATION_ORIGINS | Walk-forward validation uses one forecast origin per fold (33 origins x 24 series = 792 rows/horizon); rows within an origin share common shocks. | 33 origins; effective sample size is smaller than 792. | Report confidence intervals; consider denser origins. |
| LOW | W09_UNCHANGED_RATE_NEAR_ZERO | Unlike real procurement feeds (sticky contract prices), series have ~0% unchanged days, so 'price-change event' counts overstate genuine repricing. | change rate 98.1%. | Interpret change counts accordingly. |

## Verdict

**PRICE TRAINING DATA SUFFICIENCY: FAIL**
