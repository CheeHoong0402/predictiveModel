# Phase 9 — Forecast Target Construction

Generated 35,064 unique date/store/product rows. No model training, random splits, optimisation or target scaling.

## Forecast and observation contract
Origin: end_of_day_T, 23:59:59 Asia/Kuala_Lumpur, inherited from config/feature_engineering.yaml.
Future demand includes calendar days T+1 through T+H and never T. Cumulative targets require 100% valid coverage.
Auxiliary demand_day_H is demand on exactly T+H, for later direct-vs-cumulative evaluation; inventory targets are cumulative.
Demand source: estimated_true_demand_training_safe; null-only fallback: observed_units_sold (0 source rows).
If both are missing, the target is missing. Phase 7's final zero fallback is deliberately not used to fabricate target observations.
Retrospective demand is never read by target calculations. Demand retains units, prices MYR, changes decimal ratios.

## Price policy and data evidence
Input price observations: 35,064/35,064; observed row gap median/max: 1.0/1.0 calendar days.
Daily supplier price records include non-purchase days and are not filtered by purchase_quantity.
Default tolerance: 0 days (exact-day labels supported by complete daily source).
Configurable fallback selects the first observed price in [T+H, T+H+tolerance], strictly after T and within the origin supplier's series.
Effective carry is False; if enabled, explicit price_effective_from/until metadata and an observation after T are required; maximum age 7 days.
There is no ordinary target forward-fill. Origin baseline is the last valid same-supplier observation at/before T, at most 7 days old.
FLAT includes changes within +/-0.50%; UP/DOWN are outside that band (floating-point comparison epsilon 1e-12).

## Lifecycle and closure limits
Available explicit activity columns: none.
Earlier exports omit the skeleton activity flags, and source store/product masters have no dated closure/discontinuation records.
For this dataset, observed operational rows are assumed active, as in Phase 5; this is an explicit assumption, not verified store-opening evidence.
Missing dates stay missing; no lifecycle is inferred from future disappearance. Supplied activity, availability, closure and discontinuation flags/dates are enforced.
Null supplied flags mean unknown activity. Genuine observed zeros remain valid; unavailable products, closure, discontinuation, and missing observations are invalidated distinctly.
Stockout-corrected demand remains eligible; zero inventory alone does not mean discontinued.

## Coverage by horizon
```
 horizon  total_rows  demand_valid  demand_missing  mean_demand_coverage  future_days_required  future_valid_days_total  price_valid  price_missing  price_change_valid
       1       35064         35040              24              0.999316                     1                    35040        35040             24               35040
       3       35064         34992              72              0.998631                     3                   105048        34992             72               34992
       7       35064         34896             168              0.997262                     7                   244776        34896            168               34896
      14       35064         34728             336              0.994867                    14                   488376        34728            336               34728
      21       35064         34560             504              0.992471                    21                   730800        34560            504               34560
      30       35064         34344             720              0.989391                    30                  1040760        34344            720               34344
```

## Demand distributions (overall)
```
      target_name  horizon   scope store_id category  count  missing_count        mean   median  standard_deviation  minimum  maximum    p10       p25       p75      p90       p95       p99
 demand_target_1d        1 overall      ALL      ALL  35040             24   49.058875   41.500           27.883445    12.00   360.50   27.0   33.0000   56.0000   77.000   97.5110  175.0000
 demand_target_3d        3 overall      ALL      ALL  34992             72  147.169157  133.000           55.976800    58.00   700.00   93.5  108.5000  170.5000  217.010  252.0200  348.5000
 demand_target_7d        7 overall      ALL      ALL  34896            168  343.382458  318.080           94.899846   184.00  1032.45  248.0  273.0000  393.0100  468.500  526.9925  648.5210
demand_target_14d       14 overall      ALL      ALL  34728            336  686.740794  643.000          164.788623   428.00  1753.45  521.0  564.8600  767.5200  929.000 1017.7805 1189.0146
demand_target_21d       21 overall      ALL      ALL  34560            504 1030.198844  960.510          230.533996   673.00  2252.47  805.0  860.0000 1128.5000 1397.011 1506.5000 1713.0979
demand_target_30d       30 overall      ALL      ALL  34344            720 1471.919967 1368.185          316.301667   978.29  2894.86 1172.0 1240.3275 1594.5125 2001.990 2134.4970 2375.6653
```
Full distribution files include store, category and store/category breakdowns; price levels and percentage changes have the same quantiles.

## Separation and temporal safety
Canonical feature table and Phase 8 manifests are byte-for-byte unchanged (upstream_integrity.csv).
All-row temporal assertions pass, with feature information bounded by T using approved Phase 8 metadata.
This bound inherits the Phase 8 audit; per-feature physical source timestamps are not available and are not invented.
All chosen price observation dates are > T; all valid demand windows are T+1..T+H.

Three requested target names already occur in approved historical features: purchase_price_pct_change_1d, _7d and _30d.
It is impossible to make the literal name sets disjoint while preserving both the requested target names and authoritative manifests.
The historical features measure P(T)/P(T-H)-1; targets measure P(T+H)/P(T)-1. They remain separate, qualified dataset identities.
feature_target_name_collisions.csv documents this compatibility exception. assemble_training_view selects X only from canonical model_features and uses x__/y__ aliases.
Never select features from a bare merged target/feature frame. Target coverage, validity, observation dates, outlier flags and masks are audit metadata, not predictors.

## Review and proof artifacts
Outlier flags: 1,690 target observations; retained without clipping/removal.
Demand flags exceed within-series Q3 + 3.0 IQR; price flags exceed absolute 50% change.
These are full-history descriptive audit thresholds, never model inputs or fitted preprocessing.
Intermittent/sparse series: 0/0.
manual_target_checks.csv records exact contributing dates/values and independently reconstructed totals, plus selected price dates, baseline, changes and labels.
Seed 9 samples observations for audit only, not train/test partitions.
Horizon-specific key indexes and masks are under reports/targets; price-change eligibility is separate from price-level eligibility.

## Reproduction
From the project folder with its existing environment: `python -m src.targets.target_pipeline`.
Tests: `python -m pytest tests/test_target_construction.py tests/test_temporal_integrity.py tests/test_feature_engineering.py tests/test_demand_correction.py tests/test_master_daily_table.py`.
