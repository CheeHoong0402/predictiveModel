# Phase 11 — Demand Forecasting Models: Summary Report

## Executive Summary
Phase 11 trained, evaluated, and ranked three advanced gradient-boosted decision tree model families:
1. **Light Gradient-Boosting Machine (LightGBM)**
2. **CatBoost**
3. **eXtreme Gradient Boosting (XGBoost)**

Evaluation strictly followed the chronological 33-fold walk-forward validation scheme from Phase 10.
Zero future leakage was allowed: all training rows for horizon H used historical dates $t \le T - H$,
and predictions at origin $T$ strictly used features known by end-of-day $T$.
The final holdout period (**2024-10-01 to 2024-12-31**) remained completely untouched.

## Best Candidate Demand Model per Horizon
```text
 horizon    model     wape  baseline_wape  wape_reduction_pct                      status
       1 lightgbm 0.261526       0.273020            4.209791 BEST_CANDIDATE_DEMAND_MODEL
       3  xgboost 0.186057       0.212634           12.499020 BEST_CANDIDATE_DEMAND_MODEL
       7 catboost 0.135701       0.150883           10.062170 BEST_CANDIDATE_DEMAND_MODEL
      14 catboost 0.096595       0.107119            9.824387 BEST_CANDIDATE_DEMAND_MODEL
      21 catboost 0.079405       0.083446            4.842060 BEST_CANDIDATE_DEMAND_MODEL
      30 catboost 0.067087       0.068121            1.518534 BEST_CANDIDATE_DEMAND_MODEL
```

## Model Comparison Against Phase 10 Champion Baselines
```text
 horizon    model     wape  baseline_wape  wape_reduction_pct        mae  baseline_mae       rmse      bias
       1 catboost 0.261595       0.273020            4.184564  13.769924     14.371301  22.810004 -0.046180
       1 lightgbm 0.261526       0.273020            4.209791  13.766299     14.371301  22.990275 -0.061892
       1  xgboost 0.264651       0.273020            3.065315  13.930775     14.371301  23.101595  0.278975
       3 catboost 0.189247       0.212634           10.998578  30.326416     34.074080  44.652498 -2.782088
       3 lightgbm 0.187743       0.212634           11.706043  30.085354     34.074080  44.047033 -2.359601
       3  xgboost 0.186057       0.212634           12.499020  29.815154     34.074080  43.772793 -2.227925
       7 catboost 0.135701       0.150883           10.062170  49.432529     54.962999  66.436551 -1.817416
       7 lightgbm 0.137124       0.150883            9.119188  49.950819     54.962999  67.123758 -0.796942
       7  xgboost 0.136766       0.150883            9.356080  49.820617     54.962999  66.732725 -1.744574
      14 catboost 0.096595       0.107119            9.824387  66.770874     74.045379  86.367552 -2.034323
      14 lightgbm 0.098872       0.107119            7.699491  68.344261     74.045379  88.747140  0.026470
      14  xgboost 0.098662       0.107119            7.895280  68.199289     74.045379  88.714368 -0.233980
      21 catboost 0.079405       0.083446            4.842060  80.947051     85.065997 103.909073  0.137072
      21 lightgbm 0.080778       0.083446            3.196571  82.346802     85.065997 105.774551  3.110943
      21  xgboost 0.080035       0.083446            4.087187  81.589191     85.065997 104.801795  2.562561
      30 catboost 0.067087       0.068121            1.518534  99.115515    100.643826 127.303521  0.499176
      30 lightgbm 0.068666       0.068121           -0.799185 101.448156    100.643826 131.597233  2.368352
      30  xgboost 0.068103       0.068121            0.026840 100.616813    100.643826 129.710216  0.508594
```

## Overall Model Metrics
```text
 horizon    model  count        mae       rmse     wape     mase      bias
       1 catboost    792  13.769924  22.810004 0.261595 0.705473 -0.046180
       1 lightgbm    792  13.766299  22.990275 0.261526 0.703403 -0.061892
       1  xgboost    792  13.930775  23.101595 0.264651 0.710781  0.278975
       3 catboost    792  30.326416  44.652498 0.189247 0.520683 -2.782088
       3 lightgbm    792  30.085354  44.047033 0.187743 0.515401 -2.359601
       3  xgboost    792  29.815154  43.772793 0.186057 0.512754 -2.227925
       7 catboost    792  49.432529  66.436551 0.135701 0.364298 -1.817416
       7 lightgbm    792  49.950819  67.123758 0.137124 0.367435 -0.796942
       7  xgboost    792  49.820617  66.732725 0.136766 0.367106 -1.744574
      14 catboost    792  66.770874  86.367552 0.096595 0.247668 -2.034323
      14 lightgbm    792  68.344261  88.747140 0.098872 0.253338  0.026470
      14  xgboost    792  68.199289  88.714368 0.098662 0.253213 -0.233980
      21 catboost    792  80.947051 103.909073 0.079405 0.200876  0.137072
      21 lightgbm    792  82.346802 105.774551 0.080778 0.204677  3.110943
      21  xgboost    792  81.589191 104.801795 0.080035 0.203254  2.562561
      30 catboost    792  99.115515 127.303521 0.067087 0.173084  0.499176
      30 lightgbm    792 101.448156 131.597233 0.068666 0.176790  2.368352
      30  xgboost    792 100.616813 129.710216 0.068103 0.175689  0.508594
```

## Segmented Performance Highlights
### By Store (7-day Horizon Best Model)
```text
   store_id    model  count       mae     wape     mase
STR_JHR_003 catboost    198 50.364875 0.138410 0.362274
STR_KUL_001 catboost    198 50.344973 0.138793 0.370433
STR_PNG_004 catboost    198 45.984287 0.127487 0.331902
STR_SGR_002 catboost    198 51.035979 0.138013 0.392581
STR_JHR_003 lightgbm    198 52.122458 0.143240 0.372702
STR_KUL_001 lightgbm    198 49.338953 0.136020 0.364892
STR_PNG_004 lightgbm    198 46.679552 0.129415 0.334168
STR_SGR_002 lightgbm    198 51.662315 0.139707 0.397979
STR_JHR_003  xgboost    198 50.571133 0.138977 0.364101
STR_KUL_001  xgboost    198 49.974718 0.137772 0.368944
STR_PNG_004  xgboost    198 46.472649 0.128841 0.333277
STR_SGR_002  xgboost    198 52.263968 0.141334 0.402103
```

### By Product Category (7-day Horizon Best Model)
```text
           category    model  count       mae     wape
Cooking Oils & Fats catboost    132 45.010646 0.136665
       Dairy & Eggs catboost    132 59.359889 0.150506
        Dry Staples catboost    132 48.046684 0.151543
   Fresh Vegetables catboost    264 35.424672 0.114153
     Poultry & Meat catboost    132 73.328607 0.139887
Cooking Oils & Fats lightgbm    132 44.764765 0.135919
       Dairy & Eggs lightgbm    132 59.847413 0.151742
        Dry Staples lightgbm    132 48.934609 0.154343
   Fresh Vegetables lightgbm    264 35.426094 0.114157
     Poultry & Meat lightgbm    132 75.305943 0.143660
Cooking Oils & Fats  xgboost    132 45.624718 0.138530
       Dairy & Eggs  xgboost    132 59.866227 0.151790
        Dry Staples  xgboost    132 48.443459 0.152794
   Fresh Vegetables  xgboost    264 35.719608 0.115103
     Poultry & Meat  xgboost    132 73.550081 0.140310
```

### By Demand Volume Group
```text
  demand_type    model  count       mae     wape
  high_volume catboost    132 73.328607 0.139887
medium_volume catboost    660 44.653313 0.134380
  high_volume lightgbm    132 75.305943 0.143660
medium_volume lightgbm    660 44.879795 0.135062
  high_volume  xgboost    132 73.550081 0.140310
medium_volume  xgboost    660 45.074724 0.135648
```

### Promotion vs. Non-Promotion (7-day Horizon)
```text
promotion_period    model  count       mae     wape
    no_promotion catboost    572 49.886817 0.136337
       promotion catboost    220 48.251380 0.134021
    no_promotion lightgbm    572 50.136546 0.137019
       promotion lightgbm    220 49.467929 0.137400
    no_promotion  xgboost    572 49.815641 0.136142
       promotion  xgboost    220 49.833553 0.138415
```

### Festival vs. Normal Period (7-day Horizon)
```text
festival_period    model  count       mae     wape
       festival catboost     48 50.106503 0.135417
         normal catboost    744 49.389046 0.135720
       festival lightgbm     48 51.435614 0.139009
         normal lightgbm    744 49.855026 0.137000
       festival  xgboost     48 52.478131 0.141826
         normal  xgboost    744 49.649164 0.136434
```

## Model Training Runtime
```text
   model  total_walk_forward_runtime_seconds  folds_evaluated
lightgbm                               99.76               33
catboost                              325.93               33
 xgboost                              190.16               33
```

## Feature Counts & Explainability Preparation
```text
  task  approved_features_count                                     manifest_source  horizons_supported
demand                      275 reports/leakage/demand_features_temporally_safe.csv                   6
```

All model artifacts have been serialized under `models/demand/<family>/` with accompanying feature metadata
ready for SHAP analysis in subsequent phases. Reusable inference is available via `predict_demand()`.