# Phase 10 — Baseline Forecast Models

## Scope and temporal design
33 expanding/rolling chronological folds; one fixed forecast origin per fold and six horizons.
Configured history mode: expanding. Validation blocks contain 30 calendar days and do not overlap.
Training ends at the close of day T (Asia/Kuala_Lumpur); forecast outcomes begin T+1.
All rule parameters are fixed before evaluation. No advanced ML models, ensembles or hyperparameter search.
Each fold is frozen: none of its validation observations enters that fold's forecast. Prior completed validation periods may enter later folds' expanding history.
The last partial development block is omitted if it cannot fit all 30 days before holdout; this is disclosed in the fold ledger.
Unscored development tail: 2024-09-17 through 2024-09-30.

Final holdout: **2024-10-01 through 2024-12-31**, reserved for one later final evaluation.
No holdout forecast, score, scenario label or champion choice is produced. Read filters exclude holdout source/features and target origins with potentially holdout-spanning labels.
No random partitions are used; forecasts are identical after sorting shuffled input.

## All fold boundaries
```text
 validation_fold training_start training_end forecast_origin validation_start validation_end  number_products  number_stores  series_count  valid_demand_target_rows  valid_price_target_rows
               1     2021-01-01   2021-12-31      2021-12-31       2022-01-01     2022-01-30              6.0            4.0          24.0                     144.0                    144.0
               2     2021-01-01   2022-01-30      2022-01-30       2022-01-31     2022-03-01              6.0            4.0          24.0                     144.0                    144.0
               3     2021-01-01   2022-03-01      2022-03-01       2022-03-02     2022-03-31              6.0            4.0          24.0                     144.0                    144.0
               4     2021-01-01   2022-03-31      2022-03-31       2022-04-01     2022-04-30              6.0            4.0          24.0                     144.0                    144.0
               5     2021-01-01   2022-04-30      2022-04-30       2022-05-01     2022-05-30              6.0            4.0          24.0                     144.0                    144.0
               6     2021-01-01   2022-05-30      2022-05-30       2022-05-31     2022-06-29              6.0            4.0          24.0                     144.0                    144.0
               7     2021-01-01   2022-06-29      2022-06-29       2022-06-30     2022-07-29              6.0            4.0          24.0                     144.0                    144.0
               8     2021-01-01   2022-07-29      2022-07-29       2022-07-30     2022-08-28              6.0            4.0          24.0                     144.0                    144.0
               9     2021-01-01   2022-08-28      2022-08-28       2022-08-29     2022-09-27              6.0            4.0          24.0                     144.0                    144.0
              10     2021-01-01   2022-09-27      2022-09-27       2022-09-28     2022-10-27              6.0            4.0          24.0                     144.0                    144.0
              11     2021-01-01   2022-10-27      2022-10-27       2022-10-28     2022-11-26              6.0            4.0          24.0                     144.0                    144.0
              12     2021-01-01   2022-11-26      2022-11-26       2022-11-27     2022-12-26              6.0            4.0          24.0                     144.0                    144.0
              13     2021-01-01   2022-12-26      2022-12-26       2022-12-27     2023-01-25              6.0            4.0          24.0                     144.0                    144.0
              14     2021-01-01   2023-01-25      2023-01-25       2023-01-26     2023-02-24              6.0            4.0          24.0                     144.0                    144.0
              15     2021-01-01   2023-02-24      2023-02-24       2023-02-25     2023-03-26              6.0            4.0          24.0                     144.0                    144.0
              16     2021-01-01   2023-03-26      2023-03-26       2023-03-27     2023-04-25              6.0            4.0          24.0                     144.0                    144.0
              17     2021-01-01   2023-04-25      2023-04-25       2023-04-26     2023-05-25              6.0            4.0          24.0                     144.0                    144.0
              18     2021-01-01   2023-05-25      2023-05-25       2023-05-26     2023-06-24              6.0            4.0          24.0                     144.0                    144.0
              19     2021-01-01   2023-06-24      2023-06-24       2023-06-25     2023-07-24              6.0            4.0          24.0                     144.0                    144.0
              20     2021-01-01   2023-07-24      2023-07-24       2023-07-25     2023-08-23              6.0            4.0          24.0                     144.0                    144.0
              21     2021-01-01   2023-08-23      2023-08-23       2023-08-24     2023-09-22              6.0            4.0          24.0                     144.0                    144.0
              22     2021-01-01   2023-09-22      2023-09-22       2023-09-23     2023-10-22              6.0            4.0          24.0                     144.0                    144.0
              23     2021-01-01   2023-10-22      2023-10-22       2023-10-23     2023-11-21              6.0            4.0          24.0                     144.0                    144.0
              24     2021-01-01   2023-11-21      2023-11-21       2023-11-22     2023-12-21              6.0            4.0          24.0                     144.0                    144.0
              25     2021-01-01   2023-12-21      2023-12-21       2023-12-22     2024-01-20              6.0            4.0          24.0                     144.0                    144.0
              26     2021-01-01   2024-01-20      2024-01-20       2024-01-21     2024-02-19              6.0            4.0          24.0                     144.0                    144.0
              27     2021-01-01   2024-02-19      2024-02-19       2024-02-20     2024-03-20              6.0            4.0          24.0                     144.0                    144.0
              28     2021-01-01   2024-03-20      2024-03-20       2024-03-21     2024-04-19              6.0            4.0          24.0                     144.0                    144.0
              29     2021-01-01   2024-04-19      2024-04-19       2024-04-20     2024-05-19              6.0            4.0          24.0                     144.0                    144.0
              30     2021-01-01   2024-05-19      2024-05-19       2024-05-20     2024-06-18              6.0            4.0          24.0                     144.0                    144.0
              31     2021-01-01   2024-06-18      2024-06-18       2024-06-19     2024-07-18              6.0            4.0          24.0                     144.0                    144.0
              32     2021-01-01   2024-07-18      2024-07-18       2024-07-19     2024-08-17              6.0            4.0          24.0                     144.0                    144.0
              33     2021-01-01   2024-08-17      2024-08-17       2024-08-18     2024-09-16              6.0            4.0          24.0                     144.0                    144.0
```

## Method definitions
- last_value: latest valid daily demand, bounded by configured observation age, multiplied by H.
- seasonal_naive_7: repeat the last historical weekly cycle recursively; horizons beyond seven days never substitute realised future demand.
- same_weekday_median_4w: median of four most recent valid occurrences of each future weekday.
- moving_average_7 / moving_average_28: mean of last 7 / 28 valid observations, through T, times H. Gaps may extend calendar lookback; no missing days are converted to zero.
- median_28: median within the trailing 28 calendar days, times H, subject to minimum history.
- weekday_profile: mean by weekday over the last 84 calendar days.
- annual_naive_364: historical observations exactly 364 calendar days before each forecast day. Missing references make the annual baseline unavailable.
- recent_trend: ordinary least squares on recent calendar-day offsets; daily extrapolations clipped at zero and a training-history-based upper cap.
- croston: separate exponential smoothing of nonzero demand size and interarrival time, fixed alpha 0.1; used only for training-history intermittent series. Missing calendar days invalidate the estimator, not zero-fill it.
- empirical_quantiles: quantiles of completed historical H-day cumulative sums with matching starting weekday. Overlapping historical windows provide descriptive empirical uncertainty, not independent samples for significance tests.

Price rules preserve the origin supplier. Daily recorded supplier prices are valid observations even when purchase quantity is zero, following Phase 9.
Price last value, 7/30-day means, 30-day median and 30-day trend use observations through T; bounded origin-price age and FLAT tolerance are inherited from Phase 9.
Price history is not forward-filled. A missing or stale same-supplier price yields an explicit missing forecast.
Price empirical quantiles multiply current price by empirical historical exact-H-day price ratios whose endpoints are both <=T.
Trend prices are clipped to configured positive multiples of the current price.
Direction classifiers: always FLAT, latest historical movement, and implied movement from the 30-day trend. Classification precision/recall are reported for DOWN, FLAT and UP.

## Fallbacks and population coverage
Demand hierarchy: product_store, product_state, product_national, category_store, category_national, global; each step uses only eligible historical observations and records its level.
Category/global fallbacks are coarse pooled daily-unit heuristics; they are explicitly labelled rather than disguised as product-specific estimates.
Annual, empirical-quantile and intermittent methods remain unavailable when their method-specific requirements fail; no false seasonal or uncertainty estimate is fabricated.
Price fallback is strictly the last fresh observation from the same supplier, otherwise missing. No cross-product/supplier price pooling.
Known inactive products/stores receive no forecast. Activity assumptions from Phase 9 still apply where explicit status is absent.
Series first appearing after the origin are reported in excluded_new_series.csv; no future identities are supplied to predictors.
All missing predictions and target exclusions are enumerated in excluded_predictions.csv. Raw and Phase 9 data are never rewritten.

Phase 9's full-history intermittent report is inspected for audit context only. Eligibility and volume segments are recalculated on each training window to avoid future-population leakage.
Volume thresholds are fixed in config, not computed from validation outcomes. Segmentation tables retain low/medium/high volume separately.
Observed series-fold populations (zero means the dataset cannot validate that segment):
```text
demand_type
high_volume             132
insufficient_history      0
intermittent              0
low_volume                0
medium_volume           660
```

## Metrics and ranking
All percentage-like metrics in CSV are decimal ratios; multiply by 100 for display.
Demand WAPE = sum absolute errors / sum absolute actual demand. Zero denominator => missing, not artificial perfect accuracy.
MAE/RMSE/bias are in units for demand and MYR for price. Bias is prediction minus actual.
Demand MASE scale = H × mean(|daily demand(t) − daily demand(t−7)|) within that fold's training history; this is the documented horizon-scaled daily seasonal denominator. Zero/missing scales are excluded with mase_count retained.
Price MAPE uses strictly positive actual prices; sMAPE uses 2|error|/(|actual|+|prediction|).
Pinball losses at 0.1/0.5/0.9 and empirical 80% interval coverage/width are reported where uncertainty predictions exist.
Balanced accuracy averages recall for actual classes present; an absent class has undefined recall, while zero predicted positives give precision zero.
Champions require complete prediction coverage of the same valid target population and at least 5 folds.
Rank first by mean fold WAPE (demand) or MAE (price), then worst-fold error, then pooled error. Standard deviation, median and best/worst folds are retained.
This prevents incomplete-coverage or one-fold methods from winning by selective omission. A missing champion is reported explicitly rather than invented.

## Champion demand results

```text
 horizon            baseline  count        mae       rmse     wape     mase       bias
       1 empirical_quantiles    792  14.371301  27.122701 0.273020 0.738126  -9.735467
       3     weekday_profile    792  34.074080  51.996494 0.212634 0.584151 -12.871128
       7 empirical_quantiles    792  54.962999  80.609700 0.150883 0.404790 -34.083251
      14 empirical_quantiles    792  74.045379  97.991960 0.107119 0.275168 -16.551944
      21 empirical_quantiles    792  85.065997 108.898434 0.083446 0.211077   1.726035
      30 empirical_quantiles    792 100.643826 130.655282 0.068121 0.175509 -11.979495
```
## Champion price results

```text
 horizon                baseline  count      mae     rmse     mape      bias  direction_accuracy
       1 price_moving_average_30    792 0.203080 0.313674 0.015930  0.007766            0.616162
       3 price_moving_average_30    792 0.207279 0.320198 0.015972 -0.001261            0.625000
       7 price_moving_average_30    792 0.209460 0.325826 0.016034  0.005380            0.612374
      14         price_median_30    792 0.212361 0.318040 0.016928  0.018270            0.602273
      21 price_moving_average_30    792 0.210612 0.320388 0.016220  0.004837            0.630051
      30 price_moving_average_30    792 0.211080 0.316036 0.016480 -0.009746            0.611111
```

## Every demand baseline, WAPE percent by horizon
```text
horizon                     1       3       7       14      21      30
baseline                                                              
annual_naive_364        40.377  26.959  18.602  12.861  10.484   9.088
croston                    NaN     NaN     NaN     NaN     NaN     NaN
empirical_quantiles     27.302  21.517  15.088  10.712   8.345   6.812
last_value              43.656  40.302  35.565  35.114  34.359  33.885
median_28               32.098  26.528  20.656  16.233  14.375  15.011
moving_average_28       32.226  23.221  15.896  12.488  10.384   9.306
moving_average_7        35.850  27.377  21.869  20.024  18.101  16.159
recent_trend            37.082  30.541  27.055  28.981  30.239  32.557
same_weekday_median_4w  30.974  23.478  17.747  13.420  11.276  11.128
seasonal_naive_7        38.700  28.516  21.869  20.024  18.101  16.025
weekday_profile         30.332  21.263  15.319  11.624   9.222   7.537
```
Missing scores indicate unavailable/not-applicable methods, never perfect performance.

## Every purchase-price baseline, MAE in MYR by horizon
```text
horizon                        1       3       7       14      21      30
baseline                                                                 
price_empirical_quantiles  0.2923  0.2840  0.2909  0.2884  0.3051  0.2978
price_last_value           0.2913  0.2834  0.2895  0.2883  0.3044  0.2977
price_median_30            0.2058  0.2094  0.2107  0.2124  0.2114  0.2116
price_moving_average_30    0.2031  0.2073  0.2095  0.2124  0.2106  0.2111
price_moving_average_7     0.2142  0.2176  0.2178  0.2233  0.2248  0.2229
price_recent_trend         0.2146  0.2199  0.2292  0.2408  0.2640  0.2967
```

## Direction classifier results
```text
 horizon                baseline  target_count  excluded_predictions  count  accuracy  balanced_accuracy  macro_precision  macro_recall
       1   direction_always_flat           792                     0    792  0.133838           0.333333         0.044613      0.333333
       1 direction_latest_change           792                     0    792  0.247475           0.229634         0.227267      0.229634
       1      direction_trend_30           792                     0    792  0.604798           0.545928         0.551820      0.545928
       3   direction_always_flat           792                     0    792  0.133838           0.333333         0.044613      0.333333
       3 direction_latest_change           792                     0    792  0.233586           0.225369         0.222081      0.225369
       3      direction_trend_30           792                     0    792  0.599747           0.537381         0.545198      0.537381
       7   direction_always_flat           792                     0    792  0.141414           0.333333         0.047138      0.333333
       7 direction_latest_change           792                     0    792  0.222222           0.196404         0.196218      0.196404
       7      direction_trend_30           792                     0    792  0.573232           0.515681         0.521720      0.515681
      14   direction_always_flat           792                     0    792  0.133838           0.333333         0.044613      0.333333
      14 direction_latest_change           792                     0    792  0.219697           0.214603         0.211232      0.214603
      14      direction_trend_30           792                     0    792  0.569444           0.502949         0.506736      0.502949
      21   direction_always_flat           792                     0    792  0.098485           0.333333         0.032828      0.333333
      21 direction_latest_change           792                     0    792  0.239899           0.220663         0.211913      0.220663
      21      direction_trend_30           792                     0    792  0.549242           0.477113         0.482593      0.477113
      30   direction_always_flat           792                     0    792  0.127525           0.333333         0.042508      0.333333
      30 direction_latest_change           792                     0    792  0.252525           0.234951         0.231211      0.234951
      30      direction_trend_30           792                     0    792  0.510101           0.443867         0.446378      0.443867
```

## Scenario interpretation
Promotion and festival labels describe the future evaluation horizon, not prediction inputs. Pre-festival tags use approved known-calendar features at origin.
Price-shock, high-volatility, policy-change and commodity-shock tags are evaluation-only diagnostics. Empty scenarios are explicitly reported as NO_OBSERVATIONS.
Smaller percentage demand errors at long horizons can result from cumulative aggregation smoothing daily variation; larger absolute-unit errors should be considered alongside WAPE. This is not evidence of leakage by itself.
Store/category, volume/intermittency, promotion/festival and price-shock tables give all methods at every horizon; do not hide difficult segments behind a pooled score.

### Demand 7-day champion by store_id

```text
   store_id  count       mae     wape
STR_JHR_003    198 55.178914 0.151640
STR_KUL_001    198 54.012197 0.148903
STR_PNG_004    198 50.729015 0.140642
STR_SGR_002    198 59.931869 0.162070
```
### Demand 7-day champion by category

```text
           category  count       mae     wape
Cooking Oils & Fats    132 48.182159 0.146295
       Dairy & Eggs    132 62.071250 0.157381
        Dry Staples    132 49.891439 0.157361
   Fresh Vegetables    264 41.261515 0.132961
     Poultry & Meat    132 87.110114 0.166178
```
### Demand 7-day champion by promotion_period

```text
promotion_period  count       mae     wape
    no_promotion    572 55.072439 0.150509
       promotion    220 54.678455 0.151872
```
### Demand 7-day champion by festival_period

```text
festival_period  count       mae     wape
       festival     48 60.212083 0.162728
         normal    744 54.624348 0.150106
   pre_festival      0       NaN      NaN
```
### Price 30-day champion by store_id

```text
   store_id  count      mae
STR_JHR_003    198 0.195229
STR_KUL_001    198 0.222886
STR_PNG_004    198 0.215192
STR_SGR_002    198 0.211015
```
### Price 30-day champion by category

```text
           category  count      mae
Cooking Oils & Fats    132 0.456722
       Dairy & Eggs    132 0.203301
        Dry Staples    132 0.346174
   Fresh Vegetables    264 0.072367
     Poultry & Meat    132 0.115551
```
### Price 30-day champion by promotion_period

```text
promotion_period  count      mae
    no_promotion    392 0.211081
       promotion    400 0.211080
```
### Price 30-day champion by festival_period

```text
festival_period  count      mae
       festival    168 0.194659
         normal    552 0.215163
   pre_festival     72 0.218097
```
### Price shock diagnostics — 30-day champion

```text
       scenario  scenario_active  count      mae
    price_shock            False    792 0.211080
    price_shock             True      0      NaN
high_volatility            False    110 0.237218
high_volatility             True    682 0.206865
  policy_change            False    672 0.211597
  policy_change             True    120 0.208189
commodity_shock            False    776 0.211409
commodity_shock             True     16 0.195146
```

For business interpretation, seven-day demand MAE is the typical absolute miss in units summed across the week, while WAPE compares total absolute error to total demand. Thirty-day purchase-price MAE is the average ringgit error per procurement unit. These development results are not final-holdout performance.

## Later ML success criteria
Compare on identical horizon-specific rows and folds against the frozen development champion. Require lower demand WAPE and price MAE, check bias and direction accuracy, and demonstrate stable gains across multiple folds.
Quantify paired fold improvements and uncertainty with temporal/block-aware inference; overlapping horizons/series are not independent observations. No arbitrary improvement percentage or unsupported statistical significance is asserted now.
The untouched final holdout is for one later confirmation after all model-selection decisions.

## Artifacts and reproduction
Row-level predictions: data/predictions/baseline_demand_predictions.parquet, baseline_price_predictions.parquet and baseline_direction_predictions.parquet.
Every row records origin, keys, horizon, method, prediction/actual, error, fold, historical information bound, fallback and exclusion metadata.
Tables in this directory include full model comparisons, fold statistics, champions, conditional metrics and direction confusion matrices. Figures are PNG and SVG under reports/figures/baselines.
Interfaces accept standardised retailer datasets and configurable dates/windows; there are no demonstration store/product identifiers in forecasting logic.
The Phase 9 historical/future name collision remains isolated: prediction functions reject target columns; only the approved canonical feature artifact supplies known-calendar audit context.

Run: `python -m src.baselines.baseline_pipeline`.
Additional plotting dependency: `python -m pip install -r requirements-baselines.txt`.
Tests: `python -m pytest tests/test_baseline_models.py tests/test_temporal_integrity.py tests/test_target_construction.py tests/test_feature_engineering.py tests/test_demand_correction.py tests/test_master_daily_table.py`.

Method references: [Croston, described by Hyndman & Athanasopoulos](https://otexts.com/fpp3/counts.html); [forecast accuracy and scaled errors](https://otexts.com/fpp3/accuracy.html).
