"""Human-readable methodology and measured development results."""
import pandas as pd


def write_summary(tables, config, directory):
    champions = tables['champion_baselines.csv']
    demand = tables['demand_baseline_metrics.csv']
    price = tables['price_baseline_metrics.csv']
    folds = tables['walk_forward_folds.csv']
    sections = []
    for task, data in [('demand', demand), ('price', price)]:
        chosen = champions[champions.task == task][['horizon', 'baseline']].merge(data, on=['horizon', 'baseline'])
        columns = ['horizon', 'baseline', 'count', 'mae', 'rmse', 'wape', 'mase', 'bias'] if task == 'demand' else ['horizon', 'baseline', 'count', 'mae', 'rmse', 'mape', 'bias', 'direction_accuracy']
        sections.append(f'## Champion {task} results\n\n```text\n{chosen[columns].round(6).to_string(index=False)}\n```')
    snapshot = config['evaluation']
    demand_comparison = demand.pivot(index='baseline', columns='horizon', values='wape') * 100
    price_comparison = price.pivot(index='baseline', columns='horizon', values='mae')
    highlights = []
    for task, horizon in [('demand', 7), ('price', 30)]:
        name = champions[(champions.task == task) & (champions.horizon == horizon)].baseline.iloc[0]
        measure = 'wape' if task == 'demand' else 'mae'
        for filename, segment in [('store_performance.csv', 'store_id'), ('category_performance.csv', 'category'),
                                  ('promotion_period_performance.csv', 'promotion_period'), ('festival_period_performance.csv', 'festival_period')]:
            data = tables[filename]
            selected = data[(data.task == task) & (data.horizon == horizon) & (data.baseline == name)]
            highlights.append(f'### {task.title()} {horizon}-day champion by {segment}\n\n```text\n' + selected[[segment, 'count', 'mae', measure]].loc[:, lambda x: ~x.columns.duplicated()].round(6).to_string(index=False) + '\n```')
    shock = tables['price_shock_performance.csv']
    price_name = champions[(champions.task == 'price') & (champions.horizon == 30)].baseline.iloc[0]
    shocks = shock[(shock.horizon == 30) & (shock.baseline == price_name)]
    highlights.append('### Price shock diagnostics — 30-day champion\n\n```text\n' + shocks[['scenario', 'scenario_active', 'count', 'mae']].round(6).to_string(index=False) + '\n```')
    volume_counts = tables['population_coverage.csv'].groupby('demand_type').series_count.sum()
    text = f"""# Phase 10 — Baseline Forecast Models

## Scope and temporal design
{len(folds)} expanding/rolling chronological folds; one fixed forecast origin per fold and six horizons.
Configured history mode: {snapshot['history_mode']}. Validation blocks contain {snapshot['validation_window_days']} calendar days and do not overlap.
Training ends at the close of day T (Asia/Kuala_Lumpur); forecast outcomes begin T+1.
All rule parameters are fixed before evaluation. No advanced ML models, ensembles or hyperparameter search.
Each fold is frozen: none of its validation observations enters that fold's forecast. Prior completed validation periods may enter later folds' expanding history.
The last partial development block is omitted if it cannot fit all 30 days before holdout; this is disclosed in the fold ledger.
Unscored development tail: {(folds.validation_end.max() + pd.Timedelta(days=1)).date()} through {(pd.Timestamp(snapshot['holdout_start']) - pd.Timedelta(days=1)).date()}.

Final holdout: **{snapshot['holdout_start']} through {snapshot['holdout_end']}**, reserved for one later final evaluation.
No holdout forecast, score, scenario label or champion choice is produced. Read filters exclude holdout source/features and target origins with potentially holdout-spanning labels.
No random partitions are used; forecasts are identical after sorting shuffled input.

## All fold boundaries
```text
{folds.to_string(index=False)}
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
- croston: separate exponential smoothing of nonzero demand size and interarrival time, fixed alpha {config['intermittent']['croston_alpha']}; used only for training-history intermittent series. Missing calendar days invalidate the estimator, not zero-fill it.
- empirical_quantiles: quantiles of completed historical H-day cumulative sums with matching starting weekday. Overlapping historical windows provide descriptive empirical uncertainty, not independent samples for significance tests.

Price rules preserve the origin supplier. Daily recorded supplier prices are valid observations even when purchase quantity is zero, following Phase 9.
Price last value, 7/30-day means, 30-day median and 30-day trend use observations through T; bounded origin-price age and FLAT tolerance are inherited from Phase 9.
Price history is not forward-filled. A missing or stale same-supplier price yields an explicit missing forecast.
Price empirical quantiles multiply current price by empirical historical exact-H-day price ratios whose endpoints are both <=T.
Trend prices are clipped to configured positive multiples of the current price.
Direction classifiers: always FLAT, latest historical movement, and implied movement from the 30-day trend. Classification precision/recall are reported for DOWN, FLAT and UP.

## Fallbacks and population coverage
Demand hierarchy: {', '.join(config['fallback']['demand_levels'])}; each step uses only eligible historical observations and records its level.
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
{volume_counts.to_string()}
```

## Metrics and ranking
All percentage-like metrics in CSV are decimal ratios; multiply by 100 for display.
Demand WAPE = sum absolute errors / sum absolute actual demand. Zero denominator => missing, not artificial perfect accuracy.
MAE/RMSE/bias are in units for demand and MYR for price. Bias is prediction minus actual.
Demand MASE scale = H × mean(|daily demand(t) − daily demand(t−7)|) within that fold's training history; this is the documented horizon-scaled daily seasonal denominator. Zero/missing scales are excluded with mase_count retained.
Price MAPE uses strictly positive actual prices; sMAPE uses 2|error|/(|actual|+|prediction|).
Pinball losses at 0.1/0.5/0.9 and empirical 80% interval coverage/width are reported where uncertainty predictions exist.
Balanced accuracy averages recall for actual classes present; an absent class has undefined recall, while zero predicted positives give precision zero.
Champions require complete prediction coverage of the same valid target population and at least {snapshot['minimum_champion_folds']} folds.
Rank first by mean fold WAPE (demand) or MAE (price), then worst-fold error, then pooled error. Standard deviation, median and best/worst folds are retained.
This prevents incomplete-coverage or one-fold methods from winning by selective omission. A missing champion is reported explicitly rather than invented.

{chr(10).join(sections)}

## Every demand baseline, WAPE percent by horizon
```text
{demand_comparison.round(3).to_string()}
```
Missing scores indicate unavailable/not-applicable methods, never perfect performance.

## Every purchase-price baseline, MAE in MYR by horizon
```text
{price_comparison.round(4).to_string()}
```

## Direction classifier results
```text
{tables['price_direction_metrics.csv'].round(6).to_string(index=False)}
```

## Scenario interpretation
Promotion and festival labels describe the future evaluation horizon, not prediction inputs. Pre-festival tags use approved known-calendar features at origin.
Price-shock, high-volatility, policy-change and commodity-shock tags are evaluation-only diagnostics. Empty scenarios are explicitly reported as NO_OBSERVATIONS.
Smaller percentage demand errors at long horizons can result from cumulative aggregation smoothing daily variation; larger absolute-unit errors should be considered alongside WAPE. This is not evidence of leakage by itself.
Store/category, volume/intermittency, promotion/festival and price-shock tables give all methods at every horizon; do not hide difficult segments behind a pooled score.

{chr(10).join(highlights)}

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
"""
    (directory / 'baseline_summary.md').write_text(text, encoding='utf-8')
