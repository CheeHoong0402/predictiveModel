"""Phase 10 reproducible baseline evaluation; no advanced model fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from src.utils.paths import PROJECT_ROOT
from src.targets.target_validation import sha256, validate_manifests, validate_targets
from .fallback_baselines import load_config, prepare_history, historical_context, demand_fallback, KEYS
from .walk_forward import build_folds
from .demand_baselines import METHODS as DEMAND_METHODS, forecast_demand
from .price_baselines import METHODS as PRICE_METHODS, forecast_price, direction_forecasts, direction
from .baseline_evaluator import metric_table, rankings_and_champions, direction_metrics


def read_before(path, boundary, columns=None):
    """Predicate-filter holdout values out before any modelling/scenario work."""
    data = ds.dataset(str(path), format='parquet')
    date_type = data.schema.field('date').type
    limit = str(pd.Timestamp(boundary).date()) if pa.types.is_string(date_type) or pa.types.is_large_string(date_type) else pd.Timestamp(boundary).to_pydatetime()
    return data.to_table(columns=columns, filter=ds.field('date') < limit).to_pandas()


def evaluation_scenarios(raw, approved, origin, store, product, horizon, config):
    """Ex-post evaluation tags ONLY. Never passed to forecasting functions."""
    group = raw[(raw.store_id == store) & (raw.product_id == product)].sort_values('date')
    future = group[(group.date > origin) & (group.date <= origin + pd.Timedelta(days=horizon))]
    c = config['scenarios']
    promotion = 'promotion' if future.promotion_active.eq(1).any() else 'no_promotion'
    if future.festival_flag.eq(1).any():
        festival = 'festival'
    else:
        # Use approved known-calendar features at T to classify pre-festival
        # horizons, so this tag never consults final-holdout event records.
        origin_features = approved[(approved.store_id == store) & (approved.product_id == product) & (approved.date == origin)]
        cols = [n for n in ['days_until_cny', 'days_until_raya', 'days_until_deepavali', 'days_until_christmas'] if n in origin_features]
        next_festival = origin_features[cols].min(axis=1).iloc[0] if len(origin_features) and cols else np.nan
        festival = 'pre_festival' if np.isfinite(next_festival) and horizon < next_festival <= horizon + c['pre_festival_days'] else 'normal'
    prices = group.purchase_price.where(group.purchase_price.gt(0))
    # Supplier changes cannot be interpreted as same-supplier price volatility.
    previous = group.groupby('supplier_id').purchase_price.shift(1)
    changes = (prices / previous.where(previous.gt(0)) - 1).abs()
    volatility = bool(changes.loc[future.index].gt(c['high_volatility_daily_change']).any())
    policy_change = group.policy_type.fillna('').ne(group.policy_type.fillna('').shift(1))
    if len(policy_change):
        policy_change.iloc[0] = False
    policy = bool(policy_change.loc[future.index].any())
    commodities = group.relevant_commodity_price.where(group.relevant_commodity_price.gt(0))
    commodity_change = (commodities / commodities.shift(1) - 1).abs()
    commodity = bool(commodity_change.loc[future.index].gt(c['commodity_shock_change']).any())
    return dict(promotion_period=promotion, festival_period=festival,
                high_volatility=volatility, policy_change=policy, commodity_shock=commodity)


def evaluate_development(raw, targets, approved, config, progress=False):
    """Forecast first from training slices, then attach evaluation-only truth."""
    raw, targets, approved = raw.copy(), targets.copy(), approved.copy()
    for table in [raw, targets, approved]:
        table['date'] = pd.to_datetime(table.date)
    boundary = pd.Timestamp(config['evaluation']['holdout_start'])
    # Defense in depth for callers using in-memory datasets.
    raw, targets, approved = [x[x.date < boundary].copy() for x in [raw, targets, approved]]
    prepared = prepare_history(raw, config)
    folds = build_folds(raw.date.min(), raw.date.max(), config)
    lookup = targets.set_index(KEYS)
    demand_rows, price_rows, direction_rows, new_series = [], [], [], []
    for i, fold in folds.iterrows():
        origin = fold.training_end
        train = prepared[prepared.date.between(fold.training_start, origin)]
        assert train.date.max() <= origin < fold.validation_start
        population = train[['store_id', 'product_id']].drop_duplicates()
        future_population = raw[raw.date.between(fold.validation_start, fold.validation_end)][['store_id', 'product_id']].drop_duplicates()
        unknown = future_population.merge(population, how='left', indicator=True).query('_merge == "left_only"')
        for _, pair in unknown.iterrows():
            new_series.append(dict(validation_fold=fold.validation_fold, store_id=pair.store_id, product_id=pair.product_id,
                                   reason='series_first_observed_after_origin', excluded_horizons=len(config['horizons'])))
        folds.loc[i, 'number_products'] = population.product_id.nunique()
        folds.loc[i, 'number_stores'] = population.store_id.nunique()
        folds.loc[i, 'series_count'] = len(population)
        valid_demand, valid_price = 0, 0
        for (store, product), local in train.groupby(['store_id', 'product_id'], sort=True):
            ctx = historical_context(local, origin, store, product, config)
            ctx.fallback_daily, ctx.fallback_level = demand_fallback(train, origin, store, product, config)
            metadata = local.iloc[-1]
            for h in config['horizons']:
                base = dict(date=origin, forecast_origin=origin, store_id=store, product_id=product, supplier_id=ctx.supplier_id,
                    category=metadata.get('category', 'UNKNOWN'), state=metadata.get('state', 'UNKNOWN'),
                    demand_type=ctx.demand_type, horizon=h, validation_fold=int(fold.validation_fold),
                    training_start=fold.training_start, training_end=origin, maximum_information_date=train.date.max())
                # Forecast calls cannot see labels or any of the validation data.
                demand_forecasts = {m: forecast_demand(ctx, h, m, config) for m in DEMAND_METHODS}
                price_forecasts = {m: forecast_price(ctx, h, m, config) for m in PRICE_METHODS}
                classes = direction_forecasts(ctx, h, config)
                key = (origin, store, product)
                truth = lookup.loc[key] if key in lookup.index else pd.Series(dtype=object)
                d, p = f'demand_target_{h}d', f'purchase_price_target_{h}d'
                actual_d = truth.get(d, np.nan) if truth.get(d + '_valid', False) else np.nan
                price_date = truth.get(p + '_observation_date', pd.NaT)
                # Sparse price tolerance must not cross this fold or holdout.
                price_ok = bool(truth.get(p + '_valid', False)) and pd.notna(price_date) and origin < price_date <= fold.validation_end
                actual_p = truth.get(p, np.nan) if price_ok else np.nan
                valid_demand += int(np.isfinite(actual_d))
                valid_price += int(np.isfinite(actual_p))
                actual_direction = truth.get(f'price_direction_{h}d') if price_ok else None
                scenario = evaluation_scenarios(raw, approved, origin, store, product, h, config)
                shock = truth.get(f'purchase_price_pct_change_{h}d', np.nan)
                scenario['price_shock'] = bool(np.isfinite(shock) and abs(shock) > config['scenarios']['price_shock_change']) if price_ok else False
                for name, prediction in demand_forecasts.items():
                    demand_rows.append(dict(**base, **scenario, baseline=name, **prediction, actual=actual_d,
                        absolute_error=abs(prediction['prediction'] - actual_d), mase_scale=h * ctx.mase_daily_scale))
                for name, prediction in price_forecasts.items():
                    change = prediction['prediction'] / ctx.latest_price - 1 if np.isfinite(ctx.latest_price) else np.nan
                    price_rows.append(dict(**base, **scenario, baseline=name, **prediction, actual=actual_p,
                        target_observation_date=price_date if price_ok else pd.NaT, current_price=ctx.latest_price,
                        absolute_error=abs(prediction['prediction'] - actual_p), actual_direction=actual_direction,
                        predicted_direction=direction(change, config['targets']['price_target']['flat_tolerance'])))
                for name, label in classes.items():
                    direction_rows.append(dict(**base, baseline=name, actual_direction=actual_direction, predicted_direction=label))
        folds.loc[i, 'valid_demand_target_rows'] = valid_demand
        folds.loc[i, 'valid_price_target_rows'] = valid_price
        if progress and (i % 5 == 0 or i == len(folds) - 1):
            print(f'Fold {int(fold.validation_fold)}/{len(folds)}: origin {origin.date()}, {len(population)} series', flush=True)
    return (pd.DataFrame(demand_rows), pd.DataFrame(price_rows), pd.DataFrame(direction_rows), folds,
            pd.DataFrame(new_series, columns=['validation_fold', 'store_id', 'product_id', 'reason', 'excluded_horizons']))


def save_reports(demand, price, classes, folds, excluded, config, root):
    report = root / 'reports/baselines'
    output = root / 'data/predictions'
    figures = root / 'reports/figures/baselines'
    for directory in [report, output, figures]:
        directory.mkdir(parents=True, exist_ok=True)
    demand.to_parquet(output / 'baseline_demand_predictions.parquet', index=False)
    price.to_parquet(output / 'baseline_price_predictions.parquet', index=False)
    classes.to_parquet(output / 'baseline_direction_predictions.parquet', index=False)
    dm = metric_table(demand, 'demand', ['horizon', 'baseline'])
    pm = metric_table(price, 'price', ['horizon', 'baseline'])
    rankings, champions = rankings_and_champions(demand, price, config)
    tables = {'demand_baseline_metrics.csv': dm, 'price_baseline_metrics.csv': pm,
              'baseline_rankings.csv': rankings, 'champion_baselines.csv': champions,
              'walk_forward_folds.csv': folds, 'excluded_new_series.csv': excluded,
              'demand_predictions_sample.csv': demand.head(264), 'price_predictions_sample.csv': price.head(144),
              'fold_metrics.csv': pd.concat([metric_table(demand, 'demand', ['horizon', 'baseline', 'validation_fold']),
                                             metric_table(price, 'price', ['horizon', 'baseline', 'validation_fold'])])}
    for name, groups in [('store_performance', ['store_id']), ('category_performance', ['category']),
                         ('intermittent_demand_performance', ['demand_type']), ('promotion_period_performance', ['promotion_period']),
                         ('festival_period_performance', ['festival_period'])]:
        tables[name + '.csv'] = pd.concat([metric_table(demand, 'demand', ['horizon', 'baseline'] + groups),
                                          metric_table(price, 'price', ['horizon', 'baseline'] + groups)], ignore_index=True)
    for filename, column, states in [
        ('festival_period_performance.csv', 'festival_period', ['normal', 'festival', 'pre_festival']),
        ('intermittent_demand_performance.csv', 'demand_type', ['low_volume', 'medium_volume', 'high_volume', 'intermittent', 'insufficient_history'])]:
        original = tables[filename]
        empty = []
        for task, methods in [('demand', DEMAND_METHODS), ('price', PRICE_METHODS)]:
            for horizon in config['horizons']:
                for method in methods:
                    represented = original[(original.task == task) & (original.horizon == horizon) & (original.baseline == method)][column]
                    for state in states:
                        if state not in set(represented):
                            empty.append(dict(task=task, horizon=horizon, baseline=method, **{column: state},
                                              target_count=0, count=0, status='NO_OBSERVATIONS'))
        tables[filename] = pd.concat([original, pd.DataFrame(empty)], ignore_index=True)
    shock_rows = []
    for scenario in ['price_shock', 'high_volatility', 'policy_change', 'commodity_shock']:
        data = metric_table(price, 'price', ['horizon', 'baseline', scenario]).rename(columns={scenario: 'scenario_active'})
        data['scenario'] = scenario
        shock_rows.append(data)
        # Explicit unavailable categories are more honest than omitting them.
        for h in config['horizons']:
            for baseline in PRICE_METHODS:
                for active in [False, True]:
                    if data[(data.horizon == h) & (data.baseline == baseline) & (data.scenario_active == active)].empty:
                        shock_rows.append(pd.DataFrame([dict(task='price', horizon=h, baseline=baseline, scenario=scenario,
                            scenario_active=active, target_count=0, count=0, status='NO_OBSERVATIONS')]))
    tables['price_shock_performance.csv'] = pd.concat(shock_rows, ignore_index=True)
    population = demand.drop_duplicates(['validation_fold', 'store_id', 'product_id'])
    population_counts = []
    for fold_id, group in population.groupby('validation_fold'):
        for kind in ['low_volume', 'medium_volume', 'high_volume', 'intermittent', 'insufficient_history']:
            population_counts.append(dict(validation_fold=fold_id, demand_type=kind, series_count=int(group.demand_type.eq(kind).sum())))
    tables['population_coverage.csv'] = pd.DataFrame(population_counts)
    diagnostics = []
    for baseline, group in dm.groupby('baseline'):
        endpoints = group.set_index('horizon')
        if 1 in endpoints.index and 30 in endpoints.index:
            diagnostics.append(dict(baseline=baseline, mae_1d=endpoints.loc[1, 'mae'], mae_30d=endpoints.loc[30, 'mae'],
                wape_1d=endpoints.loc[1, 'wape'], wape_30d=endpoints.loc[30, 'wape'],
                interpretation='Cumulative units and daily units differ; inspect increasing MAE alongside aggregation-driven WAPE reduction'))
    tables['horizon_diagnostics.csv'] = pd.DataFrame(diagnostics)
    tables['holdout_policy.csv'] = pd.DataFrame([dict(holdout_start=config['evaluation']['holdout_start'],
        holdout_end=config['evaluation']['holdout_end'], evaluated=False,
        unscored_development_tail_start=(folds.validation_end.max() + pd.Timedelta(days=1)).date(),
        unscored_development_tail_end=(pd.Timestamp(config['evaluation']['holdout_start']) - pd.Timedelta(days=1)).date(),
        reason='Tail shorter than configured full validation window')])
    direction_summary, per_class, confusion = [], [], []
    for (h, name), group in classes.groupby(['horizon', 'baseline']):
        metrics, pc, matrix = direction_metrics(group.actual_direction, group.predicted_direction)
        direction_summary.append(dict(horizon=h, baseline=name, target_count=int(group.actual_direction.notna().sum()),
                                      excluded_predictions=int((group.actual_direction.notna() & group.predicted_direction.isna()).sum()), **metrics))
        per_class.append(pc.assign(horizon=h, baseline=name))
        for actual in matrix.index:
            for predicted in matrix.columns:
                confusion.append(dict(horizon=h, baseline=name, actual=actual, predicted=predicted, count=int(matrix.loc[actual, predicted])))
    tables['price_direction_metrics.csv'] = pd.DataFrame(direction_summary)
    tables['price_direction_per_class.csv'] = pd.concat(per_class, ignore_index=True)
    tables['price_direction_confusion.csv'] = pd.DataFrame(confusion)
    tables['excluded_predictions.csv'] = pd.concat([
        demand.assign(task='demand'), price.assign(task='price')], ignore_index=True).groupby(
        ['task', 'horizon', 'baseline', 'baseline_fallback_level', 'exclusion_reason'], dropna=False).agg(
            rows=('prediction', 'size'), missing_predictions=('prediction', lambda x: x.isna().sum()),
            missing_targets=('actual', lambda x: x.isna().sum())).reset_index()
    tables['fallback_usage.csv'] = tables['excluded_predictions.csv'].query("baseline_fallback_level != 'direct'")
    from .baseline_visualisations import save_figures
    save_figures(demand, price, dm, pm, champions, tables['price_direction_confusion.csv'], tables['price_shock_performance.csv'], figures)
    for name, data in tables.items():
        data.to_csv(report / name, index=False)
    from .baseline_report import write_summary
    write_summary(tables, config, report)
    return champions


def run_pipeline(root=PROJECT_ROOT):
    root = Path(root)
    config = load_config(root)
    protected = [p for folder in ['reports/leakage', 'reports/targets', 'reports/features', 'data/processed']
                 for p in (root / folder).rglob('*') if p.is_file()]
    before = {str(p.relative_to(root)): sha256(p) for p in protected}
    boundary = config['evaluation']['holdout_start']
    raw = read_before(root / 'data/processed/master_daily_demand_corrected.parquet', boundary)
    prospective_folds = build_folds(raw.date.min(), raw.date.max(), config)
    last_origin_exclusive = prospective_folds.training_end.max() + pd.Timedelta(days=1)
    targets = read_before(root / 'data/processed/forecast_targets.parquet', last_origin_exclusive)
    # Labels whose observation dates cross the holdout are never evaluated;
    # fold windows end before the boundary and per-row assertions enforce it.
    manifests = [pd.read_csv(root / f'reports/leakage/{name}_features_temporally_safe.csv') for name in ['demand', 'price']]
    names = sorted(set().union(*(set(m.feature_name) for m in manifests)))
    feature_path = root / 'data/processed/model_features.parquet'
    registry = pd.read_csv(root / 'reports/targets/target_registry.csv')
    validate_manifests(root, ds.dataset(str(feature_path)).schema.names, registry.target_name)
    approved = read_before(feature_path, boundary, KEYS + names)
    validate_targets(targets, config['targets'])
    phase9_intermittent = pd.read_csv(root / 'reports/targets/intermittent_demand_analysis.csv')
    print(f'Approved manifests: {len(manifests[0])} demand / {len(manifests[1])} price; Phase 9 flagged {int(phase9_intermittent.intermittent.sum())} intermittent series (audit only).', flush=True)
    outputs = evaluate_development(raw, targets, approved, config, progress=True)
    demand, price, _, folds, _ = outputs
    for data in [demand, price]:
        if not data.maximum_information_date.le(data.forecast_origin).all():
            raise ValueError('Future information in forecast context')
    champions = save_reports(*outputs, config, root)
    after = {name: sha256(root / name) for name in before}
    if before != after:
        raise RuntimeError('Previous phase artifact changed')
    report = root / 'reports/baselines'
    pd.DataFrame([dict(path=n, sha256_before=d, sha256_after=after[n], unchanged=True) for n, d in before.items()]).to_csv(report / 'upstream_integrity.csv', index=False)
    (report / 'temporal_validation.json').write_text(json.dumps(dict(
        status='PASS', folds=len(folds), maximum_information_date_lte_origin=True,
        validation_blocks_nonoverlapping=True, holdout_evaluated=False,
        holdout_start=boundary, immutable_artifacts=len(before), random_splits=False,
        advanced_models=False), indent=2), encoding='utf-8')
    print(champions[['task', 'horizon', 'baseline', 'mean_fold_error', 'status']].to_string(index=False), flush=True)
    return champions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT)
    run_pipeline(parser.parse_args().root)
