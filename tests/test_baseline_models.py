"""Deterministic algorithms, chronology, adversarial leakage and report tests."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.baselines.fallback_baselines import load_config, prepare_history, historical_context
from src.baselines.demand_baselines import predict_demand_baseline, METHODS as DEMAND_METHODS
from src.baselines.price_baselines import predict_price_baseline, METHODS as PRICE_METHODS, direction_forecasts
from src.baselines.intermittent_demand import croston
from src.baselines.walk_forward import build_folds
from src.baselines.baseline_evaluator import regression_metrics, direction_metrics, rankings_and_champions
from src.baselines.baseline_pipeline import evaluate_development, read_before
from src.targets.target_pipeline import construct_targets

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config(ROOT)


def history(values, start='2021-01-01', store='A', product='X'):
    n = len(values)
    return pd.DataFrame(dict(date=pd.date_range(start, periods=n), store_id=store, product_id=product,
        supplier_id='SUP', estimated_true_demand_training_safe=np.asarray(values, dtype=float),
        observed_units_sold=9999., purchase_price=np.arange(n) / 100 + 10., category='food', state='state',
        promotion_active=0, festival_flag=0, policy_type='stable', relevant_commodity_price=100.))


def demand(data, h, method, cfg, origin=None):
    return predict_demand_baseline(data, origin if origin is not None else data.date.max(), 'X', 'A', h, method, cfg)


def price(data, h, method, cfg, origin=None):
    return predict_price_baseline(data, origin if origin is not None else data.date.max(), 'X', 'A', h, method, cfg)


def test_last_value_and_given_moving_average(cfg):
    data = history([10, 12, 14, 16, 18, 20, 22])
    assert demand(data, 1, 'last_value', cfg)['prediction'] == 22
    assert demand(data, 7, 'last_value', cfg)['prediction'] == 154
    assert demand(data, 7, 'moving_average_7', cfg)['prediction'] == 112


def test_last_value_requires_only_one_valid_observation(cfg):
    assert demand(history([12]), 7, 'last_value', cfg)['prediction'] == 84


def test_given_seasonal_naive_and_recursive_horizons(cfg):
    data = history([10, 11, 12, 13, 14, 15, 16])
    assert demand(data, 7, 'seasonal_naive_7', cfg)['prediction'] == 91
    assert demand(data, 14, 'seasonal_naive_7', cfg)['prediction'] == 182
    assert demand(data, 30, 'seasonal_naive_7', cfg)['prediction'] == 4 * 91 + 10 + 11


def test_weekday_median_and_profile(cfg):
    data = history(np.arange(1, 85))
    origin = data.date.max()
    future = pd.date_range(origin + pd.Timedelta(days=1), periods=7)
    expected = sum(data.loc[data.date.dt.dayofweek.eq(d.dayofweek), 'estimated_true_demand_training_safe'].tail(4).median() for d in future)
    assert demand(data, 7, 'same_weekday_median_4w', cfg)['prediction'] == expected
    expected_mean = sum(data.loc[data.date.dt.dayofweek.eq(d.dayofweek), 'estimated_true_demand_training_safe'].mean() for d in future)
    assert demand(data, 7, 'weekday_profile', cfg)['prediction'] == expected_mean


def test_median_and_28_observation_average(cfg):
    data = history(np.arange(1, 29))
    assert demand(data, 3, 'moving_average_28', cfg)['prediction'] == 14.5 * 3
    assert demand(data, 3, 'median_28', cfg)['prediction'] == 14.5 * 3


def test_annual_calendar_reference_and_missing_history(cfg):
    data = history(np.arange(1, 401))
    origin = data.date.max()
    dates = pd.date_range(origin + pd.Timedelta(days=1), periods=7) - pd.Timedelta(days=364)
    expected = data.set_index('date').loc[dates, 'estimated_true_demand_training_safe'].sum()
    assert demand(data, 7, 'annual_naive_364', cfg)['prediction'] == expected
    assert np.isnan(demand(data.tail(30), 7, 'annual_naive_364', cfg)['prediction'])


def test_demand_linear_trend_and_nonnegative_cap(cfg):
    data = history(np.arange(1, 29))
    assert demand(data, 3, 'recent_trend', cfg)['prediction'] == pytest.approx(29 + 30 + 31)
    down = history(np.arange(28, 0, -1))
    assert demand(down, 30, 'recent_trend', cfg)['prediction'] >= 0


@pytest.mark.parametrize('method', DEMAND_METHODS)
def test_demand_future_sentinel_truncation_and_retrospective_invariance(cfg, method):
    data = history(np.arange(1, 451) % 25)
    origin = data.date.iloc[399]
    actual = demand(data, 7, method, cfg, origin)
    truncated = demand(data.iloc[:400], 7, method, cfg, origin)
    polluted = data.copy()
    polluted.loc[polluted.date > origin, ['estimated_true_demand_training_safe', 'observed_units_sold', 'purchase_price']] = 1e12
    polluted['estimated_true_demand_retrospective'] = 1e15
    sentinel = demand(polluted, 7, method, cfg, origin)
    for candidate in [truncated, sentinel]:
        assert np.allclose([actual['prediction'], actual['q10'], actual['q90']], [candidate['prediction'], candidate['q10'], candidate['q90']], equal_nan=True)


@pytest.mark.parametrize('method', PRICE_METHODS)
def test_price_future_sentinel_invariance(cfg, method):
    data = history(np.ones(450))
    origin = data.date.iloc[399]
    a = price(data, 7, method, cfg, origin)
    polluted = data.copy()
    polluted.loc[polluted.date > origin, 'purchase_price'] = 1e10
    b = price(polluted, 7, method, cfg, origin)
    assert np.allclose([a['prediction'], a['q10'], a['q90']], [b['prediction'], b['q10'], b['q90']], equal_nan=True)


def test_croston_deterministic_and_missing_days(cfg):
    assert croston([0, 2, 0, 2, 0, 2]) == 1
    assert croston([0, 0, 0]) == 0
    assert np.isnan(croston([0, np.nan, 2]))
    assert croston([0, 2, 0, 0, 4], alpha=.5) == pytest.approx(3 / 2.5)
    data = history([0, 2] * 20)
    assert demand(data, 7, 'croston', cfg)['prediction'] == 7
    regular = history([2] * 40)
    assert demand(regular, 7, 'croston', cfg)['baseline_fallback_level'] == 'not_applicable'


def test_current_price_average_median_and_trend(cfg):
    data = history(np.ones(40))
    data['purchase_price'] = np.arange(1., 41.)
    assert price(data, 7, 'price_last_value', cfg)['prediction'] == 40
    assert price(data, 7, 'price_moving_average_7', cfg)['prediction'] == 37
    assert price(data, 7, 'price_moving_average_30', cfg)['prediction'] == 25.5
    assert price(data, 7, 'price_median_30', cfg)['prediction'] == 25.5
    assert price(data, 7, 'price_recent_trend', cfg)['prediction'] == pytest.approx(47)


def test_sparse_price_gap_and_supplier_boundaries(cfg):
    data = history(np.ones(40))
    data.loc[31:, 'purchase_price'] = np.nan
    assert np.isnan(price(data, 7, 'price_last_value', cfg)['prediction'])
    data.loc[39, 'supplier_id'] = 'NEW'
    assert np.isnan(price(data, 7, 'price_last_value', cfg)['prediction'])
    data.loc[39, 'purchase_price'] = 25
    assert price(data, 7, 'price_last_value', cfg)['prediction'] == 25
    assert price(data, 7, 'price_moving_average_7', cfg)['baseline_fallback_level'] == 'same_supplier_last_value'


def test_store_product_boundaries_and_input_shuffle(cfg):
    own = history(np.arange(1, 41))
    other = history(np.ones(40) * 1e7, store='B')
    another = history(np.ones(40) * 1e8, product='Y')
    all_data = pd.concat([own, other, another], ignore_index=True)
    a = demand(all_data, 7, 'moving_average_7', cfg)
    b = demand(all_data.sample(frac=1, random_state=5), 7, 'moving_average_7', cfg)
    assert a == b
    assert a['prediction'] == 37 * 7
    assert price(all_data, 7, 'price_last_value', cfg)['prediction'] == 10.39


def test_new_product_fallback_and_unknown_history(cfg):
    existing = history(np.ones(40) * 20, store='B')
    new = history([10], start='2021-02-09')
    data = pd.concat([existing, new], ignore_index=True)
    answer = demand(data, 7, 'moving_average_28', cfg)
    assert answer['baseline_fallback_level'] == 'product_state'
    assert answer['prediction'] == pytest.approx((40 * 20 + 10) / 41 * 7)
    assert demand(new, 7, 'moving_average_28', cfg)['exclusion_reason'] == 'insufficient_history_no_fallback'


@pytest.mark.parametrize('flag,value', [('store_closed', 1), ('product_discontinued', 1), ('store_active', 0)])
def test_known_inactive_no_forecast(cfg, flag, value):
    data = history(np.ones(40))
    data[flag] = value
    assert np.isnan(demand(data, 7, 'last_value', cfg)['prediction'])
    assert np.isnan(price(data, 7, 'price_last_value', cfg)['prediction'])


def test_empirical_demand_uses_historical_cumulative_distribution(cfg):
    data = history(np.tile([10, 11, 12, 13, 14, 15, 16], 52))
    answer = demand(data, 7, 'empirical_quantiles', cfg)
    assert [answer[k] for k in ['q10', 'q50', 'q90']] == [91, 91, 91]
    # Every historical 7-day window sums to 91; this is not 7 times a
    # single arbitrary daily quantile.
    assert answer['prediction'] == 91


def test_empirical_price_ratios_and_direction(cfg):
    data = history(np.ones(100))
    data['purchase_price'] = 10 * 1.001 ** np.arange(100)
    answer = price(data, 7, 'price_empirical_quantiles', cfg)
    expected = data.purchase_price.iloc[-1] * 1.001 ** 7
    assert answer['prediction'] == pytest.approx(expected)
    assert answer['q10'] <= answer['q50'] <= answer['q90']
    ctx = historical_context(prepare_history(data, cfg), data.date.max(), 'A', 'X', cfg)
    labels = direction_forecasts(ctx, 7, cfg)
    assert labels['direction_always_flat'] == 'FLAT'
    assert labels['direction_latest_change'] == 'FLAT'
    assert labels['direction_trend_30'] == 'UP'


@pytest.mark.parametrize('column', ['demand_target_7d', 'purchase_price_target_7d', 'purchase_price_pct_change_7d', 'price_direction_7d', 'demand_day_7'])
def test_targets_rejected_from_prediction_inputs(cfg, column):
    data = history(np.ones(40))
    data[column] = 100
    with pytest.raises(ValueError, match='prohibited'):
        demand(data, 7, 'last_value', cfg)
    with pytest.raises(ValueError, match='prohibited'):
        price(data, 7, 'price_last_value', cfg)


def test_walk_forward_disjoint_training_and_holdout(cfg):
    folds = build_folds('2021-01-01', '2024-12-31', cfg)
    assert len(folds) == 33
    assert (folds.training_end < folds.validation_start).all()
    assert (folds.validation_end < pd.Timestamp('2024-10-01')).all()
    assert folds.training_start.nunique() == 1
    assert (folds.validation_start.iloc[1:].to_numpy() > folds.validation_end.iloc[:-1].to_numpy()).all()
    cfg['evaluation']['history_mode'] = 'rolling'
    cfg['evaluation']['rolling_history_days'] = 365
    rolling = build_folds('2021-01-01', '2024-12-31', cfg)
    assert ((rolling.training_end - rolling.training_start).dt.days <= 364).all()


def test_metrics_wape_mase_bias_quantiles_and_price():
    frame = pd.DataFrame(dict(actual=[10., 20.], prediction=[12., 16.], mase_scale=[2., 2.],
        q10=[8., 15.], q50=[12., 16.], q90=[14., 22.], actual_direction=['UP', 'DOWN'], predicted_direction=['UP', 'FLAT']))
    m = regression_metrics(frame, 'demand')
    assert m['mae'] == 3
    assert m['rmse'] == pytest.approx(np.sqrt(10))
    assert m['wape'] == .2
    assert m['mase'] == 1.5
    assert m['bias'] == -1
    assert m['pinball_q50'] == 1.5
    assert m['interval_coverage_80'] == 1
    p = regression_metrics(frame, 'price')
    assert p['mape'] == .2
    assert p['direction_accuracy'] == .5
    zeros = pd.DataFrame(dict(actual=[0.], prediction=[0.], mase_scale=[0.]))
    z = regression_metrics(zeros, 'demand')
    assert np.isnan(z['wape']) and np.isnan(z['mase']) and z['mase_count'] == 0


def test_classification_metrics_confusion():
    m, per_class, matrix = direction_metrics(['UP', 'UP', 'FLAT', 'DOWN'], ['UP', 'DOWN', 'FLAT', 'DOWN'])
    assert m['accuracy'] == .75
    assert m['balanced_accuracy'] == pytest.approx((.5 + 1 + 1) / 3)
    assert matrix.loc['UP', 'DOWN'] == 1
    assert per_class.set_index('label').loc['DOWN', 'precision'] == .5


def test_pipeline_fold_never_uses_validation_data_or_holdout(cfg):
    data = history(20 + np.arange(500) % 7)
    cfg['evaluation'].update(start_date='2022-01-01', holdout_start='2022-03-02', holdout_end='2022-05-15')
    _, target = construct_targets(data, cfg['targets'])
    approved = data[['date', 'store_id', 'product_id']].copy()
    a = evaluate_development(data, target, approved, cfg)
    # Alter first validation block and every holdout observation. First-fold
    # forecasts remain invariant. Later folds may learn earlier realised data.
    corrupted = data.copy()
    corrupted.loc[corrupted.date >= '2022-01-01', 'estimated_true_demand_training_safe'] = 1e8
    corrupted.loc[corrupted.date >= '2022-01-01', 'purchase_price'] = 1e7
    b = evaluate_development(corrupted, target, approved, cfg)
    for first, second in zip(a[:2], b[:2]):
        cols = ['prediction', 'q10', 'q50', 'q90']
        pd.testing.assert_frame_equal(first.loc[first.validation_fold == 1, cols], second.loc[second.validation_fold == 1, cols])
        assert (first.forecast_origin < pd.Timestamp('2022-03-02')).all()
    holdout_only = data.copy()
    holdout_only.loc[holdout_only.date >= '2022-03-02', ['estimated_true_demand_training_safe', 'purchase_price']] = 1e9
    holdout_labels = target.copy()
    # Labels for pre-holdout origins can themselves extend into holdout.
    for h in cfg['horizons']:
        crossing = holdout_labels.date + pd.Timedelta(days=h) >= pd.Timestamp('2022-03-02')
        holdout_labels.loc[crossing, [f'demand_target_{h}d', f'purchase_price_target_{h}d']] = 1e12
    c = evaluate_development(holdout_only, holdout_labels, approved, cfg)
    for first, third in zip(a[:3], c[:3]):
        pd.testing.assert_frame_equal(first, third)


def test_parquet_reader_excludes_holdout_before_materialization(tmp_path):
    data = history([1, 2, 3], start='2024-09-30')
    for as_string in [False, True]:
        frame = data.copy()
        if as_string:
            frame['date'] = frame.date.dt.strftime('%Y-%m-%d')
        path = tmp_path / f'input_{as_string}.parquet'
        frame.to_parquet(path, index=False)
        actual = read_before(path, '2024-10-01')
        assert len(actual) == 1
        assert actual.estimated_true_demand_training_safe.iloc[0] == 1


def test_saved_run_artifacts_and_champions():
    report = ROOT / 'reports/baselines'
    if not (report / 'temporal_validation.json').exists():
        pytest.skip('Run the baseline pipeline before checking delivery artifacts')
    for name in ['baseline_summary.md', 'demand_baseline_metrics.csv', 'price_baseline_metrics.csv',
                 'baseline_rankings.csv', 'champion_baselines.csv', 'store_performance.csv', 'category_performance.csv',
                 'intermittent_demand_performance.csv', 'promotion_period_performance.csv', 'festival_period_performance.csv',
                 'price_shock_performance.csv', 'walk_forward_folds.csv']:
        assert (report / name).exists()
    champions = pd.read_csv(report / 'champion_baselines.csv')
    assert len(champions) == 12
    assert champions.status.eq('DEVELOPMENT_CHAMPION').all()
    assert champions.prediction_coverage.eq(1).all()
    assert pd.read_csv(report / 'upstream_integrity.csv').unchanged.all()
    for task in ['demand', 'price']:
        predictions = pd.read_parquet(ROOT / f'data/predictions/baseline_{task}_predictions.parquet')
        assert not predictions.duplicated(['date', 'store_id', 'product_id', 'horizon', 'baseline']).any()
        assert predictions.maximum_information_date.le(predictions.forecast_origin).all()
    assert len(list((ROOT / 'reports/figures/baselines').glob('*.png'))) >= 8


def test_ranking_rejects_selective_missing_predictions(cfg):
    cfg['evaluation']['minimum_champion_folds'] = 2
    frame = pd.DataFrame(dict(horizon=[1] * 4, baseline=['reliable', 'reliable', 'selective', 'selective'],
        validation_fold=[1, 2, 1, 2], actual=[10., 10., 10., 10.], prediction=[11., 11., 10., np.nan], mase_scale=[1.] * 4))
    rankings, champions = rankings_and_champions(frame, frame, cfg)
    assert champions.baseline.eq('reliable').all()
    assert not rankings.loc[rankings.baseline.eq('selective'), 'eligible'].any()


def test_zero_training_scale_is_reported_not_fabricated(cfg):
    data = history(np.ones(40) * 12)
    ctx = historical_context(prepare_history(data, cfg), data.date.max(), 'A', 'X', cfg)
    assert ctx.mase_daily_scale == 0
    frame = pd.DataFrame(dict(actual=[10.], prediction=[12.], mase_scale=[ctx.mase_daily_scale]))
    assert np.isnan(regression_metrics(frame, 'demand')['mase'])


def test_missing_dates_do_not_become_zero_or_row_lags(cfg):
    data = history(np.arange(1, 41))
    data = data.drop(index=35)
    # Missing recent seasonal date must trigger explicit fallback, not a
    # shifted neighbouring calendar observation.
    seasonal = demand(data, 7, 'seasonal_naive_7', cfg)
    assert seasonal['baseline_fallback_level'] == 'product_store'
    assert seasonal['prediction'] == pytest.approx(data.estimated_true_demand_training_safe.mean() * 7)


def test_price_direction_inactive_is_unavailable(cfg):
    data = history(np.ones(40))
    data['product_discontinued'] = 1
    ctx = historical_context(prepare_history(data, cfg), data.date.max(), 'A', 'X', cfg)
    assert all(x is None for x in direction_forecasts(ctx, 7, cfg).values())
