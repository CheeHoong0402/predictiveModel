"""Supplier-preserving price rules and transparent direction classifiers."""
import numpy as np
import pandas as pd
from .fallback_baselines import load_config, prepare_history, historical_context, result

METHODS = ['price_last_value', 'price_moving_average_7', 'price_moving_average_30',
           'price_median_30', 'price_recent_trend', 'price_empirical_quantiles']


def direction(change, tolerance):
    if not np.isfinite(change):
        return None
    return 'UP' if change > tolerance + 1e-12 else 'DOWN' if change < -tolerance - 1e-12 else 'FLAT'


def forecast_price(ctx, horizon, method, config):
    if method not in METHODS or horizon not in config['horizons']:
        raise ValueError('Unknown price method or horizon')
    if not ctx.active:
        return result(level='missing', reason='known_inactive_at_origin')
    current = ctx.latest_price
    if not np.isfinite(current):
        return result(level='missing', reason='missing_or_stale_same_supplier_price')
    s, c = ctx.prices, config['price']
    value = np.nan
    if method == 'price_last_value':
        value = current
    elif method.startswith('price_moving_average_') or method == 'price_median_30':
        window = c['median_window_days'] if method == 'price_median_30' else int(method.rsplit('_', 1)[1])
        if method.startswith('price_moving_average_') and window not in c['moving_average_windows']:
            raise ValueError('Price moving average window disabled')
        recent = s[s.index > ctx.origin - pd.Timedelta(days=window)]
        if len(recent) >= c['minimum_window_observations']:
            value = recent.median() if method == 'price_median_30' else recent.mean()
    elif method == 'price_recent_trend':
        recent = s[s.index > ctx.origin - pd.Timedelta(days=c['trend_window_days'])]
        if len(recent) >= c['trend_min_observations']:
            slope, intercept = np.polyfit((recent.index - ctx.origin).days.to_numpy(), recent.to_numpy(), 1)
            value = np.clip(intercept + slope * horizon, current * c['trend_min_multiple'], current * c['trend_max_multiple'])
    elif method == 'price_empirical_quantiles':
        recent = s[s.index > ctx.origin - pd.Timedelta(days=config['probabilistic']['lookback_days'])]
        if len(recent):
            calendar = recent.reindex(pd.date_range(recent.index.min(), ctx.origin))
            # Endpoints both already observed by T; no Phase 9 target is used.
            ratios = (calendar / calendar.shift(horizon)).dropna()
            if len(ratios) >= config['probabilistic']['minimum_samples']:
                quantiles = current * ratios.quantile(config['probabilistic']['quantiles']).to_numpy(float)
                return result(quantiles[1], quantiles=quantiles)
        return result(level='missing', reason='insufficient_historical_horizon_pairs')
    if np.isfinite(value):
        return result(value)
    if config['fallback']['price_levels'] == ['same_supplier_last_value', 'missing']:
        return result(current, level='same_supplier_last_value')
    raise ValueError('Unknown price fallback hierarchy')


def direction_forecasts(ctx, horizon, config):
    if not ctx.active:
        return {name: None for name in ['direction_always_flat', 'direction_latest_change', 'direction_trend_30']}
    tol = config['targets']['price_target']['flat_tolerance']
    latest = None
    if len(ctx.prices) >= 2 and np.isfinite(ctx.latest_price):
        max_age = config['targets']['price_target']['current_price_maximum_age_days']
        if (ctx.prices.index[-1] - ctx.prices.index[-2]).days <= max_age:
            latest = direction(ctx.prices.iloc[-1] / ctx.prices.iloc[-2] - 1, tol)
    trend = forecast_price(ctx, horizon, 'price_recent_trend', config)
    trend_label = direction(trend['prediction'] / ctx.latest_price - 1, tol) if np.isfinite(ctx.latest_price) else None
    return {'direction_always_flat': 'FLAT', 'direction_latest_change': latest, 'direction_trend_30': trend_label}


def predict_price_baseline(history, forecast_origin, product_id, store_id, horizon, method, config=None):
    cfg = config or load_config()
    prepared = prepare_history(history, cfg)
    ctx = historical_context(prepared, forecast_origin, store_id, product_id, cfg)
    return forecast_price(ctx, horizon, method, cfg)
