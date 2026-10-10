"""Point and empirical cumulative-demand forecasts from history through T."""
import numpy as np
import pandas as pd
from .fallback_baselines import load_config, prepare_history, historical_context, demand_fallback, result
from .intermittent_demand import croston, is_intermittent

METHODS = ['last_value', 'seasonal_naive_7', 'same_weekday_median_4w', 'moving_average_7',
           'moving_average_28', 'median_28', 'weekday_profile', 'annual_naive_364',
           'recent_trend', 'croston', 'empirical_quantiles']


def forecast_demand(ctx, horizon, method, config):
    if method not in METHODS or horizon not in config['horizons']:
        raise ValueError('Unknown demand method or horizon')
    if not ctx.active:
        return result(level='missing', reason='known_inactive_at_origin')
    s, c = ctx.daily, config['demand']
    valid = s.dropna()
    future = pd.date_range(ctx.origin + pd.Timedelta(days=1), periods=horizon)
    daily = np.full(horizon, np.nan)
    quantiles = None
    if method == 'croston' and not is_intermittent(s, config):
        return result(level='not_applicable', reason='not_intermittent_from_training_history')
    if len(valid) >= c['minimum_series_observations'] or (method == 'last_value' and len(valid) >= 1):
        if method == 'last_value' and (ctx.origin - valid.index[-1]).days <= c['maximum_last_value_age_days']:
            daily[:] = valid.iloc[-1]
        elif method == 'seasonal_naive_7':
            # Recursively repeat the last seasonal cycle for h > 7 without
            # reading actual observations inside the forecast horizon.
            lag = c['seasonal_lag']
            refs = [d - pd.Timedelta(days=lag * int(np.ceil((d - ctx.origin).days / lag))) for d in future]
            daily = s.reindex(refs).to_numpy(float)
        elif method in ['same_weekday_median_4w', 'weekday_profile']:
            for i, d in enumerate(future):
                if method == 'same_weekday_median_4w':
                    # Previous four VALID observations of this weekday.
                    candidates = valid[valid.index.dayofweek == d.dayofweek].tail(c['same_weekday_weeks'])
                    minimum = c['same_weekday_weeks']
                else:
                    recent = s[s.index > ctx.origin - pd.Timedelta(days=c['weekday_profile_days'])].dropna()
                    candidates = recent[recent.index.dayofweek == d.dayofweek]
                    minimum = c['weekday_profile_min_observations']
                if len(candidates) >= minimum:
                    daily[i] = candidates.median() if method == 'same_weekday_median_4w' else candidates.mean()
        elif method.startswith('moving_average_'):
            window = int(method.rsplit('_', 1)[1])
            if window not in c['moving_average_windows']:
                raise ValueError('Moving average window disabled')
            if len(valid) >= window:
                daily[:] = valid.tail(window).mean()
        elif method == 'median_28':
            recent = s[s.index > ctx.origin - pd.Timedelta(days=c['median_window'])].dropna()
            if len(recent) >= c['minimum_series_observations']:
                daily[:] = recent.median()
        elif method == 'annual_naive_364':
            daily = s.reindex(future - pd.Timedelta(days=c['annual_lag'])).to_numpy(float)
        elif method == 'recent_trend':
            recent = s[s.index > ctx.origin - pd.Timedelta(days=c['trend_window_days'])].dropna()
            if len(recent) >= c['trend_min_observations']:
                slope, intercept = np.polyfit((recent.index - ctx.origin).days.to_numpy(), recent.to_numpy(), 1)
                daily = np.clip(intercept + slope * np.arange(1, horizon + 1), 0, max(recent.max(), recent.mean() * c['trend_cap_multiple']))
        elif method == 'croston':
            daily[:] = croston(s.to_numpy(), config['intermittent']['croston_alpha'])
        elif method == 'empirical_quantiles':
            # Quantiles of ACTUAL historical H-day sums, not sum of daily
            # quantiles. Preserve cross-day dependence and weekday alignment.
            recent = s[s.index > ctx.origin - pd.Timedelta(days=config['probabilistic']['lookback_days'])]
            sums = recent.rolling(horizon, min_periods=horizon).sum()
            starts = sums.index - pd.Timedelta(days=horizon - 1)
            samples = sums[starts.dayofweek == future[0].dayofweek].dropna()
            if len(samples) >= config['probabilistic']['minimum_samples']:
                quantiles = samples.quantile(config['probabilistic']['quantiles']).to_numpy(float)
                return result(quantiles[1], quantiles=quantiles)
    if np.isfinite(daily).all():
        return result(daily.sum())
    # Keep annual and empirical applicability honest instead of labelling a
    # pooled mean as an annual forecast or inventing uncertainty intervals.
    if method in ['annual_naive_364', 'empirical_quantiles', 'croston']:
        return result(level='missing', reason='insufficient_complete_history')
    if np.isfinite(ctx.fallback_daily):
        return result(ctx.fallback_daily * horizon, level=ctx.fallback_level, reason='')
    return result(level='missing', reason='insufficient_history_no_fallback')


def predict_demand_baseline(history, forecast_origin, product_id, store_id, horizon, method, config=None):
    cfg = config or load_config()
    prepared = prepare_history(history, cfg)
    ctx = historical_context(prepared, forecast_origin, store_id, product_id, cfg)
    ctx.fallback_daily, ctx.fallback_level = demand_fallback(prepared, forecast_origin, store_id, product_id, cfg)
    return forecast_demand(ctx, horizon, method, cfg)
