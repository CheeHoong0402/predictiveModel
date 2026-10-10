"""Shared input contract, historical context and explicit fallback hierarchy."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.utils.paths import PROJECT_ROOT
from src.targets.target_pipeline import load_config as load_target_config
from src.targets.demand_targets import prepare_source, KEYS

ACTIVITY = ['product_available', 'product_active_in_store', 'product_active', 'store_active', 'store_open',
            'product_discontinued', 'store_closed', 'product_discontinued_date', 'store_closure_date',
            'product_start_date', 'store_opening_date']


def load_config(root=PROJECT_ROOT):
    with (Path(root) / 'config/baseline_models.yaml').open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['targets'] = load_target_config(root)
    cfg['horizons'] = cfg['targets']['horizons']
    if cfg['evaluation']['history_mode'] not in ['expanding', 'rolling']:
        raise ValueError('Unknown history mode')
    if cfg['evaluation']['validation_window_days'] < max(cfg['horizons']):
        raise ValueError('Validation blocks must cover the longest forecast horizon')
    if cfg['demand']['annual_lag'] < max(cfg['horizons']):
        raise ValueError('Annual references must be historical for every horizon')
    if cfg['evaluation']['minimum_prediction_coverage'] != 1.0:
        raise ValueError('Champion comparison requires complete common coverage')
    if cfg['probabilistic']['quantiles'] != [0.1, 0.5, 0.9]:
        raise ValueError('Required quantiles are 0.1, 0.5, 0.9')
    if not 0 < cfg['intermittent']['croston_alpha'] <= 1:
        raise ValueError('Invalid Croston alpha')
    if pd.Timestamp(cfg['evaluation']['holdout_start']) <= pd.Timestamp(cfg['evaluation']['start_date']):
        raise ValueError('Holdout must follow development evaluation')
    return cfg


def prepare_history(history, config):
    """Allowlisted observational inputs; targets must never be passed here."""
    banned = ('demand_target_', 'demand_day_', 'purchase_price_target_', 'purchase_price_pct_change_', 'price_direction_', 'y__')
    if any(str(c).startswith(banned) for c in history.columns):
        raise ValueError('Phase 9 targets are prohibited prediction inputs')
    columns = KEYS + ['estimated_true_demand_training_safe', 'observed_units_sold', 'purchase_price',
                      'supplier_id', 'state', 'category', 'price_observation_valid'] + ACTIVITY
    selected = history[[c for c in columns if c in history]].copy()
    return prepare_source(selected, config['targets'])


@dataclass
class Context:
    origin: pd.Timestamp
    daily: pd.Series
    prices: pd.Series
    latest_price: float
    supplier_id: object
    observed_until: pd.Timestamp
    active: bool
    demand_type: str
    mase_daily_scale: float
    fallback_daily: float = np.nan
    fallback_level: str = 'missing'


def historical_context(prepared, origin, store_id, product_id, config):
    """Every input is trimmed to T. Gaps stay null on the calendar grid."""
    origin = pd.Timestamp(origin)
    group = prepared[(prepared.store_id == store_id) & (prepared.product_id == product_id) & (prepared.date <= origin)].sort_values('date')
    if config['evaluation']['history_mode'] == 'rolling':
        group = group[group.date > origin - pd.Timedelta(days=config['evaluation']['rolling_history_days'])]
    if group.empty:
        return Context(origin, pd.Series(dtype=float), pd.Series(dtype=float), np.nan, None, pd.NaT, True, 'insufficient_history', np.nan)
    last = group.iloc[-1]
    # Missing observation is not an explicit discontinuation. Known inactivity is.
    active = last._demand_reason not in ['STORE_CLOSED', 'DISCONTINUED_PRODUCT', 'INACTIVE_PRODUCT', 'UNAVAILABLE_PRODUCT', 'NOT_YET_ACTIVE', 'UNKNOWN_ACTIVITY']
    calendar = pd.date_range(group.date.min(), origin)
    daily = group.set_index('date')._valid_demand.reindex(calendar)
    supplier = last.get('supplier_id')
    p = group[group.supplier_id.eq(supplier)] if 'supplier_id' in group and pd.notna(supplier) else group.iloc[:0]
    prices = pd.to_numeric(p.get('purchase_price', pd.Series(dtype=float)), errors='raise')
    good = np.isfinite(prices) & prices.gt(0)
    if 'price_observation_valid' in p:
        good &= p.price_observation_valid.eq(1)
    prices = pd.Series(prices.loc[good].to_numpy(), index=pd.DatetimeIndex(p.loc[good, 'date']), dtype=float)
    max_age = config['targets']['price_target']['current_price_maximum_age_days']
    current = float(prices.iloc[-1]) if len(prices) and (origin - prices.index[-1]).days <= max_age else np.nan
    valid = daily.dropna()
    from .intermittent_demand import is_intermittent
    if len(valid) < config['intermittent']['minimum_observations']:
        kind = 'insufficient_history'
    elif is_intermittent(daily, config):
        kind = 'intermittent'
    elif valid.mean() < config['demand']['low_volume_below']:
        kind = 'low_volume'
    elif valid.mean() >= config['demand']['high_volume_at_least']:
        kind = 'high_volume'
    else:
        kind = 'medium_volume'
    scale = (daily - daily.shift(config['demand']['seasonal_lag'])).abs().mean()
    return Context(origin, daily, prices, current, supplier, group.date.max(), active, kind, float(scale))


def demand_fallback(prepared, origin, store_id, product_id, config):
    """Past-only pooled daily rates, retaining per-store/unit interpretation.
    No price pooling across products, units or suppliers is permitted.
    """
    origin = pd.Timestamp(origin)
    window = config['fallback']['window_days']
    if config['evaluation']['history_mode'] == 'rolling':
        window = min(window, config['evaluation']['rolling_history_days'])
    hist = prepared[(prepared.date <= origin) & (prepared.date > origin - pd.Timedelta(days=window))]
    own = hist[(hist.store_id == store_id) & (hist.product_id == product_id)]
    # Metadata used for pooling must itself have been observed by T.
    metadata = own.iloc[-1] if len(own) else None
    state = metadata.get('state') if metadata is not None else None
    category = metadata.get('category') if metadata is not None else None
    levels = {
        'product_store': own,
        'product_state': hist[hist.product_id.eq(product_id) & hist.get('state', pd.Series(index=hist.index, dtype=object)).eq(state)] if pd.notna(state) else hist.iloc[:0],
        'product_national': hist[hist.product_id.eq(product_id)],
        'category_store': hist[hist.store_id.eq(store_id) & hist.get('category', pd.Series(index=hist.index, dtype=object)).eq(category)] if pd.notna(category) else hist.iloc[:0],
        'category_national': hist[hist.get('category', pd.Series(index=hist.index, dtype=object)).eq(category)] if pd.notna(category) else hist.iloc[:0],
        'global': hist,
    }
    for level in config['fallback']['demand_levels']:
        values = levels[level]._valid_demand.dropna()
        if len(values) >= config['fallback']['minimum_observations']:
            return float(values.mean()), level
    return np.nan, 'missing'


def result(value=np.nan, level='direct', reason='', quantiles=None):
    q = quantiles if quantiles is not None else [np.nan] * 3
    return dict(prediction=float(value), baseline_fallback_level=level, exclusion_reason=reason,
                q10=float(q[0]), q50=float(q[1]), q90=float(q[2]))
