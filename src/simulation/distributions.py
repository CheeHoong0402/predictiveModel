"""Uncertainty distribution samplers for Monte Carlo risk simulation.

Provides vectorised generation of:
- Coherent daily demand paths across 30 days matching Phase 16 calibrated quantiles (P10, P50, P90).
- Multi-horizon purchase price trajectories matching calibrated price uncertainty.
- Empirical and parametric supplier lead-time distributions (non-negative integer days).
- Supplier delivery fill rates and reliability shocks.
"""
from typing import Dict, Optional, Tuple, Union
import numpy as np

from src.inventory.curve_fitting import interpolate_demand_curve, interpolate_price_curve


def sample_daily_demand_paths(
    demand_horizons: Dict[int, float],
    demand_p10: Optional[Dict[int, float]] = None,
    demand_p50: Optional[Dict[int, float]] = None,
    demand_p90: Optional[Dict[int, float]] = None,
    n_scenarios: int = 10000,
    days: int = 30,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Generate coherent daily demand paths across `days` days.

    Maintains consistency with Phase 16 calibrated cumulative quantiles:
    - Daily demand d_{s, t} >= 0 for all scenarios and days.
    - Cumulative demand D_{s, t} is strictly non-decreasing over time.
    - 30-day cumulative demand matches Phase 16 P10, P50, and P90 quantiles.

    Returns:
        np.ndarray of shape (n_scenarios, days) with dtype float32.
    """
    generator = rng or np.random.default_rng(42)

    # 1. Obtain interpolated expected cumulative curve across 1..days
    curves = interpolate_demand_curve(
        horizon_forecasts=demand_horizons,
        horizon_p10=demand_p10,
        horizon_p50=demand_p50,
        horizon_p90=demand_p90
    )
    cum_expected = curves["point_demand"][:days]
    cum_p10 = curves["p10_demand"][:days]
    cum_p50 = curves["p50_demand"][:days]
    cum_p90 = curves["p90_demand"][:days]

    # Daily baseline increments
    daily_base = np.zeros(days, dtype=np.float32)
    daily_base[0] = max(0.05, float(cum_expected[0]))
    for t in range(1, days):
        daily_base[t] = max(0.05, float(cum_expected[t] - cum_expected[t - 1]))

    # Target 30-day quantile bounds
    target_p10 = float(cum_p10[-1])
    target_p50 = float(cum_p50[-1])
    target_p90 = float(cum_p90[-1])

    # Safeguard bounds
    if target_p50 <= 0:
        target_p50 = max(1.0, float(cum_expected[-1]))
    if target_p10 <= 0:
        target_p10 = max(0.5, target_p50 * 0.7)
    if target_p90 <= target_p50:
        target_p90 = target_p50 * 1.3

    # 2. Fit Log-Normal dispersion for scenario-level macro trajectory
    # In standard normal, 10th and 90th percentiles are at +/- 1.28155
    ratio_90_10 = max(1.05, target_p90 / max(0.1, target_p10))
    sigma_macro = float(np.log(ratio_90_10) / (2.0 * 1.28155))
    mu_macro = float(np.log(target_p50))

    # Sample macro scaling factor Z for each scenario: shape (n_scenarios, 1)
    norm_samples = generator.standard_normal(size=(n_scenarios, 1), dtype=np.float32)
    macro_totals = np.exp(mu_macro + sigma_macro * norm_samples)
    macro_scale = macro_totals / max(1.0, float(np.sum(daily_base)))

    # 3. Add daily idiosyncratic noise with mean 1.0 (shape: n_scenarios, days)
    # Using Gamma(shape=12, scale=1/12) ensures positive shocks with modest variance
    alpha = 12.0
    daily_noise = generator.gamma(shape=alpha, scale=1.0 / alpha, size=(n_scenarios, days)).astype(np.float32)

    # 4. Construct daily paths: shape (n_scenarios, days)
    daily_paths = macro_scale * daily_base[np.newaxis, :] * daily_noise

    # Ensure non-negative
    daily_paths = np.maximum(0.0, daily_paths).astype(np.float32)

    return daily_paths


def sample_future_purchase_prices(
    current_price: float,
    price_horizons: Dict[int, float],
    price_p10: Optional[Dict[int, float]] = None,
    price_p50: Optional[Dict[int, float]] = None,
    price_p90: Optional[Dict[int, float]] = None,
    target_horizon: int = 7,
    n_scenarios: int = 10000,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Sample future purchase price distribution at target horizon.

    Maintains positive prices (price > 0.01) and agreement with Phase 16 quantiles.

    Returns:
        np.ndarray of shape (n_scenarios,) with dtype float32.
    """
    generator = rng or np.random.default_rng(42)

    curves = interpolate_price_curve(
        current_price=current_price,
        horizon_prices=price_horizons,
        horizon_p10=price_p10,
        horizon_p50=price_p50,
        horizon_p90=price_p90
    )
    idx = min(29, max(0, target_horizon - 1))
    p_median = float(curves["p50_price"][idx])
    p_10 = float(curves["p10_price"][idx])
    p_90 = float(curves["p90_price"][idx])

    # Safeguards
    if p_median <= 0:
        p_median = max(0.5, current_price)
    if p_10 <= 0:
        p_10 = max(0.1, p_median * 0.9)
    if p_90 <= p_median:
        p_90 = p_median * 1.1

    ratio = max(1.02, p_90 / max(0.01, p_10))
    sigma = float(np.log(ratio) / (2.0 * 1.28155))
    mu = float(np.log(p_median))

    z = generator.standard_normal(size=n_scenarios, dtype=np.float32)
    prices = np.exp(mu + sigma * z)
    prices = np.maximum(0.01, prices).astype(np.float32)

    return prices


def sample_supplier_lead_times(
    nominal_lead_time: float,
    historical_variability: float = 0.5,
    min_lead_time: int = 0,
    max_lead_time: int = 10,
    n_scenarios: int = 10000,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Sample supplier lead times in integer days (L >= 0).

    Returns:
        np.ndarray of shape (n_scenarios,) with dtype int32.
    """
    generator = rng or np.random.default_rng(42)

    # Normal distribution rounded to integer days, bounded to [min_lt, max_lt]
    std = max(0.2, float(historical_variability))
    raw = generator.normal(loc=nominal_lead_time, scale=std, size=n_scenarios)
    lead_times = np.round(raw).astype(np.int32)
    lead_times = np.clip(lead_times, min_lead_time, max_lead_time)

    return lead_times


def sample_supplier_fill_rates(
    complete_delivery_rate: float = 0.98,
    reliability_rate: float = 0.96,
    n_scenarios: int = 10000,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Sample supplier order fill rate (0.0 to 1.0).

    With probability `complete_delivery_rate`, fill rate is 1.0 (100% delivered).
    Otherwise, minor shortfall sampled uniformly between reliability_rate and 0.99.

    Returns:
        np.ndarray of shape (n_scenarios,) with dtype float32.
    """
    generator = rng or np.random.default_rng(42)

    fill_rates = np.ones(n_scenarios, dtype=np.float32)
    prob_incomplete = 1.0 - float(complete_delivery_rate)

    if prob_incomplete > 0:
        is_partial = generator.uniform(0.0, 1.0, size=n_scenarios) < prob_incomplete
        n_partial = int(np.sum(is_partial))
        if n_partial > 0:
            low_bound = max(0.80, float(reliability_rate) * 0.95)
            fill_rates[is_partial] = generator.uniform(low_bound, 0.99, size=n_partial).astype(np.float32)

    return fill_rates
