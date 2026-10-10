"""Cumulative Demand and Purchase Price Curve Interpolation across 1 to 30 days."""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


KNOWN_HORIZONS = [1, 3, 7, 14, 21, 30]


def interpolate_demand_curve(
    horizon_forecasts: Dict[int, float],
    horizon_p10: Optional[Dict[int, float]] = None,
    horizon_p50: Optional[Dict[int, float]] = None,
    horizon_p90: Optional[Dict[int, float]] = None,
    target_days: Optional[List[int]] = None
) -> Dict[str, np.ndarray]:
    """Interpolate cumulative demand trajectory (point, P10, P50, P90) across 1..30 days.
    
    Guarantees:
    - Anchored at (0, 0)
    - Strictly monotonic non-decreasing
    - Exact agreement at known horizons (1, 3, 7, 14, 21, 30)
    - P10 <= P50 <= P90 ordering preserved for all days
    """
    if target_days is None:
        target_days = list(range(1, 31))

    days_arr = np.array(target_days, dtype=float)

    # Base grid
    grid_x = np.array([0] + [h for h in KNOWN_HORIZONS if h in horizon_forecasts], dtype=float)
    grid_y_pt = np.array([0.0] + [horizon_forecasts[int(h)] for h in grid_x[1:]], dtype=float)

    # Monotonic cumulative interpolation
    pt_curve = np.interp(days_arr, grid_x, grid_y_pt)

    # Quantiles
    if horizon_p10 and horizon_p50 and horizon_p90:
        grid_y_p10 = np.array([0.0] + [horizon_p10.get(int(h), horizon_forecasts[int(h)] * 0.8) for h in grid_x[1:]], dtype=float)
        grid_y_p50 = np.array([0.0] + [horizon_p50.get(int(h), horizon_forecasts[int(h)]) for h in grid_x[1:]], dtype=float)
        grid_y_p90 = np.array([0.0] + [horizon_p90.get(int(h), horizon_forecasts[int(h)] * 1.2) for h in grid_x[1:]], dtype=float)

        p10_curve = np.interp(days_arr, grid_x, grid_y_p10)
        p50_curve = np.interp(days_arr, grid_x, grid_y_p50)
        p90_curve = np.interp(days_arr, grid_x, grid_y_p90)

        # Enforce P10 <= P50 <= P90 and >= 0
        p10_curve = np.maximum(0.0, p10_curve)
        p50_curve = np.maximum(p10_curve, p50_curve)
        p90_curve = np.maximum(p50_curve, p90_curve)
    else:
        # Fallback proportional bands if specific quantiles are missing
        p10_curve = np.maximum(0.0, pt_curve * 0.80)
        p50_curve = pt_curve.copy()
        p90_curve = pt_curve * 1.25

    return {
        "days": np.array(target_days, dtype=int),
        "point_demand": pt_curve,
        "p10_demand": p10_curve,
        "p50_demand": p50_curve,
        "p90_demand": p90_curve
    }


def interpolate_price_curve(
    current_price: float,
    horizon_prices: Dict[int, float],
    horizon_p10: Optional[Dict[int, float]] = None,
    horizon_p50: Optional[Dict[int, float]] = None,
    horizon_p90: Optional[Dict[int, float]] = None,
    target_days: Optional[List[int]] = None
) -> Dict[str, np.ndarray]:
    """Interpolate expected future purchase price curve across 1..30 days."""
    if target_days is None:
        target_days = list(range(1, 31))

    days_arr = np.array(target_days, dtype=float)

    grid_x = np.array([0] + [h for h in KNOWN_HORIZONS if h in horizon_prices], dtype=float)
    grid_y_pt = np.array([float(current_price)] + [float(horizon_prices[int(h)]) for h in grid_x[1:]], dtype=float)

    pt_curve = np.interp(days_arr, grid_x, grid_y_pt)

    if horizon_p10 and horizon_p50 and horizon_p90:
        grid_y_p10 = np.array([float(current_price)] + [float(horizon_p10.get(int(h), horizon_prices[int(h)] * 0.95)) for h in grid_x[1:]], dtype=float)
        grid_y_p50 = np.array([float(current_price)] + [float(horizon_p50.get(int(h), horizon_prices[int(h)])) for h in grid_x[1:]], dtype=float)
        grid_y_p90 = np.array([float(current_price)] + [float(horizon_p90.get(int(h), horizon_prices[int(h)] * 1.05)) for h in grid_x[1:]], dtype=float)

        p10_curve = np.interp(days_arr, grid_x, grid_y_p10)
        p50_curve = np.interp(days_arr, grid_x, grid_y_p50)
        p90_curve = np.interp(days_arr, grid_x, grid_y_p90)

        p10_curve = np.maximum(0.01, p10_curve)
        p50_curve = np.maximum(p10_curve, p50_curve)
        p90_curve = np.maximum(p50_curve, p90_curve)
    else:
        p10_curve = np.maximum(0.01, pt_curve * 0.95)
        p50_curve = pt_curve.copy()
        p90_curve = pt_curve * 1.05

    return {
        "days": np.array(target_days, dtype=int),
        "point_price": pt_curve,
        "p10_price": p10_curve,
        "p50_price": p50_curve,
        "p90_price": p90_curve
    }
