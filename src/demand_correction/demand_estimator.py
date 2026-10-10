"""
src/demand_correction/demand_estimator.py
=========================================
Estimates true customer demand for censored inventory observations:
- Robust central estimate: median of comparable unconstrained observations
- Percentile lower and upper bounds (25th and 75th percentiles)
- Available-fraction extrapolation for partial-day stockouts
- Ensures estimated true demand is never below observed sales
- Ensures estimated lost sales are never negative: max(true_demand - observed_sales, 0)
"""

import numpy as np
from typing import Dict, Any, Tuple


class DemandEstimator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        est_cfg = config.get("estimation_rules", {})
        self.lower_pct = float(est_cfg.get("lower_bound_percentile", 0.25))
        self.upper_pct = float(est_cfg.get("upper_bound_percentile", 0.75))
        self.min_avail_frac = float(est_cfg.get("min_available_fraction_for_extrapolation", 0.20))
        self.max_ratio = float(est_cfg.get("max_correction_ratio_over_observed", 3.5))

    def estimate_demand(
        self,
        observed_sales: float,
        is_censored: bool,
        is_partial: bool,
        stockout_fraction_of_day: float,
        ref_sales: np.ndarray,
        ref_count: int,
        method: str
    ) -> Dict[str, Any]:
        """
        Calculates estimated true demand, bounds, and lost sales for a single observation.
        """
        # Case 1: Unconstrained
        if not is_censored:
            return {
                "estimated_true_demand": float(observed_sales),
                "lower_bound": float(observed_sales),
                "upper_bound": float(observed_sales),
                "estimated_lost_sales": 0.0,
                "correction_applied": 0,
                "correction_method": "UNCONSTRAINED"
            }

        # Case 2: Censored but no reference observations found (e.g., initial days)
        if ref_count == 0 or len(ref_sales) == 0:
            return {
                "estimated_true_demand": float(observed_sales),
                "lower_bound": float(observed_sales),
                "upper_bound": float(observed_sales),
                "estimated_lost_sales": 0.0,
                "correction_applied": 0,
                "correction_method": "INSUFFICIENT_HISTORY"
            }

        # Case 3: Censored with valid reference observations
        med_ref = float(np.median(ref_sales))
        q_low = float(np.percentile(ref_sales, self.lower_pct * 100))
        q_high = float(np.percentile(ref_sales, self.upper_pct * 100))

        if is_partial and stockout_fraction_of_day > 0:
            # Partial stockout: available fraction extrapolation
            avail_frac = max(1.0 - stockout_fraction_of_day, self.min_avail_frac)
            extrapolated = observed_sales / avail_frac
            # Bounded by reference median and reasonable upper cap
            est_demand = max(extrapolated, med_ref, observed_sales)
            # Guard against extreme partial multiplier if observed sales had high burst
            if observed_sales > 0 and (est_demand / observed_sales) > self.max_ratio:
                est_demand = max(med_ref, observed_sales * self.max_ratio)
            method_used = f"{method}_PARTIAL_EXTRAPOLATION"
        else:
            # Full day stockout: take reference median
            est_demand = max(med_ref, observed_sales)
            method_used = f"{method}_FULL_MEDIAN"

        # Bounds
        lower_bound = max(q_low, observed_sales)
        upper_bound = max(q_high, est_demand)

        # Lost sales: max(estimated - observed, 0)
        lost_sales = max(est_demand - observed_sales, 0.0)
        correction_applied = 1 if lost_sales > 0.001 else 0

        return {
            "estimated_true_demand": float(np.round(est_demand, 2)),
            "lower_bound": float(np.round(lower_bound, 2)),
            "upper_bound": float(np.round(upper_bound, 2)),
            "estimated_lost_sales": float(np.round(lost_sales, 2)),
            "correction_applied": correction_applied,
            "correction_method": method_used
        }
