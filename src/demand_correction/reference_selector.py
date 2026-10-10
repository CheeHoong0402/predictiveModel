"""
src/demand_correction/reference_selector.py
===========================================
Selects comparable unconstrained historical reference observations:
- Strict past-only filter for training safety (ref_date < date_T)
- Only unconstrained observations (censored_demand_flag == 0)
- Fallback hierarchy across weekday, promotion status, and price tolerance
- Supports both training-safe (past-only) and retrospective (symmetric) reference sets
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional


class ReferenceSelector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        ref_cfg = config.get("reference_selection", {})
        self.preferred_window_days = int(ref_cfg.get("preferred_historical_window_days", 84))
        self.max_window_days = int(ref_cfg.get("maximum_historical_window_days", 182))
        self.min_observations = int(ref_cfg.get("minimum_reference_observations", 4))
        self.price_tolerance = float(ref_cfg.get("price_tolerance_pct", 0.10))

    def select_references_for_series(
        self,
        series_df: pd.DataFrame,
        past_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fast series-level reference selector for a single (store_id, product_id) series.
        series_df is sorted by date ascending.
        Returns a list of dicts with reference sales and metadata for each observation in series_df.
        """
        n = len(series_df)
        dates = pd.to_datetime(series_df["date"]).values
        day_of_weeks = pd.to_datetime(series_df["date"]).dt.dayofweek.values
        censored = series_df["censored_demand_flag"].values
        sales = series_df["observed_units_sold"].values
        promos = series_df["promotion_active"].values
        prices = series_df["selling_price"].values

        # Mask of strictly unconstrained candidate observations
        unconstrained_mask = (censored == 0)

        results = []

        # Convert dates to integer day counts for rapid diffs
        date_days = (dates - dates[0]) / np.timedelta64(1, "D")

        for i in range(n):
            if not censored[i]:
                # Unconstrained row needs no reference search
                results.append({
                    "ref_sales": np.array([sales[i]]),
                    "ref_count": 0,
                    "method": "UNCONSTRAINED",
                    "window_used": "NONE"
                })
                continue

            current_day = date_days[i]
            current_dow = day_of_weeks[i]
            current_promo = promos[i]
            current_price = prices[i]

            # Candidate time pool
            if past_only:
                time_mask = unconstrained_mask & (date_days < current_day)
            else:
                time_mask = unconstrained_mask & (date_days != current_day)

            if not np.any(time_mask):
                results.append({
                    "ref_sales": np.array([]),
                    "ref_count": 0,
                    "method": "INSUFFICIENT_HISTORY",
                    "window_used": "NONE"
                })
                continue

            # Level 1: Same DOW, preferred window, same promo, price band
            days_diff = np.abs(date_days - current_day)
            price_low = current_price * (1.0 - self.price_tolerance)
            price_high = current_price * (1.0 + self.price_tolerance)

            l1_mask = (
                time_mask &
                (day_of_weeks == current_dow) &
                (days_diff <= self.preferred_window_days) &
                (promos == current_promo) &
                (prices >= price_low) &
                (prices <= price_high)
            )
            if np.sum(l1_mask) >= self.min_observations:
                results.append({
                    "ref_sales": sales[l1_mask],
                    "ref_count": int(np.sum(l1_mask)),
                    "method": "L1_RECENT_DOW_PRICE_PROMO",
                    "window_used": f"{self.preferred_window_days}d"
                })
                continue

            # Level 2: Same DOW, max window, same promo
            l2_mask = (
                time_mask &
                (day_of_weeks == current_dow) &
                (days_diff <= self.max_window_days) &
                (promos == current_promo)
            )
            if np.sum(l2_mask) >= self.min_observations:
                results.append({
                    "ref_sales": sales[l2_mask],
                    "ref_count": int(np.sum(l2_mask)),
                    "method": "L2_EXTENDED_DOW_PROMO",
                    "window_used": f"{self.max_window_days}d"
                })
                continue

            # Level 3: Same DOW, all past unconstrained history, same promo
            l3_mask = time_mask & (day_of_weeks == current_dow) & (promos == current_promo)
            if np.sum(l3_mask) >= self.min_observations:
                results.append({
                    "ref_sales": sales[l3_mask],
                    "ref_count": int(np.sum(l3_mask)),
                    "method": "L3_ALL_TIME_DOW_PROMO",
                    "window_used": "all_time"
                })
                continue

            # Level 4: All weekdays, preferred window, same promo
            l4_mask = time_mask & (days_diff <= self.preferred_window_days) & (promos == current_promo)
            if np.sum(l4_mask) >= self.min_observations:
                results.append({
                    "ref_sales": sales[l4_mask],
                    "ref_count": int(np.sum(l4_mask)),
                    "method": "L4_ALL_WEEKDAYS_RECENT_PROMO",
                    "window_used": f"{self.preferred_window_days}d"
                })
                continue

            # Level 5: Any past unconstrained observations
            l5_mask = time_mask
            if np.sum(l5_mask) > 0:
                results.append({
                    "ref_sales": sales[l5_mask],
                    "ref_count": int(np.sum(l5_mask)),
                    "method": "L5_FALLBACK_ALL_UNCONSTRAINED",
                    "window_used": "all_time"
                })
            else:
                results.append({
                    "ref_sales": np.array([]),
                    "ref_count": 0,
                    "method": "INSUFFICIENT_HISTORY",
                    "window_used": "NONE"
                })

        return results
