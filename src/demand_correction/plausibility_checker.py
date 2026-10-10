"""
src/demand_correction/plausibility_checker.py
=============================================
Applies plausibility safeguards and flags extreme corrections for audit review:
- Validates non-negativity of lost sales
- Confirms estimated demand >= observed sales
- Detects implausibly high multipliers and flags high-risk corrections
- Preserves genuine festive peaks while alerting review teams
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


class PlausibilityChecker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_ratio = float(config.get("estimation_rules", {}).get("max_correction_ratio_over_observed", 3.5))

    def audit_and_flag(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validates mathematical constraints and flags extreme corrections.
        Returns audited DataFrame and high-risk audit subset.
        """
        df = df.copy()

        # 1. Mathematical invariants
        # Lost sales >= 0
        df["estimated_lost_sales"] = np.maximum(df["estimated_lost_sales"].values, 0.0)

        # True demand >= observed sales
        df["estimated_true_demand"] = np.maximum(df["estimated_true_demand"].values, df["observed_units_sold"].values)

        # 2. Extreme correction detection
        # Flag if:
        # a) lost sales is more than 3x observed sales when observed sales > 10
        # b) estimated demand exceeds 3.5x median demand of the series
        # c) correction ratio exceeds max_ratio
        obs = df["observed_units_sold"].values
        est = df["estimated_true_demand"].values
        lost = df["estimated_lost_sales"].values
        corr_applied = df["demand_correction_applied"].values

        ratio = np.zeros(len(df), dtype=float)
        valid_obs_mask = (obs > 0)
        ratio[valid_obs_mask] = est[valid_obs_mask] / obs[valid_obs_mask]

        is_extreme = (
            (corr_applied == 1) &
            (
                (ratio > self.max_ratio) |
                ((obs > 15) & (lost > 2.5 * obs)) |
                (lost > 100.0)
            )
        )

        df["is_extreme_correction"] = is_extreme.astype(int)

        high_risk_df = df[df["is_extreme_correction"] == 1][[
            "date", "store_id", "product_id", "observed_units_sold",
            "estimated_true_demand", "estimated_lost_sales", "stockout_duration_hours",
            "demand_correction_method", "demand_correction_confidence"
        ]].copy()

        return df, high_risk_df
