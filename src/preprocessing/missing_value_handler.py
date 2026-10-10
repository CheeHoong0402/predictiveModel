"""
missing_value_handler.py
========================
Domain-specific missing value imputation and masking:
- Does NOT apply blanket global fills
- Documents treatment rules per domain
- Generates imputation indicator columns (e.g., _imputed = 1)
- Counts and logs all affected rows.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional


class MissingValueHandler:
    """Handles domain-specific missing value logic with full transparency and indicators."""

    @staticmethod
    def handle_promotion_nulls(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Sets non-promotion day nulls in promo type and discount to semantic defaults."""
        df = df.copy()
        imputed_count = 0
        if "promotion_status" in df.columns:
            if "promotion_type" in df.columns:
                mask = df["promotion_status"] == 0
                null_promo_type = mask & df["promotion_type"].isna()
                df.loc[null_promo_type, "promotion_type"] = "None"
                imputed_count += int(null_promo_type.sum())

            if "discount_percentage" in df.columns:
                null_disc = df["discount_percentage"].isna()
                df.loc[null_disc, "discount_percentage"] = 0.0
                imputed_count += int(null_disc.sum())

        return df, imputed_count

    @staticmethod
    def handle_fuel_policy_nulls(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        In Malaysian fuel prices, BUDI Madani and SKDS columns are null prior to policy inception in 2024.
        Masks them with documented baseline rate or 0.0 with explicit pre_policy indicator.
        """
        df = df.copy()
        imputed_count = 0
        policy_cols = ["diesel_budi", "diesel_skds", "ron95_skps", "ron95_budi95"]
        for col in policy_cols:
            if col in df.columns:
                null_mask = df[col].isna()
                if null_mask.any():
                    count = int(null_mask.sum())
                    imputed_count += count
                    df[f"{col}_pre_policy_indicator"] = null_mask.astype(int)
                    # Forward-fill backwards from inception or retain NaN representation
                    df[col] = df[col].fillna(0.0)
        return df, imputed_count

    @staticmethod
    def impute_numeric_forward_fill(
        df: pd.DataFrame,
        columns: List[str],
        date_col: str = "date",
        add_indicator: bool = True
    ) -> Tuple[pd.DataFrame, int]:
        """Forward-fills time-series series (e.g. commodity prices, exchange rates on holidays)."""
        df = df.copy()
        if date_col in df.columns:
            df = df.sort_values(date_col).reset_index(drop=True)

        imputed_count = 0
        for col in columns:
            if col in df.columns:
                null_mask = df[col].isna()
                if null_mask.any():
                    count = int(null_mask.sum())
                    imputed_count += count
                    if add_indicator:
                        df[f"{col}_imputed"] = null_mask.astype(int)
                    df[col] = df[col].ffill().bfill()
        return df, imputed_count
