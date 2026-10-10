"""
outlier_handler.py
==================
Detects and flags extreme values without destroying genuine structural shocks
(such as festival demand surges, subsidy changes, and price floating).
Categorizes outliers into:
1. Clearly invalid (e.g., negative prices, unphysical quantities)
2. Suspicious but plausible (flagged with indicator)
3. Plausible extreme economic events (preserved for predictive value)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any


class OutlierHandler:
    """Flags outliers while preserving valid historical structural shifts."""

    @staticmethod
    def flag_iqr_outliers(
        df: pd.DataFrame,
        numeric_columns: List[str],
        multiplier: float = 3.0
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Flags extreme statistical outliers exceeding Q3 + multiplier*IQR or below Q1 - multiplier*IQR.
        Appends a binary flag column `<col>_outlier_flag` and preserves original value.
        """
        df = df.copy()
        flagged_counts = {}
        for col in numeric_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].dropna()
                if len(series) < 10:
                    continue
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - (multiplier * iqr)
                upper = q3 + (multiplier * iqr)

                outlier_mask = (df[col] < lower) | (df[col] > upper)
                count = int(outlier_mask.sum())
                flagged_counts[col] = count
                df[f"{col}_extreme_outlier_flag"] = outlier_mask.astype(int)

        return df, flagged_counts
