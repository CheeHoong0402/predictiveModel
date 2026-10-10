"""
numeric_cleaner.py
==================
Cleans and standardizes numerical and monetary values:
- Strips Malaysian Ringgit symbols ("RM", "MYR", ",") into float numbers
- Handles impossible negative values in non-negative columns
- Detects and replaces infinite values
- Applies numeric plausibility bounds.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any


class NumericCleaner:
    """Standardizes numeric and monetary variables."""

    @staticmethod
    def parse_monetary_value(val: Any) -> Optional[float]:
        """Parses monetary strings like 'RM 12.50', 'RM12.50', '1,250.00' to float 12.50."""
        if pd.isna(val) or val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        
        s = str(val).strip()
        # Remove currency symbols and formatting commas
        s = re.sub(r"[^\d.-]", "", s)
        try:
            return float(s)
        except ValueError:
            return None

    def clean_monetary_columns(self, df: pd.DataFrame, monetary_columns: List[str]) -> Tuple[pd.DataFrame, int]:
        df = df.copy()
        corrected_count = 0
        for col in monetary_columns:
            if col in df.columns:
                orig = df[col].copy()
                df[col] = df[col].apply(self.parse_monetary_value)
                # Count changed non-null entries
                changed = (orig.astype(str) != df[col].astype(str)) & orig.notnull()
                corrected_count += int(changed.sum())
        return df, corrected_count

    def clean_non_negative_columns(self, df: pd.DataFrame, non_negative_columns: List[str]) -> Tuple[pd.DataFrame, int]:
        """Flags or sets negative values in strictly non-negative columns to 0 or NaN with logging."""
        df = df.copy()
        corrected_count = 0
        for col in non_negative_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                neg_mask = df[col] < 0
                if neg_mask.any():
                    count = int(neg_mask.sum())
                    corrected_count += count
                    # Flag negative anomaly and clip to 0
                    df[f"{col}_negative_corrected"] = neg_mask.astype(int)
                    df.loc[neg_mask, col] = 0.0
        return df, corrected_count

    def clean_infinite_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Replaces inf and -inf with NaN."""
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = 0
        for col in num_cols:
            inf_mask = np.isinf(df[col])
            if inf_mask.any():
                inf_count += int(inf_mask.sum())
                df.loc[inf_mask, col] = np.nan
        return df, inf_count
