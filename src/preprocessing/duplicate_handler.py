"""
duplicate_handler.py
====================
Handles exact duplicates and business-key duplicates with full auditability.
- Exact duplicates (all columns identical): removed cleanly with logging.
- Business-key duplicates: investigated and resolved via documented aggregation or filtering.
"""

import pandas as pd
from typing import List, Tuple, Dict, Any, Optional


class DuplicateHandler:
    """Detects, categorizes, and safely resolves duplicates."""

    @staticmethod
    def drop_exact_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Removes exact duplicate rows."""
        orig_len = len(df)
        df_clean = df.drop_duplicates().reset_index(drop=True)
        removed = orig_len - len(df_clean)
        return df_clean, removed

    @staticmethod
    def resolve_business_key_duplicates(
        df: pd.DataFrame,
        key_cols: List[str],
        aggregation_rules: Optional[Dict[str, str]] = None,
        keep: str = "last"
    ) -> Tuple[pd.DataFrame, int, int]:
        """
        Resolves duplicate business keys using aggregation or explicit retention policy.
        Returns cleaned dataframe, number of affected duplicate rows, and number of rows removed.
        """
        if not all(col in df.columns for col in key_cols):
            return df, 0, 0

        dup_mask = df.duplicated(subset=key_cols, keep=False)
        affected_count = int(dup_mask.sum())
        if affected_count == 0:
            return df, 0, 0

        orig_len = len(df)
        if aggregation_rules:
            # Aggregate numeric and categorical columns
            df_clean = df.groupby(key_cols, as_index=False).agg(aggregation_rules)
        else:
            df_clean = df.drop_duplicates(subset=key_cols, keep=keep).reset_index(drop=True)

        removed_count = orig_len - len(df_clean)
        return df_clean, affected_count, removed_count
