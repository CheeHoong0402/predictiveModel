"""
schema_standardizer.py
======================
Standardizes DataFrame column names into canonical snake_case format,
strips whitespace, removes special characters, and validates schema integrity.
"""

import re
import pandas as pd
from typing import List, Dict, Any


class SchemaStandardizer:
    """Enforces consistent column naming and schema standards."""

    @staticmethod
    def to_snake_case(col_name: str) -> str:
        """Converts arbitrary column strings to clean snake_case."""
        name = str(col_name).strip()
        # Replace non-alphanumeric (except underscore) with underscore
        name = re.sub(r"[^\w\s]", "_", name)
        # Handle camelCase to snake_case
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        # Replace multiple spaces/underscores with single underscore
        name = re.sub(r"[\s_]+", "_", name)
        return name.lower().strip("_")

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies snake_case standardization to all columns in a dataframe."""
        df = df.copy()
        new_cols = {col: self.to_snake_case(col) for col in df.columns}
        df = df.rename(columns=new_cols)
        return df
