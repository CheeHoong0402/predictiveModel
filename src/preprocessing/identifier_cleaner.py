"""
identifier_cleaner.py
=====================
Cleans and standardizes business keys and identifiers (store, product, supplier, item_code, etc.).
Ensures all identifiers are strings, strips extraneous whitespace, and strictly preserves leading zeroes.
"""

import pandas as pd
from typing import List


class IdentifierCleaner:
    """Standardizes identifier columns without dropping leading zeroes or corrupting keys."""

    def __init__(self, identifier_columns: List[str] = None):
        self.default_identifiers = [
            "store_id", "product_id", "supplier_id", "promotion_id",
            "item_code", "premise_code", "purchase_order_id"
        ]
        self.identifier_columns = identifier_columns or self.default_identifiers

    def clean_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if col in self.identifier_columns or any(k in col.lower() for k in ["_id", "item_code", "premise_code"]):
                # Convert to string without truncating leading zeroes
                # For floats (e.g. 123.0 from CSV reader), convert to integer string first if whole number
                def format_id(val):
                    if pd.isna(val) or val is None or str(val).strip().lower() in ["nan", "none", "null", ""]:
                        return None
                    if isinstance(val, float) and val.is_integer():
                        return str(int(val)).strip()
                    return str(val).strip()

                df[col] = df[col].apply(format_id)

        return df
