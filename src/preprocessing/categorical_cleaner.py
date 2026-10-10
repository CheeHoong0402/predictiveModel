"""
categorical_cleaner.py
======================
Standardizes categorical variables, boolean flags, and Malaysian geographic data.
Maps states canonically (e.g., 'Penang' -> 'Pulau Pinang', 'Kuala Lumpur' -> 'W.P. Kuala Lumpur').
Standardizes booleans to binary 1 and 0.
"""

import pandas as pd
from typing import Dict, List, Tuple


class CategoricalCleaner:
    """Cleans categorical labels, geographic names, and boolean representations."""

    STATE_MAPPINGS = {
        "penang": "Pulau Pinang",
        "pulau pinang": "Pulau Pinang",
        "kuala lumpur": "W.P. Kuala Lumpur",
        "w.p. kuala lumpur": "W.P. Kuala Lumpur",
        "wp kuala lumpur": "W.P. Kuala Lumpur",
        "wilayah persekutuan kuala lumpur": "W.P. Kuala Lumpur",
        "kl": "W.P. Kuala Lumpur",
        "selangor": "Selangor",
        "johor": "Johor",
        "perak": "Perak",
        "pahang": "Pahang",
        "sabah": "Sabah",
        "sarawak": "Sarawak",
        "kedah": "Kedah",
        "melaka": "Melaka",
        "malacca": "Melaka",
        "negeri sembilan": "Negeri Sembilan",
        "terengganu": "Terengganu",
        "kelantan": "Kelantan",
        "perlis": "Perlis",
        "labuan": "W.P. Labuan",
        "w.p. labuan": "W.P. Labuan",
        "putrajaya": "W.P. Putrajaya",
        "w.p. putrajaya": "W.P. Putrajaya",
    }

    TRUTHY_VALUES = {"yes", "y", "true", "1", 1, True}
    FALSY_VALUES = {"no", "n", "false", "0", 0, False}

    def standardize_booleans(self, df: pd.DataFrame, bool_columns: List[str]) -> Tuple[pd.DataFrame, int]:
        """Converts diverse boolean representations (Yes/No, True/False) to canonical 1 and 0."""
        df = df.copy()
        corrected_count = 0
        for col in bool_columns:
            if col in df.columns:
                def map_bool(val):
                    if pd.isna(val) or val is None:
                        return 0
                    if str(val).strip().lower() in self.TRUTHY_VALUES:
                        return 1
                    elif str(val).strip().lower() in self.FALSY_VALUES:
                        return 0
                    return 0

                orig = df[col].copy()
                df[col] = df[col].apply(map_bool).astype(int)
                changed = (orig.astype(str) != df[col].astype(str)) & orig.notnull()
                corrected_count += int(changed.sum())
        return df, corrected_count

    def standardize_states(self, df: pd.DataFrame, state_columns: List[str] = None) -> Tuple[pd.DataFrame, int]:
        """Standardizes Malaysian state names using canonical naming dictionary."""
        df = df.copy()
        target_cols = state_columns or [c for c in df.columns if "state" in c.lower()]
        corrected_count = 0

        for col in target_cols:
            if col in df.columns:
                def map_state(val):
                    if pd.isna(val) or val is None:
                        return None
                    s = str(val).strip().lower()
                    return self.STATE_MAPPINGS.get(s, str(val).strip())

                orig = df[col].copy()
                df[col] = df[col].apply(map_state)
                changed = (orig != df[col]) & orig.notnull()
                corrected_count += int(changed.sum())

        return df, corrected_count

    def clean_text_categories(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """Strips whitespace and standardizes formatting on general categorical text."""
        df = df.copy()
        for col in text_columns:
            if col in df.columns and df[col].dtype == "object":
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace({"nan": None, "None": None, "null": None, "": None})
        return df
