"""
date_cleaner.py
===============
Validates, standardizes, and parses date and timestamp fields into canonical YYYY-MM-DD.
Enforces chronological constraints and logs invalid dates without silent coercion.
"""

import pandas as pd
from typing import Tuple, List, Optional
from datetime import datetime


class DateCleaner:
    """Standardizes date columns, verifies chronological integrity, and captures failures."""

    def __init__(self, daily_format: str = "%Y-%m-%d"):
        self.daily_format = daily_format

    def parse_and_standardize_dates(
        self,
        df: pd.DataFrame,
        date_columns: List[str],
        allow_null: bool = True
    ) -> Tuple[pd.DataFrame, List[dict]]:
        """
        Parses target date columns to datetime and formats as canonical YYYY-MM-DD strings.
        Returns cleaned dataframe and list of any invalid date failure records.
        """
        df = df.copy()
        failures = []

        for col in date_columns:
            if col not in df.columns:
                continue

            # Original values
            raw_series = df[col]

            # Attempt safe parsing with mixed format support
            parsed = pd.to_datetime(raw_series, errors="coerce", format="mixed")

            # Check invalid dates
            for idx, (raw, dt) in enumerate(zip(raw_series, parsed)):
                raw_str = str(raw).strip() if pd.notna(raw) else ""
                if pd.isna(dt) and raw_str != "" and raw_str.lower() not in ["none", "nan", "nat", "null", ""]:
                    failures.append({
                        "column": col,
                        "row_index": idx,
                        "raw_value": raw,
                        "error": "Unparseable date format"
                    })
                elif not pd.isna(dt):
                    if dt.year < 1980 or dt.year > 2035:
                        failures.append({
                            "column": col,
                            "row_index": idx,
                            "raw_value": raw,
                            "error": f"Year {dt.year} outside valid range [1980, 2035]"
                        })

            # Format to canonical YYYY-MM-DD string, preserving None for genuine nulls
            df[col] = pd.Series(
                [d.strftime(self.daily_format) if pd.notna(d) else None for d in parsed],
                index=df.index,
                dtype=object
            )

        return df, failures

    def validate_chronology(
        self,
        df: pd.DataFrame,
        earlier_col: str,
        later_col: str
    ) -> Tuple[pd.DataFrame, int]:
        """
        Checks that earlier_col <= later_col.
        Returns dataframe with chronology_inversion_flag and count of inverted rows.
        """
        df = df.copy()
        if earlier_col not in df.columns or later_col not in df.columns:
            return df, 0

        d_early = pd.to_datetime(df[earlier_col], errors="coerce")
        d_late = pd.to_datetime(df[later_col], errors="coerce")

        inversion = (d_early > d_late) & d_early.notnull() & d_late.notnull()
        inversion_count = int(inversion.sum())

        df[f"{earlier_col}_{later_col}_chronology_valid"] = (~inversion).astype(int)
        return df, inversion_count
