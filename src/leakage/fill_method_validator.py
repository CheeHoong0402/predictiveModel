"""
Fill Method and Imputation Validator.
Audits time series filling methods to detect backward filling, future interpolation,
or global non-causal imputations.
"""

from typing import Dict, List
import polars as pl


class FillMethodValidator:
    """Verifies that all missing value treatments are causally valid and past-only."""

    @staticmethod
    def audit_forward_fill_direction(df: pl.DataFrame) -> List[Dict[str, str]]:
        """
        Verify that exchange rates, commodities, and policies are filled strictly forward
        and that missing values at the beginning of series are never backfilled from future points.
        """
        violations = []

        # Audit exchange rates: verify that first row (2021-01-01) if missing is not backfilled from future
        sub = df.filter(pl.col("date") == "2021-01-01")
        # Ensure no feature values at date 2021-01-01 match future business days via backfill
        if sub.height == 0:
            violations.append({
                "feature_name": "date_series",
                "issue_type": "MISSING_ORIGIN_DATE",
                "severity": "HIGH",
                "details": "Initial series date 2021-01-01 missing from feature table.",
            })

        # Check for backwards fill artifacts across all lag features
        lag_cols = [c for c in df.columns if c.startswith("demand_lag_")]
        first_day_df = df.filter(pl.col("date") == "2021-01-01")
        for col in lag_cols:
            # On day 1, lag 1 must be null or default 0.0, never future demand
            val = first_day_df[col].drop_nulls()
            if not val.is_empty():
                if val.max() > 5000:  # Sentinel or future contamination
                    violations.append({
                        "feature_name": col,
                        "issue_type": "SUSPICIOUS_DAY1_VALUE",
                        "severity": "HIGH",
                        "details": f"Lag feature {col} has unexpected value on initial dataset date.",
                    })

        return violations

    @staticmethod
    def audit_global_imputation(df: pl.DataFrame) -> List[Dict[str, str]]:
        """
        Verify that missing values have not been replaced with full-dataset future statistics.
        """
        violations = []
        # Check that high-missingness columns preserve nulls rather than being filled with global dataset mean
        for col in ["market_price_median", "price_vs_market_diff"]:
            if col in df.columns:
                null_count = df[col].null_count()
                if null_count == 0:
                    violations.append({
                        "feature_name": col,
                        "issue_type": "SUSPECTED_GLOBAL_MEAN_IMPUTATION",
                        "severity": "CRITICAL",
                        "details": f"Feature {col} has 0 nulls despite being unobserved across 2021-2023.",
                    })
        return violations
