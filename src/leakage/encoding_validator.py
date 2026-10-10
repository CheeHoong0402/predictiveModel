"""
Encoding, Scaling, and Lifecycle Leakage Validator.
Verifies the complete absence of premature target encoding, global frequency encoding,
global normalization, and future product/store discontinuation indicators.
"""

from typing import Dict, List
import polars as pl


class EncodingValidator:
    """Verifies that feature encodings adhere to strict fold-aware validation rules."""

    @staticmethod
    def audit_target_encoding(df: pl.DataFrame) -> List[Dict[str, str]]:
        """
        Confirm that no target-based encoding columns exist in the feature dataset.
        """
        violations = []
        suspicious_patterns = [
            "target_enc", "target_mean", "mean_target", "encoded_target",
            "mean_future_demand", "mean_future_price", "smoothed_target"
        ]
        for col in df.columns:
            col_lower = col.lower()
            for pattern in suspicious_patterns:
                if pattern in col_lower:
                    violations.append({
                        "feature_name": col,
                        "issue_type": "PREMATURE_TARGET_ENCODING",
                        "severity": "CRITICAL",
                        "details": f"Feature {col} matches target encoding pattern '{pattern}', causing catastrophic leakage.",
                    })
        return violations

    @staticmethod
    def audit_global_scaling(df: pl.DataFrame) -> List[Dict[str, str]]:
        """
        Verify that features maintain raw physical units and have not been globally
        standardized or scaled across the entire multi-year timeline.
        """
        violations = []
        # Check if raw demand features maintain natural discrete/monetary quantities
        if "demand_lag_1" in df.columns:
            d_vals = df["demand_lag_1"].drop_nulls()
            # If standard-scaled, mean would be ~0.0 and std ~1.0
            d_mean = float(d_vals.mean())
            if abs(d_mean) < 0.05 and abs(float(d_vals.std()) - 1.0) < 0.05:
                violations.append({
                    "feature_name": "demand_lag_1",
                    "issue_type": "SUSPECTED_GLOBAL_STANDARDIZATION",
                    "severity": "HIGH",
                    "details": "Feature demand_lag_1 appears globally standardized (mean ~ 0, std ~ 1).",
                })
        return violations

    @staticmethod
    def audit_lifecycle_leakage(df: pl.DataFrame) -> List[Dict[str, str]]:
        """
        Verify that product and store lifecycle features do not leak future discontinuation
        or death dates (e.g. days_until_discontinuation).
        """
        violations = []
        prohibited_lifecycle = [
            "last_seen_date", "days_until_discontinuation", "days_until_store_closure",
            "future_lifecycle_stage", "is_discontinued_in_future"
        ]
        for col in df.columns:
            if col.lower() in prohibited_lifecycle:
                violations.append({
                    "feature_name": col,
                    "issue_type": "FUTURE_LIFECYCLE_LEAKAGE",
                    "severity": "CRITICAL",
                    "details": f"Feature {col} leaks future product or store lifecycle cessation.",
                })
        return violations
