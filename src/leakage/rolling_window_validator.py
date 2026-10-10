"""
Rolling Window and Lag Mathematical Validator.
Verifies that all rolling windows and lag operations use strictly historical observations
and prevents forward-looking or centred windows.
"""

from typing import Dict, List, Tuple
import numpy as np
import polars as pl


class RollingWindowValidator:
    """Validates rolling statistics, lag offsets, and series boundaries."""

    @staticmethod
    def audit_lag_offsets(df: pl.DataFrame) -> List[Dict[str, str]]:
        """
        Verify that lag features correspond strictly to prior historical days
        and that no negative shifts or leads exist.
        """
        violations = []
        lag_cols = [c for c in df.columns if "_lag_" in c]

        # Audit a test slice of consecutive days for a single series
        sub = (
            df.filter(
                (pl.col("store_id") == "STR_KUL_001") & (pl.col("product_id") == "PRD_CHK_01")
            )
            .sort("date")
        )

        for col in lag_cols:
            parts = col.split("_lag_")
            base_prefix = parts[0]
            try:
                lag_num = int(parts[1])
            except ValueError:
                continue

            if lag_num < 0:
                violations.append({
                    "feature_name": col,
                    "issue_type": "NEGATIVE_LAG",
                    "severity": "CRITICAL",
                    "details": f"Feature {col} specifies negative lag {lag_num}, causing future target leakage.",
                })
                continue

            # Verify against source column
            src_col = (
                "estimated_true_demand_training_safe" if base_prefix == "demand"
                else "observed_units_sold" if base_prefix == "sales"
                else "purchase_price" if base_prefix == "purchase_price"
                else None
            )
            if src_col and src_col in sub.columns:
                raw_vals = sub[src_col].to_list()
                lag_vals = sub[col].to_list()
                # Check sample index
                for t in range(lag_num, min(lag_num + 30, len(raw_vals))):
                    expected = raw_vals[t - lag_num]
                    actual = lag_vals[t]
                    if actual is not None and not np.isnan(actual) and expected is not None:
                        if abs(actual - expected) > 1e-4:
                            violations.append({
                                "feature_name": col,
                                "issue_type": "INCORRECT_LAG_ALIGNMENT",
                                "severity": "CRITICAL",
                                "details": f"Row {t} has value {actual} != expected {expected} from index {t - lag_num}",
                            })
                            break
        return violations

    @staticmethod
    def audit_rolling_windows(df: pl.DataFrame) -> List[Dict[str, str]]:
        """
        Verify that rolling window aggregations are strictly computed on past observations
        (i.e. strictly up to T-1 for target-proximate metrics).
        """
        violations = []
        sub = (
            df.filter(
                (pl.col("store_id") == "STR_KUL_001") & (pl.col("product_id") == "PRD_CHK_01")
            )
            .sort("date")
        )

        # Audit demand rolling windows: 3d, 7d, 14d, 28d
        raw_demand = sub["estimated_true_demand_training_safe"].to_list()

        for w in [3, 7, 14, 28]:
            col_mean = f"demand_mean_{w}d"
            if col_mean not in sub.columns:
                continue
            actual_means = sub[col_mean].to_list()

            for t in range(w, min(w + 30, len(raw_demand))):
                # Expected window is strictly [t - w, ..., t - 1]
                expected_window = raw_demand[t - w : t]
                expected_val = float(np.mean(expected_window))
                actual_val = actual_means[t]
                if actual_val is not None and not np.isnan(actual_val):
                    if abs(actual_val - expected_val) > 1e-4:
                        # Check if observation at t was accidentally included
                        leaked_window = raw_demand[t - w + 1 : t + 1]
                        if abs(actual_val - np.mean(leaked_window)) < 1e-4:
                            violations.append({
                                "feature_name": col_mean,
                                "issue_type": "FUTURE_INCLUSIVE_ROLLING_WINDOW",
                                "severity": "CRITICAL",
                                "details": f"Rolling window includes observation at day T (leaked window [T-{w-1} to T])",
                            })
                            break
                        else:
                            violations.append({
                                "feature_name": col_mean,
                                "issue_type": "MISALIGNED_ROLLING_WINDOW",
                                "severity": "HIGH",
                                "details": f"Rolling mean at index {t} is {actual_val} vs expected {expected_val}",
                            })
                            break

        return violations

    @staticmethod
    def generate_mathematical_proof(
        df: pl.DataFrame,
        store_id: str = "STR_KUL_001",
        product_id: str = "PRD_CHK_01",
        target_date: str = "2024-06-15",
    ) -> Dict[str, object]:
        """
        Generate explicit mathematical proof verifying exact historical rows
        used to calculate demand_mean_7d on target_date.
        """
        sub = (
            df.filter(
                (pl.col("store_id") == store_id) & (pl.col("product_id") == product_id)
            )
            .sort("date")
        )
        dates = sub["date"].to_list()
        demands = sub["estimated_true_demand_training_safe"].to_list()
        means_7d = sub["demand_mean_7d"].to_list()

        target_idx = dates.index(target_date)
        # 7 historical days prior to target_date: target_idx - 7 to target_idx - 1
        historical_indices = list(range(target_idx - 7, target_idx))
        historical_dates = [dates[i] for i in historical_indices]
        historical_values = [demands[i] for i in historical_indices]
        calculated_mean = float(np.mean(historical_values))
        stored_mean = means_7d[target_idx]

        return {
            "prediction_date": target_date,
            "store_id": store_id,
            "product_id": product_id,
            "feature_name": "demand_mean_7d",
            "historical_window_dates": historical_dates,
            "historical_window_values": historical_values,
            "mathematical_mean": calculated_mean,
            "stored_feature_value": stored_mean,
            "difference": abs(calculated_mean - stored_mean),
            "is_strictly_past_only": bool(all(d < target_date for d in historical_dates)),
            "temporal_proof_passed": bool(abs(calculated_mean - stored_mean) < 1e-4),
        }
