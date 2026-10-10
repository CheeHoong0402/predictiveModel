"""
Temporal Audit Engine and Adversarial Invariance Validator.
Provides comprehensive temporal availability classification, adversarial date-shift
sentinel testing, and truncation invariance verification.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import polars as pl

from src.features.demand_features import generate_demand_features
from src.features.price_features import generate_price_features


class AuditStatus(str, Enum):
    SAFE = "SAFE"
    SAFE_WITH_ASSUMPTION = "SAFE_WITH_ASSUMPTION"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    LEAKED = "LEAKED"
    EXCLUDED = "EXCLUDED"


class LeakageSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


FORECAST_ORIGIN_DEFINITION = {
    "convention_name": "end_of_day_T",
    "timing_timestamp": "23:59:59 Asia/Kuala_Lumpur",
    "rule": "At the conclusion of calendar day T, using only information genuinely available by that moment, predict future outcomes for horizon starting after T (T+1 onwards).",
    "lag_definition": "Lag k corresponds strictly to observation at calendar day T-k.",
    "rolling_window_definition": "Rolling window of length W is computed strictly over historical days [T-W, ..., T-1] via prior-day shift.",
}


class TemporalAuditEngine:
    """Core auditing engine enforcing temporal validity and absence of future data leakage."""

    def __init__(self, output_dir: Path = Path("reports/leakage")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.violations: List[Dict[str, str]] = []

    def audit_feature_temporal_metadata(
        self, feature_registry_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extend feature registry with complete temporal availability metadata and classifications.
        """
        temporal_records = []

        for _, row in feature_registry_df.iterrows():
            fname = row["name"]
            family = row["feature_family"]
            target_app = row.get("target_model_applicability", "both")

            # Determine temporal characteristics by family
            is_known_in_advance = family in [
                "calendar", "holiday_proximity", "festival_proximity",
                "school_holiday", "salary_cycle", "static_product", "static_store"
            ]
            is_historical_only = family in [
                "demand_history", "rolling_demand", "demand_momentum",
                "weekly_seasonality", "annual_seasonality", "observed_sales",
                "stockout_history", "replenishment_history", "purchase_price_history",
                "purchase_price_rolling", "price_momentum", "weather_rolling",
                "weather_spells"
            ]

            pub_lag = (
                21 if family == "macro_inflation" and "cpi" in fname else
                30 if family == "macro_inflation" and "ppi" in fname else
                0
            )

            # Classify audit status
            if family in ["quarantined_demand", "future_weather", "future_market", "forecast_target"]:
                status = AuditStatus.EXCLUDED.value
                risk = "critical"
            elif family in ["macro_inflation", "market_price"]:
                status = AuditStatus.SAFE_WITH_ASSUMPTION.value
                risk = "low"
            else:
                status = AuditStatus.SAFE.value
                risk = "none"

            temporal_records.append({
                "feature_name": fname,
                "feature_family": family,
                "source_dataset": row.get("source_columns", "master_daily"),
                "source_frequency": "Daily" if family not in ["macro_inflation", "school_holiday"] else "Monthly/Gazetted",
                "observation_timing": "End of Day T (23:59:59 MYT)",
                "publication_lag_days": pub_lag,
                "effective_date_logic": "Synchronous on effective date" if "active" in fname else "N/A",
                "known_in_advance_status": is_known_in_advance,
                "historical_only_status": is_historical_only,
                "potentially_future_known_status": False,
                "leakage_risk": risk,
                "target_model_applicability": target_app,
                "final_audit_status": status,
            })

        df_temporal = pd.DataFrame(temporal_records)
        df_temporal.to_csv(self.output_dir / "feature_temporal_registry.csv", index=False)
        return df_temporal

    def run_adversarial_sentinel_test(
        self,
        df: pl.DataFrame,
        cutoff_date: str = "2023-06-30",
        store_id: str = "STR_KUL_001",
        product_id: str = "PRD_CHK_01",
    ) -> Dict[str, object]:
        """
        Adversarial test: Inject extreme sentinel values (999,999,999) into future data (date > cutoff).
        Assert that all features computed for dates <= cutoff remain 100% IDENTICAL to baseline.
        """
        # Baseline slice for series
        sub_base = df.filter(
            (pl.col("store_id") == store_id) & (pl.col("product_id") == product_id)
        ).sort("date")

        # Create adversarial test copy with contaminated future
        sentinel_val = 999999999.0
        df_adv = sub_base.with_columns(
            pl.when(pl.col("date") > cutoff_date)
            .then(pl.lit(sentinel_val))
            .otherwise(pl.col("estimated_true_demand_training_safe"))
            .alias("estimated_true_demand_training_safe"),
            pl.when(pl.col("date") > cutoff_date)
            .then(pl.lit(sentinel_val))
            .otherwise(pl.col("purchase_price"))
            .alias("purchase_price"),
        )

        # Regenerate demand and price features on adversarial data
        df_adv, _ = generate_demand_features(df_adv)
        df_adv, _ = generate_price_features(df_adv)

        # Compare pre-cutoff features: must be 100% identical
        base_pre = sub_base.filter(pl.col("date") <= cutoff_date)
        adv_pre = df_adv.filter(pl.col("date") <= cutoff_date)

        test_features = ["demand_lag_1", "demand_lag_7", "demand_mean_7d", "demand_mean_28d", "purchase_price_lag_7"]
        max_discrepancies = {}
        all_passed = True

        for feat in test_features:
            if feat in base_pre.columns and feat in adv_pre.columns:
                b_vals = np.array(base_pre[feat].drop_nulls().to_list())
                a_vals = np.array(adv_pre[feat].drop_nulls().to_list())
                diff = float(np.max(np.abs(b_vals - a_vals))) if len(b_vals) > 0 else 0.0
                max_discrepancies[feat] = diff
                if diff > 1e-4:
                    all_passed = False
                    self.violations.append({
                        "feature_name": feat,
                        "issue_type": "ADVERSARIAL_SENTINEL_FAILURE",
                        "severity": LeakageSeverity.CRITICAL.value,
                        "details": f"Future sentinel values leaked into pre-cutoff date ({cutoff_date}). Diff: {diff}",
                    })

        return {
            "cutoff_date": cutoff_date,
            "tested_series": f"{store_id} / {product_id}",
            "sentinel_value_injected": sentinel_val,
            "discrepancies": max_discrepancies,
            "adversarial_test_passed": all_passed,
        }

    def run_truncation_invariance_test(
        self,
        df: pl.DataFrame,
        cutoff_dates: List[str] = ["2022-12-31", "2023-12-31"],
        store_id: str = "STR_KUL_001",
        product_id: str = "PRD_CHK_01",
    ) -> List[Dict[str, object]]:
        """
        Truncation invariance test: Compare features generated with full multi-year history vs
        features generated when the dataset is physically truncated at cutoff date T.
        Features up to T must be strictly invariant.
        """
        results = []
        sub_full = df.filter(
            (pl.col("store_id") == store_id) & (pl.col("product_id") == product_id)
        ).sort("date")

        for cutoff in cutoff_dates:
            # Physical truncation strictly at cutoff
            sub_trunc = sub_full.filter(pl.col("date") <= cutoff)

            # Regenerate features on truncated history
            sub_trunc_gen, _ = generate_demand_features(sub_trunc)
            sub_trunc_gen, _ = generate_price_features(sub_trunc_gen)

            # Compare against full run pre-cutoff slice
            full_slice = sub_full.filter(pl.col("date") <= cutoff)

            test_features = ["demand_lag_1", "demand_lag_7", "demand_mean_7d", "purchase_price_lag_7"]
            diffs = {}
            passed = True

            for feat in test_features:
                if feat in full_slice.columns and feat in sub_trunc_gen.columns:
                    f_vals = np.array(full_slice[feat].drop_nulls().to_list())
                    t_vals = np.array(sub_trunc_gen[feat].drop_nulls().to_list())
                    diff = float(np.max(np.abs(f_vals - t_vals))) if len(f_vals) > 0 else 0.0
                    diffs[feat] = diff
                    if diff > 1e-4:
                        passed = False

            results.append({
                "cutoff_date": cutoff,
                "tested_series": f"{store_id} / {product_id}",
                "rows_evaluated": sub_trunc.height,
                "feature_differences": diffs,
                "truncation_invariance_passed": passed,
            })

        # Save report
        df_trunc_res = pd.DataFrame(results)
        df_trunc_res.to_csv(self.output_dir / "truncation_invariance_results.csv", index=False)
        return results

    def export_approved_manifests(
        self, temporal_registry_df: pd.DataFrame
    ) -> Tuple[Path, Path, Path]:
        """
        Export approved model-safe manifests and quarantined leakage exclusions.
        """
        # Safe demand features (status in SAFE, SAFE_WITH_ASSUMPTION) and applicability in (demand, both)
        safe_demand = temporal_registry_df[
            (temporal_registry_df["final_audit_status"].isin([AuditStatus.SAFE.value, AuditStatus.SAFE_WITH_ASSUMPTION.value])) &
            (temporal_registry_df["target_model_applicability"].isin(["demand", "both"]))
        ]
        demand_manifest_path = self.output_dir / "demand_features_temporally_safe.csv"
        safe_demand.to_csv(demand_manifest_path, index=False)

        # Safe price features
        safe_price = temporal_registry_df[
            (temporal_registry_df["final_audit_status"].isin([AuditStatus.SAFE.value, AuditStatus.SAFE_WITH_ASSUMPTION.value])) &
            (temporal_registry_df["target_model_applicability"].isin(["price", "both"]))
        ]
        price_manifest_path = self.output_dir / "price_features_temporally_safe.csv"
        safe_price.to_csv(price_manifest_path, index=False)

        # Excluded / Leaked features
        excluded_df = temporal_registry_df[
            temporal_registry_df["final_audit_status"].isin([AuditStatus.EXCLUDED.value, AuditStatus.LEAKED.value])
        ]
        excluded_path = self.output_dir / "excluded_leakage_features.csv"
        excluded_df.to_csv(excluded_path, index=False)

        # Leakage violations report
        df_violations = pd.DataFrame(self.violations)
        if df_violations.empty:
            df_violations = pd.DataFrame(columns=["feature_name", "issue_type", "severity", "details"])
        violations_path = self.output_dir / "leakage_violations.csv"
        df_violations.to_csv(violations_path, index=False)

        return demand_manifest_path, price_manifest_path, excluded_path
