"""
Procurement and Supplier Features Module.
Generates historical supplier lead time metrics, supplier reliability,
on-time delivery rates, and order size constraints.
"""

from typing import List, Tuple
import polars as pl
from src.features.feature_registry import FeatureDefinition


def generate_procurement_features(
    df: pl.DataFrame,
) -> Tuple[pl.DataFrame, List[FeatureDefinition]]:
    """
    Generate leakage-safe historical supplier procurement features.
    """
    definitions: List[FeatureDefinition] = []

    df = df.sort(["store_id", "product_id", "date"])

    # 1. Lead times and basic supplier fields
    exp_lt = pl.col("expected_lead_time_days").cast(pl.Float64)
    act_lt = pl.col("actual_lead_time_days").cast(pl.Float64)

    df = df.with_columns([
        exp_lt.alias("supplier_lead_time_expected"),
        act_lt.alias("supplier_lead_time_actual"),
        (act_lt - exp_lt).alias("supplier_lead_time_gap"),
        (act_lt <= exp_lt).cast(pl.Float64).alias("__on_time_flag"),
        pl.col("minimum_order_quantity").cast(pl.Float64).alias("supplier_moq"),
        pl.col("supplier_pack_size").cast(pl.Float64).alias("supplier_pack_size_units"),
    ])
    definitions.extend([
        FeatureDefinition(
            name="supplier_lead_time_expected",
            description="Contracted/expected supplier lead time in calendar days",
            source_columns="expected_lead_time_days",
            calculation="Direct static/contractual lead time attribute",
            data_type="float64",
            feature_family="procurement_supplier",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="supplier_lead_time_actual",
            description="Actual lead time in calendar days for latest order",
            source_columns="actual_lead_time_days",
            calculation="Recorded actual order lead time",
            data_type="float64",
            feature_family="procurement_supplier",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="supplier_lead_time_gap",
            description="Discrepancy between actual and contracted lead time (days)",
            source_columns="actual_lead_time_days, expected_lead_time_days",
            calculation="actual_lead_time_days - expected_lead_time_days",
            data_type="float64",
            feature_family="procurement_supplier",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="supplier_moq",
            description="Minimum Order Quantity (MOQ) enforced by supplier",
            source_columns="minimum_order_quantity",
            calculation="Direct contractual attribute",
            data_type="float64",
            feature_family="procurement_supplier",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="supplier_pack_size_units",
            description="Supplier standard case/pack size in base units",
            source_columns="supplier_pack_size",
            calculation="Direct contractual attribute",
            data_type="float64",
            feature_family="procurement_supplier",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
    ])

    # 2. Historical rolling lead time and reliability (shifted by 1 day)
    df = df.with_columns([
        act_lt.shift(1).over(["store_id", "product_id"]).alias("__act_lt_t1"),
        pl.col("__on_time_flag").shift(1).over(["store_id", "product_id"]).alias("__on_time_t1"),
    ])

    roll_exprs = [
        pl.col("__act_lt_t1").rolling_mean(window_size=30).over(["store_id", "product_id"]).alias("supplier_lead_time_mean_30d"),
        pl.col("__act_lt_t1").rolling_std(window_size=30).over(["store_id", "product_id"]).alias("supplier_lead_time_std_30d"),
        pl.col("__act_lt_t1").rolling_mean(window_size=90).over(["store_id", "product_id"]).alias("supplier_lead_time_mean_90d"),
        pl.col("__on_time_t1").rolling_mean(window_size=30).over(["store_id", "product_id"]).alias("supplier_on_time_rate_30d"),
        pl.col("__on_time_t1").rolling_mean(window_size=90).over(["store_id", "product_id"]).alias("supplier_on_time_rate_90d"),
    ]
    definitions.extend([
        FeatureDefinition(
            name="supplier_lead_time_mean_30d",
            description="30-day rolling mean of historical lead time prior to T",
            source_columns="actual_lead_time_days",
            calculation="rolling_mean(30) on actual lead time shifted by 1",
            data_type="float64",
            feature_family="procurement_supplier",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="supplier_lead_time_std_30d",
            description="30-day rolling standard deviation of lead time prior to T",
            source_columns="actual_lead_time_days",
            calculation="rolling_std(30) on actual lead time shifted by 1",
            data_type="float64",
            feature_family="procurement_supplier",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="supplier_lead_time_mean_90d",
            description="90-day rolling mean of historical lead time prior to T",
            source_columns="actual_lead_time_days",
            calculation="rolling_mean(90) on actual lead time shifted by 1",
            data_type="float64",
            feature_family="procurement_supplier",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="supplier_on_time_rate_30d",
            description="Proportion of orders delivered on-time in the prior 30 days",
            source_columns="actual_lead_time_days, expected_lead_time_days",
            calculation="rolling_mean(30) on (actual <= expected) shifted by 1",
            data_type="float64",
            feature_family="procurement_supplier",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="supplier_on_time_rate_90d",
            description="Proportion of orders delivered on-time in the prior 90 days",
            source_columns="actual_lead_time_days, expected_lead_time_days",
            calculation="rolling_mean(90) on (actual <= expected) shifted by 1",
            data_type="float64",
            feature_family="procurement_supplier",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
    ])
    df = df.with_columns(roll_exprs)
    df = df.drop(["__on_time_flag", "__act_lt_t1", "__on_time_t1"])

    return df, definitions
