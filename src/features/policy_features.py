"""
Policy Features Module.
Generates structured policy, subsidy, price ceiling, minimum wage,
and taxation regime indicators from national budget and ministry gazettes.
"""

from typing import List, Tuple
import polars as pl
from src.features.feature_registry import FeatureDefinition


def generate_policy_features(
    df: pl.DataFrame,
) -> Tuple[pl.DataFrame, List[FeatureDefinition]]:
    """
    Generate structured policy, subsidy, price controls, and regulatory regime features.
    """
    definitions: List[FeatureDefinition] = []

    df = df.sort(["store_id", "product_id", "date"])

    # 1. Base announcement and active status flags
    sub_ann = pl.col("subsidy_announced_flag").cast(pl.Float64)
    sub_act = pl.col("subsidy_active_flag").cast(pl.Float64)
    pol_ann = pl.col("policy_announced_flag").cast(pl.Float64)
    pol_act = pl.col("policy_active_flag").cast(pl.Float64)

    df = df.with_columns([
        sub_ann.alias("subsidy_announced_flag_indicator"),
        sub_act.alias("subsidy_active_flag_indicator"),
        pol_ann.alias("policy_announced_flag_indicator"),
        pol_act.alias("policy_active_flag_indicator"),
        pl.int_range(0, pl.len()).over(["store_id", "product_id"]).alias("__t_idx"),
    ])
    definitions.extend([
        FeatureDefinition(
            name="subsidy_announced_flag_indicator",
            description="Binary flag if a government subsidy policy was publicly announced on date T",
            source_columns="subsidy_announced_flag",
            calculation="Direct gazette announcement event",
            data_type="float64",
            feature_family="policy_subsidy",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="subsidy_active_flag_indicator",
            description="Binary flag if a government subsidy regime is actively in force on date T",
            source_columns="subsidy_active_flag",
            calculation="Active operational regime indicator",
            data_type="float64",
            feature_family="policy_subsidy",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="policy_announced_flag_indicator",
            description="Binary flag if a national budget or fiscal policy was announced on date T",
            source_columns="policy_announced_flag",
            calculation="Direct national policy announcement event",
            data_type="float64",
            feature_family="policy_subsidy",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="policy_active_flag_indicator",
            description="Binary flag if a national budget policy regime is currently active on date T",
            source_columns="policy_active_flag",
            calculation="Active national policy regime indicator",
            data_type="float64",
            feature_family="policy_subsidy",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
    ])

    # 2. Specific regulatory regimes and price ceilings
    sub_type_col = pl.col("subsidy_type")
    pol_type_col = pl.col("policy_type")
    date_col = pl.col("date")

    df = df.with_columns([
        (sub_type_col.str.contains("(?i)Chicken") & (sub_act == 1.0)).cast(pl.Float64).alias("is_chicken_subsidy_active"),
        (sub_type_col.str.contains("(?i)Diesel") & (sub_act == 1.0)).cast(pl.Float64).alias("is_diesel_rationalisation_active"),
        (pol_type_col.str.contains("(?i)Electricity") & (pol_act == 1.0)).cast(pl.Float64).alias("is_electricity_discount_active"),
        (date_col >= "2022-05-01").cast(pl.Float64).alias("is_minimum_wage_rm1500_active"),
        (date_col >= "2024-03-01").cast(pl.Float64).alias("is_service_tax_8pct_active"),
    ])
    definitions.extend([
        FeatureDefinition(
            name="is_chicken_subsidy_active",
            description="Binary indicator if chicken and egg price ceiling / breeder subsidy is active",
            source_columns="subsidy_type, subsidy_active_flag",
            calculation="subsidy_type contains 'Chicken' and active==1",
            data_type="float64",
            feature_family="policy_subsidy",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="is_diesel_rationalisation_active",
            description="Binary indicator if targeted diesel subsidy rationalisation (retail float) is active",
            source_columns="subsidy_type, subsidy_active_flag",
            calculation="subsidy_type contains 'Diesel' and active==1",
            data_type="float64",
            feature_family="policy_subsidy",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="is_electricity_discount_active",
            description="Binary indicator if TNB commercial electricity discount package is in effect",
            source_columns="policy_type, policy_active_flag",
            calculation="policy_type contains 'Electricity' and active==1",
            data_type="float64",
            feature_family="policy_subsidy",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="is_minimum_wage_rm1500_active",
            description="Binary indicator for national minimum wage regime RM1,500/month (effective 1 May 2022)",
            source_columns="date, policy_type",
            calculation="date >= '2022-05-01'",
            data_type="float64",
            feature_family="policy_subsidy",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="is_service_tax_8pct_active",
            description="Binary indicator for Service Tax rate increase from 6% to 8% (effective 1 March 2024)",
            source_columns="date, policy_type",
            calculation="date >= '2024-03-01'",
            data_type="float64",
            feature_family="policy_subsidy",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
    ])

    # 3. Days since policy / subsidy announcement
    df = df.with_columns([
        pl.when(sub_ann == 1.0).then(pl.col("__t_idx")).otherwise(None).forward_fill().over(["store_id", "product_id"]).alias("__last_sub_ann_t"),
        pl.when(pol_ann == 1.0).then(pl.col("__t_idx")).otherwise(None).forward_fill().over(["store_id", "product_id"]).alias("__last_pol_ann_t"),
    ])
    df = df.with_columns([
        (pl.col("__t_idx") - pl.col("__last_sub_ann_t")).fill_null(999.0).cast(pl.Float64).alias("days_since_subsidy_announced"),
        (pl.col("__t_idx") - pl.col("__last_pol_ann_t")).fill_null(999.0).cast(pl.Float64).alias("days_since_policy_announced"),
    ])
    definitions.extend([
        FeatureDefinition(
            name="days_since_subsidy_announced",
            description="Days elapsed since the latest government subsidy announcement",
            source_columns="subsidy_announced_flag",
            calculation="t - last_announcement_t (default 999)",
            data_type="float64",
            feature_family="policy_subsidy",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="days_since_policy_announced",
            description="Days elapsed since the latest national fiscal policy announcement",
            source_columns="policy_announced_flag",
            calculation="t - last_announcement_t (default 999)",
            data_type="float64",
            feature_family="policy_subsidy",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
    ])

    df = df.drop(["__t_idx", "__last_sub_ann_t", "__last_pol_ann_t"])

    return df, definitions
