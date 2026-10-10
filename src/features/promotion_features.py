"""
Promotion Features Module.
Generates historical and scheduled promotion features including active status,
duration, days active, promo frequency, and advance planned promotion flags.
"""

from typing import List, Tuple
import polars as pl
from src.features.feature_registry import FeatureDefinition


def generate_promotion_features(
    df: pl.DataFrame,
) -> Tuple[pl.DataFrame, List[FeatureDefinition]]:
    """
    Generate promotion indicators, duration metrics, historical frequency,
    and planned upcoming promotional schedule features.
    """
    definitions: List[FeatureDefinition] = []

    df = df.sort(["store_id", "product_id", "date"])

    # 1. Basic promotion flags
    promo_act = pl.col("promotion_active").cast(pl.Float64)
    disc_pct = pl.col("discount_percentage").cast(pl.Float64)

    df = df.with_columns([
        promo_act.alias("promo_active_flag"),
        disc_pct.alias("promo_discount_percentage"),
        pl.int_range(0, pl.len()).over(["store_id", "product_id"]).alias("__t_idx"),
    ])
    definitions.extend([
        FeatureDefinition(
            name="promo_active_flag",
            description="Binary indicator if an active promotional campaign is in effect on date T",
            source_columns="promotion_active",
            calculation="Direct observation",
            data_type="float64",
            feature_family="promotion",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="promo_discount_percentage",
            description="Campaign discount percentage in effect on date T",
            source_columns="discount_percentage",
            calculation="Direct observation",
            data_type="float64",
            feature_family="promotion",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
    ])

    # 2. Promotion block duration and days active so far
    df = df.with_columns(
        (pl.col("promo_active_flag") != pl.col("promo_active_flag").shift(1).over(["store_id", "product_id"]))
        .fill_null(True)
        .cum_sum()
        .over(["store_id", "product_id"])
        .alias("__promo_block_id")
    )
    df = df.with_columns([
        pl.when(pl.col("promo_active_flag") == 1.0)
        .then(pl.col("__t_idx") - pl.col("__t_idx").min().over(["store_id", "product_id", "__promo_block_id"]) + 1)
        .otherwise(0.0)
        .cast(pl.Float64)
        .alias("promo_days_active_so_far"),
        pl.when(pl.col("promo_active_flag") == 1.0)
        .then(pl.col("__t_idx").max().over(["store_id", "product_id", "__promo_block_id"]) - pl.col("__t_idx"))
        .otherwise(0.0)
        .cast(pl.Float64)
        .alias("promo_days_until_end"),
    ])
    definitions.extend([
        FeatureDefinition(
            name="promo_days_active_so_far",
            description="Elapsed consecutive calendar days current promotion has been active (1 on start day)",
            source_columns="promotion_active",
            calculation="t - block_min_t + 1 when active, else 0",
            data_type="float64",
            feature_family="promotion",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="promo_days_until_end",
            description="Days remaining until the scheduled end of current active promotion",
            source_columns="promotion_active",
            calculation="block_max_t - t when active, else 0",
            data_type="float64",
            feature_family="promotion",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
    ])

    # 3. Historical promotion frequency and days since last promotion (shifted by 1)
    df = df.with_columns(
        promo_act.shift(1).over(["store_id", "product_id"]).alias("__promo_hist_t1")
    )
    df = df.with_columns([
        pl.col("__promo_hist_t1").rolling_sum(window_size=30).over(["store_id", "product_id"]).alias("promo_frequency_30d"),
        pl.col("__promo_hist_t1").rolling_sum(window_size=90).over(["store_id", "product_id"]).alias("promo_frequency_90d"),
    ])
    definitions.extend([
        FeatureDefinition(
            name="promo_frequency_30d",
            description="Total number of promotion days in prior 30 days",
            source_columns="promotion_active",
            calculation="rolling_sum(30) on promo_active shifted by 1",
            data_type="float64",
            feature_family="promotion",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="promo_frequency_90d",
            description="Total number of promotion days in prior 90 days",
            source_columns="promotion_active",
            calculation="rolling_sum(90) on promo_active shifted by 1",
            data_type="float64",
            feature_family="promotion",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
    ])

    # Days since previous promotion
    df = df.with_columns(
        pl.when(pl.col("__promo_hist_t1") == 1.0)
        .then(pl.col("__t_idx") - 1)
        .otherwise(None)
        .forward_fill()
        .over(["store_id", "product_id"])
        .alias("__last_promo_t")
    )
    df = df.with_columns(
        (pl.col("__t_idx") - pl.col("__last_promo_t")).fill_null(999.0).cast(pl.Float64).alias("days_since_previous_promo")
    )
    definitions.append(
        FeatureDefinition(
            name="days_since_previous_promo",
            description="Days elapsed since the previous active promotion concluded",
            source_columns="promotion_active",
            calculation="t - last_promo_t shifted by 1 (default 999)",
            data_type="float64",
            feature_family="promotion",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        )
    )

    # 4. Advance planned upcoming promotions (known promotional calendar)
    # Reverse series, rolling max over forward window
    df = df.reverse()
    df = df.with_columns([
        pl.col("promo_active_flag").shift(1).rolling_max(window_size=7).over(["store_id", "product_id"]).fill_null(0.0).alias("planned_promo_within_7d"),
        pl.col("promo_active_flag").shift(1).rolling_max(window_size=14).over(["store_id", "product_id"]).fill_null(0.0).alias("planned_promo_within_14d"),
        pl.col("promo_active_flag").shift(1).rolling_max(window_size=30).over(["store_id", "product_id"]).fill_null(0.0).alias("planned_promo_within_30d"),
    ])
    df = df.reverse()
    definitions.extend([
        FeatureDefinition(
            name="planned_promo_within_7d",
            description="Binary flag if a known promotion is scheduled within the next 7 days",
            source_columns="promotion_active",
            calculation="forward rolling max(7) of scheduled promotion active",
            data_type="float64",
            feature_family="promotion",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="planned_promo_within_14d",
            description="Binary flag if a known promotion is scheduled within the next 14 days",
            source_columns="promotion_active",
            calculation="forward rolling max(14) of scheduled promotion active",
            data_type="float64",
            feature_family="promotion",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="planned_promo_within_30d",
            description="Binary flag if a known promotion is scheduled within the next 30 days",
            source_columns="promotion_active",
            calculation="forward rolling max(30) of scheduled promotion active",
            data_type="float64",
            feature_family="promotion",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
    ])

    df = df.drop(["__t_idx", "__promo_block_id", "__promo_hist_t1", "__last_promo_t"])

    return df, definitions
