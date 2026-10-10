"""
Weather Features Module.
Generates historical local meteorological conditions, rolling precipitation,
and heatwave/rain spells strictly using past weather history.
"""

from typing import List, Tuple
import polars as pl
from src.features.feature_registry import FeatureDefinition


def generate_weather_features(
    df: pl.DataFrame,
) -> Tuple[pl.DataFrame, List[FeatureDefinition]]:
    """
    Generate store-level weather indicators and historical rolling metrics.
    All rolling metrics are shifted by 1 day to ensure zero leakage.
    """
    definitions: List[FeatureDefinition] = []

    df = df.sort(["store_id", "product_id", "date"])

    # 1. Base weather metrics
    weather_cols = [
        ("maximum_temperature", "weather_temp_max", "Daily maximum ambient temperature in Celsius"),
        ("minimum_temperature", "weather_temp_min", "Daily minimum ambient temperature in Celsius"),
        ("average_temperature", "weather_temp_avg", "Daily mean ambient temperature in Celsius"),
        ("rainfall", "weather_rainfall_mm", "Daily total precipitation rainfall in millimetres"),
        ("humidity", "weather_humidity_pct", "Daily relative humidity percentage"),
        ("precipitation_duration", "weather_precip_hours", "Hours of recorded precipitation duration"),
    ]

    base_exprs = []
    for raw_col, new_col, desc in weather_cols:
        base_exprs.append(pl.col(raw_col).cast(pl.Float64).alias(new_col))
        definitions.append(
            FeatureDefinition(
                name=new_col,
                description=desc,
                source_columns=raw_col,
                calculation="Direct meteorological station measurement",
                data_type="float64",
                feature_family="weather",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="both",
                leakage_risk="none",
            )
        )
    df = df.with_columns(base_exprs)

    # 2. Historical rolling weather (shifted by 1 day)
    df = df.with_columns([
        pl.col("weather_rainfall_mm").shift(1).over(["store_id", "product_id"]).alias("__rain_t1"),
        pl.col("weather_temp_avg").shift(1).over(["store_id", "product_id"]).alias("__temp_avg_t1"),
        pl.col("weather_temp_max").shift(1).over(["store_id", "product_id"]).alias("__temp_max_t1"),
        pl.int_range(0, pl.len()).over(["store_id", "product_id"]).alias("__t_idx"),
    ])

    roll_exprs = [
        pl.col("__rain_t1").rolling_sum(window_size=3).over(["store_id", "product_id"]).alias("weather_rainfall_sum_3d"),
        pl.col("__rain_t1").rolling_sum(window_size=7).over(["store_id", "product_id"]).alias("weather_rainfall_sum_7d"),
        pl.col("__rain_t1").rolling_sum(window_size=14).over(["store_id", "product_id"]).alias("weather_rainfall_sum_14d"),
        pl.col("__temp_avg_t1").rolling_mean(window_size=7).over(["store_id", "product_id"]).alias("weather_temp_mean_7d"),
        pl.col("__temp_max_t1").rolling_max(window_size=7).over(["store_id", "product_id"]).alias("weather_temp_max_7d"),
    ]
    definitions.extend([
        FeatureDefinition(
            name="weather_rainfall_sum_3d",
            description="Total cumulative rainfall in the prior 3 days (T-3 to T-1)",
            source_columns="rainfall",
            calculation="rolling_sum(3) on rainfall shifted by 1",
            data_type="float64",
            feature_family="weather_rolling",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="weather_rainfall_sum_7d",
            description="Total cumulative rainfall in the prior 7 days (T-7 to T-1)",
            source_columns="rainfall",
            calculation="rolling_sum(7) on rainfall shifted by 1",
            data_type="float64",
            feature_family="weather_rolling",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="weather_rainfall_sum_14d",
            description="Total cumulative rainfall in the prior 14 days (T-14 to T-1)",
            source_columns="rainfall",
            calculation="rolling_sum(14) on rainfall shifted by 1",
            data_type="float64",
            feature_family="weather_rolling",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="weather_temp_mean_7d",
            description="7-day rolling mean ambient temperature prior to T",
            source_columns="average_temperature",
            calculation="rolling_mean(7) on average_temperature shifted by 1",
            data_type="float64",
            feature_family="weather_rolling",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="weather_temp_max_7d",
            description="7-day rolling peak maximum temperature prior to T",
            source_columns="maximum_temperature",
            calculation="rolling_max(7) on maximum_temperature shifted by 1",
            data_type="float64",
            feature_family="weather_rolling",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
    ])
    df = df.with_columns(roll_exprs)

    # 3. Consecutive rainy days and hot days up to T-1
    df = df.with_columns([
        (pl.col("__rain_t1") > 1.0).cast(pl.Int64).fill_null(0).alias("__is_rain_hist"),
        (pl.col("__temp_max_t1") > 33.0).cast(pl.Int64).fill_null(0).alias("__is_hot_hist"),
    ])

    df = df.with_columns([
        (pl.col("__is_rain_hist") != pl.col("__is_rain_hist").shift(1).over(["store_id", "product_id"]))
        .fill_null(True)
        .cum_sum()
        .over(["store_id", "product_id"])
        .alias("__rain_blk"),
        (pl.col("__is_hot_hist") != pl.col("__is_hot_hist").shift(1).over(["store_id", "product_id"]))
        .fill_null(True)
        .cum_sum()
        .over(["store_id", "product_id"])
        .alias("__hot_blk"),
    ])

    df = df.with_columns([
        pl.when(pl.col("__is_rain_hist") == 1)
        .then(pl.col("__t_idx") - pl.col("__t_idx").min().over(["store_id", "product_id", "__rain_blk"]) + 1)
        .otherwise(0)
        .cast(pl.Float64)
        .alias("weather_consecutive_rain_days"),
        pl.when(pl.col("__is_hot_hist") == 1)
        .then(pl.col("__t_idx") - pl.col("__t_idx").min().over(["store_id", "product_id", "__hot_blk"]) + 1)
        .otherwise(0)
        .cast(pl.Float64)
        .alias("weather_consecutive_hot_days"),
    ])
    definitions.extend([
        FeatureDefinition(
            name="weather_consecutive_rain_days",
            description="Number of consecutive rainy days (>1.0mm) ending on day T-1",
            source_columns="rainfall",
            calculation="Consecutive days rainfall > 1.0mm shifted by 1",
            data_type="float64",
            feature_family="weather_spells",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="weather_consecutive_hot_days",
            description="Number of consecutive hot days (>33.0C) ending on day T-1",
            source_columns="maximum_temperature",
            calculation="Consecutive days max temp > 33.0C shifted by 1",
            data_type="float64",
            feature_family="weather_spells",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
    ])

    df = df.drop([
        "__rain_t1", "__temp_avg_t1", "__temp_max_t1",
        "__t_idx", "__is_rain_hist", "__is_hot_hist",
        "__rain_blk", "__hot_blk"
    ])

    return df, definitions
