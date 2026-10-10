"""
Calendar Features Module.
Generates calendar indicators, cyclical trigonometric representations,
weekend and month boundaries.
"""

import math
from typing import List, Tuple
import polars as pl
from src.features.feature_registry import FeatureDefinition


def generate_calendar_features(
    df: pl.DataFrame,
    date_col: str = "date",
) -> Tuple[pl.DataFrame, List[FeatureDefinition]]:
    """
    Generate calendar and cyclical trigonometric features from calendar date.
    All calendar properties are fully deterministic and known at prediction time.
    """
    definitions: List[FeatureDefinition] = []

    # Parse date if string
    d_expr = pl.col(date_col).str.strptime(pl.Date, "%Y-%m-%d")

    df = df.with_columns([
        d_expr.dt.weekday().cast(pl.Float64).alias("cal_day_of_week"),
        d_expr.dt.day().cast(pl.Float64).alias("cal_day_of_month"),
        d_expr.dt.week().cast(pl.Float64).alias("cal_week_of_year"),
        d_expr.dt.month().cast(pl.Float64).alias("cal_month"),
        d_expr.dt.quarter().cast(pl.Float64).alias("cal_quarter"),
        d_expr.dt.year().cast(pl.Float64).alias("cal_year"),
    ])
    definitions.extend([
        FeatureDefinition(
            name="cal_day_of_week",
            description="Day of week (1=Monday, ..., 7=Sunday)",
            source_columns=date_col,
            calculation="date.weekday()",
            data_type="float64",
            feature_family="calendar",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="cal_day_of_month",
            description="Day of calendar month (1 to 31)",
            source_columns=date_col,
            calculation="date.day()",
            data_type="float64",
            feature_family="calendar",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="cal_week_of_year",
            description="ISO week number of year (1 to 53)",
            source_columns=date_col,
            calculation="date.week()",
            data_type="float64",
            feature_family="calendar",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="cal_month",
            description="Calendar month (1 to 12)",
            source_columns=date_col,
            calculation="date.month()",
            data_type="float64",
            feature_family="calendar",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="cal_quarter",
            description="Calendar quarter (1 to 4)",
            source_columns=date_col,
            calculation="date.quarter()",
            data_type="float64",
            feature_family="calendar",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="cal_year",
            description="Calendar year",
            source_columns=date_col,
            calculation="date.year()",
            data_type="float64",
            feature_family="calendar",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
    ])

    # Weekend and month boundary indicators
    df = df.with_columns([
        (pl.col("cal_day_of_week") >= 6.0).cast(pl.Float64).alias("cal_is_weekend"),
        (pl.col("cal_day_of_month") <= 3.0).cast(pl.Float64).alias("cal_is_month_start"),
        (pl.col("cal_day_of_month") >= 28.0).cast(pl.Float64).alias("cal_is_month_end"),
    ])
    definitions.extend([
        FeatureDefinition(
            name="cal_is_weekend",
            description="Binary weekend indicator (1 for Saturday or Sunday)",
            source_columns=date_col,
            calculation="cal_day_of_week >= 6",
            data_type="float64",
            feature_family="calendar",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="cal_is_month_start",
            description="Binary flag for beginning of month (days 1-3)",
            source_columns=date_col,
            calculation="cal_day_of_month <= 3",
            data_type="float64",
            feature_family="calendar",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="cal_is_month_end",
            description="Binary flag for conclusion of month (days 28-31)",
            source_columns=date_col,
            calculation="cal_day_of_month >= 28",
            data_type="float64",
            feature_family="calendar",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
    ])

    # Cyclical trigonometric transformations
    two_pi = 2.0 * math.pi
    df = df.with_columns([
        (pl.col("cal_day_of_week") * (two_pi / 7.0)).sin().alias("cal_sin_day_of_week"),
        (pl.col("cal_day_of_week") * (two_pi / 7.0)).cos().alias("cal_cos_day_of_week"),
        (pl.col("cal_month") * (two_pi / 12.0)).sin().alias("cal_sin_month"),
        (pl.col("cal_month") * (two_pi / 12.0)).cos().alias("cal_cos_month"),
    ])
    definitions.extend([
        FeatureDefinition(
            name="cal_sin_day_of_week",
            description="Sine transformation of day of week (cyclical weekly periodicity)",
            source_columns=date_col,
            calculation="sin(2 * pi * cal_day_of_week / 7)",
            data_type="float64",
            feature_family="calendar",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="cal_cos_day_of_week",
            description="Cosine transformation of day of week (cyclical weekly periodicity)",
            source_columns=date_col,
            calculation="cos(2 * pi * cal_day_of_week / 7)",
            data_type="float64",
            feature_family="calendar",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="cal_sin_month",
            description="Sine transformation of calendar month (cyclical annual seasonality)",
            source_columns=date_col,
            calculation="sin(2 * pi * cal_month / 12)",
            data_type="float64",
            feature_family="calendar",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="cal_cos_month",
            description="Cosine transformation of calendar month (cyclical annual seasonality)",
            source_columns=date_col,
            calculation="cos(2 * pi * cal_month / 12)",
            data_type="float64",
            feature_family="calendar",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
    ])

    return df, definitions
