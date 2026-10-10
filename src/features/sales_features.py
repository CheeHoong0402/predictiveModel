"""
Observed Sales Features Module.
Generates historical lags and rolling window statistics from observed Point-of-Sale sales.
Enables ablation and comparative experiments against corrected demand.
"""

from typing import List, Tuple
import polars as pl
from src.features.feature_registry import FeatureDefinition


def generate_sales_features(
    df: pl.DataFrame,
    sales_col: str = "observed_units_sold",
) -> Tuple[pl.DataFrame, List[FeatureDefinition]]:
    """
    Generate leakage-safe historical observed sales features.
    
    All calculations are strictly shifted by 1 day relative to observation date T.
    """
    definitions: List[FeatureDefinition] = []

    df = df.sort(["store_id", "product_id", "date"])

    # Base series shifted by 1
    df = df.with_columns(
        pl.col(sales_col)
        .cast(pl.Float64)
        .shift(1)
        .over(["store_id", "product_id"])
        .alias("__sales_hist_t1")
    )

    # 1. Sales Lags
    lag_days = [1, 2, 3, 7, 14, 28, 364]
    lag_exprs = []
    for lag in lag_days:
        col_name = f"sales_lag_{lag}"
        lag_exprs.append(
            pl.col("__sales_hist_t1")
            .shift(lag - 1)
            .over(["store_id", "product_id"])
            .alias(col_name)
        )
        definitions.append(
            FeatureDefinition(
                name=col_name,
                description=f"Observed sales units {lag} day(s) prior to date T",
                source_columns=sales_col,
                calculation=f"shift({lag}) of observed sales units",
                data_type="float64",
                feature_family="observed_sales",
                known_at_prediction_time=True,
                expected_missingness="low" if lag <= 28 else "medium",
                target_model_applicability="demand",
                leakage_risk="none",
            )
        )
    df = df.with_columns(lag_exprs)

    # 2. Sales Rolling Statistics (7d and 28d)
    rolling_windows = [7, 28]
    roll_exprs = []
    for w in rolling_windows:
        # Mean
        col_mean = f"sales_mean_{w}d"
        roll_exprs.append(
            pl.col("__sales_hist_t1")
            .rolling_mean(window_size=w)
            .over(["store_id", "product_id"])
            .alias(col_mean)
        )
        definitions.append(
            FeatureDefinition(
                name=col_mean,
                description=f"Rolling {w}-day mean of observed sales units prior to T",
                source_columns=sales_col,
                calculation=f"rolling_mean({w}) on observed sales shifted by 1",
                data_type="float64",
                feature_family="observed_sales",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="demand",
                leakage_risk="none",
            )
        )

        # Standard deviation
        col_std = f"sales_std_{w}d"
        roll_exprs.append(
            pl.col("__sales_hist_t1")
            .rolling_std(window_size=w)
            .over(["store_id", "product_id"])
            .alias(col_std)
        )
        definitions.append(
            FeatureDefinition(
                name=col_std,
                description=f"Rolling {w}-day std of observed sales units prior to T",
                source_columns=sales_col,
                calculation=f"rolling_std({w}) on observed sales shifted by 1",
                data_type="float64",
                feature_family="observed_sales",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="demand",
                leakage_risk="none",
            )
        )

        # Sum
        col_sum = f"sales_sum_{w}d"
        roll_exprs.append(
            pl.col("__sales_hist_t1")
            .rolling_sum(window_size=w)
            .over(["store_id", "product_id"])
            .alias(col_sum)
        )
        definitions.append(
            FeatureDefinition(
                name=col_sum,
                description=f"Rolling {w}-day sum of observed sales units prior to T",
                source_columns=sales_col,
                calculation=f"rolling_sum({w}) on observed sales shifted by 1",
                data_type="float64",
                feature_family="observed_sales",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="demand",
                leakage_risk="none",
            )
        )

    df = df.with_columns(roll_exprs)
    df = df.drop("__sales_hist_t1")

    return df, definitions
