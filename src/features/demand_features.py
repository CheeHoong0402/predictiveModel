"""
Demand Features Module.
Generates historical demand lags, rolling window statistics, momentum,
weekly seasonality, and annual seasonality using training-safe true demand.
"""

from typing import List, Tuple
import polars as pl
from src.features.feature_registry import FeatureDefinition


def generate_demand_features(
    df: pl.DataFrame,
    demand_col: str = "estimated_true_demand_training_safe",
    fallback_col: str = "observed_units_sold",
) -> Tuple[pl.DataFrame, List[FeatureDefinition]]:
    """
    Generate leakage-safe historical demand features.
    
    All calculations are strictly shifted by at least 1 day relative to observation date T,
    guaranteeing that only past information available at or before T-1 is utilized.
    """
    definitions: List[FeatureDefinition] = []

    # Ensure data is sorted by series and date
    df = df.sort(["store_id", "product_id", "date"])

    # 1. Base historical series shifted by 1 day
    # Coalesce with fallback if needed
    df = df.with_columns(
        pl.coalesce([pl.col(demand_col), pl.col(fallback_col), pl.lit(0.0)])
        .shift(1)
        .over(["store_id", "product_id"])
        .alias("__demand_hist_t1")
    )

    # 2. Lag features
    lag_days = [1, 2, 3, 7, 14, 21, 28, 35, 56, 84, 168, 364]
    lag_exprs = []
    for lag in lag_days:
        col_name = f"demand_lag_{lag}"
        lag_exprs.append(
            pl.col("__demand_hist_t1")
            .shift(lag - 1)
            .over(["store_id", "product_id"])
            .alias(col_name)
        )
        definitions.append(
            FeatureDefinition(
                name=col_name,
                description=f"Estimated true demand {lag} day(s) prior to observation date T",
                source_columns=demand_col,
                calculation=f"shift({lag}) of training-safe demand grouped by store and product",
                data_type="float64",
                feature_family="demand_history",
                known_at_prediction_time=True,
                expected_missingness="low" if lag <= 28 else "medium",
                target_model_applicability="demand",
                leakage_risk="none",
            )
        )

    df = df.with_columns(lag_exprs)

    # 3. Rolling window statistics
    rolling_windows = [3, 7, 14, 28, 56, 84]
    roll_exprs = []
    for w in rolling_windows:
        # Mean
        col_mean = f"demand_mean_{w}d"
        roll_exprs.append(
            pl.col("__demand_hist_t1")
            .rolling_mean(window_size=w)
            .over(["store_id", "product_id"])
            .alias(col_mean)
        )
        definitions.append(
            FeatureDefinition(
                name=col_mean,
                description=f"Rolling {w}-day mean of historical demand (prior to T)",
                source_columns=demand_col,
                calculation=f"rolling_mean({w}) on demand shifted by 1",
                data_type="float64",
                feature_family="rolling_demand",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="demand",
                leakage_risk="none",
            )
        )

        # Median
        col_med = f"demand_median_{w}d"
        roll_exprs.append(
            pl.col("__demand_hist_t1")
            .rolling_median(window_size=w)
            .over(["store_id", "product_id"])
            .alias(col_med)
        )
        definitions.append(
            FeatureDefinition(
                name=col_med,
                description=f"Rolling {w}-day median of historical demand (prior to T)",
                source_columns=demand_col,
                calculation=f"rolling_median({w}) on demand shifted by 1",
                data_type="float64",
                feature_family="rolling_demand",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="demand",
                leakage_risk="none",
            )
        )

        # Standard Deviation
        col_std = f"demand_std_{w}d"
        roll_exprs.append(
            pl.col("__demand_hist_t1")
            .rolling_std(window_size=w)
            .over(["store_id", "product_id"])
            .alias(col_std)
        )
        definitions.append(
            FeatureDefinition(
                name=col_std,
                description=f"Rolling {w}-day standard deviation of historical demand",
                source_columns=demand_col,
                calculation=f"rolling_std({w}) on demand shifted by 1",
                data_type="float64",
                feature_family="rolling_demand",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="demand",
                leakage_risk="none",
            )
        )

        # Min & Max
        col_min = f"demand_min_{w}d"
        col_max = f"demand_max_{w}d"
        roll_exprs.extend([
            pl.col("__demand_hist_t1").rolling_min(window_size=w).over(["store_id", "product_id"]).alias(col_min),
            pl.col("__demand_hist_t1").rolling_max(window_size=w).over(["store_id", "product_id"]).alias(col_max),
        ])
        definitions.extend([
            FeatureDefinition(
                name=col_min,
                description=f"Rolling {w}-day minimum historical demand",
                source_columns=demand_col,
                calculation=f"rolling_min({w}) on demand shifted by 1",
                data_type="float64",
                feature_family="rolling_demand",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="demand",
                leakage_risk="none",
            ),
            FeatureDefinition(
                name=col_max,
                description=f"Rolling {w}-day maximum historical demand",
                source_columns=demand_col,
                calculation=f"rolling_max({w}) on demand shifted by 1",
                data_type="float64",
                feature_family="rolling_demand",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="demand",
                leakage_risk="none",
            ),
        ])

        # Sum
        col_sum = f"demand_sum_{w}d"
        roll_exprs.append(
            pl.col("__demand_hist_t1")
            .rolling_sum(window_size=w)
            .over(["store_id", "product_id"])
            .alias(col_sum)
        )
        definitions.append(
            FeatureDefinition(
                name=col_sum,
                description=f"Rolling {w}-day sum of historical demand",
                source_columns=demand_col,
                calculation=f"rolling_sum({w}) on demand shifted by 1",
                data_type="float64",
                feature_family="rolling_demand",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="demand",
                leakage_risk="none",
            )
        )

        # Coefficient of variation definition registered here, computed after roll_exprs
        col_cv = f"demand_cv_{w}d"
        definitions.append(
            FeatureDefinition(
                name=col_cv,
                description=f"Rolling {w}-day coefficient of variation of historical demand",
                source_columns=demand_col,
                calculation=f"demand_std_{w}d / (demand_mean_{w}d + 1e-6)",
                data_type="float64",
                feature_family="rolling_demand",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="demand",
                leakage_risk="none",
            )
        )

    df = df.with_columns(roll_exprs)

    cv_exprs = [
        (pl.col(f"demand_std_{w}d") / (pl.col(f"demand_mean_{w}d") + 1e-6)).alias(f"demand_cv_{w}d")
        for w in rolling_windows
    ]
    df = df.with_columns(cv_exprs)

    # 4. Demand Momentum and Trend Features
    momentum_exprs = [
        # Ratio 7d / 28d
        (pl.col("demand_mean_7d") / (pl.col("demand_mean_28d") + 1e-6)).alias("demand_ratio_7d_28d"),
        # Ratio 14d / 56d
        (pl.col("demand_mean_14d") / (pl.col("demand_mean_56d") + 1e-6)).alias("demand_ratio_14d_56d"),
        # Growth 7d
        ((pl.col("demand_mean_7d") - pl.col("demand_lag_7")) / (pl.col("demand_lag_7") + 1e-6)).alias("demand_growth_7d"),
        # Growth 28d
        ((pl.col("demand_mean_28d") - pl.col("demand_lag_28")) / (pl.col("demand_lag_28") + 1e-6)).alias("demand_growth_28d"),
        # Trend 28d (slope indicator)
        ((pl.col("demand_mean_7d") - pl.col("demand_mean_28d")) / (pl.col("demand_mean_28d") + 1e-6)).alias("demand_trend_28d"),
        # Recent demand vs 28d median
        (pl.col("demand_lag_1") / (pl.col("demand_median_28d") + 1e-6)).alias("demand_vs_median_28d"),
    ]
    definitions.extend([
        FeatureDefinition(
            name="demand_ratio_7d_28d",
            description="Ratio of short-term 7-day demand to medium-term 28-day demand",
            source_columns=demand_col,
            calculation="demand_mean_7d / (demand_mean_28d + 1e-6)",
            data_type="float64",
            feature_family="demand_momentum",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="demand_ratio_14d_56d",
            description="Ratio of 14-day demand to 56-day demand",
            source_columns=demand_col,
            calculation="demand_mean_14d / (demand_mean_56d + 1e-6)",
            data_type="float64",
            feature_family="demand_momentum",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="demand_growth_7d",
            description="7-day demand growth relative to lag 7",
            source_columns=demand_col,
            calculation="(demand_mean_7d - demand_lag_7) / (demand_lag_7 + 1e-6)",
            data_type="float64",
            feature_family="demand_momentum",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="demand_growth_28d",
            description="28-day demand growth relative to lag 28",
            source_columns=demand_col,
            calculation="(demand_mean_28d - demand_lag_28) / (demand_lag_28 + 1e-6)",
            data_type="float64",
            feature_family="demand_momentum",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="demand_trend_28d",
            description="Normalized trend indicator comparing 7d mean with 28d mean",
            source_columns=demand_col,
            calculation="(demand_mean_7d - demand_mean_28d) / (demand_mean_28d + 1e-6)",
            data_type="float64",
            feature_family="demand_momentum",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="demand_vs_median_28d",
            description="Most recent observed demand relative to 28-day historical median",
            source_columns=demand_col,
            calculation="demand_lag_1 / (demand_median_28d + 1e-6)",
            data_type="float64",
            feature_family="demand_momentum",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
    ])
    df = df.with_columns(momentum_exprs)

    # 5. Weekly Seasonality Features
    # Median of same weekday over prior 4 weeks and 8 weeks
    df = df.with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d").dt.weekday().alias("__dow_temp")
    )
    df = df.sort(["store_id", "product_id", "__dow_temp", "date"])

    dow_exprs = [
        pl.col("__demand_hist_t1")
        .rolling_median(window_size=4)
        .over(["store_id", "product_id", "__dow_temp"])
        .alias("demand_dow_median_4w"),
        pl.col("__demand_hist_t1")
        .rolling_median(window_size=8)
        .over(["store_id", "product_id", "__dow_temp"])
        .alias("demand_dow_median_8w"),
    ]
    df = df.with_columns(dow_exprs)
    df = df.sort(["store_id", "product_id", "date"])

    definitions.extend([
        FeatureDefinition(
            name="demand_dow_median_4w",
            description="Median demand for the same weekday over the prior 4 weeks",
            source_columns=demand_col,
            calculation="rolling_median(4) on same weekday demand shifted by 1",
            data_type="float64",
            feature_family="weekly_seasonality",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="demand_dow_median_8w",
            description="Median demand for the same weekday over the prior 8 weeks",
            source_columns=demand_col,
            calculation="rolling_median(8) on same weekday demand shifted by 1",
            data_type="float64",
            feature_family="weekly_seasonality",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
    ])

    # 6. Annual Seasonality Features
    annual_exprs = [
        pl.col("__demand_hist_t1").shift(364).over(["store_id", "product_id"]).alias("demand_lag_365"),
        pl.col("__demand_hist_t1").shift(363).rolling_mean(window_size=7).over(["store_id", "product_id"]).alias("demand_prior_year_week_mean"),
    ]
    df = df.with_columns(annual_exprs)
    definitions.extend([
        FeatureDefinition(
            name="demand_lag_365",
            description="Estimated true demand exactly 365 days prior (calendar year lag)",
            source_columns=demand_col,
            calculation="shift(365) of training-safe demand",
            data_type="float64",
            feature_family="annual_seasonality",
            known_at_prediction_time=True,
            expected_missingness="medium",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="demand_prior_year_week_mean",
            description="7-day mean demand centered around 52 weeks prior",
            source_columns=demand_col,
            calculation="7-day rolling mean around lag 364",
            data_type="float64",
            feature_family="annual_seasonality",
            known_at_prediction_time=True,
            expected_missingness="medium",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
    ])

    # Drop temporary calculation columns
    df = df.drop(["__demand_hist_t1", "__dow_temp"])

    return df, definitions
