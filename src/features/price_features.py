"""
Price Features Module.
Generates purchase-price lags, rolling statistics, price momentum,
selling-price dynamics, and price elasticity proxies.
"""

from typing import List, Tuple
import polars as pl
from src.features.feature_registry import FeatureDefinition


def generate_price_features(
    df: pl.DataFrame,
    purchase_price_col: str = "purchase_price",
    selling_price_col: str = "selling_price",
    normal_selling_price_col: str = "normal_selling_price",
) -> Tuple[pl.DataFrame, List[FeatureDefinition]]:
    """
    Generate leakage-safe purchase price and selling price features.
    
    Purchase price lags use observations known up to T (for lag 1: shift(1) is T-1).
    Rolling statistics use shift(1) to avoid target-period contamination.
    """
    definitions: List[FeatureDefinition] = []

    df = df.sort(["store_id", "product_id", "date"])

    # Base shifted purchase price
    df = df.with_columns(
        pl.col(purchase_price_col).cast(pl.Float64).shift(1).over(["store_id", "product_id"]).alias("__pp_hist_t1")
    )

    # 1. Purchase price lags
    pp_lags = [1, 3, 7, 14, 21, 30, 60, 90]
    lag_exprs = []
    for lag in pp_lags:
        col_name = f"purchase_price_lag_{lag}"
        lag_exprs.append(
            pl.col("__pp_hist_t1")
            .shift(lag - 1)
            .over(["store_id", "product_id"])
            .alias(col_name)
        )
        definitions.append(
            FeatureDefinition(
                name=col_name,
                description=f"Purchase price {lag} day(s) prior to observation date T",
                source_columns=purchase_price_col,
                calculation=f"shift({lag}) of purchase_price",
                data_type="float64",
                feature_family="purchase_price_history",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="both",
                leakage_risk="none",
            )
        )
    df = df.with_columns(lag_exprs)

    # 2. Purchase price rolling statistics
    pp_windows = [7, 14, 30, 60, 90]
    roll_exprs = []
    for w in pp_windows:
        col_mean = f"purchase_price_mean_{w}d"
        col_std = f"purchase_price_std_{w}d"
        col_min = f"purchase_price_min_{w}d"
        col_max = f"purchase_price_max_{w}d"
        col_vol = f"purchase_price_volatility_{w}d"

        roll_exprs.extend([
            pl.col("__pp_hist_t1").rolling_mean(window_size=w).over(["store_id", "product_id"]).alias(col_mean),
            pl.col("__pp_hist_t1").rolling_std(window_size=w).over(["store_id", "product_id"]).alias(col_std),
            pl.col("__pp_hist_t1").rolling_min(window_size=w).over(["store_id", "product_id"]).alias(col_min),
            pl.col("__pp_hist_t1").rolling_max(window_size=w).over(["store_id", "product_id"]).alias(col_max),
        ])
        definitions.extend([
            FeatureDefinition(
                name=col_mean,
                description=f"Rolling {w}-day mean of purchase price prior to T",
                source_columns=purchase_price_col,
                calculation=f"rolling_mean({w}) on purchase_price shifted by 1",
                data_type="float64",
                feature_family="purchase_price_rolling",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="both",
                leakage_risk="none",
            ),
            FeatureDefinition(
                name=col_std,
                description=f"Rolling {w}-day standard deviation of purchase price",
                source_columns=purchase_price_col,
                calculation=f"rolling_std({w}) on purchase_price shifted by 1",
                data_type="float64",
                feature_family="purchase_price_rolling",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="both",
                leakage_risk="none",
            ),
            FeatureDefinition(
                name=col_min,
                description=f"Rolling {w}-day minimum purchase price",
                source_columns=purchase_price_col,
                calculation=f"rolling_min({w}) on purchase_price shifted by 1",
                data_type="float64",
                feature_family="purchase_price_rolling",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="both",
                leakage_risk="none",
            ),
            FeatureDefinition(
                name=col_max,
                description=f"Rolling {w}-day maximum purchase price",
                source_columns=purchase_price_col,
                calculation=f"rolling_max({w}) on purchase_price shifted by 1",
                data_type="float64",
                feature_family="purchase_price_rolling",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="both",
                leakage_risk="none",
            ),
        ])
    df = df.with_columns(roll_exprs)

    # Volatility for 30d
    df = df.with_columns(
        (pl.col("purchase_price_std_30d") / (pl.col("purchase_price_mean_30d") + 1e-6)).alias("purchase_price_volatility_30d")
    )
    definitions.append(
        FeatureDefinition(
            name="purchase_price_volatility_30d",
            description="Rolling 30-day volatility (std/mean) of purchase price",
            source_columns=purchase_price_col,
            calculation="purchase_price_std_30d / (purchase_price_mean_30d + 1e-6)",
            data_type="float64",
            feature_family="purchase_price_rolling",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        )
    )

    # 3. Price momentum features
    pp_cur = pl.col(purchase_price_col).cast(pl.Float64)
    momentum_exprs = [
        ((pp_cur - pl.col("purchase_price_lag_1")) / (pl.col("purchase_price_lag_1") + 1e-6)).alias("purchase_price_pct_change_1d"),
        ((pp_cur - pl.col("purchase_price_lag_7")) / (pl.col("purchase_price_lag_7") + 1e-6)).alias("purchase_price_pct_change_7d"),
        ((pp_cur - pl.col("purchase_price_lag_30")) / (pl.col("purchase_price_lag_30") + 1e-6)).alias("purchase_price_pct_change_30d"),
        (pp_cur / (pl.col("purchase_price_mean_30d") + 1e-6)).alias("purchase_price_vs_mean_30d"),
        (pp_cur / (pl.col("purchase_price_mean_90d") + 1e-6)).alias("purchase_price_vs_mean_90d"),
    ]
    definitions.extend([
        FeatureDefinition(
            name="purchase_price_pct_change_1d",
            description="1-day percentage change in purchase price",
            source_columns=purchase_price_col,
            calculation="(purchase_price - lag_1) / (lag_1 + 1e-6)",
            data_type="float64",
            feature_family="price_momentum",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="purchase_price_pct_change_7d",
            description="7-day percentage change in purchase price",
            source_columns=purchase_price_col,
            calculation="(purchase_price - lag_7) / (lag_7 + 1e-6)",
            data_type="float64",
            feature_family="price_momentum",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="purchase_price_pct_change_30d",
            description="30-day percentage change in purchase price",
            source_columns=purchase_price_col,
            calculation="(purchase_price - lag_30) / (lag_30 + 1e-6)",
            data_type="float64",
            feature_family="price_momentum",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="purchase_price_vs_mean_30d",
            description="Current purchase price relative to 30-day historical mean",
            source_columns=purchase_price_col,
            calculation="purchase_price / (purchase_price_mean_30d + 1e-6)",
            data_type="float64",
            feature_family="price_momentum",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="purchase_price_vs_mean_90d",
            description="Current purchase price relative to 90-day historical mean",
            source_columns=purchase_price_col,
            calculation="purchase_price / (purchase_price_mean_90d + 1e-6)",
            data_type="float64",
            feature_family="price_momentum",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
    ])
    df = df.with_columns(momentum_exprs)

    # Price acceleration
    df = df.with_columns(
        (pl.col("purchase_price_pct_change_7d") - pl.col("purchase_price_pct_change_7d").shift(7).over(["store_id", "product_id"])).alias("purchase_price_acceleration")
    )
    definitions.append(
        FeatureDefinition(
            name="purchase_price_acceleration",
            description="Change in 7-day purchase price momentum over prior 7 days",
            source_columns=purchase_price_col,
            calculation="pct_change_7d - shift(7) of pct_change_7d",
            data_type="float64",
            feature_family="price_momentum",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        )
    )

    # 4. Selling price features
    sp_cur = pl.col(selling_price_col).cast(pl.Float64)
    nsp_cur = pl.col(normal_selling_price_col).cast(pl.Float64)
    df = df.with_columns(
        sp_cur.shift(1).over(["store_id", "product_id"]).alias("__sp_hist_t1")
    )

    selling_exprs = [
        (nsp_cur - sp_cur).alias("discount_amount"),
        ((nsp_cur - sp_cur) / (nsp_cur + 1e-6) * 100.0).alias("discount_percentage_calculated"),
        (sp_cur / (nsp_cur + 1e-6)).alias("selling_price_vs_normal_ratio"),
        pl.col("__sp_hist_t1").rolling_mean(window_size=7).over(["store_id", "product_id"]).alias("selling_price_mean_7d"),
        pl.col("__sp_hist_t1").rolling_mean(window_size=28).over(["store_id", "product_id"]).alias("selling_price_mean_28d"),
    ]
    definitions.extend([
        FeatureDefinition(
            name="discount_amount",
            description="Absolute currency difference between normal and actual selling price",
            source_columns=f"{normal_selling_price_col}, {selling_price_col}",
            calculation="normal_selling_price - selling_price",
            data_type="float64",
            feature_family="selling_price",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="discount_percentage_calculated",
            description="Percentage discount calculated from normal selling price",
            source_columns=f"{normal_selling_price_col}, {selling_price_col}",
            calculation="(normal_price - selling_price) / normal_price * 100",
            data_type="float64",
            feature_family="selling_price",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="selling_price_vs_normal_ratio",
            description="Ratio of current selling price to regular normal price",
            source_columns=f"{selling_price_col}, {normal_selling_price_col}",
            calculation="selling_price / (normal_selling_price + 1e-6)",
            data_type="float64",
            feature_family="selling_price",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="selling_price_mean_7d",
            description="7-day rolling mean of historical selling prices prior to T",
            source_columns=selling_price_col,
            calculation="rolling_mean(7) on selling_price shifted by 1",
            data_type="float64",
            feature_family="selling_price",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="selling_price_mean_28d",
            description="28-day rolling mean of historical selling prices prior to T",
            source_columns=selling_price_col,
            calculation="rolling_mean(28) on selling_price shifted by 1",
            data_type="float64",
            feature_family="selling_price",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
    ])
    df = df.with_columns(selling_exprs)

    # Selling price relative and changes
    df = df.with_columns([
        (sp_cur / (pl.col("selling_price_mean_7d") + 1e-6)).alias("selling_price_vs_mean_7d"),
        ((sp_cur - pl.col("__sp_hist_t1").shift(6).over(["store_id", "product_id"])) / (pl.col("__sp_hist_t1").shift(6).over(["store_id", "product_id"]) + 1e-6)).alias("selling_price_pct_change_7d"),
    ])
    definitions.extend([
        FeatureDefinition(
            name="selling_price_vs_mean_7d",
            description="Current selling price relative to prior 7-day average selling price",
            source_columns=selling_price_col,
            calculation="selling_price / (selling_price_mean_7d + 1e-6)",
            data_type="float64",
            feature_family="selling_price",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="selling_price_pct_change_7d",
            description="7-day percentage change in selling price",
            source_columns=selling_price_col,
            calculation="(selling_price - lag_7) / (lag_7 + 1e-6)",
            data_type="float64",
            feature_family="selling_price",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
    ])

    # 5. Price Elasticity Proxies
    # Selling price relative to product's historical median price
    df = df.with_columns(
        sp_cur.median().over("product_id").alias("__prod_median_price")
    )
    df = df.with_columns(
        (sp_cur / (pl.col("__prod_median_price") + 1e-6)).alias("selling_price_vs_product_median")
    )
    definitions.append(
        FeatureDefinition(
            name="selling_price_vs_product_median",
            description="Selling price relative to long-term median price for this product",
            source_columns=selling_price_col,
            calculation="selling_price / (median(selling_price over product) + 1e-6)",
            data_type="float64",
            feature_family="price_elasticity_proxy",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        )
    )

    # Elasticity proxy: demand_growth_7d / (selling_price_pct_change_7d + 1e-6) clipped to [-10, 10]
    if "demand_growth_7d" in df.columns:
        df = df.with_columns(
            pl.min_horizontal([
                pl.max_horizontal([
                    pl.col("demand_growth_7d") / (pl.col("selling_price_pct_change_7d") + 1e-6),
                    pl.lit(-10.0)
                ]),
                pl.lit(10.0)
            ]).alias("price_elasticity_proxy_7d")
        )
        definitions.append(
            FeatureDefinition(
                name="price_elasticity_proxy_7d",
                description="Descriptive 7-day arc elasticity proxy clipped to [-10, 10]",
                source_columns="demand_growth_7d, selling_price_pct_change_7d",
                calculation="clip(demand_growth_7d / (selling_price_pct_change_7d + 1e-6), -10, 10)",
                data_type="float64",
                feature_family="price_elasticity_proxy",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="demand",
                leakage_risk="none",
            )
        )

    df = df.drop(["__pp_hist_t1", "__sp_hist_t1", "__prod_median_price"])

    return df, definitions
