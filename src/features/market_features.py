"""
Market Features Module.
Generates competitive benchmarks from PriceCatcher market prices including
median/mean prices, retailer price differentials, and market volatility.
"""

from typing import List, Tuple
import polars as pl
from src.features.feature_registry import FeatureDefinition


def generate_market_features(
    df: pl.DataFrame,
    selling_price_col: str = "selling_price",
) -> Tuple[pl.DataFrame, List[FeatureDefinition]]:
    """
    Generate PriceCatcher market price features and competitive price differentials.
    """
    definitions: List[FeatureDefinition] = []

    df = df.sort(["store_id", "product_id", "date"])

    # 1. Base market price columns
    m_med = pl.col("market_median_price").cast(pl.Float64)
    m_mean = pl.col("market_mean_price").cast(pl.Float64)
    m_min = pl.col("market_minimum_price").cast(pl.Float64)
    m_max = pl.col("market_maximum_price").cast(pl.Float64)
    m_cnt = pl.col("market_observation_count").cast(pl.Float64)
    sp_cur = pl.col(selling_price_col).cast(pl.Float64)

    df = df.with_columns([
        m_med.alias("market_price_median"),
        m_mean.alias("market_price_mean"),
        m_min.alias("market_price_min"),
        m_max.alias("market_price_max"),
        m_cnt.alias("market_price_observation_count"),
        (sp_cur - m_med).alias("price_vs_market_diff"),
        (sp_cur / (m_med + 1e-6)).alias("price_vs_market_ratio"),
    ])
    definitions.extend([
        FeatureDefinition(
            name="market_price_median",
            description="Aggregated market median price across premise competitors from PriceCatcher",
            source_columns="market_median_price",
            calculation="Market benchmark median",
            data_type="float64",
            feature_family="market_price",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="market_price_mean",
            description="Aggregated market mean price across premise competitors from PriceCatcher",
            source_columns="market_mean_price",
            calculation="Market benchmark mean",
            data_type="float64",
            feature_family="market_price",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="market_price_min",
            description="Minimum competitor price observed in PriceCatcher market survey",
            source_columns="market_minimum_price",
            calculation="Market benchmark min",
            data_type="float64",
            feature_family="market_price",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="market_price_max",
            description="Maximum competitor price observed in PriceCatcher market survey",
            source_columns="market_maximum_price",
            calculation="Market benchmark max",
            data_type="float64",
            feature_family="market_price",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="market_price_observation_count",
            description="Total survey observation count in PriceCatcher on observation date",
            source_columns="market_observation_count",
            calculation="Survey observation sample size",
            data_type="float64",
            feature_family="market_price",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="price_vs_market_diff",
            description="Retailer selling price minus market median price (premium or discount)",
            source_columns=f"{selling_price_col}, market_median_price",
            calculation="selling_price - market_median_price",
            data_type="float64",
            feature_family="market_price",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="price_vs_market_ratio",
            description="Ratio of retailer selling price to market median price",
            source_columns=f"{selling_price_col}, market_median_price",
            calculation="selling_price / (market_median_price + 1e-6)",
            data_type="float64",
            feature_family="market_price",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
    ])

    # 2. Market price historical changes and volatility (shifted by 1 day)
    df = df.with_columns(
        m_med.shift(1).over(["store_id", "product_id"]).alias("__m_med_t1")
    )

    roll_exprs = [
        ((pl.col("__m_med_t1") - pl.col("__m_med_t1").shift(6).over(["store_id", "product_id"])) / (pl.col("__m_med_t1").shift(6).over(["store_id", "product_id"]) + 1e-6)).alias("market_median_change_7d"),
        ((pl.col("__m_med_t1") - pl.col("__m_med_t1").shift(29).over(["store_id", "product_id"])) / (pl.col("__m_med_t1").shift(29).over(["store_id", "product_id"]) + 1e-6)).alias("market_median_change_30d"),
        pl.col("__m_med_t1").rolling_mean(window_size=30).over(["store_id", "product_id"]).alias("__m_med_mean_30d"),
        pl.col("__m_med_t1").rolling_std(window_size=30).over(["store_id", "product_id"]).alias("__m_med_std_30d"),
    ]
    df = df.with_columns(roll_exprs)

    df = df.with_columns(
        (pl.col("__m_med_std_30d") / (pl.col("__m_med_mean_30d") + 1e-6)).alias("market_median_volatility_30d")
    )
    definitions.extend([
        FeatureDefinition(
            name="market_median_change_7d",
            description="7-day percentage change in market median price prior to T",
            source_columns="market_median_price",
            calculation="(m_med_t1 - shift(7)) / shift(7)",
            data_type="float64",
            feature_family="market_price",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="market_median_change_30d",
            description="30-day percentage change in market median price prior to T",
            source_columns="market_median_price",
            calculation="(m_med_t1 - shift(30)) / shift(30)",
            data_type="float64",
            feature_family="market_price",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="market_median_volatility_30d",
            description="30-day rolling volatility (std/mean) of market median price prior to T",
            source_columns="market_median_price",
            calculation="rolling_std(30) / (rolling_mean(30) + 1e-6)",
            data_type="float64",
            feature_family="market_price",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
    ])

    df = df.drop(["__m_med_t1", "__m_med_mean_30d", "__m_med_std_30d"])

    return df, definitions
