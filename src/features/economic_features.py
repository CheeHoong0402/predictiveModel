"""
Economic Features Module.
Generates fuel prices, Brent crude oil benchmarks, foreign exchange rates,
agricultural commodity prices, Consumer Price Index, and Producer Price Index.
"""

from typing import List, Tuple
import polars as pl
from src.features.feature_registry import FeatureDefinition


def generate_economic_features(
    df: pl.DataFrame,
) -> Tuple[pl.DataFrame, List[FeatureDefinition]]:
    """
    Generate macroeconomic and commodity features respecting historical publication timing.
    """
    definitions: List[FeatureDefinition] = []

    df = df.sort(["store_id", "product_id", "date"])

    # 1. Fuel Prices
    petrol_sub = pl.col("subsidised_petrol_price").cast(pl.Float64)
    petrol_unsub = pl.col("unsubsidised_petrol_price").cast(pl.Float64)
    diesel_sub = pl.col("subsidised_diesel_price").cast(pl.Float64)
    diesel_unsub = pl.col("unsubsidised_diesel_price").cast(pl.Float64)

    # Effective fuel price (subsidised if > 0 else unsubsidised)
    eff_diesel = pl.when(diesel_sub > 0.0).then(diesel_sub).otherwise(diesel_unsub)
    eff_petrol = pl.when(petrol_sub > 0.0).then(petrol_sub).otherwise(petrol_unsub)

    df = df.with_columns([
        petrol_sub.alias("fuel_petrol_subsidised"),
        petrol_unsub.alias("fuel_petrol_unsubsidised"),
        diesel_sub.alias("fuel_diesel_subsidised"),
        diesel_unsub.alias("fuel_diesel_unsubsidised"),
        eff_diesel.alias("fuel_diesel_effective"),
        eff_petrol.alias("fuel_petrol_effective"),
        pl.int_range(0, pl.len()).over(["store_id", "product_id"]).alias("__t_idx"),
    ])
    definitions.extend([
        FeatureDefinition(
            name="fuel_petrol_subsidised",
            description="Official gazetted subsidised petrol price (RON95) in MYR/litre",
            source_columns="subsidised_petrol_price",
            calculation="Official weekly gazetted price",
            data_type="float64",
            feature_family="macro_fuel",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="fuel_petrol_unsubsidised",
            description="Market unsubsidised petrol price (RON97) in MYR/litre",
            source_columns="unsubsidised_petrol_price",
            calculation="Official weekly gazetted price",
            data_type="float64",
            feature_family="macro_fuel",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="fuel_diesel_subsidised",
            description="Official subsidised diesel price in MYR/litre (0.0 after rationalisation float)",
            source_columns="subsidised_diesel_price",
            calculation="Official weekly gazetted price",
            data_type="float64",
            feature_family="macro_fuel",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="fuel_diesel_unsubsidised",
            description="Market retail diesel price (Euro 5 B10/B7) in MYR/litre",
            source_columns="unsubsidised_diesel_price",
            calculation="Official weekly gazetted price",
            data_type="float64",
            feature_family="macro_fuel",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="fuel_diesel_effective",
            description="Effective retail diesel price paid at pump in MYR/litre",
            source_columns="subsidised_diesel_price, unsubsidised_diesel_price",
            calculation="subsidised if > 0 else unsubsidised",
            data_type="float64",
            feature_family="macro_fuel",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="fuel_petrol_effective",
            description="Effective retail petrol price paid at pump in MYR/litre",
            source_columns="subsidised_petrol_price, unsubsidised_petrol_price",
            calculation="subsidised if > 0 else unsubsidised",
            data_type="float64",
            feature_family="macro_fuel",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
    ])

    # Fuel percentage changes
    df = df.with_columns([
        ((pl.col("fuel_petrol_effective") - pl.col("fuel_petrol_effective").shift(7).over(["store_id", "product_id"])) / (pl.col("fuel_petrol_effective").shift(7).over(["store_id", "product_id"]) + 1e-6) * 100.0).alias("fuel_petrol_pct_change_7d"),
        ((pl.col("fuel_petrol_effective") - pl.col("fuel_petrol_effective").shift(28).over(["store_id", "product_id"])) / (pl.col("fuel_petrol_effective").shift(28).over(["store_id", "product_id"]) + 1e-6) * 100.0).alias("fuel_petrol_pct_change_28d"),
        ((pl.col("fuel_diesel_effective") - pl.col("fuel_diesel_effective").shift(7).over(["store_id", "product_id"])) / (pl.col("fuel_diesel_effective").shift(7).over(["store_id", "product_id"]) + 1e-6) * 100.0).alias("fuel_diesel_pct_change_7d"),
        ((pl.col("fuel_diesel_effective") - pl.col("fuel_diesel_effective").shift(28).over(["store_id", "product_id"])) / (pl.col("fuel_diesel_effective").shift(28).over(["store_id", "product_id"]) + 1e-6) * 100.0).alias("fuel_diesel_pct_change_28d"),
    ])
    definitions.extend([
        FeatureDefinition(
            name="fuel_petrol_pct_change_7d",
            description="7-day percentage change in effective petrol price",
            source_columns="subsidised_petrol_price",
            calculation="(petrol - lag_7) / lag_7 * 100",
            data_type="float64",
            feature_family="macro_fuel",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="fuel_petrol_pct_change_28d",
            description="28-day percentage change in effective petrol price",
            source_columns="subsidised_petrol_price",
            calculation="(petrol - lag_28) / lag_28 * 100",
            data_type="float64",
            feature_family="macro_fuel",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="fuel_diesel_pct_change_7d",
            description="7-day percentage change in effective diesel price",
            source_columns="unsubsidised_diesel_price",
            calculation="(diesel - lag_7) / lag_7 * 100",
            data_type="float64",
            feature_family="macro_fuel",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="fuel_diesel_pct_change_28d",
            description="28-day percentage change in effective diesel price",
            source_columns="unsubsidised_diesel_price",
            calculation="(diesel - lag_28) / lag_28 * 100",
            data_type="float64",
            feature_family="macro_fuel",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
    ])

    # 2. Foreign Exchange Rates
    fx_cur = pl.col("relevant_exchange_rate").cast(pl.Float64)
    df = df.with_columns(
        fx_cur.shift(1).over(["store_id", "product_id"]).alias("__fx_t1")
    )

    df = df.with_columns([
        fx_cur.alias("fx_rate"),
        (fx_cur - pl.col("__fx_t1")).alias("fx_change_1d"),
        (fx_cur - pl.col("__fx_t1").shift(6).over(["store_id", "product_id"])).alias("fx_change_7d"),
        (fx_cur - pl.col("__fx_t1").shift(29).over(["store_id", "product_id"])).alias("fx_change_30d"),
        ((fx_cur - pl.col("__fx_t1").shift(6).over(["store_id", "product_id"])) / (pl.col("__fx_t1").shift(6).over(["store_id", "product_id"]) + 1e-6) * 100.0).alias("fx_pct_change_7d"),
        ((fx_cur - pl.col("__fx_t1").shift(29).over(["store_id", "product_id"])) / (pl.col("__fx_t1").shift(29).over(["store_id", "product_id"]) + 1e-6) * 100.0).alias("fx_pct_change_30d"),
        pl.col("__fx_t1").rolling_mean(window_size=30).over(["store_id", "product_id"]).alias("fx_mean_30d"),
        pl.col("__fx_t1").rolling_std(window_size=30).over(["store_id", "product_id"]).alias("__fx_std_30d"),
    ])
    df = df.with_columns([
        (pl.col("__fx_std_30d") / (pl.col("fx_mean_30d") + 1e-6)).alias("fx_volatility_30d"),
        (fx_cur / (pl.col("fx_mean_30d") + 1e-6)).alias("fx_vs_mean_30d"),
    ])
    definitions.extend([
        FeatureDefinition(
            name="fx_rate",
            description="Relevant bilateral foreign exchange rate per MYR",
            source_columns="relevant_exchange_rate",
            calculation="BNM official reference exchange rate",
            data_type="float64",
            feature_family="macro_currency",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="fx_change_1d",
            description="1-day change in relevant exchange rate",
            source_columns="relevant_exchange_rate",
            calculation="fx_rate - lag_1",
            data_type="float64",
            feature_family="macro_currency",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="fx_pct_change_7d",
            description="7-day percentage change in relevant exchange rate",
            source_columns="relevant_exchange_rate",
            calculation="(fx_rate - lag_7) / lag_7 * 100",
            data_type="float64",
            feature_family="macro_currency",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="fx_pct_change_30d",
            description="30-day percentage change in relevant exchange rate",
            source_columns="relevant_exchange_rate",
            calculation="(fx_rate - lag_30) / lag_30 * 100",
            data_type="float64",
            feature_family="macro_currency",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="fx_mean_30d",
            description="30-day rolling mean exchange rate prior to T",
            source_columns="relevant_exchange_rate",
            calculation="rolling_mean(30) on fx_rate shifted by 1",
            data_type="float64",
            feature_family="macro_currency",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="fx_volatility_30d",
            description="30-day rolling volatility (std/mean) of exchange rate prior to T",
            source_columns="relevant_exchange_rate",
            calculation="rolling_std(30) / (rolling_mean(30) + 1e-6)",
            data_type="float64",
            feature_family="macro_currency",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="fx_vs_mean_30d",
            description="Current exchange rate relative to 30-day historical mean",
            source_columns="relevant_exchange_rate",
            calculation="fx_rate / (fx_mean_30d + 1e-6)",
            data_type="float64",
            feature_family="macro_currency",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
    ])

    # 3. Mapped Commodity Prices
    comm_cur = pl.col("relevant_commodity_price").cast(pl.Float64)
    df = df.with_columns(
        comm_cur.shift(1).over(["store_id", "product_id"]).alias("__comm_t1")
    )

    df = df.with_columns([
        comm_cur.alias("commodity_price"),
        pl.col("__comm_t1").rolling_mean(window_size=7).over(["store_id", "product_id"]).alias("commodity_mean_7d"),
        pl.col("__comm_t1").rolling_mean(window_size=30).over(["store_id", "product_id"]).alias("commodity_mean_30d"),
        pl.col("__comm_t1").rolling_std(window_size=30).over(["store_id", "product_id"]).alias("__comm_std_30d"),
        ((comm_cur - pl.col("__comm_t1").shift(6).over(["store_id", "product_id"])) / (pl.col("__comm_t1").shift(6).over(["store_id", "product_id"]) + 1e-6) * 100.0).alias("commodity_pct_change_7d"),
        ((comm_cur - pl.col("__comm_t1").shift(29).over(["store_id", "product_id"])) / (pl.col("__comm_t1").shift(29).over(["store_id", "product_id"]) + 1e-6) * 100.0).alias("commodity_pct_change_30d"),
    ])
    df = df.with_columns([
        (pl.col("__comm_std_30d") / (pl.col("commodity_mean_30d") + 1e-6)).alias("commodity_volatility_30d"),
        (comm_cur / (pl.col("commodity_mean_30d") + 1e-6)).alias("commodity_vs_mean_30d"),
    ])
    definitions.extend([
        FeatureDefinition(
            name="commodity_price",
            description="Relevant global commodity market price mapped to product category",
            source_columns="relevant_commodity_price",
            calculation="Mapped benchmark commodity spot price",
            data_type="float64",
            feature_family="macro_commodity",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="commodity_mean_7d",
            description="7-day rolling mean of mapped commodity price prior to T",
            source_columns="relevant_commodity_price",
            calculation="rolling_mean(7) on commodity_price shifted by 1",
            data_type="float64",
            feature_family="macro_commodity",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="commodity_mean_30d",
            description="30-day rolling mean of mapped commodity price prior to T",
            source_columns="relevant_commodity_price",
            calculation="rolling_mean(30) on commodity_price shifted by 1",
            data_type="float64",
            feature_family="macro_commodity",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="commodity_pct_change_7d",
            description="7-day percentage change in mapped commodity price",
            source_columns="relevant_commodity_price",
            calculation="(commodity - lag_7) / lag_7 * 100",
            data_type="float64",
            feature_family="macro_commodity",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="commodity_pct_change_30d",
            description="30-day percentage change in mapped commodity price",
            source_columns="relevant_commodity_price",
            calculation="(commodity - lag_30) / lag_30 * 100",
            data_type="float64",
            feature_family="macro_commodity",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="commodity_volatility_30d",
            description="30-day rolling volatility (std/mean) of mapped commodity price prior to T",
            source_columns="relevant_commodity_price",
            calculation="rolling_std(30) / (rolling_mean(30) + 1e-6)",
            data_type="float64",
            feature_family="macro_commodity",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="commodity_vs_mean_30d",
            description="Current mapped commodity price relative to 30-day historical mean",
            source_columns="relevant_commodity_price",
            calculation="commodity_price / (commodity_mean_30d + 1e-6)",
            data_type="float64",
            feature_family="macro_commodity",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
    ])

    # 4. Consumer Price Index and Producer Price Index
    cpi_cur = pl.col("Consumer_Price_Index").cast(pl.Float64)
    ppi_cur = pl.col("Producer_Price_Index").cast(pl.Float64)
    cpi_cat = pl.col("relevant_category_Consumer_Price_Index").cast(pl.Float64)
    ppi_ind = pl.col("relevant_industry_Producer_Price_Index").cast(pl.Float64)

    df = df.with_columns([
        cpi_cur.alias("cpi_headline"),
        cpi_cat.alias("cpi_category"),
        ppi_cur.alias("ppi_headline"),
        ppi_ind.alias("ppi_industry"),
        ((cpi_cur - cpi_cur.shift(30).over(["store_id", "product_id"])) / (cpi_cur.shift(30).over(["store_id", "product_id"]) + 1e-6) * 100.0).alias("cpi_pct_change_30d"),
        ((cpi_cur - cpi_cur.shift(365).over(["store_id", "product_id"])) / (cpi_cur.shift(365).over(["store_id", "product_id"]) + 1e-6) * 100.0).alias("cpi_pct_change_365d"),
        ((ppi_cur - ppi_cur.shift(30).over(["store_id", "product_id"])) / (ppi_cur.shift(30).over(["store_id", "product_id"]) + 1e-6) * 100.0).alias("ppi_pct_change_30d"),
        ((ppi_cur - ppi_cur.shift(365).over(["store_id", "product_id"])) / (ppi_cur.shift(365).over(["store_id", "product_id"]) + 1e-6) * 100.0).alias("ppi_pct_change_365d"),
    ])
    definitions.extend([
        FeatureDefinition(
            name="cpi_headline",
            description="Headline Malaysia Consumer Price Index (base 2010=100)",
            source_columns="Consumer_Price_Index",
            calculation="Official DOSM published monthly CPI (lagged to publication release)",
            data_type="float64",
            feature_family="macro_inflation",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="cpi_category",
            description="Category-specific Consumer Price Index (Food & Non-Alcoholic Beverages)",
            source_columns="relevant_category_Consumer_Price_Index",
            calculation="Official DOSM published monthly category CPI",
            data_type="float64",
            feature_family="macro_inflation",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="ppi_headline",
            description="Overall Malaysia Producer Price Index for Local Production (base 2010=100)",
            source_columns="Producer_Price_Index",
            calculation="Official DOSM published monthly PPI",
            data_type="float64",
            feature_family="macro_inflation",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="ppi_industry",
            description="Industry-specific Producer Price Index (Agriculture/Manufacturing)",
            source_columns="relevant_industry_Producer_Price_Index",
            calculation="Official DOSM published monthly industry PPI",
            data_type="float64",
            feature_family="macro_inflation",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="cpi_pct_change_30d",
            description="30-day percentage change in headline Consumer Price Index",
            source_columns="Consumer_Price_Index",
            calculation="(cpi - shift(30)) / shift(30) * 100",
            data_type="float64",
            feature_family="macro_inflation",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="cpi_pct_change_365d",
            description="Annual (year-over-year) percentage inflation in headline CPI",
            source_columns="Consumer_Price_Index",
            calculation="(cpi - shift(365)) / shift(365) * 100",
            data_type="float64",
            feature_family="macro_inflation",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="ppi_pct_change_30d",
            description="30-day percentage change in headline Producer Price Index",
            source_columns="Producer_Price_Index",
            calculation="(ppi - shift(30)) / shift(30) * 100",
            data_type="float64",
            feature_family="macro_inflation",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="ppi_pct_change_365d",
            description="Annual (year-over-year) percentage change in headline PPI",
            source_columns="Producer_Price_Index",
            calculation="(ppi - shift(365)) / shift(365) * 100",
            data_type="float64",
            feature_family="macro_inflation",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
    ])

    df = df.drop(["__t_idx", "__fx_t1", "__fx_std_30d", "__comm_t1", "__comm_std_30d"])

    return df, definitions
