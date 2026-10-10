"""
Interaction Features Module.
Generates economically meaningful and interpretable cross-domain interaction terms.
"""

from typing import List, Tuple
import polars as pl
from src.features.feature_registry import FeatureDefinition


def generate_interaction_features(
    df: pl.DataFrame,
) -> Tuple[pl.DataFrame, List[FeatureDefinition]]:
    """
    Generate domain-specific interaction features combining promotions,
    festivals, weather, energy prices, exchange rates, and cash transfers.
    """
    definitions: List[FeatureDefinition] = []

    # Safe references
    promo_flag = pl.col("promo_active_flag") if "promo_active_flag" in df.columns else pl.lit(0.0)
    fest_window = pl.col("major_festival_window_active") if "major_festival_window_active" in df.columns else pl.lit(0.0)
    disc_pct = pl.col("promo_discount_percentage") if "promo_discount_percentage" in df.columns else pl.lit(0.0)
    rain_7d = pl.col("weather_rainfall_sum_7d") if "weather_rainfall_sum_7d" in df.columns else pl.lit(0.0)
    temp_7d = pl.col("weather_temp_mean_7d") if "weather_temp_mean_7d" in df.columns else pl.lit(0.0)
    diesel_chg = pl.col("fuel_diesel_pct_change_7d") if "fuel_diesel_pct_change_7d" in df.columns else pl.lit(0.0)
    lt_exp = pl.col("supplier_lead_time_expected") if "supplier_lead_time_expected" in df.columns else pl.lit(0.0)
    fx_chg = pl.col("fx_pct_change_7d") if "fx_pct_change_7d" in df.columns else pl.lit(0.0)
    imp_flag = pl.col("product_imported_flag") if "product_imported_flag" in df.columns else pl.lit(0.0)
    comm_chg = pl.col("commodity_pct_change_7d") if "commodity_pct_change_7d" in df.columns else pl.lit(0.0)
    is_comm = (pl.col("commodity_price") > 0.0).cast(pl.Float64) if "commodity_price" in df.columns else pl.lit(0.0)
    post_gov = pl.col("is_post_gov_assist_7d") if "is_post_gov_assist_7d" in df.columns else pl.lit(0.0)
    ctrl_goods = pl.col("product_is_controlled_goods") if "product_is_controlled_goods" in df.columns else pl.lit(0.0)
    price_gap = pl.col("price_vs_market_diff") if "price_vs_market_diff" in df.columns else pl.lit(0.0)

    is_veg = (pl.col("category") == "Fresh Vegetables").cast(pl.Float64)
    is_poul = (pl.col("category") == "Poultry & Meat").cast(pl.Float64)

    df = df.with_columns([
        (promo_flag * fest_window).alias("interaction_promo_x_festival"),
        (promo_flag * (disc_pct / 100.0)).alias("interaction_promo_x_discount"),
        (rain_7d * is_veg).alias("interaction_rainfall_x_vegetables"),
        (temp_7d * is_poul).alias("interaction_temp_x_poultry"),
        (diesel_chg * lt_exp).alias("interaction_diesel_change_x_lead_time"),
        (fx_chg * imp_flag).alias("interaction_fx_change_x_imported"),
        (comm_chg * is_comm).alias("interaction_commodity_change_x_product"),
        (post_gov * ctrl_goods).alias("interaction_gov_assist_x_essential"),
        (price_gap * promo_flag).alias("interaction_price_gap_x_promo"),
    ])

    definitions.extend([
        FeatureDefinition(
            name="interaction_promo_x_festival",
            description="Interaction between active promotion and festive season proximity window",
            source_columns="promo_active_flag, major_festival_window_active",
            calculation="promo_active_flag * major_festival_window_active",
            data_type="float64",
            feature_family="interactions",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="interaction_promo_x_discount",
            description="Magnitude of promotion discount intensity",
            source_columns="promo_active_flag, promo_discount_percentage",
            calculation="promo_active_flag * (discount_percentage / 100)",
            data_type="float64",
            feature_family="interactions",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="interaction_rainfall_x_vegetables",
            description="Recent 7-day cumulative rainfall interaction with perishable fresh vegetables",
            source_columns="weather_rainfall_sum_7d, category",
            calculation="rainfall_sum_7d * is_vegetable_category",
            data_type="float64",
            feature_family="interactions",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="interaction_temp_x_poultry",
            description="7-day mean ambient temperature interaction with fresh poultry category",
            source_columns="weather_temp_mean_7d, category",
            calculation="temp_mean_7d * is_poultry_category",
            data_type="float64",
            feature_family="interactions",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="interaction_diesel_change_x_lead_time",
            description="Diesel fuel price percentage shock scaled by supplier logistical lead time",
            source_columns="fuel_diesel_pct_change_7d, supplier_lead_time_expected",
            calculation="diesel_pct_change_7d * supplier_lead_time_expected",
            data_type="float64",
            feature_family="interactions",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="interaction_fx_change_x_imported",
            description="Bilateral exchange rate percentage change impacting imported SKUs",
            source_columns="fx_pct_change_7d, product_imported_flag",
            calculation="fx_pct_change_7d * product_imported_flag",
            data_type="float64",
            feature_family="interactions",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="interaction_commodity_change_x_product",
            description="Global benchmark commodity percentage change for mapped ingredient products",
            source_columns="commodity_pct_change_7d, commodity_price",
            calculation="commodity_pct_change_7d * (commodity_price > 0)",
            data_type="float64",
            feature_family="interactions",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="interaction_gov_assist_x_essential",
            description="Government cash aid disbursement window interacting with price-controlled essential food",
            source_columns="is_post_gov_assist_7d, product_is_controlled_goods",
            calculation="is_post_gov_assist_7d * product_is_controlled_goods",
            data_type="float64",
            feature_family="interactions",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="interaction_price_gap_x_promo",
            description="Retailer price gap relative to market benchmark during promotional campaigns",
            source_columns="price_vs_market_diff, promo_active_flag",
            calculation="price_vs_market_diff * promo_active_flag",
            data_type="float64",
            feature_family="interactions",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="demand",
            leakage_risk="none",
        ),
    ])

    return df, definitions
