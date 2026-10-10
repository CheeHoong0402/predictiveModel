"""
Product Features Module.
Generates static and derived SKU master features, shelf life classifications,
perishability categories, and import statuses.
"""

from typing import List, Tuple
import polars as pl
from src.features.feature_registry import FeatureDefinition


def generate_product_features(
    df: pl.DataFrame,
) -> Tuple[pl.DataFrame, List[FeatureDefinition]]:
    """
    Generate product metadata and perishability classification features.
    """
    definitions: List[FeatureDefinition] = []

    shelf_life = pl.col("shelf_life_days").cast(pl.Float64)
    imp_flag = pl.col("imported_flag").cast(pl.Float64)

    df = df.with_columns([
        shelf_life.alias("product_shelf_life_days"),
        imp_flag.alias("product_imported_flag"),
        (shelf_life <= 14.0).cast(pl.Float64).alias("product_is_perishable"),
        (shelf_life <= 5.0).cast(pl.Float64).alias("product_is_highly_perishable"),
        # Controlled goods in Malaysia: fresh whole chicken, eggs, cooking oil
        (pl.col("product_id").is_in(["PRD_CHK_01", "PRD_EGG_02", "PRD_OIL_05"])).cast(pl.Float64).alias("product_is_controlled_goods"),
    ])
    definitions.extend([
        FeatureDefinition(
            name="product_shelf_life_days",
            description="Nominal product shelf life duration in calendar days",
            source_columns="shelf_life_days",
            calculation="Product master attribute",
            data_type="float64",
            feature_family="static_product",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="product_imported_flag",
            description="Binary flag if product is imported (1) versus locally produced (0)",
            source_columns="imported_flag",
            calculation="Product master attribute",
            data_type="float64",
            feature_family="static_product",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="product_is_perishable",
            description="Binary flag if product has limited shelf life <= 14 days",
            source_columns="shelf_life_days",
            calculation="shelf_life_days <= 14",
            data_type="float64",
            feature_family="static_product",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="product_is_highly_perishable",
            description="Binary flag if product is highly perishable <= 5 days (poultry/leafy vegetables)",
            source_columns="shelf_life_days",
            calculation="shelf_life_days <= 5",
            data_type="float64",
            feature_family="static_product",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="product_is_controlled_goods",
            description="Binary indicator if item is gazetted as price-controlled essential food in Malaysia",
            source_columns="product_id",
            calculation="product_id in ['PRD_CHK_01', 'PRD_EGG_02', 'PRD_OIL_05']",
            data_type="float64",
            feature_family="static_product",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
    ])

    return df, definitions
