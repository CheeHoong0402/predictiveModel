"""
Store Features Module.
Generates store attribute features, geographical location coordinates,
and capacity scale indicators.
"""

from typing import List, Tuple
import polars as pl
from src.features.feature_registry import FeatureDefinition


def generate_store_features(
    df: pl.DataFrame,
) -> Tuple[pl.DataFrame, List[FeatureDefinition]]:
    """
    Generate store metadata and scale indicators.
    """
    definitions: List[FeatureDefinition] = []

    cap = pl.col("storage_capacity").cast(pl.Float64)
    lat = pl.col("latitude").cast(pl.Float64)
    lon = pl.col("longitude").cast(pl.Float64)

    df = df.with_columns([
        cap.alias("store_storage_capacity_units"),
        lat.alias("store_geo_latitude"),
        lon.alias("store_geo_longitude"),
        (cap / cap.mean()).alias("store_relative_capacity_ratio"),
    ])
    definitions.extend([
        FeatureDefinition(
            name="store_storage_capacity_units",
            description="Total designated physical warehouse/storage capacity of store premise",
            source_columns="storage_capacity",
            calculation="Store master static attribute",
            data_type="float64",
            feature_family="static_store",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="store_geo_latitude",
            description="Geographical latitude coordinate of store premise",
            source_columns="latitude",
            calculation="Store master geographical coordinate",
            data_type="float64",
            feature_family="static_store",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="store_geo_longitude",
            description="Geographical longitude coordinate of store premise",
            source_columns="longitude",
            calculation="Store master geographical coordinate",
            data_type="float64",
            feature_family="static_store",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="store_relative_capacity_ratio",
            description="Ratio of store capacity relative to enterprise average store capacity",
            source_columns="storage_capacity",
            calculation="storage_capacity / mean(storage_capacity)",
            data_type="float64",
            feature_family="static_store",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
    ])

    return df, definitions
