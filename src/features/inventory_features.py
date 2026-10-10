"""
Inventory Features Module.
Generates historical inventory state, stockout frequencies, inventory coverage days,
and replenishment metrics.
"""

from typing import List, Tuple
import polars as pl
from src.features.feature_registry import FeatureDefinition


def generate_inventory_features(
    df: pl.DataFrame,
) -> Tuple[pl.DataFrame, List[FeatureDefinition]]:
    """
    Generate leakage-safe historical inventory and stockout features.
    """
    definitions: List[FeatureDefinition] = []

    df = df.sort(["store_id", "product_id", "date"])

    # Integer sequence for vectorized days-since calculation
    df = df.with_columns(
        pl.int_range(0, pl.len()).over(["store_id", "product_id"]).alias("__t_idx")
    )

    # 1. Basic inventory levels
    basic_exprs = [
        pl.col("opening_inventory").cast(pl.Float64).alias("inventory_opening"),
        pl.col("current_inventory").cast(pl.Float64).alias("inventory_current"),
        pl.col("closing_inventory").cast(pl.Float64).shift(1).over(["store_id", "product_id"]).alias("inventory_prior_closing"),
        pl.col("incoming_inventory").cast(pl.Float64).alias("inventory_incoming"),
        pl.col("backorder_quantity").cast(pl.Float64).alias("inventory_backorder"),
    ]
    definitions.extend([
        FeatureDefinition(
            name="inventory_opening",
            description="Opening inventory quantity on observation date T",
            source_columns="opening_inventory",
            calculation="Direct observation known at start of day T",
            data_type="float64",
            feature_family="inventory_state",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="inventory_current",
            description="Current inventory stock quantity known on date T",
            source_columns="current_inventory",
            calculation="Direct inventory snapshot",
            data_type="float64",
            feature_family="inventory_state",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="inventory_prior_closing",
            description="Closing inventory quantity of previous day T-1",
            source_columns="closing_inventory",
            calculation="shift(1) of closing_inventory",
            data_type="float64",
            feature_family="inventory_state",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="inventory_incoming",
            description="Known incoming inventory quantity in transit",
            source_columns="incoming_inventory",
            calculation="Direct observation known at T",
            data_type="float64",
            feature_family="inventory_state",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="inventory_backorder",
            description="Outstanding backorder quantity on date T",
            source_columns="backorder_quantity",
            calculation="Direct observation known at T",
            data_type="float64",
            feature_family="inventory_state",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
    ])
    df = df.with_columns(basic_exprs)

    # 2. Stockout history and frequencies (using shifted stockout flag)
    # Check if full_day_stockout_flag or stockout_flag exists
    stockout_col = "stockout_flag" if "stockout_flag" in df.columns else "full_day_stockout_flag"
    df = df.with_columns(
        pl.col(stockout_col).cast(pl.Float64).shift(1).over(["store_id", "product_id"]).alias("__stockout_t1")
    )

    stockout_windows = [7, 28, 56]
    stockout_exprs = [
        pl.col("__stockout_t1").alias("stockout_flag_prior_day")
    ]
    definitions.append(
        FeatureDefinition(
            name="stockout_flag_prior_day",
            description="Binary stockout flag on previous day T-1",
            source_columns=stockout_col,
            calculation="shift(1) of stockout_flag",
            data_type="float64",
            feature_family="stockout_history",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        )
    )

    for w in stockout_windows:
        col_count = f"stockout_count_{w}d"
        col_rate = f"stockout_rate_{w}d"
        stockout_exprs.extend([
            pl.col("__stockout_t1").rolling_sum(window_size=w).over(["store_id", "product_id"]).alias(col_count),
            (pl.col("__stockout_t1").rolling_sum(window_size=w).over(["store_id", "product_id"]) / float(w)).alias(col_rate),
        ])
        definitions.extend([
            FeatureDefinition(
                name=col_count,
                description=f"Number of stockout days in the prior {w} days",
                source_columns=stockout_col,
                calculation=f"rolling_sum({w}) of stockout_flag shifted by 1",
                data_type="float64",
                feature_family="stockout_history",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="both",
                leakage_risk="none",
            ),
            FeatureDefinition(
                name=col_rate,
                description=f"Stockout rate (proportion of days) in the prior {w} days",
                source_columns=stockout_col,
                calculation=f"stockout_count_{w}d / {w}",
                data_type="float64",
                feature_family="stockout_history",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="both",
                leakage_risk="none",
            ),
        ])
    df = df.with_columns(stockout_exprs)

    # 3. Days since last stockout
    df = df.with_columns(
        pl.when(pl.col("__stockout_t1") == 1.0)
        .then(pl.col("__t_idx") - 1)
        .otherwise(None)
        .forward_fill()
        .over(["store_id", "product_id"])
        .alias("__last_stockout_t")
    )
    df = df.with_columns(
        (pl.col("__t_idx") - pl.col("__last_stockout_t")).fill_null(999.0).cast(pl.Float64).alias("days_since_last_stockout")
    )
    definitions.append(
        FeatureDefinition(
            name="days_since_last_stockout",
            description="Days elapsed since the previous recorded stockout event prior to T",
            source_columns=stockout_col,
            calculation="t - last_stockout_t shifted by 1 (default 999)",
            data_type="float64",
            feature_family="stockout_history",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        )
    )

    # 4. Replenishment metrics (quantity_received)
    df = df.with_columns(
        pl.col("quantity_received").cast(pl.Float64).shift(1).over(["store_id", "product_id"]).alias("__received_t1")
    )
    # Days since last replenishment
    df = df.with_columns(
        pl.when(pl.col("__received_t1") > 0.0)
        .then(pl.col("__t_idx") - 1)
        .otherwise(None)
        .forward_fill()
        .over(["store_id", "product_id"])
        .alias("__last_receipt_t")
    )
    df = df.with_columns(
        (pl.col("__t_idx") - pl.col("__last_receipt_t")).fill_null(999.0).cast(pl.Float64).alias("days_since_last_replenishment")
    )
    definitions.append(
        FeatureDefinition(
            name="days_since_last_replenishment",
            description="Days elapsed since the previous inventory delivery receipt prior to T",
            source_columns="quantity_received",
            calculation="t - last_receipt_t shifted by 1 (default 999)",
            data_type="float64",
            feature_family="replenishment_history",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        )
    )

    # Rolling receipts
    receipt_exprs = [
        pl.col("__received_t1").rolling_sum(window_size=7).over(["store_id", "product_id"]).alias("quantity_received_7d_sum"),
        pl.col("__received_t1").rolling_sum(window_size=28).over(["store_id", "product_id"]).alias("quantity_received_28d_sum"),
        (pl.col("__received_t1") > 0.0).cast(pl.Float64).rolling_sum(window_size=30).over(["store_id", "product_id"]).alias("receipt_count_30d"),
    ]
    definitions.extend([
        FeatureDefinition(
            name="quantity_received_7d_sum",
            description="Total inventory units received in prior 7 days",
            source_columns="quantity_received",
            calculation="rolling_sum(7) of quantity_received shifted by 1",
            data_type="float64",
            feature_family="replenishment_history",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="quantity_received_28d_sum",
            description="Total inventory units received in prior 28 days",
            source_columns="quantity_received",
            calculation="rolling_sum(28) of quantity_received shifted by 1",
            data_type="float64",
            feature_family="replenishment_history",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="receipt_count_30d",
            description="Number of delivery receipts in the prior 30 days",
            source_columns="quantity_received",
            calculation="rolling_sum(30) of (quantity_received > 0) shifted by 1",
            data_type="float64",
            feature_family="replenishment_history",
            known_at_prediction_time=True,
            expected_missingness="low",
            target_model_applicability="both",
            leakage_risk="none",
        ),
    ])
    df = df.with_columns(receipt_exprs)

    # 5. Inventory coverage days (current stock / recent mean demand)
    # Use demand_mean_7d and demand_mean_28d if present, or compute safe ratio
    mean_7d_col = "demand_mean_7d" if "demand_mean_7d" in df.columns else None
    mean_28d_col = "demand_mean_28d" if "demand_mean_28d" in df.columns else None

    cov_exprs = []
    if mean_7d_col:
        cov_exprs.append(
            pl.min_horizontal([
                (pl.col("inventory_current") / (pl.col(mean_7d_col) + 1e-6)),
                pl.lit(999.0)
            ]).alias("inventory_coverage_days_7d")
        )
        definitions.append(
            FeatureDefinition(
                name="inventory_coverage_days_7d",
                description="Estimated days of stock based on current inventory and 7-day mean demand (capped at 999)",
                source_columns="current_inventory, demand_mean_7d",
                calculation="min(current_inventory / (demand_mean_7d + 1e-6), 999.0)",
                data_type="float64",
                feature_family="inventory_coverage",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="demand",
                leakage_risk="none",
            )
        )
    if mean_28d_col:
        cov_exprs.append(
            pl.min_horizontal([
                (pl.col("inventory_current") / (pl.col(mean_28d_col) + 1e-6)),
                pl.lit(999.0)
            ]).alias("inventory_coverage_days_28d")
        )
        definitions.append(
            FeatureDefinition(
                name="inventory_coverage_days_28d",
                description="Estimated days of stock based on current inventory and 28-day mean demand (capped at 999)",
                source_columns="current_inventory, demand_mean_28d",
                calculation="min(current_inventory / (demand_mean_28d + 1e-6), 999.0)",
                data_type="float64",
                feature_family="inventory_coverage",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="demand",
                leakage_risk="none",
            )
        )

    if cov_exprs:
        df = df.with_columns(cov_exprs)

    # Clean up temp columns
    df = df.drop(["__t_idx", "__stockout_t1", "__last_stockout_t", "__received_t1", "__last_receipt_t"])

    return df, definitions
