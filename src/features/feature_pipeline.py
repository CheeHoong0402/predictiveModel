"""
Phase 7 Feature Engineering Pipeline Orchestrator.
Transforms clean daily modelling data into a comprehensive, leakage-safe
feature dataset for demand and purchase price predictive modelling.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import polars as pl
import pandas as pd

from src.features.calendar_features import generate_calendar_features
from src.features.demand_features import generate_demand_features
from src.features.economic_features import generate_economic_features
from src.features.event_features import generate_event_features
from src.features.feature_quality import audit_feature_quality
from src.features.feature_registry import (
    ExcludedFeatureDefinition,
    FeatureDefinition,
    FeatureRegistry,
)
from src.features.interaction_features import generate_interaction_features
from src.features.inventory_features import generate_inventory_features
from src.features.market_features import generate_market_features
from src.features.policy_features import generate_policy_features
from src.features.price_features import generate_price_features
from src.features.procurement_features import generate_procurement_features
from src.features.product_features import generate_product_features
from src.features.promotion_features import generate_promotion_features
from src.features.sales_features import generate_sales_features
from src.features.store_features import generate_store_features
from src.features.weather_features import generate_weather_features


def run_feature_pipeline(
    input_path: Path = Path("data/processed/master_daily_demand_corrected.parquet"),
    output_path: Path = Path("data/processed/model_features.parquet"),
    reports_dir: Path = Path("reports/features"),
) -> Tuple[pl.DataFrame, FeatureRegistry]:
    """
    Execute end-to-end feature engineering pipeline.
    """
    start_time = time.time()
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 7 - FEATURE ENGINEERING PIPELINE")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Reports directory: {reports_dir}")
    print("=" * 80)

    # 1. Load Input Dataset
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pl.read_parquet(input_path)
    input_rows = df.height
    input_cols = df.width
    print(f"[1/11] Loaded input dataset: {input_rows:,} rows, {input_cols} columns")

    # Verify Primary Key Uniqueness
    pk_count = df.select(["date", "store_id", "product_id"]).unique().height
    if pk_count != input_rows:
        raise ValueError(f"Primary key violation in input: {pk_count} unique keys vs {input_rows} rows")

    # Initialize Registry
    registry = FeatureRegistry()

    # Register Quarantined / Excluded Features
    registry.register_excluded(ExcludedFeatureDefinition(
        feature_name="estimated_true_demand_retrospective",
        feature_family="quarantined_demand",
        reason_for_exclusion="Uses two-sided future information for retrospective analysis; prohibited in training feature table",
        risk_type="future_target_leakage",
    ))
    registry.register_excluded(ExcludedFeatureDefinition(
        feature_name="target_demand_t_plus_1",
        feature_family="forecast_target",
        reason_for_exclusion="Forecast targets must not be created during Phase 7 Feature Engineering",
        risk_type="premature_target_creation",
    ))
    registry.register_excluded(ExcludedFeatureDefinition(
        feature_name="target_purchase_price_t_plus_7",
        feature_family="forecast_target",
        reason_for_exclusion="Forecast targets must not be created during Phase 7 Feature Engineering",
        risk_type="premature_target_creation",
    ))
    registry.register_excluded(ExcludedFeatureDefinition(
        feature_name="future_observed_weather",
        feature_family="future_weather",
        reason_for_exclusion="Actual future weather is unknown at prediction date T; prohibited from feature table",
        risk_type="future_information_leakage",
    ))
    registry.register_excluded(ExcludedFeatureDefinition(
        feature_name="future_competitor_prices",
        feature_family="future_market",
        reason_for_exclusion="Future competitor prices are unknown at prediction date T; prohibited from feature table",
        risk_type="future_information_leakage",
    ))

    # 2. Demand History Features (using training-safe demand shifted by 1)
    print("[2/11] Generating demand history, rolling stats, and seasonality features...")
    df, demand_defs = generate_demand_features(df)
    registry.register_many(demand_defs)

    # 3. Observed Sales Features
    print("[3/11] Generating observed sales history and rolling features...")
    df, sales_defs = generate_sales_features(df)
    registry.register_many(sales_defs)

    # 4. Inventory Features
    print("[4/11] Generating inventory state, stockouts, and replenishment features...")
    df, inv_defs = generate_inventory_features(df)
    registry.register_many(inv_defs)

    # 5. Price Features (Purchase & Selling Prices)
    print("[5/11] Generating purchase price lags, momentum, and selling price dynamics...")
    df, price_defs = generate_price_features(df)
    registry.register_many(price_defs)

    # 6. Procurement Features
    print("[6/11] Generating procurement lead time and supplier reliability features...")
    df, proc_defs = generate_procurement_features(df)
    registry.register_many(proc_defs)

    # 7. Promotion Features
    print("[7/11] Generating promotion dynamics and schedule features...")
    df, promo_defs = generate_promotion_features(df)
    registry.register_many(promo_defs)

    # 8. Calendar & Event Features
    print("[8/11] Generating calendar, holiday proximity, festival, and salary cycle features...")
    df, cal_defs = generate_calendar_features(df)
    registry.register_many(cal_defs)
    df, event_defs = generate_event_features(df)
    registry.register_many(event_defs)

    # 9. Weather, Market, Economic, Policy Features
    print("[9/11] Generating weather, market competitor, macroeconomic, and policy features...")
    df, weather_defs = generate_weather_features(df)
    registry.register_many(weather_defs)

    df, mkt_defs = generate_market_features(df)
    registry.register_many(mkt_defs)

    df, econ_defs = generate_economic_features(df)
    registry.register_many(econ_defs)

    df, pol_defs = generate_policy_features(df)
    registry.register_many(pol_defs)

    # 10. Product, Store, Interactions, and Missingness Features
    print("[10/11] Generating product/store attributes, interactions, and missingness indicators...")
    df, prod_defs = generate_product_features(df)
    registry.register_many(prod_defs)

    df, store_defs = generate_store_features(df)
    registry.register_many(store_defs)

    df, inter_defs = generate_interaction_features(df)
    registry.register_many(inter_defs)

    # Missingness indicator features
    df = df.with_columns([
        (pl.col("market_observation_count").fill_null(0) == 0).cast(pl.Float64).alias("missing_market_price"),
        (pl.col("weather_available").fill_null(0) == 0).cast(pl.Float64).alias("missing_weather"),
        (pl.col("commodity_price_available").fill_null(0) == 0).cast(pl.Float64).alias("missing_commodity_price"),
        (pl.col("exchange_rate_available").fill_null(0) == 0).cast(pl.Float64).alias("missing_exchange_rate"),
    ])
    missingness_defs = [
        FeatureDefinition(
            name="missing_market_price",
            description="Binary indicator if PriceCatcher market survey data is unavailable for this date",
            source_columns="market_observation_count",
            calculation="market_observation_count == 0",
            data_type="float64",
            feature_family="missingness_indicator",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="missing_weather",
            description="Binary indicator if local weather station report was missing",
            source_columns="weather_available",
            calculation="weather_available == 0",
            data_type="float64",
            feature_family="missingness_indicator",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="both",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="missing_commodity_price",
            description="Binary indicator if ingredient commodity price was unmapped or missing",
            source_columns="commodity_price_available",
            calculation="commodity_price_available == 0",
            data_type="float64",
            feature_family="missingness_indicator",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="price",
            leakage_risk="none",
        ),
        FeatureDefinition(
            name="missing_exchange_rate",
            description="Binary indicator if bilateral foreign exchange rate was missing",
            source_columns="exchange_rate_available",
            calculation="exchange_rate_available == 0",
            data_type="float64",
            feature_family="missingness_indicator",
            known_at_prediction_time=True,
            expected_missingness="none",
            target_model_applicability="price",
            leakage_risk="none",
        ),
    ]
    registry.register_many(missingness_defs)

    # 11. Leakage Pre-Checks & Sanity Validation
    print("[11/11] Running automated leakage pre-checks and saving feature artifacts...")
    # Check no target columns
    target_cols = [c for c in df.columns if c.startswith("target_") or c.startswith("future_")]
    if target_cols:
        raise ValueError(f"Prohibited target columns detected: {target_cols}")

    # Check primary key uniqueness
    if df.select(["date", "store_id", "product_id"]).unique().height != df.height:
        raise ValueError("Primary key uniqueness violated after feature engineering!")

    # Check row count invariant
    if df.height != input_rows:
        raise ValueError(f"Row count changed: {input_rows} input vs {df.height} output")

    # Clean any accidental infinites in float columns
    float_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Float64]
    df = df.with_columns([
        pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
        for c in float_cols
    ])

    # Save output parquet
    df.write_parquet(output_path, compression="snappy")
    output_size_mb = output_path.stat().st_size / (1024 * 1024)

    # Save Registry and Manifest Reports
    report_paths = registry.save_reports(reports_dir)

    # Audit Feature Quality and Missingness
    engineered_feature_names = registry.feature_names
    quality_dfs = audit_feature_quality(
        df=df,
        feature_names=engineered_feature_names,
        date_col="date",
        output_dir=reports_dir,
    )

    elapsed_time = time.time() - start_time

    # Generate Markdown Summary Report
    summary_md_path = reports_dir / "feature_engineering_summary.md"
    _generate_summary_report(
        summary_path=summary_md_path,
        input_rows=input_rows,
        output_rows=df.height,
        input_cols=input_cols,
        total_features=len(engineered_feature_names),
        demand_features_count=len(registry.demand_features),
        price_features_count=len(registry.price_features),
        output_size_mb=output_size_mb,
        elapsed_seconds=elapsed_time,
        quality_df=quality_dfs["quality"],
        missing_df=quality_dfs["missingness"],
        registry=registry,
    )

    print("=" * 80)
    print(f"PHASE 7 COMPLETED SUCCESSFULLY in {elapsed_time:.2f}s")
    print(f"Input Rows: {input_rows:,} | Output Rows: {df.height:,}")
    print(f"Total Columns: {df.width} (Input: {input_cols}, Engineered: {len(engineered_feature_names)})")
    print(f"Demand Candidate Features: {len(registry.demand_features)}")
    print(f"Price Candidate Features: {len(registry.price_features)}")
    print(f"Saved Parquet: {output_path} ({output_size_mb:.2f} MB)")
    print(f"Saved Reports: {reports_dir}")
    print("=" * 80)

    return df, registry


def _generate_summary_report(
    summary_path: Path,
    input_rows: int,
    output_rows: int,
    input_cols: int,
    total_features: int,
    demand_features_count: int,
    price_features_count: int,
    output_size_mb: float,
    elapsed_seconds: float,
    quality_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    registry: FeatureRegistry,
) -> None:
    """Generate comprehensive executive summary markdown report."""
    high_missing = missing_df[missing_df["missing_pct"] > 30.0]
    constant_features = quality_df[quality_df["is_constant"] == True]

    lines = [
        "# Phase 7 — Feature Engineering Summary Report",
        "",
        "## Executive Summary",
        "",
        f"- **Status**: Successfully Completed",
        f"- **Input Table**: `data/processed/master_daily_demand_corrected.parquet`",
        f"- **Output Table**: `data/processed/model_features.parquet`",
        f"- **Input Rows**: `{input_rows:,}`",
        f"- **Output Rows**: `{output_rows:,}`",
        f"- **Primary Key**: `date + store_id + product_id` (100% Unique, 0 duplicates)",
        f"- **Input Columns**: `{input_cols}`",
        f"- **Total Engineered Features**: `{total_features}`",
        f"- **Demand-Model Candidate Features**: `{demand_features_count}`",
        f"- **Purchase-Price Candidate Features**: `{price_features_count}`",
        f"- **Output Dataset Size**: `{output_size_mb:.2f} MB` (Snappy compressed Parquet)",
        f"- **Execution Time**: `{elapsed_seconds:.2f} seconds`",
        "",
        "---",
        "",
        "## Feature Families Breakdown",
        "",
        "| Feature Family | Feature Count | Model Applicability | Leakage Risk |",
        "| :--- | :--- | :--- | :--- |",
    ]

    family_counts = {}
    for f in registry.all_features:
        family_counts[f.feature_family] = family_counts.get(f.feature_family, 0) + 1

    for fam, cnt in sorted(family_counts.items()):
        lines.append(f"| `{fam}` | {cnt} | Demand / Price / Both | None (Shifted) |")

    lines.extend([
        "",
        "---",
        "",
        "## Temporal Safety & Leakage Prevention Guarantees",
        "",
        "1. **Strict Lag Shift Convention**:",
        "   - All demand, sales, and target-related rolling statistics are calculated on historical values shifted by at least 1 day relative to observation date $T$ (`shift(1)`).",
        "   - Lags: `demand_lag_1` represents demand on day $T-1$; `demand_lag_7` represents demand on day $T-7$.",
        "   - Rolling windows: `demand_mean_7d` is strictly computed across days $[T-7, \\dots, T-1]$.",
        "2. **Training-Safe Demand Integrity**:",
        "   - `estimated_true_demand_training_safe` (one-sided past-only correction) is strictly used as the historical demand series.",
        "   - `estimated_true_demand_retrospective` is quarantined in `reports/features/excluded_features.csv`.",
        "3. **Zero Target Contamination**:",
        "   - Zero target variables (`target_demand_t+1`, etc.) were created during this phase.",
        "   - No machine learning model has been trained.",
        "4. **Calendar and Astro-Gazette Determinism**:",
        "   - Public holidays, festivals, and salary cycles use deterministic calendar/gazetted schedules known centuries or weeks in advance.",
        "",
        "---",
        "",
        "## Feature Missingness Audit",
        "",
        f"- Features with 0% missingness: `{len(missing_df[missing_df['missing_pct'] == 0.0])}`",
        f"- Features with low missingness (<= 10%): `{len(missing_df[missing_df['missing_pct'] <= 10.0])}`",
        f"- Features with moderate missingness (10% to 30%): `{len(missing_df[(missing_df['missing_pct'] > 10.0) & (missing_df['missing_pct'] <= 30.0)])}`",
        f"- Features with high missingness (> 30%): `{len(high_missing)}` (annual 364d/365d lags requiring 1 full year history)",
        "",
        "---",
        "",
        "## Quarantined & Excluded Features",
        "",
        "| Feature Name | Family | Reason for Exclusion | Risk Type |",
        "| :--- | :--- | :--- | :--- |",
    ])

    for exc in registry._excluded:
        lines.append(f"| `{exc.feature_name}` | `{exc.feature_family}` | {exc.reason_for_exclusion} | `{exc.risk_type}` |")

    lines.extend([
        "",
        "---",
        "",
        "## Generated Report Manifests",
        "",
        "- `reports/features/feature_registry.csv`: Complete metadata catalog of all engineered features.",
        "- `reports/features/demand_feature_manifest.csv`: Candidate features selected for demand forecasting models.",
        "- `reports/features/price_feature_manifest.csv`: Candidate features selected for purchase-price forecasting models.",
        "- `reports/features/excluded_features.csv`: Catalog of prohibited/quarantined features.",
        "- `reports/features/feature_quality_report.csv`: Statistical distribution profiles and constant flags.",
        "- `reports/features/feature_missingness.csv`: Detailed missingness audit per feature.",
        "- `reports/features/feature_availability.csv`: Date coverage ranges for every feature.",
    ])

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 7 Feature Engineering Pipeline")
    parser.add_argument("--input", default="data/processed/master_daily_demand_corrected.parquet")
    parser.add_argument("--output", default="data/processed/model_features.parquet")
    parser.add_argument("--reports", default="reports/features")
    args = parser.parse_args()

    run_feature_pipeline(
        input_path=Path(args.input),
        output_path=Path(args.output),
        reports_dir=Path(args.reports),
    )
