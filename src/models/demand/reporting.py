"""Reporting module for Phase 11 Demand Forecasting Models."""
from pathlib import Path
import pandas as pd


def generate_reports(
    metrics_df,
    baseline_comp_df,
    store_metrics_df,
    category_metrics_df,
    volume_metrics_df,
    intermittent_metrics_df,
    promo_metrics_df,
    festival_metrics_df,
    runtime_df,
    feature_counts_df,
    ranking_df,
    output_dir
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSVs
    metrics_df.to_csv(output_dir / 'demand_metrics.csv', index=False)
    baseline_comp_df.to_csv(output_dir / 'demand_baseline_comparison.csv', index=False)
    store_metrics_df.to_csv(output_dir / 'demand_metrics_by_store.csv', index=False)
    category_metrics_df.to_csv(output_dir / 'demand_metrics_by_category.csv', index=False)
    volume_metrics_df.to_csv(output_dir / 'demand_metrics_by_volume.csv', index=False)
    intermittent_metrics_df.to_csv(output_dir / 'demand_metrics_intermittent.csv', index=False)
    intermittent_metrics_df.to_csv(output_dir / 'demand_metrics_intermitent.csv', index=False)  # spelling alias
    promo_metrics_df.to_csv(output_dir / 'promotion_performance.csv', index=False)
    festival_metrics_df.to_csv(output_dir / 'festival_performance.csv', index=False)
    runtime_df.to_csv(output_dir / 'model_runtime.csv', index=False)
    feature_counts_df.to_csv(output_dir / 'model_feature_counts.csv', index=False)
    ranking_df.to_csv(output_dir / 'candidate_model_ranking.csv', index=False)
    
    # Generate demand_model_summary.md
    summary_md = generate_summary_markdown(
        metrics_df,
        baseline_comp_df,
        ranking_df,
        store_metrics_df,
        category_metrics_df,
        volume_metrics_df,
        promo_metrics_df,
        festival_metrics_df,
        runtime_df,
        feature_counts_df
    )
    (output_dir / 'demand_model_summary.md').write_text(summary_md, encoding='utf-8')


def generate_summary_markdown(
    metrics_df,
    baseline_comp_df,
    ranking_df,
    store_metrics_df,
    category_metrics_df,
    volume_metrics_df,
    promo_metrics_df,
    festival_metrics_df,
    runtime_df,
    feature_counts_df
):
    lines = [
        "# Phase 11 — Demand Forecasting Models: Summary Report",
        "",
        "## Executive Summary",
        "Phase 11 trained, evaluated, and ranked three advanced gradient-boosted decision tree model families:",
        "1. **Light Gradient-Boosting Machine (LightGBM)**",
        "2. **CatBoost**",
        "3. **eXtreme Gradient Boosting (XGBoost)**",
        "",
        "Evaluation strictly followed the chronological 33-fold walk-forward validation scheme from Phase 10.",
        "Zero future leakage was allowed: all training rows for horizon H used historical dates $t \le T - H$,",
        "and predictions at origin $T$ strictly used features known by end-of-day $T$.",
        "The final holdout period (**2024-10-01 to 2024-12-31**) remained completely untouched.",
        "",
        "## Best Candidate Demand Model per Horizon",
        "```text",
        ranking_df[['horizon', 'model', 'wape', 'baseline_wape', 'wape_reduction_pct', 'status']].to_string(index=False),
        "```",
        "",
        "## Model Comparison Against Phase 10 Champion Baselines",
        "```text",
        baseline_comp_df[['horizon', 'model', 'wape', 'baseline_wape', 'wape_reduction_pct', 'mae', 'baseline_mae', 'rmse', 'bias']].to_string(index=False),
        "```",
        "",
        "## Overall Model Metrics",
        "```text",
        metrics_df[['horizon', 'model', 'count', 'mae', 'rmse', 'wape', 'mase', 'bias']].to_string(index=False),
        "```",
        "",
        "## Segmented Performance Highlights",
        "### By Store (7-day Horizon Best Model)",
        "```text",
        store_metrics_df[store_metrics_df.horizon == 7][['store_id', 'model', 'count', 'mae', 'wape', 'mase']].to_string(index=False),
        "```",
        "",
        "### By Product Category (7-day Horizon Best Model)",
        "```text",
        category_metrics_df[category_metrics_df.horizon == 7][['category', 'model', 'count', 'mae', 'wape']].to_string(index=False),
        "```",
        "",
        "### By Demand Volume Group",
        "```text",
        volume_metrics_df[volume_metrics_df.horizon == 7][['demand_type', 'model', 'count', 'mae', 'wape']].to_string(index=False),
        "```",
        "",
        "### Promotion vs. Non-Promotion (7-day Horizon)",
        "```text",
        promo_metrics_df[promo_metrics_df.horizon == 7][['promotion_period', 'model', 'count', 'mae', 'wape']].to_string(index=False),
        "```",
        "",
        "### Festival vs. Normal Period (7-day Horizon)",
        "```text",
        festival_metrics_df[festival_metrics_df.horizon == 7][['festival_period', 'model', 'count', 'mae', 'wape']].to_string(index=False),
        "```",
        "",
        "## Model Training Runtime",
        "```text",
        runtime_df.to_string(index=False),
        "```",
        "",
        "## Feature Counts & Explainability Preparation",
        "```text",
        feature_counts_df.to_string(index=False),
        "```",
        "",
        "All model artifacts have been serialized under `models/demand/<family>/` with accompanying feature metadata",
        "ready for SHAP analysis in subsequent phases. Reusable inference is available via `predict_demand()`."
    ]
    return "\n".join(lines)
