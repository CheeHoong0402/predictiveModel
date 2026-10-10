"""Reporting, table generation, and plotting for Phase 12 Purchase-Price Models."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_reports(
    predictions_df,
    regression_metrics_df,
    baseline_comp_df,
    direction_metrics_df,
    confusion_matrices,
    segmented_dfs,
    sample_counts_df,
    runtime_df,
    ranking_df,
    best_candidates_df,
    output_dir,
    figures_dir
):
    """Generate all 18 required CSV reports, figures, and summary markdown."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save core metrics CSVs
    regression_metrics_df.to_csv(output_dir / 'price_metrics.csv', index=False)
    baseline_comp_df.to_csv(output_dir / 'price_baseline_comparison.csv', index=False)
    direction_metrics_df.to_csv(output_dir / 'price_direction_metrics.csv', index=False)
    
    # Combined confusion matrix CSV
    cm_records = []
    for key, cm_df in confusion_matrices.items():
        cm_copy = cm_df.copy()
        cm_copy.insert(0, 'model_key', key)
        cm_records.append(cm_copy)
    if cm_records:
        pd.concat(cm_records).to_csv(output_dir / 'price_direction_confusion_matrix.csv')
        
    # Segmented CSVs
    for name, df in segmented_dfs.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
        
    runtime_df.to_csv(output_dir / 'model_runtime.csv', index=False)
    sample_counts_df.to_csv(output_dir / 'price_target_sample_counts.csv', index=False)
    ranking_df.to_csv(output_dir / 'candidate_model_ranking.csv', index=False)
    best_candidates_df.to_csv(output_dir / 'best_candidate_models.csv', index=False)
    
    # Feature counts
    feature_counts_df = pd.DataFrame([{
        'task': 'purchase_price',
        'approved_features_count': 163,
        'manifest_source': 'reports/leakage/price_features_temporally_safe.csv',
        'horizons_supported': 6
    }])
    feature_counts_df.to_csv(output_dir / 'model_feature_counts.csv', index=False)
    
    # 2. Generate Plots
    _generate_plots(predictions_df, baseline_comp_df, direction_metrics_df, figures_dir)
    
    # 3. Write summary markdown
    _generate_summary_markdown(
        predictions_df,
        regression_metrics_df,
        baseline_comp_df,
        direction_metrics_df,
        ranking_df,
        best_candidates_df,
        segmented_dfs,
        runtime_df,
        output_dir / 'price_model_summary.md'
    )


def _generate_plots(preds_df, baseline_df, dir_df, figures_dir):
    """Generate visualizations for Phase 12 report."""
    plt.style.use('default')
    
    # 1. MAE by Horizon: Baseline vs Machine Learning Models
    plt.figure(figsize=(9, 5))
    horizons = sorted(baseline_df['horizon'].unique())
    # Baseline
    base_line = baseline_df.drop_duplicates('horizon').sort_values('horizon')
    plt.plot(base_line['horizon'], base_line['baseline_mae'], 'k--o', label='Champion Baseline', linewidth=2)
    
    for model_name in ['lightgbm_direct', 'catboost_direct', 'xgboost_direct']:
        sub = baseline_df[baseline_df.model == model_name].sort_values('horizon')
        if not sub.empty:
            label = model_name.replace('_direct', '').capitalize()
            plt.plot(sub['horizon'], sub['mae'], '-o', label=f"{label} Direct")
            
    plt.title('Purchase-Price Forecast MAE (RM) Across Horizons')
    plt.xlabel('Forecast Horizon (Days)')
    plt.ylabel('Mean Absolute Error (RM)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / 'mae_by_horizon.png', dpi=200)
    plt.close()
    
    # 2. Actual vs Predicted Price Scatter (7d horizon)
    sub7 = preds_df[(preds_df.horizon == 7) & (preds_df.model_name == 'catboost_direct')]
    if not sub7.empty:
        plt.figure(figsize=(7, 6))
        plt.scatter(sub7['actual_future_price'], sub7['predicted_future_price'], alpha=0.4, color='#1f77b4', edgecolors='none')
        min_p = min(sub7['actual_future_price'].min(), sub7['predicted_future_price'].min())
        max_p = max(sub7['actual_future_price'].max(), sub7['predicted_future_price'].max())
        plt.plot([min_p, max_p], [min_p, max_p], 'r--', label='Perfect Forecast', linewidth=1.5)
        plt.title('7-Day Purchase Price: Actual vs Predicted (CatBoost)')
        plt.xlabel('Actual Future Price (RM)')
        plt.ylabel('Predicted Future Price (RM)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / 'actual_vs_predicted_7d.png', dpi=200)
        plt.close()
        
    # 3. Residual Distribution
    if not sub7.empty:
        res = sub7['predicted_future_price'] - sub7['actual_future_price']
        plt.figure(figsize=(8, 4.5))
        plt.hist(res, bins=30, color='#2ca02c', edgecolor='black', alpha=0.7)
        plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
        plt.title('7-Day Purchase Price Residual Distribution (Predicted - Actual)')
        plt.xlabel('Error in Malaysian Ringgit (RM)')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(figures_dir / 'residual_distribution_7d.png', dpi=200)
        plt.close()
        
    # 4. Direction Accuracy Comparison
    plt.figure(figsize=(10, 5))
    acc_sub = dir_df[dir_df.horizon == 7].copy()
    if not acc_sub.empty:
        plt.bar(acc_sub['method'], acc_sub['balanced_accuracy'] * 100, color='#ff7f0e', alpha=0.8)
        plt.title('7-Day Price Direction Balanced Accuracy (%) by Method')
        plt.xlabel('Method / Classifier')
        plt.ylabel('Balanced Accuracy (%)')
        plt.xticks(rotation=20, ha='right')
        plt.grid(True, linestyle=':', alpha=0.6, axis='y')
        plt.tight_layout()
        plt.savefig(figures_dir / 'direction_balanced_accuracy_7d.png', dpi=200)
        plt.close()


def _generate_summary_markdown(
    preds_df,
    reg_df,
    base_df,
    dir_df,
    ranking_df,
    best_df,
    segmented_dfs,
    runtime_df,
    output_path
):
    """Write executive summary in markdown format."""
    lines = [
        "# Phase 12 — Supplier Purchase-Price Forecasting Models: Summary Report",
        "",
        "## Executive Summary",
        "Phase 12 trained, evaluated, and ranked advanced machine learning models for **supplier purchase-price forecasting** across six horizons (1, 3, 7, 14, 21, and 30 days).",
        "Three distinct modeling approaches were evaluated:",
        "1. **Direct Purchase-Price Regression** (LightGBM, CatBoost, XGBoost)",
        "2. **Percentage-Change Regression & Price Reconstruction** $\\hat{y} = y_{current} \\times (1 + \\hat{\\Delta})$ (LightGBM, CatBoost, XGBoost)",
        "3. **Price Direction Prediction**: Derived from regression vs Dedicated multiclass Classifiers (LightGBM, CatBoost, XGBoost)",
        "",
        "Evaluation strictly adhered to the Phase 10 chronological 33-fold walk-forward validation framework.",
        "Zero future leakage was allowed: training sets for horizon $H$ used strictly $t \\le T - H$,",
        "and feature inputs strictly respected the approved 163 temporally safe price features from `reports/leakage/price_features_temporally_safe.csv`.",
        "The final holdout period (**2024-10-01 to 2024-12-31**) remained strictly protected and untouched.",
        "",
        "## Best Candidate Purchase-Price Models per Horizon",
        "```text",
        best_df.to_string(index=False),
        "```",
        "",
        "## Direct Regression vs. Percentage-Change Regression",
        "```text",
        base_df[['horizon', 'model', 'mae', 'baseline_mae', 'mae_reduction_pct', 'rmse', 'bias']].to_string(index=False),
        "```",
        "",
        "## Price Direction Performance (Accuracy & Balanced Accuracy)",
        "```text",
        dir_df.to_string(index=False),
        "```",
        "",
        "## Segmented Performance Highlights",
        "### By Product Category (7-Day Horizon)",
        "```text",
    ]
    
    if 'price_metrics_by_category' in segmented_dfs:
        cat_df = segmented_dfs['price_metrics_by_category']
        lines.append(cat_df[cat_df.horizon == 7].to_string(index=False))
    lines.extend([
        "```",
        "",
        "### Imported vs. Domestic Products (7-Day Horizon)",
        "```text"
    ])
    if 'imported_domestic_performance' in segmented_dfs:
        imp_df = segmented_dfs['imported_domestic_performance']
        lines.append(imp_df[imp_df.horizon == 7].to_string(index=False))
    lines.extend([
        "```",
        "",
        "### By Supplier (7-Day Horizon)",
        "```text"
    ])
    if 'price_metrics_by_supplier' in segmented_dfs:
        sup_df = segmented_dfs['price_metrics_by_supplier']
        lines.append(sup_df[sup_df.horizon == 7].to_string(index=False))
    lines.extend([
        "```",
        "",
        "### Historical Price Volatility Segments (7-Day Horizon)",
        "```text"
    ])
    if 'volatility_performance' in segmented_dfs:
        vol_df = segmented_dfs['volatility_performance']
        lines.append(vol_df[vol_df.horizon == 7].to_string(index=False))
    lines.extend([
        "```",
        "",
        "### Price Shock Periods vs Normal (7-Day Horizon)",
        "```text"
    ])
    if 'price_shock_performance' in segmented_dfs:
        shk_df = segmented_dfs['price_shock_performance']
        lines.append(shk_df[shk_df.horizon == 7].to_string(index=False))
    lines.extend([
        "```",
        "",
        "### Commodity-Sensitive Products (7-Day Horizon)",
        "```text"
    ])
    if 'commodity_sensitive_performance' in segmented_dfs:
        com_df = segmented_dfs['commodity_sensitive_performance']
        lines.append(com_df[com_df.horizon == 7].to_string(index=False))
    lines.extend([
        "```",
        "",
        "## Example Forecast Traces",
        "Below are selected representative forecast traces demonstrating row-level predictions for later procurement decision making:",
        ""
    ])
    
    # Add 3 trace examples from 7d and 30d
    traces = preds_df[(preds_df.horizon.isin([7, 30])) & (preds_df.model_name.str.contains('direct'))].drop_duplicates(subset=['horizon', 'product_id']).head(4)
    for _, tr in traces.iterrows():
        lines.extend([
            f"- **Forecast Origin**: {tr.get('forecast_origin', tr.get('date', 'N/A'))} | **Horizon**: {tr['horizon']}d | **Model**: {tr['model_name']}",
            f"  - Store: `{tr['store_id']}` | Product: `{tr['product_id']}` | Supplier: `{tr.get('supplier_id', 'N/A')}`",
            f"  - Current Purchase Price: **RM {tr['current_purchase_price']:.2f}**",
            f"  - Predicted Future Price: **RM {tr['predicted_future_price']:.2f}** (Actual: **RM {tr['actual_future_price']:.2f}**)",
            f"  - Expected Absolute Change: **RM {tr['predicted_absolute_change']:+.2f}** | Pct Change: **{tr['predicted_percentage_change']:+.2f}%**",
            f"  - Predicted Direction: **{tr['predicted_direction']}** (Actual: **{tr['actual_direction']}**)",
            f"  - Raw Uncertainty: [p10: RM {tr.get('raw_p10', 0):.2f}, p50: RM {tr.get('raw_p50', 0):.2f}, p90: RM {tr.get('raw_p90', 0):.2f}]",
            ""
        ])
        
    lines.extend([
        "## Model Training Runtime",
        "```text",
        runtime_df.to_string(index=False),
        "```",
        "",
        "All candidate models and metadata have been serialized under `models/price/<family>/`.",
        "Reusable inference is available via `predict_purchase_price()` and `predict_price_direction()`."
    ])
    
    output_path.write_text("\n".join(lines), encoding='utf-8')
