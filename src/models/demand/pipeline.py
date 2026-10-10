"""End-to-end reproducible pipeline for Phase 11 Demand Forecasting Models."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.utils.paths import PROJECT_ROOT
from .config import load_config, load_demand_feature_manifest, validate_feature_input
from .evaluator import (
    compute_regression_metrics,
    compute_segmented_metrics,
    compute_baseline_comparison,
    compute_residual_quantiles
)
from .reporting import generate_reports

import lightgbm as lgb
import xgboost as xgb
import catboost as cb


def instantiate_models(config):
    """Instantiate fresh model instances with conservative fixed parameters."""
    cfg = config['models']
    models = {
        'lightgbm': lgb.LGBMRegressor(**cfg['lightgbm']),
        'catboost': cb.CatBoostRegressor(**cfg['catboost']),
        'xgboost': xgb.XGBRegressor(**cfg['xgboost'])
    }
    return models


def read_before(path, boundary, columns=None):
    """Filter rows before boundary, handling string vs timestamp date schema types."""
    import pyarrow as pa
    data = ds.dataset(str(path), format='parquet')
    date_type = data.schema.field('date').type
    if pa.types.is_string(date_type) or pa.types.is_large_string(date_type):
        limit = str(pd.Timestamp(boundary).date())
    else:
        limit = pd.Timestamp(boundary).to_pydatetime()
    return data.to_table(columns=columns, filter=ds.field('date') < limit).to_pandas()


def run_pipeline(root=PROJECT_ROOT, verbose=True):
    root = Path(root)
    cfg = load_config(root)
    horizons = cfg['evaluation']['horizons']
    holdout_start = pd.Timestamp(cfg['evaluation']['holdout_start'])
    
    if verbose:
        print("=== Phase 11 Demand Forecasting Pipeline ===", flush=True)
        print(f"Supported horizons: {horizons}", flush=True)
        print(f"Holdout start: {holdout_start.date()}", flush=True)
        
    manifest = load_demand_feature_manifest(root)
    if verbose:
        print(f"Approved safe demand features: {len(manifest)}", flush=True)
        
    # Read pre-holdout features
    features_df = read_before(
        root / 'data/processed/model_features.parquet',
        holdout_start,
        columns=['date', 'store_id', 'product_id'] + manifest
    )
    features_df['date'] = pd.to_datetime(features_df.date)
    
    # Read pre-holdout targets
    target_cols = [f'demand_target_{h}d' for h in horizons] + [f'demand_target_{h}d_valid' for h in horizons]
    targets_df = read_before(
        root / 'data/processed/forecast_targets.parquet',
        holdout_start,
        columns=['date', 'store_id', 'product_id'] + target_cols
    )
    targets_df['date'] = pd.to_datetime(targets_df.date)
    
    # Merge features with targets
    data_df = features_df.merge(targets_df, on=['date', 'store_id', 'product_id'])
    
    # Load Phase 10 champion baselines and baseline predictions
    champions_df = pd.read_csv(root / 'reports/baselines/champion_baselines.csv')
    demand_champs = champions_df[champions_df.task == 'demand'][['horizon', 'baseline']]
    
    base_preds_df = pd.read_parquet(root / 'data/predictions/baseline_demand_predictions.parquet')
    base_preds_df['date'] = pd.to_datetime(base_preds_df.date)
    champ_preds = base_preds_df.merge(demand_champs, on=['horizon', 'baseline'])
    
    # Load walk-forward folds
    folds_df = pd.read_csv(root / 'reports/baselines/walk_forward_folds.csv')
    folds_df['training_start'] = pd.to_datetime(folds_df.training_start)
    folds_df['training_end'] = pd.to_datetime(folds_df.training_end)
    folds_df['forecast_origin'] = pd.to_datetime(folds_df.forecast_origin)
    
    prediction_rows = []
    runtimes = {'lightgbm': 0.0, 'catboost': 0.0, 'xgboost': 0.0}
    
    if verbose:
        print(f"Beginning 33-fold walk-forward validation across {len(horizons)} horizons...", flush=True)
        
    start_all = time.time()
    
    for f_idx, fold in folds_df.iterrows():
        fold_id = int(fold.validation_fold)
        origin = fold.forecast_origin
        t_start = fold.training_start
        
        # Eval slice at origin
        eval_slice = data_df[data_df.date == origin].copy()
        if eval_slice.empty:
            continue
            
        X_eval = eval_slice[manifest].to_numpy(dtype=np.float32)
        
        for h in horizons:
            t_col = f'demand_target_{h}d'
            v_col = f'demand_target_{h}d_valid'
            
            # Training cutoff: date + h <= origin
            train_cutoff = origin - pd.Timedelta(days=h)
            train_mask = (data_df.date >= t_start) & (data_df.date <= train_cutoff) & data_df[v_col].eq(True)
            train_slice = data_df.loc[train_mask]
            
            if train_slice.empty:
                continue
                
            X_train = train_slice[manifest].to_numpy(dtype=np.float32)
            y_train = train_slice[t_col].to_numpy(dtype=np.float32)
            
            models = instantiate_models(cfg)
            
            for m_name, model in models.items():
                t0 = time.time()
                model.fit(X_train, y_train)
                runtimes[m_name] += (time.time() - t0)
                
                # Training residuals for raw quantiles
                train_preds = model.predict(X_train)
                train_residuals = y_train - train_preds
                
                # Predictions at origin
                raw_pred = model.predict(X_eval)
                pred_demand = np.maximum(0.0, np.asarray(raw_pred, dtype=float))
                
                raw_q10, raw_q50, raw_q90 = compute_residual_quantiles(train_residuals, pred_demand)
                
                for i, (_, row) in enumerate(eval_slice.iterrows()):
                    prediction_rows.append({
                        'date': origin,
                        'forecast_origin': origin,
                        'store_id': row['store_id'],
                        'product_id': row['product_id'],
                        'horizon': h,
                        'validation_fold': fold_id,
                        'model': m_name,
                        'predicted_demand': pred_demand[i],
                        'raw_q10': raw_q10[i],
                        'raw_q50': raw_q50[i],
                        'raw_q90': raw_q90[i],
                        'quantile_status': 'raw_uncalibrated_quantiles'
                    })
                    
        if verbose and (fold_id % 5 == 0 or fold_id == len(folds_df)):
            print(f"Completed Fold {fold_id}/{len(folds_df)} (Origin: {origin.date()})", flush=True)
            
    total_time = time.time() - start_all
    if verbose:
        print(f"Walk-forward evaluation completed in {total_time:.2f}s", flush=True)
        
    preds_df = pd.DataFrame(prediction_rows)
    
    # Merge with champion baseline predictions to attach actuals, mase_scale, baseline_prediction, and scenario tags
    merge_keys = ['date', 'store_id', 'product_id', 'horizon', 'validation_fold']
    join_cols = [
        'date', 'store_id', 'product_id', 'horizon', 'validation_fold',
        'prediction', 'actual', 'mase_scale', 'category', 'state', 'demand_type',
        'promotion_period', 'festival_period', 'high_volatility'
    ]
    meta_df = champ_preds[[c for c in join_cols if c in champ_preds.columns]].rename(
        columns={'prediction': 'baseline_prediction', 'actual': 'actual_demand'}
    )
    
    final_preds_df = preds_df.merge(meta_df, on=merge_keys, how='left')
    final_preds_df['absolute_error'] = (final_preds_df['predicted_demand'] - final_preds_df['actual_demand']).abs()
    
    # Save predictions parquet
    pred_dir = root / 'data/predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    final_preds_df.to_parquet(pred_dir / 'demand_model_predictions.parquet', index=False)
    if verbose:
        print(f"Saved row-level predictions to: {pred_dir / 'demand_model_predictions.parquet'}", flush=True)
        
    # Compute Metrics
    metric_rows = []
    for (h, m_name), grp in final_preds_df.groupby(['horizon', 'model']):
        m = compute_regression_metrics(grp['actual_demand'], grp['predicted_demand'], grp['mase_scale'])
        metric_rows.append({'horizon': h, 'model': m_name, **m})
    metrics_df = pd.DataFrame(metric_rows).sort_values(['horizon', 'model'])
    
    # Compute Baseline Comparison
    # Extract baseline metrics from Phase 10 demand champion table
    dm_path = root / 'reports/baselines/demand_baseline_metrics.csv'
    base_metrics_df = pd.read_csv(dm_path)
    base_champ_metrics = base_metrics_df.merge(demand_champs, on=['horizon', 'baseline'])
    
    baseline_comp_df = compute_baseline_comparison(metrics_df, base_champ_metrics)
    
    # Segmented Metrics
    store_metrics_df = compute_segmented_metrics(final_preds_df, ['horizon', 'model', 'store_id'])
    category_metrics_df = compute_segmented_metrics(final_preds_df, ['horizon', 'model', 'category'])
    volume_metrics_df = compute_segmented_metrics(final_preds_df, ['horizon', 'model', 'demand_type'])
    intermittent_metrics_df = volume_metrics_df.copy() # includes low/medium/high/intermittent
    promo_metrics_df = compute_segmented_metrics(final_preds_df, ['horizon', 'model', 'promotion_period'])
    festival_metrics_df = compute_segmented_metrics(final_preds_df, ['horizon', 'model', 'festival_period'])
    
    # Model ranking and champion selection per horizon
    ranking_rows = []
    for h, grp in baseline_comp_df.groupby('horizon'):
        sorted_grp = grp.sort_values('wape')
        best = sorted_grp.iloc[0]
        ranking_rows.append({
            'horizon': h,
            'model': best['model'],
            'wape': best['wape'],
            'baseline_wape': best['baseline_wape'],
            'wape_reduction_pct': best['wape_reduction_pct'],
            'mae': best['mae'],
            'rmse': best['rmse'],
            'status': 'BEST_CANDIDATE_DEMAND_MODEL'
        })
    ranking_df = pd.DataFrame(ranking_rows).sort_values('horizon')
    
    # Runtime & Feature Counts
    runtime_df = pd.DataFrame([
        {'model': m, 'total_walk_forward_runtime_seconds': round(runtimes[m], 2), 'folds_evaluated': len(folds_df)}
        for m in runtimes
    ])
    feature_counts_df = pd.DataFrame([{
        'task': 'demand',
        'approved_features_count': len(manifest),
        'manifest_source': 'reports/leakage/demand_features_temporally_safe.csv',
        'horizons_supported': len(horizons)
    }])
    
    # Save Model Artifacts
    # Retrain on all pre-holdout development data for each horizon
    models_dir = root / 'models/demand'
    for m_name in ['lightgbm', 'catboost', 'xgboost']:
        (models_dir / m_name).mkdir(parents=True, exist_ok=True)
        
    if verbose:
        print("Retraining production candidate models on full development data...", flush=True)
        
    for h in horizons:
        t_col = f'demand_target_{h}d'
        v_col = f'demand_target_{h}d_valid'
        # Must have completed target before holdout
        prod_cutoff = holdout_start - pd.Timedelta(days=h)
        mask = (data_df.date <= prod_cutoff) & data_df[v_col].eq(True)
        prod_data = data_df.loc[mask]
        
        X_prod = prod_data[manifest].to_numpy(dtype=np.float32)
        y_prod = prod_data[t_col].to_numpy(dtype=np.float32)
        
        prod_models = instantiate_models(cfg)
        
        for m_name, model in prod_models.items():
            model.fit(X_prod, y_prod)
            preds_prod = model.predict(X_prod)
            residuals_prod = y_prod - preds_prod
            q10_off = float(np.quantile(residuals_prod, 0.1))
            q50_off = float(np.quantile(residuals_prod, 0.5))
            q90_off = float(np.quantile(residuals_prod, 0.9))
            
            # Save artifacts
            m_dir = models_dir / m_name
            joblib.dump(model, m_dir / f"{m_name}_demand_{h}d.joblib")
            
            if m_name == 'catboost':
                model.save_model(str(m_dir / f"{m_name}_demand_{h}d.cbm"))
            elif m_name == 'xgboost':
                model.save_model(str(m_dir / f"{m_name}_demand_{h}d.json"))
            elif m_name == 'lightgbm':
                model.booster_.save_model(str(m_dir / f"{m_name}_demand_{h}d.txt"))
                
            metadata = {
                'horizon': h,
                'model_family': m_name,
                'training_samples': len(y_prod),
                'training_features_count': len(manifest),
                'q10_offset': q10_off,
                'q50_offset': q50_off,
                'q90_offset': q90_off,
                'training_date_cutoff': str(prod_cutoff.date()),
                'holdout_start': str(holdout_start.date())
            }
            with (m_dir / f"{m_name}_metadata_{h}d.json").open('w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
                
    # Save feature names list for SHAP explainability readiness
    with (models_dir / 'feature_names.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        
    # Generate all reports
    reports_dir = root / 'reports/models/demand'
    generate_reports(
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
        reports_dir
    )
    
    if verbose:
        print(f"Generated all reports in: {reports_dir}", flush=True)
        print("\n=== Best Candidate Demand Model per Horizon ===", flush=True)
        print(ranking_df.to_string(index=False), flush=True)
        
    return ranking_df, final_preds_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Phase 11 Demand Model Pipeline")
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    run_pipeline(args.root)
