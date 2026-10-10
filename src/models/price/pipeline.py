"""End-to-end reproducible pipeline for Phase 12 Supplier Purchase-Price Models."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from src.utils.paths import PROJECT_ROOT
from .config import load_config, load_price_feature_manifest, validate_feature_input
from .evaluator import (
    compute_regression_metrics,
    compute_direction_metrics,
    compute_confusion_matrix_df,
    compute_baseline_comparison,
    compute_residual_quantiles,
    compute_error_distribution,
    derive_direction_from_prices,
    derive_direction_from_pct_change,
    CLASS_TO_INT,
    INT_TO_CLASS,
    DIRECTION_CLASSES
)
from .reporting import generate_reports

import lightgbm as lgb
import xgboost as xgb
import catboost as cb


def instantiate_models(config):
    """Instantiate fresh regressors and classifiers with conservative parameters."""
    reg_cfg = config['models']['regression']
    cls_cfg = config['models']['classification']
    
    models = {
        'lightgbm_direct': lgb.LGBMRegressor(**reg_cfg['lightgbm']),
        'catboost_direct': cb.CatBoostRegressor(**reg_cfg['catboost']),
        'xgboost_direct': xgb.XGBRegressor(**reg_cfg['xgboost']),
        'lightgbm_pct': lgb.LGBMRegressor(**reg_cfg['lightgbm']),
        'catboost_pct': cb.CatBoostRegressor(**reg_cfg['catboost']),
        'xgboost_pct': xgb.XGBRegressor(**reg_cfg['xgboost']),
        'lightgbm_cls': lgb.LGBMClassifier(**cls_cfg['lightgbm']),
        'catboost_cls': cb.CatBoostClassifier(**cls_cfg['catboost']),
        'xgboost_cls': xgb.XGBClassifier(**cls_cfg['xgboost']),
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


def load_development_dataset(root=PROJECT_ROOT, holdout_start='2024-10-01'):
    """Load features and targets strictly prior to the final holdout period."""
    root = Path(root)
    feat_df = read_before(root / 'data/processed/model_features.parquet', holdout_start)
    targ_df = read_before(root / 'data/processed/forecast_targets.parquet', holdout_start)
    
    feat_df['date'] = pd.to_datetime(feat_df['date'])
    targ_df['date'] = pd.to_datetime(targ_df['date'])
    
    key_cols = ['date', 'store_id', 'product_id']
    targ_cols = [c for c in targ_df.columns if c not in key_cols and c not in feat_df.columns]
    overlap = [c for c in targ_df.columns if c in feat_df.columns and c not in key_cols]
    targ_subset = targ_df[key_cols + targ_cols + overlap].copy()
    
    merged = pd.merge(feat_df, targ_subset, on=key_cols, how='inner', suffixes=('', '__targ'))
    return merged


def run_pipeline(root=PROJECT_ROOT, verbose=True):
    root = Path(root)
    cfg = load_config(root)
    horizons = cfg['evaluation']['horizons']
    holdout_start = pd.Timestamp(cfg['evaluation']['holdout_start'])
    flat_tol = cfg['evaluation'].get('flat_tolerance', 0.005)
    
    if verbose:
        print("=== Phase 12 Supplier Purchase-Price Pipeline ===", flush=True)
        print(f"Horizons: {horizons}", flush=True)
        print(f"Holdout start: {holdout_start.date()}", flush=True)
        
    start_time = time.time()
    
    # 1. Load safe feature manifest
    approved_features = load_price_feature_manifest(root)
    if verbose:
        print(f"Loaded {len(approved_features)} approved price features.", flush=True)
        
    # 2. Load dataset before holdout
    dev_df = load_development_dataset(root, holdout_start=holdout_start)
    if verbose:
        print(f"Loaded development dataset: {len(dev_df)} rows.", flush=True)
        
    # 3. Load walk-forward folds
    folds_df = pd.read_csv(root / 'reports/baselines/walk_forward_folds.csv')
    folds_df['training_start'] = pd.to_datetime(folds_df['training_start'])
    folds_df['training_end'] = pd.to_datetime(folds_df['training_end'])
    folds_df['forecast_origin'] = pd.to_datetime(folds_df['forecast_origin'])
    
    # 4. Load Phase 10 champion baselines and baseline predictions
    champ_df = pd.read_csv(root / 'reports/baselines/champion_baselines.csv')
    price_champ = champ_df[champ_df.task == 'price'][['horizon', 'baseline', 'primary_metric', 'pooled_error']].copy()
    price_champ = price_champ.rename(columns={'pooled_error': 'mae'})
    
    base_preds_df = pd.read_parquet(root / 'data/predictions/baseline_price_predictions.parquet')
    base_preds_df['date'] = pd.to_datetime(base_preds_df['date'])
    champ_base_preds = base_preds_df.merge(
        price_champ[['horizon', 'baseline']],
        on=['horizon', 'baseline'],
        how='inner'
    )
    
    # Storage for predictions and runtimes
    prediction_rows = []
    model_runtimes = {'lightgbm': 0.0, 'catboost': 0.0, 'xgboost': 0.0}
    
    if verbose:
        print(f"Beginning 33-fold walk-forward validation across {len(horizons)} horizons...", flush=True)
        
    # Walk-forward validation loop: fold -> origin -> horizon
    for f_idx, fold in folds_df.iterrows():
        fold_id = int(fold['validation_fold'])
        origin = fold['forecast_origin']
        t_start = fold['training_start']
        
        eval_slice = dev_df[dev_df.date == origin].copy()
        if eval_slice.empty:
            continue
            
        X_eval = eval_slice[approved_features].to_numpy(dtype=np.float32)
        
        for h in horizons:
            targ_col = f"purchase_price_target_{h}d"
            valid_mask_col = f"purchase_price_target_{h}d_valid"
            pct_targ_col = f"purchase_price_pct_change_{h}d__targ" if f"purchase_price_pct_change_{h}d__targ" in dev_df.columns else f"purchase_price_pct_change_{h}d"
            dir_targ_col = f"price_direction_{h}d"
            
            # Temporal training cutoff: date + h <= origin
            train_cutoff = origin - pd.Timedelta(days=h)
            train_mask = (dev_df.date >= t_start) & (dev_df.date <= train_cutoff) & (dev_df[valid_mask_col] == 1)
            train_slice = dev_df.loc[train_mask]
            
            if train_slice.empty:
                continue
                
            X_train = train_slice[approved_features].to_numpy(dtype=np.float32)
            y_train_direct = train_slice[targ_col].to_numpy(dtype=np.float32)
            y_train_pct = train_slice[pct_targ_col].to_numpy(dtype=np.float32)
            y_train_dir = train_slice[dir_targ_col].map(CLASS_TO_INT).fillna(1).to_numpy(dtype=int)
            
            # Balanced class weights for direction classifier
            classes, counts = np.unique(y_train_dir, return_counts=True)
            n_samples = len(y_train_dir)
            n_classes = len(classes)
            class_weight_map = {c: n_samples / (n_classes * cnt) for c, cnt in zip(classes, counts)}
            sample_weights = np.array([class_weight_map.get(c, 1.0) for c in y_train_dir], dtype=np.float32)
            
            models = instantiate_models(cfg)
            
            # 1. Direct Regressors
            for fam in ['lightgbm', 'catboost', 'xgboost']:
                m_direct = models[f"{fam}_direct"]
                t0 = time.time()
                m_direct.fit(X_train, y_train_direct)
                model_runtimes[fam] += time.time() - t0
                
                pred_direct = np.maximum(0.0, np.asarray(m_direct.predict(X_eval), dtype=float))
                
                # Residuals for quantiles
                train_res = y_train_direct - m_direct.predict(X_train)
                raw_q10, raw_q50, raw_q90, _ = compute_residual_quantiles(train_res, pred_direct)
                
                for i, (_, row_i) in enumerate(eval_slice.iterrows()):
                    prediction_rows.append({
                        'date': origin,
                        'forecast_origin': origin,
                        'store_id': row_i['store_id'],
                        'product_id': row_i['product_id'],
                        'horizon': h,
                        'validation_fold': fold_id,
                        'model_name': f"{fam}_direct",
                        'family': fam,
                        'approach': 'direct',
                        'predicted_price': pred_direct[i],
                        'predicted_pct_change': np.nan,
                        'predicted_class_label': None,
                        'raw_p10': raw_q10[i],
                        'raw_p50': raw_q50[i],
                        'raw_p90': raw_q90[i],
                    })
                    
            # 2. Percentage-Change Regressors
            for fam in ['lightgbm', 'catboost', 'xgboost']:
                m_pct = models[f"{fam}_pct"]
                t0 = time.time()
                m_pct.fit(X_train, y_train_pct)
                model_runtimes[fam] += time.time() - t0
                
                pred_pct = np.asarray(m_pct.predict(X_eval), dtype=float)
                
                for i, (_, row_i) in enumerate(eval_slice.iterrows()):
                    prediction_rows.append({
                        'date': origin,
                        'forecast_origin': origin,
                        'store_id': row_i['store_id'],
                        'product_id': row_i['product_id'],
                        'horizon': h,
                        'validation_fold': fold_id,
                        'model_name': f"{fam}_pct_change",
                        'family': fam,
                        'approach': 'percentage_change',
                        'predicted_price': np.nan,  # Reconstructed after merging with current_price
                        'predicted_pct_change': pred_pct[i],
                        'predicted_class_label': None,
                        'raw_p10': np.nan,
                        'raw_p50': np.nan,
                        'raw_p90': np.nan,
                    })
                    
            # 3. Dedicated Direction Classifiers
            for fam in ['lightgbm', 'catboost', 'xgboost']:
                m_cls = models[f"{fam}_cls"]
                t0 = time.time()
                m_cls.fit(X_train, y_train_dir, sample_weight=sample_weights)
                model_runtimes[fam] += time.time() - t0
                
                raw_cls = m_cls.predict(X_eval)
                if hasattr(raw_cls, 'ndim') and raw_cls.ndim > 1:
                    raw_cls = np.argmax(raw_cls, axis=1)
                raw_cls = np.asarray(raw_cls).ravel()
                cls_labels = [INT_TO_CLASS.get(int(c), 'FLAT') for c in raw_cls]
                
                for i, (_, row_i) in enumerate(eval_slice.iterrows()):
                    prediction_rows.append({
                        'date': origin,
                        'forecast_origin': origin,
                        'store_id': row_i['store_id'],
                        'product_id': row_i['product_id'],
                        'horizon': h,
                        'validation_fold': fold_id,
                        'model_name': f"{fam}_classifier",
                        'family': fam,
                        'approach': 'classifier',
                        'predicted_price': np.nan,
                        'predicted_pct_change': np.nan,
                        'predicted_class_label': cls_labels[i],
                        'raw_p10': np.nan,
                        'raw_p50': np.nan,
                        'raw_p90': np.nan,
                    })
                    
        if verbose and (fold_id % 5 == 0 or fold_id == len(folds_df)):
            print(f"Completed Fold {fold_id}/{len(folds_df)} (Origin: {origin.date()})", flush=True)
            
    eval_time = time.time() - start_time
    if verbose:
        print(f"Walk-forward evaluation completed in {eval_time:.2f}s", flush=True)
        
    preds_df = pd.DataFrame(prediction_rows)
    
    # 5. Merge with champion baseline predictions for complete metadata, actuals, and baseline comparisons
    merge_keys = ['date', 'store_id', 'product_id', 'horizon', 'validation_fold']
    join_cols = [
        'date', 'store_id', 'product_id', 'horizon', 'validation_fold',
        'prediction', 'actual', 'current_price', 'actual_direction', 'supplier_id',
        'category', 'state', 'demand_type', 'promotion_period', 'festival_period',
        'high_volatility', 'policy_change', 'commodity_shock', 'price_shock'
    ]
    meta_df = champ_base_preds[[c for c in join_cols if c in champ_base_preds.columns]].rename(
        columns={
            'prediction': 'baseline_prediction',
            'actual': 'actual_future_price',
            'current_price': 'current_purchase_price'
        }
    )
    
    final_preds_df = preds_df.merge(meta_df, on=merge_keys, how='left')
    
    # Fill in prices for percentage change approach: current_purchase_price * (1 + pct_change)
    pct_mask = final_preds_df['approach'] == 'percentage_change'
    final_preds_df.loc[pct_mask, 'predicted_price'] = np.maximum(
        0.0,
        final_preds_df.loc[pct_mask, 'current_purchase_price'] * (1.0 + final_preds_df.loc[pct_mask, 'predicted_pct_change'])
    )
    
    # Add absolute and percentage changes
    final_preds_df['predicted_future_price'] = final_preds_df['predicted_price']
    final_preds_df['predicted_absolute_change'] = final_preds_df['predicted_future_price'] - final_preds_df['current_purchase_price']
    final_preds_df['actual_absolute_change'] = final_preds_df['actual_future_price'] - final_preds_df['current_purchase_price']
    final_preds_df['predicted_percentage_change'] = (final_preds_df['predicted_absolute_change'] / (final_preds_df['current_purchase_price'] + 1e-9)) * 100.0
    final_preds_df['actual_percentage_change'] = (final_preds_df['actual_absolute_change'] / (final_preds_df['current_purchase_price'] + 1e-9)) * 100.0
    
    # Determine predicted direction
    # For classifiers: predicted_class_label
    # For direct & percentage regressors: derive direction using flat_tolerance
    dir_derived = derive_direction_from_prices(
        final_preds_df['predicted_future_price'],
        final_preds_df['current_purchase_price'],
        flat_tolerance=flat_tol
    )
    final_preds_df['predicted_direction'] = np.where(
        final_preds_df['approach'] == 'classifier',
        final_preds_df['predicted_class_label'],
        dir_derived
    )
    
    # Map imported_flag from product
    final_preds_df['imported_flag'] = (final_preds_df['product_id'] == 'PRD_ONN_06').astype(int)
    
    # 6. Compute Regression Metrics for Direct and Percentage Change Models
    reg_rows = []
    reg_sub = final_preds_df[final_preds_df['approach'].isin(['direct', 'percentage_change'])]
    for (h, m_name), grp in reg_sub.groupby(['horizon', 'model_name']):
        reg_m = compute_regression_metrics(grp['actual_future_price'], grp['predicted_future_price'])
        reg_rows.append({
            'horizon': h,
            'model': m_name,
            'family': grp['family'].iloc[0],
            'approach': grp['approach'].iloc[0],
            **reg_m
        })
    pooled_reg_df = pd.DataFrame(reg_rows)
    
    # Baseline comparison
    baseline_comp_df = compute_baseline_comparison(pooled_reg_df, price_champ)
    
    # 7. Compute Direction Metrics (Classifiers vs Regression-derived)
    dir_rows = []
    confusion_matrices = {}
    for (h, m_name), grp in final_preds_df.groupby(['horizon', 'model_name']):
        dir_m = compute_direction_metrics(grp['actual_direction'], grp['predicted_direction'])
        dir_rows.append({
            'horizon': h,
            'method': m_name,
            'family': grp['family'].iloc[0],
            'approach': grp['approach'].iloc[0],
            **dir_m
        })
        cm = compute_confusion_matrix_df(grp['actual_direction'], grp['predicted_direction'])
        confusion_matrices[f"{m_name}_{h}d"] = cm
    pooled_dir_df = pd.DataFrame(dir_rows)
    
    # 8. Rank Models and Select Best Candidates
    ranking_records = []
    best_candidates = []
    for h in horizons:
        h_comp = baseline_comp_df[baseline_comp_df.horizon == h].sort_values('mae')
        for rank_idx, (_, row) in enumerate(h_comp.iterrows(), start=1):
            is_best = (rank_idx == 1)
            status = 'BEST_CANDIDATE_PRICE_MODEL' if is_best else 'CANDIDATE_PRICE_MODEL'
            ranking_records.append({
                'horizon': h,
                'task': 'regression',
                'rank': rank_idx,
                'model': row['model'],
                'family': row['family'],
                'approach': row['approach'],
                'mae': row['mae'],
                'baseline_mae': row['baseline_mae'],
                'mae_reduction_pct': row['mae_reduction_pct'],
                'rmse': row['rmse'],
                'bias': row['bias'],
                'status': status
            })
            if is_best:
                best_candidates.append({
                    'horizon': h,
                    'task': 'regression',
                    'champion_model': row['model'],
                    'family': row['family'],
                    'approach': row['approach'],
                    'mae': row['mae'],
                    'baseline_mae': row['baseline_mae'],
                    'mae_reduction_pct': row['mae_reduction_pct'],
                    'status': 'BEST_CANDIDATE_PRICE_MODEL'
                })
    ranking_df = pd.DataFrame(ranking_records)
    best_candidates_df = pd.DataFrame(best_candidates)
    
    # 9. Segmented Evaluations
    segmented_dfs = _compute_segmented_metrics(final_preds_df, dev_df)
    
    # 10. Target Sample Counts Report
    sample_counts = []
    for h in horizons:
        valid_cnt = int((dev_df[f"purchase_price_target_{h}d_valid"] == 1).sum())
        total_cnt = len(dev_df)
        sample_counts.append({
            'horizon': h,
            'total_potential_rows': total_cnt,
            'valid_target_rows': valid_cnt,
            'missing_target_rows': total_cnt - valid_cnt,
            'usable_percentage': (valid_cnt / total_cnt) * 100.0,
            'median_days_between_observations': 1.0
        })
    sample_counts_df = pd.DataFrame(sample_counts)
    
    # 11. Model Runtimes
    runtime_records = []
    for fam, sec in model_runtimes.items():
        runtime_records.append({
            'model_family': fam,
            'total_walk_forward_runtime_seconds': round(sec, 2),
            'folds_evaluated': len(folds_df)
        })
    runtime_df = pd.DataFrame(runtime_records)
    
    # 12. Full Development Retraining & Serialization
    if verbose:
        print("\n--- Retraining Candidate Models on Full Development History ---", flush=True)
        
    models_dir = root / 'models/price'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    for h in horizons:
        targ_col = f"purchase_price_target_{h}d"
        valid_mask_col = f"purchase_price_target_{h}d_valid"
        pct_targ_col = f"purchase_price_pct_change_{h}d__targ" if f"purchase_price_pct_change_{h}d__targ" in dev_df.columns else f"purchase_price_pct_change_{h}d"
        dir_targ_col = f"price_direction_{h}d"
        
        train_full = dev_df[dev_df[valid_mask_col] == 1]
        X_full = train_full[approved_features].to_numpy(dtype=np.float32)
        y_full_direct = train_full[targ_col].to_numpy(dtype=np.float32)
        y_full_pct = train_full[pct_targ_col].to_numpy(dtype=np.float32)
        y_full_dir = train_full[dir_targ_col].map(CLASS_TO_INT).fillna(1).to_numpy(dtype=int)
        
        models = instantiate_models(cfg)
        
        for fam in ['lightgbm', 'catboost', 'xgboost']:
            fam_dir = models_dir / fam
            fam_dir.mkdir(parents=True, exist_ok=True)
            
            # Direct Regressor
            reg_model = models[f"{fam}_direct"]
            reg_model.fit(X_full, y_full_direct)
            joblib.dump(reg_model, fam_dir / f"{fam}_direct_{h}d.joblib")
            
            # Percentage Regressor
            pct_model = models[f"{fam}_pct"]
            pct_model.fit(X_full, y_full_pct)
            joblib.dump(pct_model, fam_dir / f"{fam}_pct_{h}d.joblib")
            
            # Dedicated Classifier
            cls_model = models[f"{fam}_cls"]
            cls_model.fit(X_full, y_full_dir)
            joblib.dump(cls_model, fam_dir / f"{fam}_classifier_{h}d.joblib")
            
            # Validation residuals for raw quantiles
            sub_res = final_preds_df[(final_preds_df.horizon == h) & (final_preds_df.model_name == f"{fam}_direct")]
            if not sub_res.empty:
                residuals = sub_res['actual_future_price'] - sub_res['predicted_future_price']
                _, _, _, offsets = compute_residual_quantiles(residuals, [0.0])
            else:
                offsets = (0.0, 0.0, 0.0)
                
            metadata = {
                'task': 'purchase_price',
                'horizon': h,
                'model_family': fam,
                'target': targ_col,
                'unit': 'MYR',
                'training_cutoff': str(holdout_start.date()),
                'features_count': len(approved_features),
                'q10_offset': offsets[0],
                'q50_offset': offsets[1],
                'q90_offset': offsets[2],
                'quantile_status': 'RAW_UNCALIBRATED'
            }
            with (fam_dir / f"{fam}_direct_metadata_{h}d.json").open('w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
                
    with (models_dir / 'feature_names.json').open('w', encoding='utf-8') as f:
        json.dump(approved_features, f, indent=2)
        
    # Save Predictions Parquet
    pred_path = root / 'data/predictions/price_model_predictions.parquet'
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure final column types
    final_preds_df['forecast_origin'] = final_preds_df['forecast_origin'].astype(str)
    final_preds_df['date'] = final_preds_df['date'].astype(str)
    final_preds_df.to_parquet(pred_path, index=False)
    if verbose:
        print(f"Saved row-level predictions to: {pred_path}", flush=True)
        
    # 13. Generate Reports & Visualizations
    output_dir = root / 'reports/models/price'
    figures_dir = root / 'reports/figures/models/price'
    generate_reports(
        predictions_df=final_preds_df,
        regression_metrics_df=pooled_reg_df,
        baseline_comp_df=baseline_comp_df,
        direction_metrics_df=pooled_dir_df,
        confusion_matrices=confusion_matrices,
        segmented_dfs=segmented_dfs,
        sample_counts_df=sample_counts_df,
        runtime_df=runtime_df,
        ranking_df=ranking_df,
        best_candidates_df=best_candidates_df,
        output_dir=output_dir,
        figures_dir=figures_dir
    )
    
    total_time = time.time() - start_time
    if verbose:
        print(f"Phase 12 Pipeline successfully completed in {total_time:.2f} seconds.", flush=True)
        print(f"Reports saved to: {output_dir}", flush=True)
        print(f"Figures saved to: {figures_dir}", flush=True)


def _compute_segmented_metrics(preds_df, dev_df):
    """Compute detailed segmentations for Section 46 CSVs."""
    # Filter to regression models for price metrics
    reg_df = preds_df[preds_df['approach'].isin(['direct', 'percentage_change'])].copy()
    
    # 1. Price metrics by horizon
    by_h_records = []
    for (h, m), grp in reg_df.groupby(['horizon', 'model_name']):
        reg = compute_regression_metrics(grp['actual_future_price'], grp['predicted_future_price'])
        dir_m = compute_direction_metrics(grp['actual_direction'], grp['predicted_direction'])
        err_dist = compute_error_distribution(grp['actual_future_price'], grp['predicted_future_price'])
        by_h_records.append({
            'horizon': h,
            'model': m,
            **reg,
            'direction_accuracy': dir_m['accuracy'],
            'direction_bal_accuracy': dir_m['balanced_accuracy'],
            **err_dist
        })
    by_h_df = pd.DataFrame(by_h_records)
    
    # 2. Category segmentation
    by_cat_records = []
    for (h, cat, m), grp in reg_df.groupby(['horizon', 'category', 'model_name']):
        reg = compute_regression_metrics(grp['actual_future_price'], grp['predicted_future_price'])
        dir_m = compute_direction_metrics(grp['actual_direction'], grp['predicted_direction'])
        by_cat_records.append({
            'horizon': h,
            'category': cat,
            'model': m,
            **reg,
            'direction_accuracy': dir_m['accuracy']
        })
    by_cat_df = pd.DataFrame(by_cat_records)
    
    # 3. Supplier segmentation
    by_sup_records = []
    for (h, sup, m), grp in reg_df.groupby(['horizon', 'supplier_id', 'model_name']):
        reg = compute_regression_metrics(grp['actual_future_price'], grp['predicted_future_price'])
        dir_m = compute_direction_metrics(grp['actual_direction'], grp['predicted_direction'])
        by_sup_records.append({
            'horizon': h,
            'supplier_id': sup,
            'model': m,
            **reg,
            'direction_accuracy': dir_m['accuracy']
        })
    by_sup_df = pd.DataFrame(by_sup_records)
    
    # 4. Imported vs Domestic
    by_imp_records = []
    for (h, imp, m), grp in reg_df.groupby(['horizon', 'imported_flag', 'model_name']):
        reg = compute_regression_metrics(grp['actual_future_price'], grp['predicted_future_price'])
        dir_m = compute_direction_metrics(grp['actual_direction'], grp['predicted_direction'])
        by_imp_records.append({
            'horizon': h,
            'imported_flag': imp,
            'origin_type': 'Imported' if imp == 1 else 'Domestic',
            'model': m,
            **reg,
            'direction_accuracy': dir_m['accuracy']
        })
    by_imp_df = pd.DataFrame(by_imp_records)
    
    # 5. Volatility segmentation
    vol_map = {
        'PRD_CHK_01': 'MEDIUM',
        'PRD_EGG_02': 'LOW',
        'PRD_OIL_05': 'MEDIUM',
        'PRD_ONN_06': 'HIGH',
        'PRD_RIC_04': 'LOW',
        'PRD_VCH_03': 'HIGH'
    }
    reg_df['volatility_group'] = reg_df['product_id'].map(vol_map).fillna('MEDIUM')
    by_vol_records = []
    for (h, vol, m), grp in reg_df.groupby(['horizon', 'volatility_group', 'model_name']):
        reg = compute_regression_metrics(grp['actual_future_price'], grp['predicted_future_price'])
        by_vol_records.append({
            'horizon': h,
            'volatility_group': vol,
            'model': m,
            **reg
        })
    by_vol_df = pd.DataFrame(by_vol_records)
    
    # 6. Price shock periods (|actual pct change| >= 5%)
    reg_df['is_price_shock'] = np.abs(reg_df['actual_percentage_change']) >= 5.0
    by_shk_records = []
    for (h, shk, m), grp in reg_df.groupby(['horizon', 'is_price_shock', 'model_name']):
        reg = compute_regression_metrics(grp['actual_future_price'], grp['predicted_future_price'])
        by_shk_records.append({
            'horizon': h,
            'price_shock_period': 'Shock (|change| >= 5%)' if shk else 'Normal (< 5%)',
            'model': m,
            **reg
        })
    by_shk_df = pd.DataFrame(by_shk_records)
    
    # 7. Commodity-sensitive performance
    comm_products = {
        'PRD_OIL_05': 'Cooking Oil (Palm Oil)',
        'PRD_CHK_01': 'Chicken (Feed Grain)',
        'PRD_EGG_02': 'Eggs (Feed Grain)',
        'PRD_RIC_04': 'Rice (Rice)'
    }
    reg_df['commodity_sensitive'] = reg_df['product_id'].map(comm_products).fillna('Other Non-Commodity')
    by_com_records = []
    for (h, com, m), grp in reg_df.groupby(['horizon', 'commodity_sensitive', 'model_name']):
        reg = compute_regression_metrics(grp['actual_future_price'], grp['predicted_future_price'])
        by_com_records.append({
            'horizon': h,
            'commodity_mapping': com,
            'model': m,
            **reg
        })
    by_com_df = pd.DataFrame(by_com_records)
    
    # 8. Policy-event performance
    reg_df['policy_active'] = reg_df['product_id'].isin(['PRD_CHK_01', 'PRD_EGG_02', 'PRD_OIL_05'])
    by_pol_records = []
    for (h, pol, m), grp in reg_df.groupby(['horizon', 'policy_active', 'model_name']):
        reg = compute_regression_metrics(grp['actual_future_price'], grp['predicted_future_price'])
        by_pol_records.append({
            'horizon': h,
            'policy_period': 'Controlled/Subsidised Good' if pol else 'Uncontrolled Good',
            'model': m,
            **reg
        })
    by_pol_df = pd.DataFrame(by_pol_records)
    
    # 9. Currency-sensitive performance
    reg_df['currency_sensitive'] = reg_df['imported_flag'] == 1
    by_curr_records = []
    for (h, curr, m), grp in reg_df.groupby(['horizon', 'currency_sensitive', 'model_name']):
        reg = compute_regression_metrics(grp['actual_future_price'], grp['predicted_future_price'])
        by_curr_records.append({
            'horizon': h,
            'currency_sensitive': 'Imported (FX Exposed)' if curr else 'Domestic',
            'model': m,
            **reg
        })
    by_curr_df = pd.DataFrame(by_curr_records)
    
    return {
        'price_metrics_by_horizon': by_h_df,
        'price_metrics_by_category': by_cat_df,
        'price_metrics_by_supplier': by_sup_df,
        'imported_domestic_performance': by_imp_df,
        'volatility_performance': by_vol_df,
        'price_shock_performance': by_shk_df,
        'commodity_sensitive_performance': by_com_df,
        'policy_event_performance': by_pol_df,
        'currency_sensitive_performance': by_curr_df,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Phase 12 Price Forecasting Pipeline")
    parser.add_argument('--root', type=str, default=str(PROJECT_ROOT))
    args = parser.parse_args()
    run_pipeline(root=args.root)
