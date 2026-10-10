"""Reusable inference interface for future user-uploaded Point-of-Sale and Supplier data."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from src.utils.paths import PROJECT_ROOT
from .config import load_config, load_price_feature_manifest, validate_feature_input
from .evaluator import derive_direction_from_prices, INT_TO_CLASS


def load_model(horizon, model_name=None, task_type='regression', root=PROJECT_ROOT):
    """Load serialized model artifact for a specified horizon, model family, and task type."""
    root = Path(root)
    models_dir = root / 'models/price'
    
    # Standardize model_name
    if model_name is None or model_name in ('champion', 'best_candidate'):
        ranking_file = root / 'reports/models/price/candidate_model_ranking.csv'
        if ranking_file.exists():
            ranking_df = pd.read_csv(ranking_file)
            match = ranking_df[(ranking_df.horizon == horizon) & (ranking_df.task == task_type)]
            if not match.empty:
                model_name = match.iloc[0]['model']
            else:
                model_name = 'catboost_direct' if task_type == 'regression' else 'catboost_classifier'
        else:
            model_name = 'catboost_direct' if task_type == 'regression' else 'catboost_classifier'
            
    model_name = str(model_name).lower()
    # Normalize aliases
    family = 'catboost'
    if 'lightgbm' in model_name or 'lgbm' in model_name:
        family = 'lightgbm'
    elif 'xgboost' in model_name or 'xgb' in model_name:
        family = 'xgboost'
    elif 'catboost' in model_name or 'cb' in model_name:
        family = 'catboost'
        
    model_path = models_dir / family / f"{model_name}_{horizon}d.joblib"
    if not model_path.exists():
        # Fallback to family-specific artifact
        pattern = f"*{family}*_{horizon}d.joblib"
        matches = list((models_dir / family).glob(pattern))
        if matches:
            model_path = matches[0]
            model_name = model_path.stem.replace(f"_{horizon}d", "")
        else:
            raise FileNotFoundError(f"Model artifact not found at: {model_path}")
            
    model = joblib.load(model_path)
    return model, model_name, family


def predict_purchase_price(
    feature_frame,
    horizon,
    model_name=None,
    return_quantiles=False,
    root=PROJECT_ROOT
):
    """Generic inference function for supplier purchase price prediction.
    
    Compatible with retailer-uploaded datasets after feature preprocessing.
    """
    root = Path(root)
    cfg = load_config(root)
    if horizon not in cfg['evaluation']['horizons']:
        raise ValueError(f"Unsupported horizon: {horizon}. Must be one of {cfg['evaluation']['horizons']}")
        
    manifest = load_price_feature_manifest(root)
    
    # Input validation
    input_cols = list(feature_frame.columns)
    validate_feature_input(input_cols, manifest)
    
    # Ensure all approved features exist; fill any missing with NaN
    missing_features = [f for f in manifest if f not in input_cols]
    if missing_features:
        feature_frame = feature_frame.copy()
        for f in missing_features:
            feature_frame[f] = np.nan
            
    X = feature_frame[manifest]
    model, used_model_name, family = load_model(horizon, model_name, task_type='regression', root=root)
    
    # Generate predictions
    raw_preds = model.predict(X)
    preds = np.maximum(0.0, np.asarray(raw_preds, dtype=float))
    
    # Check if this was a percentage change model
    if 'pct_change' in used_model_name:
        current_price = None
        for col_name in ['purchase_price_lag_1', 'current_purchase_price', 'purchase_price']:
            if col_name in feature_frame.columns:
                current_price = feature_frame[col_name].to_numpy(dtype=float)
                break
        if current_price is not None:
            # Reconstruct: current_price * (1 + pct_change)
            preds = np.maximum(0.0, current_price * (1.0 + raw_preds))
            
    if not return_quantiles:
        return preds
        
    # Read stored metadata for raw uncalibrated quantiles
    meta_path = root / 'models/price' / family / f"{used_model_name}_metadata_{horizon}d.json"
    if meta_path.exists():
        with meta_path.open(encoding='utf-8') as f:
            meta = json.load(f)
        q10_offset = meta.get('q10_offset', 0.0)
        q50_offset = meta.get('q50_offset', 0.0)
        q90_offset = meta.get('q90_offset', 0.0)
        q10 = np.maximum(0.0, preds + q10_offset)
        q50 = np.maximum(q10, preds + q50_offset)
        q90 = np.maximum(q50, preds + q90_offset)
    else:
        q10 = preds
        q50 = preds
        q90 = preds
        
    res_df = pd.DataFrame({
        'predicted_future_price': preds,
        'raw_p10': q10,
        'raw_p50': q50,
        'raw_p90': q90,
        'quantile_status': 'RAW_UNCALIBRATED'
    })
    
    # If current price available, also include change and direction
    current_p = None
    for col_name in ['purchase_price_lag_1', 'current_purchase_price', 'purchase_price']:
        if col_name in feature_frame.columns:
            current_p = feature_frame[col_name].to_numpy(dtype=float)
            break
            
    if current_p is not None:
        res_df['current_purchase_price'] = current_p
        res_df['expected_absolute_change'] = preds - current_p
        res_df['expected_percentage_change'] = ((preds - current_p) / (current_p + 1e-9)) * 100.0
        res_df['predicted_direction'] = derive_direction_from_prices(
            preds, current_p, flat_tolerance=cfg['evaluation'].get('flat_tolerance', 0.005)
        )
        
    return res_df


def predict_price_direction(
    feature_frame,
    horizon,
    model_name=None,
    method='classifier',
    root=PROJECT_ROOT
):
    """Predict price direction ('UP', 'FLAT', 'DOWN').
    
    Parameters:
    -----------
    method : {'classifier', 'derived'}
        'classifier' uses the dedicated multiclass classifier.
        'derived' predicts price via regression and applies flat_tolerance.
    """
    root = Path(root)
    cfg = load_config(root)
    if horizon not in cfg['evaluation']['horizons']:
        raise ValueError(f"Unsupported horizon: {horizon}. Must be one of {cfg['evaluation']['horizons']}")
        
    manifest = load_price_feature_manifest(root)
    input_cols = list(feature_frame.columns)
    validate_feature_input(input_cols, manifest)
    
    if method == 'derived':
        # Regression-derived direction
        preds = predict_purchase_price(feature_frame, horizon, model_name=model_name, root=root)
        current_p = None
        for col_name in ['purchase_price_lag_1', 'current_purchase_price', 'purchase_price']:
            if col_name in feature_frame.columns:
                current_p = feature_frame[col_name].to_numpy(dtype=float)
                break
        if current_p is None:
            raise ValueError("Current purchase price required in feature frame to derive direction.")
        flat_tol = cfg['evaluation'].get('flat_tolerance', 0.005)
        return derive_direction_from_prices(preds, current_p, flat_tolerance=flat_tol)
        
    # Dedicated Classifier
    model, _, _ = load_model(horizon, model_name, task_type='classification', root=root)
    missing_features = [f for f in manifest if f not in input_cols]
    if missing_features:
        feature_frame = feature_frame.copy()
        for f in missing_features:
            feature_frame[f] = np.nan
            
    X = feature_frame[manifest]
    raw_class_preds = model.predict(X)
    
    # Handle both 1D array of class indices/labels and 2D probability matrices
    if hasattr(raw_class_preds, 'ndim') and raw_class_preds.ndim > 1:
        raw_class_preds = np.argmax(raw_class_preds, axis=1)
        
    raw_class_preds = np.asarray(raw_class_preds).ravel()
    
    # Map int back to string class if needed
    if np.issubdtype(raw_class_preds.dtype, np.integer) or np.issubdtype(raw_class_preds.dtype, np.floating):
        labels = np.array([INT_TO_CLASS.get(int(c), 'FLAT') for c in raw_class_preds])
    else:
        labels = np.array([str(c).upper() for c in raw_class_preds])
        
    return labels
