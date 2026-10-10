"""Reusable inference interface for future user-uploaded Point-of-Sale data."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from src.utils.paths import PROJECT_ROOT
from .config import load_config, load_demand_feature_manifest, validate_feature_input


def load_model(horizon, model_name=None, root=PROJECT_ROOT):
    """Load serialized model artifact for a specified horizon and model family."""
    root = Path(root)
    models_dir = root / 'models/demand'
    
    # If model_name is None or 'champion' / 'best_candidate', read candidate ranking
    if model_name is None or model_name in ('champion', 'best_candidate'):
        ranking_file = root / 'reports/models/demand/candidate_model_ranking.csv'
        if ranking_file.exists():
            ranking_df = pd.read_csv(ranking_file)
            match = ranking_df[ranking_df.horizon == horizon]
            if not match.empty:
                model_name = match.iloc[0]['model']
            else:
                model_name = 'lightgbm'
        else:
            model_name = 'lightgbm'
            
    model_name = model_name.lower()
    if model_name in ('lgbm', 'lightgbm'):
        model_name = 'lightgbm'
    elif model_name in ('xgb', 'xgboost'):
        model_name = 'xgboost'
    elif model_name in ('cat', 'catboost', 'cb'):
        model_name = 'catboost'
    model_path = models_dir / model_name / f"{model_name}_demand_{horizon}d.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {model_path}")
        
    model = joblib.load(model_path)
    return model, model_name


def predict_demand(feature_frame, horizon, model_name=None, return_quantiles=False, root=PROJECT_ROOT):
    """Generic inference function compatible with user-uploaded POS feature data.
    
    Parameters:
    -----------
    feature_frame : pd.DataFrame
        DataFrame containing approved demand features for prediction origin(s).
        Must not contain target columns.
    horizon : int
        Forecast horizon in days: 1, 3, 7, 14, 21, or 30.
    model_name : str, optional
        Model family to use ('lightgbm', 'catboost', 'xgboost', or None for best candidate).
    return_quantiles : bool, default False
        Whether to return raw uncalibrated quantiles (q10, q50, q90).
    root : Path or str, optional
        Project root directory.
        
    Returns:
    --------
    pd.DataFrame or np.ndarray
        Array or DataFrame of non-negative predicted cumulative demand.
    """
    root = Path(root)
    cfg = load_config(root)
    if horizon not in cfg['evaluation']['horizons']:
        raise ValueError(f"Unsupported horizon: {horizon}. Must be one of {cfg['evaluation']['horizons']}")
        
    manifest = load_demand_feature_manifest(root)
    
    # Input validation
    input_cols = list(feature_frame.columns)
    validate_feature_input(input_cols, manifest)
    
    # Check that all approved features exist; if any missing from uploaded data, fill with NaN
    missing_features = [f for f in manifest if f not in input_cols]
    if missing_features:
        # Create a copy and fill missing required features with NaN
        feature_frame = feature_frame.copy()
        for f in missing_features:
            feature_frame[f] = np.nan
            
    X = feature_frame[manifest]
    model, used_model_name = load_model(horizon, model_name, root)
    
    # Generate predictions
    raw_preds = model.predict(X)
    preds = np.maximum(0.0, np.asarray(raw_preds, dtype=float))
    
    if not return_quantiles:
        return preds
        
    # Read stored metadata for raw uncalibrated quantiles
    meta_path = root / 'models/demand' / used_model_name / f"{used_model_name}_metadata_{horizon}d.json"
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
        
    return pd.DataFrame({
        'predicted_demand': preds,
        'raw_q10': q10,
        'raw_q50': q50,
        'raw_q90': q90,
        'quantile_status': 'raw_uncalibrated_quantiles'
    })
