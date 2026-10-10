"""TreeSHAP computation module supporting native C++ implementations for LightGBM, CatBoost, and XGBoost.

Supports single models and linearly weighted ensembles with exact mathematical additivity:
    sum(phi_j) + phi_0 == y_pred
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

from src.utils.paths import PROJECT_ROOT
from src.optimisation.config import load_optimisation_feature_manifest
from src.ensemble.inference import _resolve_model_path, load_cached_model


def resolve_model_features(model: Any, task: str, root: Path = PROJECT_ROOT) -> List[str]:
    """Resolve the exact list of feature names expected by a model."""
    # 1. Direct attribute on model
    for attr in ["feature_names_", "feature_name_", "feature_names_in_"]:
        if hasattr(model, attr) and getattr(model, attr) is not None:
            val = getattr(model, attr)
            if len(val) > 0:
                return list(val)

    # 2. Booster feature names
    if hasattr(model, "booster_") and hasattr(model.booster_, "feature_name"):
        names = model.booster_.feature_name()
        if names and len(names) > 0:
            return list(names)

    num_f = None
    if hasattr(model, "get_booster"):
        b = model.get_booster()
        if b.feature_names and len(b.feature_names) > 0:
            return list(b.feature_names)
        num_f = b.num_features()
    elif hasattr(model, "n_features_"):
        num_f = model.n_features_
    elif hasattr(model, "n_features_in_"):
        num_f = model.n_features_in_

    # Match by count against full feature manifest
    manifest_path = Path(root) / f"reports/features/{task}_feature_manifest.csv"
    if manifest_path.exists():
        m_df = pd.read_csv(manifest_path)
        col = "name" if "name" in m_df.columns else m_df.columns[0]
        full_feats = m_df[col].tolist()
        if num_f is None or num_f == len(full_feats):
            return full_feats

    # Optimised manifest
    opt_feats = load_optimisation_feature_manifest(task, root=root)
    if num_f is None or num_f == len(opt_feats):
        return opt_feats

    if num_f is not None:
        return [f"f_{i}" for i in range(num_f)]

    return opt_feats


def _compute_single_model_shap(
    model: Any,
    X_mat: np.ndarray,
    feature_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute TreeSHAP contributions and base value for a single model artifact.
    
    Returns:
        phi: Array of shape (N, n_features) representing feature contributions.
        base_value: Array of shape (N,) representing expected model bias / base value.
    """
    model_cls = model.__class__.__name__

    if "LGBM" in model_cls or hasattr(model, "booster_"):
        # LightGBM booster pred_contrib
        booster = getattr(model, "booster_", model)
        contribs = booster.predict(X_mat, pred_contrib=True)
        # Last column is the expected base value
        phi = contribs[:, :-1]
        base_value = contribs[:, -1]
        return phi, base_value

    elif "CatBoost" in model_cls:
        # CatBoost get_feature_importance ShapValues
        pool = Pool(X_mat, feature_names=feature_names)
        contribs = model.get_feature_importance(pool, type="ShapValues")
        phi = contribs[:, :-1]
        base_value = contribs[:, -1]
        return phi, base_value

    elif "XGB" in model_cls or hasattr(model, "get_booster"):
        # XGBoost booster predict pred_contribs
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        dmat = xgb.DMatrix(X_mat, feature_names=feature_names)
        contribs = booster.predict(dmat, pred_contribs=True)
        phi = contribs[:, :-1]
        base_value = contribs[:, -1]
        return phi, base_value

    else:
        raise TypeError(f"Unsupported model architecture for native TreeSHAP: {model_cls}")


def compute_tree_shap(
    task: str,
    horizon: Union[int, str],
    feature_frame: pd.DataFrame,
    root: Path = PROJECT_ROOT
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Compute exact TreeSHAP values for the selected final model or ensemble.
    
    Args:
        task: 'demand' or 'price'
        horizon: Horizon integer (1, 3, 7, 14, 21, 30) or string ('1d', etc.)
        feature_frame: DataFrame containing input features
        root: Project root directory
        
    Returns:
        shap_df: DataFrame of shape (N, n_features) with SHAP values for each feature.
        base_value: 1D numpy array of shape (N,) containing baseline predictions.
        feature_names: List of feature names corresponding to shap_df columns.
    """
    h = int(str(horizon).replace("d", ""))
    reg_path = Path(root) / "reports/ensemble/final_precalibration_model_registry.csv"
    if not reg_path.exists():
        raise FileNotFoundError(f"Model registry not found at {reg_path}")

    reg_df = pd.read_csv(reg_path)
    row = reg_df[(reg_df["task"] == task) & (reg_df["horizon"] == h)]
    if row.empty:
        raise ValueError(f"No registry entry found for {task} horizon {h}d")
    reg_entry = row.iloc[0]

    arch = reg_entry["selected_architecture"]
    comp_models = json.loads(reg_entry["component_models"].replace("'", '"'))
    comp_weights = json.loads(reg_entry["component_weights"].replace("'", '"'))

    # Collect SHAP values and base values for each component
    comp_shap_maps: List[Dict[str, np.ndarray]] = []
    comp_base_list: List[np.ndarray] = []
    all_seen_features: List[str] = []

    for m_name in comp_models:
        if task == "price":
            m_pct_name = m_name.replace("pct_change", "pct")
            m_path = _resolve_model_path(task, m_pct_name, h, root, allow_quarantined_price=True)
        else:
            m_path = _resolve_model_path(task, m_name, h, root)

        model = load_cached_model(m_path)
        m_feats = resolve_model_features(model, task=task, root=root)

        # Track unique features across all components while preserving order
        for f in m_feats:
            if f not in all_seen_features:
                all_seen_features.append(f)

        # Build feature matrix for this component
        missing_m = [f for f in m_feats if f not in feature_frame.columns]
        if missing_m:
            missing_df = pd.DataFrame(0.0, index=feature_frame.index, columns=missing_m, dtype=np.float32)
            X_df_m = pd.concat([feature_frame, missing_df], axis=1)
        else:
            X_df_m = feature_frame
        X_mat_m = X_df_m[m_feats].to_numpy(dtype=np.float32)

        m_phi, m_base = _compute_single_model_shap(model, X_mat_m, m_feats)

        # Store in dict for easy feature-by-feature alignment
        feat_map = {feat: m_phi[:, idx] for idx, feat in enumerate(m_feats)}
        comp_shap_maps.append(feat_map)
        comp_base_list.append(m_base)

    N = len(feature_frame)

    if arch == "SINGLE_MODEL" or len(comp_models) == 1:
        # Single model
        phi_mat = np.zeros((N, len(all_seen_features)), dtype=np.float32)
        feat_map = comp_shap_maps[0]
        for idx, feat in enumerate(all_seen_features):
            phi_mat[:, idx] = feat_map.get(feat, np.zeros(N, dtype=np.float32))
        base_value = comp_base_list[0]
    else:
        # Weighted ensemble
        weights = [float(comp_weights.get(m, 1.0 / len(comp_models))) for m in comp_models]
        w_sum = sum(weights)
        if w_sum > 0:
            weights = [w / w_sum for w in weights]

        phi_mat = np.zeros((N, len(all_seen_features)), dtype=np.float32)
        base_value = np.zeros(N, dtype=np.float32)

        for w, feat_map, m_base in zip(weights, comp_shap_maps, comp_base_list):
            base_value += w * m_base
            for idx, feat in enumerate(all_seen_features):
                if feat in feat_map:
                    phi_mat[:, idx] += w * feat_map[feat]

    shap_df = pd.DataFrame(phi_mat, columns=all_seen_features, index=feature_frame.index)
    return shap_df, base_value, all_seen_features
