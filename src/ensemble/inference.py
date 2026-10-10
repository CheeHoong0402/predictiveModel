"""Reusable final inference interface for Demand and Purchase-Price forecasting."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import joblib
import numpy as np
import pandas as pd

from src.utils.paths import PROJECT_ROOT
from src.optimisation.config import load_optimisation_feature_manifest
from src.ensemble.combiners import (
    combine_simple_mean,
    combine_median,
    combine_weighted,
    post_process_demand,
    post_process_price
)

# Global in-memory cache for loaded model objects
_MODEL_CACHE: Dict[str, Any] = {}


def load_cached_model(model_path: Union[str, Path]) -> Any:
    """Load a model artifact from disk or return cached instance."""
    path_str = str(model_path)
    if path_str not in _MODEL_CACHE:
        if not Path(path_str).exists():
            raise FileNotFoundError(f"Model artifact not found at {path_str}")
        _MODEL_CACHE[path_str] = joblib.load(path_str)
    return _MODEL_CACHE[path_str]


def clear_model_cache() -> None:
    """Clear in-memory model cache."""
    _MODEL_CACHE.clear()


def _resolve_model_path(task: str, model_name: str, horizon: int, root: Path, allow_quarantined_price: bool = False) -> Path:
    """Locate model joblib file on disk.
    
    Guarantees that contaminated or unvalidated price models are NEVER silently loaded.
    """
    from src.models.governance import price_model_usable_for_decision, TargetLeakageError

    h_str = f"{horizon}d"
    base_m = model_name.split("_")[0]
    
    if task == "demand":
        # 1. Check models/optimised/demand
        p1 = root / f"models/optimised/demand/{base_m}_{h_str}.joblib"
        if p1.exists():
            return p1
        # 2. Check models/demand/<base_m>/<base_m>_demand_<h_str>.joblib
        p2 = root / f"models/demand/{base_m}/{base_m}_demand_{h_str}.joblib"
        if p2.exists():
            return p2
        # 3. Check models/demand/<base_m>/<model_name>_<h_str>.joblib
        p3 = root / f"models/demand/{base_m}/{model_name}_{h_str}.joblib"
        if p3.exists():
            return p3
    else:  # price
        # Check validation gate first
        if not allow_quarantined_price and not price_model_usable_for_decision(root=root):
            raise PermissionError(
                f"Price model resolution BLOCKED by validation gate for horizon {h_str}. "
                f"Price models are currently NOT_VALIDATED_FOR_DECISION_USE."
            )

        # STRICT PROHIBITION: models/optimised/price/ contains target leakage for 1d, 7d, 30d
        # Never prioritize models/optimised/price!
        # Fall back to Phase 12 clean models only if explicitly requested for analysis
        # 1. Check clean models/price/<base_m>/<base_m>_pct_<h_str>.joblib
        p2 = root / f"models/price/{base_m}/{base_m}_pct_{h_str}.joblib"
        if p2.exists():
            return p2
        # 2. Check direct models
        p3 = root / f"models/price/{base_m}/{model_name}_{h_str}.joblib"
        if p3.exists():
            return p3
        # 3. Check models/optimised/price only if allow_quarantined_price is True
        if allow_quarantined_price:
            p1 = root / f"models/optimised/price/{base_m}_pct_{h_str}.joblib"
            if p1.exists():
                return p1

    raise FileNotFoundError(f"Could not locate model artifact for {task} {model_name} {h_str} under {root}")


def resolve_model_features(model: Any, task: str, root: Path = PROJECT_ROOT) -> List[str]:
    """Resolve the exact list of feature names expected by a model artifact."""
    for attr in ["feature_names_", "feature_name_", "feature_names_in_"]:
        if hasattr(model, attr) and getattr(model, attr) is not None:
            val = getattr(model, attr)
            if len(val) > 0:
                return list(val)

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

    manifest_path = Path(root) / f"reports/features/{task}_feature_manifest.csv"
    if manifest_path.exists():
        m_df = pd.read_csv(manifest_path)
        col = "name" if "name" in m_df.columns else m_df.columns[0]
        full_feats = m_df[col].tolist()
        if num_f is None or num_f == len(full_feats):
            return full_feats

    opt_feats = load_optimisation_feature_manifest(task, root=root)
    if num_f is None or num_f == len(opt_feats):
        return opt_feats

    if num_f is not None:
        return [f"f_{i}" for i in range(num_f)]

    return opt_feats


def predict_final_demand(
    feature_frame: pd.DataFrame,
    horizon: Union[int, str],
    root: Path = PROJECT_ROOT
) -> pd.DataFrame:
    """Generate final standardized demand predictions for arbitrary feature frames.
    
    Compatible with user-uploaded data frames. Does not require target columns.
    """
    h = int(str(horizon).replace("d", ""))
    reg_path = Path(root) / "reports/ensemble/final_precalibration_model_registry.csv"
    if not reg_path.exists():
        raise FileNotFoundError(f"Final model registry not found at {reg_path}")

    reg_df = pd.read_csv(reg_path)
    row = reg_df[(reg_df["task"] == "demand") & (reg_df["horizon"] == h)]
    if row.empty:
        raise ValueError(f"No registry entry found for demand horizon {h}d")
    reg_entry = row.iloc[0]

    arch = reg_entry["selected_architecture"]
    comp_models = json.loads(reg_entry["component_models"].replace("'", '"'))
    comp_weights = json.loads(reg_entry["component_weights"].replace("'", '"'))

    preds_list = []
    for m_name in comp_models:
        m_path = _resolve_model_path("demand", m_name, h, root)
        model = load_cached_model(m_path)
        m_feats = resolve_model_features(model, task="demand", root=root)
        missing_m = [f for f in m_feats if f not in feature_frame.columns]
        if missing_m:
            missing_df = pd.DataFrame(0.0, index=feature_frame.index, columns=missing_m, dtype=np.float32)
            X_df_m = pd.concat([feature_frame, missing_df], axis=1)
        else:
            X_df_m = feature_frame
        X_mat_m = X_df_m[list(m_feats)].to_numpy(dtype=np.float32)
        m_preds = model.predict(X_mat_m)
        preds_list.append(np.asarray(m_preds, dtype=float))

    preds_mat = np.column_stack(preds_list)

    if arch == "SINGLE_MODEL" or len(comp_models) == 1:
        final_preds = preds_mat[:, 0]
    elif "median" in reg_entry["selected_single_model_or_ensemble"].lower():
        final_preds = combine_median(preds_mat)
    else:  # weighted or mean
        weights = np.array([comp_weights.get(m, 1.0 / len(comp_models)) for m in comp_models], dtype=float)
        final_preds = combine_weighted(preds_mat, weights)

    final_preds = post_process_demand(final_preds)

    # Build standardized output
    out_df = pd.DataFrame()
    for col in ["date", "forecast_origin", "store_id", "product_id"]:
        if col in feature_frame.columns:
            out_df[col] = feature_frame[col].values
        else:
            out_df[col] = "UNKNOWN"

    out_df["horizon"] = h
    out_df["selected_architecture"] = arch
    out_df["selected_model"] = reg_entry["selected_single_model_or_ensemble"]
    out_df["final_point_forecast"] = final_preds

    return out_df


def predict_final_purchase_price(
    feature_frame: pd.DataFrame,
    horizon: Union[int, str],
    root: Path = PROJECT_ROOT
) -> pd.DataFrame:
    """Generate final standardized purchase-price predictions for arbitrary feature frames.
    
    Compatible with user-uploaded data frames. Does not require target columns.
    """
    h = int(str(horizon).replace("d", ""))
    reg_path = Path(root) / "reports/ensemble/final_precalibration_model_registry.csv"
    if not reg_path.exists():
        raise FileNotFoundError(f"Final model registry not found at {reg_path}")

    reg_df = pd.read_csv(reg_path)
    row = reg_df[(reg_df["task"] == "price") & (reg_df["horizon"] == h)]
    if row.empty:
        raise ValueError(f"No registry entry found for price horizon {h}d")
    reg_entry = row.iloc[0]

    default_features = load_optimisation_feature_manifest("price", root=root)

    # Current purchase price reference
    curr_price_col = "purchase_price" if "purchase_price" in feature_frame.columns else "current_purchase_price"
    if curr_price_col in feature_frame.columns:
        p_curr = feature_frame[curr_price_col].to_numpy(dtype=float)
    else:
        p_curr = np.ones(len(feature_frame), dtype=float) * 10.0

    arch = reg_entry["selected_architecture"]
    comp_models = json.loads(reg_entry["component_models"].replace("'", '"'))
    comp_weights = json.loads(reg_entry["component_weights"].replace("'", '"'))

    pct_preds_list = []
    for m_name in comp_models:
        # Resolve percentage change model
        m_pct_name = m_name.replace("pct_change", "pct")
        m_path = _resolve_model_path("price", m_pct_name, h, root, allow_quarantined_price=True)
        model = load_cached_model(m_path)
        m_feats = resolve_model_features(model, task="price", root=root)
        missing_m = [f for f in m_feats if f not in feature_frame.columns]
        if missing_m:
            missing_df = pd.DataFrame(0.0, index=feature_frame.index, columns=missing_m, dtype=np.float32)
            X_df_m = pd.concat([feature_frame, missing_df], axis=1)
        else:
            X_df_m = feature_frame
        X_mat_m = X_df_m[list(m_feats)].to_numpy(dtype=np.float32)
        m_preds = model.predict(X_mat_m)
        pct_preds_list.append(np.asarray(m_preds, dtype=float))

    pct_mat = np.column_stack(pct_preds_list)

    if arch == "SINGLE_MODEL" or len(comp_models) == 1:
        final_pct = pct_mat[:, 0]
    elif "median" in reg_entry["selected_single_model_or_ensemble"].lower():
        final_pct = combine_median(pct_mat)
    else:
        weights = np.array([comp_weights.get(m, 1.0 / len(comp_models)) for m in comp_models], dtype=float)
        final_pct = combine_weighted(pct_mat, weights)

    final_price = post_process_price(p_curr * (1.0 + final_pct))

    # Derived direction
    dirs = []
    for p_diff in final_pct:
        if p_diff > 0.01:
            dirs.append("UP")
        elif p_diff < -0.01:
            dirs.append("DOWN")
        else:
            dirs.append("FLAT")

    out_df = pd.DataFrame()
    for col in ["date", "forecast_origin", "store_id", "product_id", "supplier_id"]:
        if col in feature_frame.columns:
            out_df[col] = feature_frame[col].values
        else:
            out_df[col] = "UNKNOWN"

    out_df["horizon"] = h
    out_df["selected_architecture"] = arch
    out_df["selected_model"] = reg_entry["selected_single_model_or_ensemble"]
    out_df["current_purchase_price"] = p_curr
    out_df["predicted_percentage_change"] = final_pct
    out_df["final_predicted_future_price"] = final_price
    out_df["predicted_direction"] = dirs

    return out_df
