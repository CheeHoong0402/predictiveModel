"""Inference interface for demand and purchase-price forecasting with calibrated predictive uncertainty."""
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd

from src.utils.paths import PROJECT_ROOT
from src.ensemble.inference import predict_final_demand, predict_final_purchase_price
from src.calibration.calibrators import BaseCalibrator

# In-memory cache for loaded calibrators
_CALIBRATOR_CACHE: Dict[str, BaseCalibrator] = {}


def load_cached_calibrator(task: str, horizon: int, root: Path = PROJECT_ROOT) -> BaseCalibrator:
    """Retrieve calibrator instance from cache or load from disk."""
    key = f"{task}_{horizon}d"
    if key not in _CALIBRATOR_CACHE:
        calib_path = Path(root) / f"models/calibration/{task}/calibrator_h{horizon}.joblib"
        if not calib_path.exists():
            raise FileNotFoundError(
                f"Calibrator for {task} horizon {horizon}d not found at {calib_path}. "
                "Ensure calibration pipeline has been executed."
            )
        _CALIBRATOR_CACHE[key] = BaseCalibrator.load(calib_path)
    return _CALIBRATOR_CACHE[key]


def clear_calibrator_cache() -> None:
    """Clear in-memory calibrator cache."""
    _CALIBRATOR_CACHE.clear()


def predict_demand_with_uncertainty(
    feature_frame: pd.DataFrame,
    horizon: Union[int, str],
    root: Path = PROJECT_ROOT
) -> pd.DataFrame:
    """Generate final demand predictions along with calibrated 80% prediction interval (P10, P50, P90).
    
    Compatible with user-uploaded data frames. Does not require target columns.
    
    Returns DataFrame containing:
    - store_id, product_id, horizon
    - point_prediction
    - p10, p50, p90
    """
    h = int(str(horizon).replace("d", ""))
    
    # 1. Base point forecast from Phase 15 ensemble / best single
    base_df = predict_final_demand(feature_frame=feature_frame, horizon=h, root=root)
    point_pred = base_df["final_point_forecast"].to_numpy(dtype=float)
    
    # 2. Check for pre-existing raw quantiles if present in feature frame
    raw_q10 = feature_frame["raw_q10"].to_numpy(dtype=float) if "raw_q10" in feature_frame.columns else None
    raw_q50 = feature_frame["raw_q50"].to_numpy(dtype=float) if "raw_q50" in feature_frame.columns else None
    raw_q90 = feature_frame["raw_q90"].to_numpy(dtype=float) if "raw_q90" in feature_frame.columns else None

    # 3. Load calibrator
    calibrator = load_cached_calibrator(task="demand", horizon=h, root=root)
    
    # 4. Predict calibrated uncertainty
    p10, p50, p90 = calibrator.predict_uncertainty(
        point_pred=point_pred,
        raw_q10=raw_q10,
        raw_q50=raw_q50,
        raw_q90=raw_q90
    )

    # 5. Assemble standardized output
    out_df = base_df.copy()
    out_df["point_prediction"] = point_pred
    out_df["p10"] = p10
    out_df["p50"] = p50
    out_df["p90"] = p90
    out_df["interval_width"] = p90 - p10

    return out_df


def predict_price_with_uncertainty(
    feature_frame: pd.DataFrame,
    horizon: Union[int, str],
    root: Path = PROJECT_ROOT
) -> pd.DataFrame:
    """Generate final purchase-price predictions along with calibrated 80% prediction interval (P10, P50, P90).
    
    Compatible with user-uploaded data frames. Does not require target columns.
    
    Returns DataFrame containing:
    - store_id, product_id, supplier_id, horizon
    - current_purchase_price
    - point_prediction
    - predicted_direction
    - p10, p50, p90
    """
    h = int(str(horizon).replace("d", ""))

    # 1. Base point forecast from Phase 15
    base_df = predict_final_purchase_price(feature_frame=feature_frame, horizon=h, root=root)
    point_pred = base_df["final_predicted_future_price"].to_numpy(dtype=float)

    # 2. Check for pre-existing raw quantiles if present
    raw_p10 = feature_frame["raw_p10"].to_numpy(dtype=float) if "raw_p10" in feature_frame.columns else None
    raw_p50 = feature_frame["raw_p50"].to_numpy(dtype=float) if "raw_p50" in feature_frame.columns else None
    raw_p90 = feature_frame["raw_p90"].to_numpy(dtype=float) if "raw_p90" in feature_frame.columns else None

    # 3. Load calibrator
    calibrator = load_cached_calibrator(task="price", horizon=h, root=root)

    # 4. Predict calibrated uncertainty
    p10, p50, p90 = calibrator.predict_uncertainty(
        point_pred=point_pred,
        raw_q10=raw_p10,
        raw_q50=raw_p50,
        raw_q90=raw_p90
    )

    # 5. Assemble standardized output
    out_df = base_df.copy()
    out_df["point_prediction"] = point_pred
    out_df["p10"] = p10
    out_df["p50"] = p50
    out_df["p90"] = p90
    out_df["interval_width_myr"] = p90 - p10
    out_df["interval_width_pct"] = np.where(
        out_df["current_purchase_price"] > 0,
        (p90 - p10) / out_df["current_purchase_price"] * 100.0,
        0.0
    )

    return out_df
