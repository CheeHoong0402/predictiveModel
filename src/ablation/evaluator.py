"""Evaluation metrics, differential impact calculations, and classification for Feature Ablation."""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute WAPE, MAE, and RMSE for demand or price predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid_mask):
        return {"wape": np.nan, "mae": np.nan, "rmse": np.nan}
        
    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]
    
    err = np.abs(yt - yp)
    mae = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    
    sum_true = float(np.sum(yt))
    if sum_true > 0:
        wape = float(np.sum(err) / sum_true)
    else:
        wape = np.nan
        
    return {
        "wape": wape,
        "mae": mae,
        "rmse": rmse
    }


def compute_differential(
    full_metric: float,
    ablated_metric: float
) -> Tuple[float, float]:
    """Compute (delta, relative_percentage_change).
    
    Positive delta means ablated error is higher than full model (removing family degraded performance -> family adds value).
    Negative delta means ablated error is lower than full model (removing family improved performance -> family is harmful).
    """
    delta = ablated_metric - full_metric
    if full_metric != 0 and np.isfinite(full_metric):
        rel_pct = (delta / full_metric) * 100.0
    else:
        rel_pct = 0.0
    return delta, rel_pct


def classify_family_value(
    mean_rel_impact_pct: float,
    fold_impacts: List[float],
    thresholds: Optional[Dict[str, float]] = None
) -> str:
    """Classify feature family value based on controlled relative degradation and fold stability.
    
    thresholds default:
      critical: 5.0%
      high_value: 2.0%
      moderate_value: 0.5%
      low_value: 0.05%
      bound: 0.05%
      harmful: -0.05%
    """
    if thresholds is None:
        th = {
            "critical": 5.0,
            "high_value": 2.0,
            "moderate_value": 0.5,
            "low_value": 0.05,
            "bound": 0.05,
            "harmful": -0.05
        }
    else:
        th = thresholds

    # Check fold stability
    if len(fold_impacts) >= 3:
        std_impact = float(np.std(fold_impacts))
        # If standard deviation is greater than 3x the absolute mean and fold signs diverge wildly
        pos_folds = sum(1 for x in fold_impacts if x > 0)
        neg_folds = sum(1 for x in fold_impacts if x < 0)
        if abs(mean_rel_impact_pct) < 0.1 and pos_folds > 0 and neg_folds > 0 and (std_impact > 3 * abs(mean_rel_impact_pct)):
            return "INCONCLUSIVE"

    if mean_rel_impact_pct < th["harmful"]:
        return "HARMFUL"
    elif abs(mean_rel_impact_pct) < th["bound"]:
        return "NO MEASURABLE VALUE"
    elif mean_rel_impact_pct >= th["critical"]:
        return "CRITICAL"
    elif mean_rel_impact_pct >= th["high_value"]:
        return "HIGH VALUE"
    elif mean_rel_impact_pct >= th["moderate_value"]:
        return "MODERATE VALUE"
    elif mean_rel_impact_pct >= th["low_value"]:
        return "LOW VALUE"
    else:
        return "INCONCLUSIVE"


def compute_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute mean and 95% confidence bounds across folds."""
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return mean, mean, mean
    sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
    z = 1.96 if confidence == 0.95 else 1.645
    lower = mean - z * sem
    upper = mean + z * sem
    return mean, lower, upper
