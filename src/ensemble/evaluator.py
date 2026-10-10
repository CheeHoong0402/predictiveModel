"""Evaluation engine, stability analysis, and segmented error metrics for Phase 15."""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def compute_demand_metrics(actuals: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """Compute standard demand forecasting metrics."""
    act = np.asarray(actuals, dtype=float)
    pred = np.asarray(predictions, dtype=float)
    err = np.abs(act - pred)
    sum_act = float(np.sum(act))

    wape = float(np.sum(err) / sum_act) if sum_act > 0 else 0.0
    mae = float(np.mean(err))
    rmse = float(np.sqrt(np.mean((act - pred) ** 2)))
    bias = float(np.mean(pred - act))
    p90_err = float(np.percentile(err, 90))
    p95_err = float(np.percentile(err, 95))
    max_err = float(np.max(err))

    return {
        "wape": wape,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "p90_error": p90_err,
        "p95_error": p95_err,
        "max_error": max_err
    }


def compute_price_metrics(
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    actual_directions: Optional[pd.Series] = None,
    predicted_directions: Optional[pd.Series] = None
) -> Dict[str, float]:
    """Compute standard purchase-price forecasting metrics."""
    act = np.asarray(actual_prices, dtype=float)
    pred = np.asarray(predicted_prices, dtype=float)
    err = np.abs(act - pred)

    mae = float(np.mean(err))
    rmse = float(np.sqrt(np.mean((act - pred) ** 2)))
    # MAPE with protection against division by zero
    non_zero = act > 1e-4
    mape = float(np.mean(err[non_zero] / act[non_zero])) if np.any(non_zero) else 0.0
    bias = float(np.mean(pred - act))
    p90_err = float(np.percentile(err, 90))
    p95_err = float(np.percentile(err, 95))
    max_err = float(np.max(err))

    dir_acc = np.nan
    if actual_directions is not None and predicted_directions is not None:
        dir_acc = float(np.mean(actual_directions.values == predicted_directions.values))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "bias": bias,
        "p90_error": p90_err,
        "p95_error": p95_err,
        "max_error": max_err,
        "direction_accuracy": dir_acc
    }


def evaluate_fold_stability(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    task: str = "demand",
    fold_col: str = "validation_fold"
) -> Dict[str, float]:
    """Calculate fold-level error distribution (mean, std, best, worst fold)."""
    fold_errors = []
    for f, grp in df.groupby(fold_col):
        act = grp[actual_col].to_numpy(dtype=float)
        pred = grp[pred_col].to_numpy(dtype=float)
        if task == "demand":
            s_act = float(np.sum(act))
            err = float(np.sum(np.abs(act - pred)) / s_act) if s_act > 0 else 0.0
        else:  # price
            err = float(np.mean(np.abs(act - pred)))
        fold_errors.append(err)

    fold_errors = np.array(fold_errors, dtype=float)
    return {
        "mean_fold_error": float(np.mean(fold_errors)),
        "std_fold_error": float(np.std(fold_errors)),
        "best_fold_error": float(np.min(fold_errors)),
        "worst_fold_error": float(np.max(fold_errors))
    }


def evaluate_segmented_performance(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    segment_col: str,
    task: str = "demand"
) -> pd.DataFrame:
    """Compute error metric grouped by business segment."""
    records = []
    for seg_val, grp in df.groupby(segment_col):
        act = grp[actual_col].to_numpy(dtype=float)
        pred = grp[pred_col].to_numpy(dtype=float)
        n = len(grp)
        if task == "demand":
            s_act = float(np.sum(act))
            err = float(np.sum(np.abs(act - pred)) / s_act) if s_act > 0 else 0.0
            metric_name = "wape"
        else:
            err = float(np.mean(np.abs(act - pred)))
            metric_name = "mae"

        records.append({
            "segment_name": segment_col,
            "segment_value": str(seg_val),
            "sample_count": n,
            f"segment_{metric_name}": err
        })

    return pd.DataFrame(records)
