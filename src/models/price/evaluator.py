"""Evaluation metrics, segmented analysis, and quantile utilities for Phase 12 Purchase-Price Models."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

DIRECTION_CLASSES = ['DOWN', 'FLAT', 'UP']
CLASS_TO_INT = {'DOWN': 0, 'FLAT': 1, 'UP': 2}
INT_TO_CLASS = {0: 'DOWN', 1: 'FLAT', 2: 'UP'}


def compute_regression_metrics(actual, predicted):
    """Compute regression metrics: MAE (RM), RMSE, MAPE, sMAPE, and Bias."""
    y = np.asarray(actual, dtype=float)
    y_hat = np.asarray(predicted, dtype=float)
    n = len(y)
    if n == 0:
        return {'count': 0, 'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'smape': np.nan, 'bias': np.nan}
    
    errors = y_hat - y
    abs_errors = np.abs(errors)
    
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    
    # Avoid zero division in MAPE
    denom = np.where(np.abs(y) > 1e-6, np.abs(y), np.nan)
    mape = float(np.nanmean(abs_errors / denom))
    
    smape_denom = (np.abs(y) + np.abs(y_hat)) / 2.0
    smape_denom = np.where(smape_denom > 1e-6, smape_denom, np.nan)
    smape = float(np.nanmean(abs_errors / smape_denom))
    
    bias = float(np.mean(errors))
    
    return {
        'count': n,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'smape': smape,
        'bias': bias
    }


def compute_direction_metrics(actual_direction, predicted_direction):
    """Compute direction classification metrics."""
    y_true = np.asarray(actual_direction, dtype=object)
    y_pred = np.asarray(predicted_direction, dtype=object)
    
    # Filter out missing/None values
    valid_mask = pd.notna(y_true) & pd.notna(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    n = len(y_true)
    if n == 0:
        return {
            'count': 0,
            'accuracy': np.nan,
            'balanced_accuracy': np.nan,
            'macro_f1': np.nan,
            'weighted_f1': np.nan,
            'macro_precision': np.nan,
            'macro_recall': np.nan
        }
        
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=DIRECTION_CLASSES, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=DIRECTION_CLASSES, average='weighted', zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, labels=DIRECTION_CLASSES, average='macro', zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, labels=DIRECTION_CLASSES, average='macro', zero_division=0))
    
    return {
        'count': n,
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec
    }


def compute_confusion_matrix_df(actual_direction, predicted_direction):
    """Generate labeled 3x3 confusion matrix DataFrame."""
    valid_mask = pd.notna(actual_direction) & pd.notna(predicted_direction)
    y_true = np.asarray(actual_direction)[valid_mask]
    y_pred = np.asarray(predicted_direction)[valid_mask]
    
    cm = confusion_matrix(y_true, y_pred, labels=DIRECTION_CLASSES)
    return pd.DataFrame(
        cm,
        index=[f"actual_{c}" for c in DIRECTION_CLASSES],
        columns=[f"pred_{c}" for c in DIRECTION_CLASSES]
    )


def derive_direction_from_prices(predicted_price, current_price, flat_tolerance=0.005):
    """Derive UP, FLAT, DOWN label from predicted and current prices using flat_tolerance."""
    p_hat = np.asarray(predicted_price, dtype=float)
    p_curr = np.asarray(current_price, dtype=float)
    
    pct_change = (p_hat - p_curr) / (p_curr + 1e-9)
    labels = np.where(
        pct_change > flat_tolerance + 1e-12,
        'UP',
        np.where(pct_change < -flat_tolerance - 1e-12, 'DOWN', 'FLAT')
    )
    return labels


def derive_direction_from_pct_change(predicted_pct_change, flat_tolerance=0.005):
    """Derive UP, FLAT, DOWN label directly from predicted percentage change."""
    pct = np.asarray(predicted_pct_change, dtype=float)
    labels = np.where(
        pct > flat_tolerance + 1e-12,
        'UP',
        np.where(pct < -flat_tolerance - 1e-12, 'DOWN', 'FLAT')
    )
    return labels


def compute_baseline_comparison(model_metrics_df, baseline_df):
    """Compare model regression MAE against Phase 10 champion baselines."""
    merged = pd.merge(
        model_metrics_df,
        baseline_df[['horizon', 'mae']].rename(columns={'mae': 'baseline_mae'}),
        on='horizon',
        how='left'
    )
    merged['mae_reduction_rm'] = merged['baseline_mae'] - merged['mae']
    merged['mae_reduction_pct'] = (merged['mae_reduction_rm'] / merged['baseline_mae']) * 100.0
    merged['outperforms_baseline'] = merged['mae'] < merged['baseline_mae']
    return merged


def compute_residual_quantiles(validation_residuals, point_predictions):
    """Compute empirical uncalibrated raw quantiles (p10, p50, p90) with non-crossing guarantee."""
    res = np.asarray(validation_residuals, dtype=float)
    res = res[np.isfinite(res)]
    if len(res) == 0:
        res = np.array([0.0])
        
    q10_offset = float(np.quantile(res, 0.10))
    q50_offset = float(np.quantile(res, 0.50))
    q90_offset = float(np.quantile(res, 0.90))
    
    p = np.asarray(point_predictions, dtype=float)
    q10 = np.maximum(0.0, p + q10_offset)
    q50 = np.maximum(q10, p + q50_offset)
    q90 = np.maximum(q50, p + q90_offset)
    
    return q10, q50, q90, (q10_offset, q50_offset, q90_offset)


def compute_error_distribution(actual, predicted):
    """Compute error percentiles: median, 75th, 90th, 95th absolute error."""
    y = np.asarray(actual, dtype=float)
    y_hat = np.asarray(predicted, dtype=float)
    abs_errors = np.abs(y_hat - y)
    
    return {
        'median_abs_error': float(np.percentile(abs_errors, 50)),
        'p75_abs_error': float(np.percentile(abs_errors, 75)),
        'p90_abs_error': float(np.percentile(abs_errors, 90)),
        'p95_abs_error': float(np.percentile(abs_errors, 95)),
        'max_abs_error': float(np.max(abs_errors))
    }
