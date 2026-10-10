"""Evaluation metrics, segmented performance, and baseline comparisons for Phase 11."""
import numpy as np
import pandas as pd


def compute_regression_metrics(actual, predicted, mase_scale=None):
    """Compute standard retail forecast regression metrics."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not mask.any():
        return {
            'count': 0, 'mae': np.nan, 'rmse': np.nan,
            'wape': np.nan, 'mase': np.nan, 'bias': np.nan
        }
    
    act = actual[mask]
    pred = predicted[mask]
    error = pred - act
    abs_error = np.abs(error)
    
    mae = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    
    denom_wape = float(np.sum(np.abs(act)))
    wape = float(np.sum(abs_error) / denom_wape) if denom_wape > 0 else np.nan
    bias = float(np.mean(error))
    
    mase = np.nan
    if mase_scale is not None:
        scales = np.asarray(mase_scale, dtype=float)[mask]
        valid_scale = np.isfinite(scales) & (scales > 0)
        if valid_scale.any():
            mase = float(np.mean(abs_error[valid_scale] / scales[valid_scale]))
            
    return {
        'count': int(mask.sum()),
        'mae': mae,
        'rmse': rmse,
        'wape': wape,
        'mase': mase,
        'bias': bias
    }


def compute_segmented_metrics(df, group_cols):
    """Compute metrics grouped by arbitrary segmentation columns."""
    rows = []
    for key, group in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        m = compute_regression_metrics(group['actual_demand'], group['predicted_demand'], group.get('mase_scale'))
        row = dict(zip(group_cols, key))
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_baseline_comparison(model_metrics, baseline_metrics):
    """Merge model metrics with champion baseline metrics and calculate relative error reduction."""
    merged = model_metrics.merge(
        baseline_metrics[['horizon', 'wape', 'mae', 'rmse']].rename(
            columns={'wape': 'baseline_wape', 'mae': 'baseline_mae', 'rmse': 'baseline_rmse'}
        ),
        on='horizon',
        how='left'
    )
    merged['wape_reduction_pct'] = (merged['baseline_wape'] - merged['wape']) / merged['baseline_wape'] * 100.0
    merged['mae_reduction_pct'] = (merged['baseline_mae'] - merged['mae']) / merged['baseline_mae'] * 100.0
    merged['outperforms_baseline'] = merged['wape'] < merged['baseline_wape']
    return merged


def compute_residual_quantiles(train_residuals, preds, quantiles=(0.1, 0.5, 0.9)):
    """Compute raw empirical residual quantiles on fold training residuals."""
    residuals = np.asarray(train_residuals, dtype=float)
    valid_res = residuals[np.isfinite(residuals)]
    if len(valid_res) == 0:
        q_vals = [0.0] * len(quantiles)
    else:
        q_vals = [float(np.quantile(valid_res, q)) for q in quantiles]
        
    p = np.asarray(preds, dtype=float)
    raw_q10 = np.maximum(0.0, p + q_vals[0])
    raw_q50 = np.maximum(0.0, p + q_vals[1])
    raw_q90 = np.maximum(0.0, p + q_vals[2])
    
    # Monotonicity check
    raw_q50 = np.maximum(raw_q10, raw_q50)
    raw_q90 = np.maximum(raw_q50, raw_q90)
    
    return raw_q10, raw_q50, raw_q90
