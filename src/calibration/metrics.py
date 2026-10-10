"""Uncertainty calibration metrics, scoring functions, and monotonic repair utilities."""
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


def calculate_coverage(
    y_true: Union[np.ndarray, pd.Series],
    lower: Union[np.ndarray, pd.Series],
    upper: Union[np.ndarray, pd.Series]
) -> float:
    """Calculate empirical prediction interval coverage percentage: P(lower <= y <= upper)."""
    y = np.asarray(y_true, dtype=float)
    l = np.asarray(lower, dtype=float)
    u = np.asarray(upper, dtype=float)
    if len(y) == 0:
        return 0.0
    covered = (y >= l) & (y <= u)
    return float(np.mean(covered))


def calculate_mean_interval_width(
    lower: Union[np.ndarray, pd.Series],
    upper: Union[np.ndarray, pd.Series]
) -> float:
    """Calculate arithmetic mean interval width: E[upper - lower]."""
    l = np.asarray(lower, dtype=float)
    u = np.asarray(upper, dtype=float)
    if len(l) == 0:
        return 0.0
    return float(np.mean(u - l))


def calculate_median_interval_width(
    lower: Union[np.ndarray, pd.Series],
    upper: Union[np.ndarray, pd.Series]
) -> float:
    """Calculate median interval width: Median(upper - lower)."""
    l = np.asarray(lower, dtype=float)
    u = np.asarray(upper, dtype=float)
    if len(l) == 0:
        return 0.0
    return float(np.median(u - l))


def calculate_p90_interval_width(
    lower: Union[np.ndarray, pd.Series],
    upper: Union[np.ndarray, pd.Series]
) -> float:
    """Calculate 90th percentile interval width."""
    l = np.asarray(lower, dtype=float)
    u = np.asarray(upper, dtype=float)
    if len(l) == 0:
        return 0.0
    return float(np.percentile(u - l, 90))


def winkler_interval_score(
    y_true: Union[np.ndarray, pd.Series],
    lower: Union[np.ndarray, pd.Series],
    upper: Union[np.ndarray, pd.Series],
    alpha: float = 0.20
) -> float:
    """Compute the Winkler Interval Score for a nominal (1 - alpha) prediction interval.
    
    Penalizes both interval width and observations outside the interval bounds:
    - If L <= y <= U: Score = (U - L)
    - If y < L: Score = (U - L) + (2 / alpha) * (L - y)
    - If y > U: Score = (U - L) + (2 / alpha) * (y - U)
    Lower score represents better calibrated and sharper interval.
    """
    y = np.asarray(y_true, dtype=float)
    l = np.asarray(lower, dtype=float)
    u = np.asarray(upper, dtype=float)
    if len(y) == 0:
        return 0.0

    width = u - l
    penalty_lower = np.maximum(0.0, l - y) * (2.0 / alpha)
    penalty_upper = np.maximum(0.0, y - u) * (2.0 / alpha)
    return float(np.mean(width + penalty_lower + penalty_upper))


def pinball_loss(
    y_true: Union[np.ndarray, pd.Series],
    q_pred: Union[np.ndarray, pd.Series],
    tau: float
) -> float:
    """Compute Pinball (quantile) loss for quantile tau in (0, 1).
    
    Loss = max(tau * (y - q), (tau - 1) * (y - q))
    """
    y = np.asarray(y_true, dtype=float)
    q = np.asarray(q_pred, dtype=float)
    if len(y) == 0:
        return 0.0

    err = y - q
    return float(np.mean(np.maximum(tau * err, (tau - 1.0) * err)))


def detect_quantile_crossings(
    q10: Union[np.ndarray, pd.Series],
    q50: Union[np.ndarray, pd.Series],
    q90: Union[np.ndarray, pd.Series]
) -> int:
    """Count the number of instances where monotonic quantile ordering is violated:
    q10 > q50 or q50 > q90.
    """
    p10 = np.asarray(q10, dtype=float)
    p50 = np.asarray(q50, dtype=float)
    p90 = np.asarray(q90, dtype=float)
    if len(p10) == 0:
        return 0

    crossings = (p10 > p50) | (p50 > p90)
    return int(np.sum(crossings))


def enforce_monotonic_quantiles(
    q10: Union[np.ndarray, pd.Series],
    q50: Union[np.ndarray, pd.Series],
    q90: Union[np.ndarray, pd.Series],
    task: str = "demand",
    min_val: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enforce strict monotonic ordering (q10 <= q50 <= q90) and domain non-negativity/positivity.
    
    Applies sorting across quantiles to resolve crossings and clips to valid domain:
    - Demand: q >= 0.0
    - Price: q >= 0.01 (MYR)
    """
    p10 = np.asarray(q10, dtype=float)
    p50 = np.asarray(q50, dtype=float)
    p90 = np.asarray(q90, dtype=float)
    
    # 1. Monotonic ordering via column-wise sorting
    mat = np.vstack([p10, p50, p90])
    mat_sorted = np.sort(mat, axis=0)
    
    # 2. Domain floor clipping
    if min_val is not None:
        floor = float(min_val)
    elif task.lower() == "demand":
        floor = 0.0
    else:  # price
        floor = 0.01
        
    mat_clipped = np.maximum(floor, mat_sorted)
    
    return mat_clipped[0], mat_clipped[1], mat_clipped[2]


def classify_calibration_status(
    coverage: float,
    target: float = 0.80,
    tolerance: float = 0.03,
    n_samples: Optional[int] = None,
    min_samples: int = 24
) -> str:
    """Classify the empirical calibration status of a prediction interval against nominal target.
    
    Returns:
    - WELL_CALIBRATED: within target +- tolerance (e.g. 77% - 83% for 80% nominal)
    - SLIGHT_UNDERCOVERAGE: (target - 0.08) <= coverage < (target - tolerance)
    - SLIGHT_OVERCOVERAGE: (target + tolerance) < coverage <= (target + 0.08)
    - POORLY_CALIBRATED: error > 0.08
    - INSUFFICIENT_DATA: sample size below min_samples
    """
    if n_samples is not None and n_samples < min_samples:
        return "INSUFFICIENT_DATA"
        
    cov_err = coverage - target
    if abs(cov_err) <= tolerance:
        return "WELL_CALIBRATED"
    elif -0.08 <= cov_err < -tolerance:
        return "SLIGHT_UNDERCOVERAGE"
    elif tolerance < cov_err <= 0.08:
        return "SLIGHT_OVERCOVERAGE"
    else:
        return "POORLY_CALIBRATED"


def calculate_error_quantiles(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """Calculate absolute error quantiles: 50th, 75th, 90th, 95th, 99th percentiles."""
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if len(y) == 0:
        return {
            "abs_err_p50": 0.0,
            "abs_err_p75": 0.0,
            "abs_err_p90": 0.0,
            "abs_err_p95": 0.0,
            "abs_err_p99": 0.0
        }
    abs_err = np.abs(y - pred)
    return {
        "abs_err_p50": float(np.percentile(abs_err, 50)),
        "abs_err_p75": float(np.percentile(abs_err, 75)),
        "abs_err_p90": float(np.percentile(abs_err, 90)),
        "abs_err_p95": float(np.percentile(abs_err, 95)),
        "abs_err_p99": float(np.percentile(abs_err, 99))
    }


def calculate_all_uncertainty_metrics(
    y_true: Union[np.ndarray, pd.Series],
    q10: Union[np.ndarray, pd.Series],
    q50: Union[np.ndarray, pd.Series],
    q90: Union[np.ndarray, pd.Series],
    alpha: float = 0.20,
    task: str = "demand",
    min_samples: int = 24
) -> Dict[str, Any]:
    """Calculate comprehensive suite of uncertainty calibration metrics."""
    y = np.asarray(y_true, dtype=float)
    p10 = np.asarray(q10, dtype=float)
    p50 = np.asarray(q50, dtype=float)
    p90 = np.asarray(q90, dtype=float)
    n = len(y)

    crossings_before = detect_quantile_crossings(p10, p50, p90)
    p10_m, p50_m, p90_m = enforce_monotonic_quantiles(p10, p50, p90, task=task)
    crossings_after = detect_quantile_crossings(p10_m, p50_m, p90_m)

    cov = calculate_coverage(y, p10_m, p90_m)
    nom = 1.0 - alpha
    cov_err = cov - nom
    m_width = calculate_mean_interval_width(p10_m, p90_m)
    med_width = calculate_median_interval_width(p10_m, p90_m)
    p90_width = calculate_p90_interval_width(p10_m, p90_m)
    winkler = winkler_interval_score(y, p10_m, p90_m, alpha=alpha)
    loss_p10 = pinball_loss(y, p10_m, tau=0.10)
    loss_p50 = pinball_loss(y, p50_m, tau=0.50)
    loss_p90 = pinball_loss(y, p90_m, tau=0.90)
    status = classify_calibration_status(cov, target=nom, tolerance=0.03, n_samples=n, min_samples=min_samples)

    err_quantiles = calculate_error_quantiles(y, p50_m)

    return {
        "sample_count": n,
        "nominal_coverage": nom,
        "observed_coverage": round(cov, 4),
        "coverage_error": round(cov_err, 4),
        "mean_interval_width": round(m_width, 4),
        "median_interval_width": round(med_width, 4),
        "p90_interval_width": round(p90_width, 4),
        "winkler_score": round(winkler, 4),
        "pinball_p10": round(loss_p10, 4),
        "pinball_p50": round(loss_p50, 4),
        "pinball_p90": round(loss_p90, 4),
        "quantile_crossings_before": crossings_before,
        "quantile_crossings_after": crossings_after,
        "status": status,
        **err_quantiles
    }
