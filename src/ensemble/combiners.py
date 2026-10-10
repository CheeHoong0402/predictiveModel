"""Deterministic combiners and weight optimizers for Phase 15 Ensemble Modelling."""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def combine_simple_mean(predictions_matrix: np.ndarray) -> np.ndarray:
    """Calculate deterministic simple mean across columns (models).
    
    predictions_matrix: shape (n_samples, n_models)
    """
    if predictions_matrix.ndim == 1:
        return predictions_matrix.copy()
    return np.mean(predictions_matrix, axis=1)


def combine_median(predictions_matrix: np.ndarray) -> np.ndarray:
    """Calculate median prediction across columns (models).
    
    predictions_matrix: shape (n_samples, n_models)
    """
    if predictions_matrix.ndim == 1:
        return predictions_matrix.copy()
    return np.median(predictions_matrix, axis=1)


def calculate_validation_weights(metric_values: Dict[str, float]) -> Dict[str, float]:
    """Calculate inverse-error validation weights normalized to sum to 1.
    
    metric_values: mapping of model_name -> validation error metric (WAPE or MAE)
    """
    models = list(metric_values.keys())
    errors = np.array([max(1e-6, float(metric_values[m])) for m in models], dtype=float)
    inv_errors = 1.0 / errors
    weights = inv_errors / np.sum(inv_errors)
    return {m: float(w) for m, w in zip(models, weights)}


def combine_weighted(predictions_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Calculate weighted linear combination.
    
    predictions_matrix: shape (n_samples, n_models)
    weights: shape (n_models,), normalized to sum to 1.
    """
    if predictions_matrix.ndim == 1:
        return predictions_matrix.copy()
    w = np.asarray(weights, dtype=float)
    if not np.isclose(np.sum(w), 1.0, atol=1e-4):
        w = w / np.sum(w)
    return np.dot(predictions_matrix, w)


def optimize_constrained_weights(
    predictions_matrix: np.ndarray,
    actuals: np.ndarray,
    loss_fn: str = "wape"
) -> np.ndarray:
    """Find constrained optimal weights on out-of-fold development predictions.
    
    Constraints:
      weight_i >= 0
      sum(weight_i) == 1
    loss_fn: 'wape' or 'mae'
    """
    n_models = predictions_matrix.shape[1] if predictions_matrix.ndim > 1 else 1
    if n_models == 1:
        return np.array([1.0], dtype=float)

    sum_act = float(np.sum(actuals))

    def objective(w: np.ndarray) -> float:
        combined = np.dot(predictions_matrix, w)
        if loss_fn == "wape":
            if sum_act > 0:
                return float(np.sum(np.abs(actuals - combined)) / sum_act)
            return float(np.mean(np.abs(actuals - combined)))
        else:  # mae
            return float(np.mean(np.abs(actuals - combined)))

    # Initial equal weights
    init_w = np.ones(n_models, dtype=float) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        objective,
        init_w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-6}
    )

    if res.success and np.all(res.x >= -1e-5):
        w_opt = np.maximum(0.0, res.x)
        w_opt = w_opt / np.sum(w_opt)
        return w_opt

    # Fallback to initial equal weights if optimization diverges
    return init_w


def combine_quantiles(
    q10_matrix: np.ndarray,
    q50_matrix: np.ndarray,
    q90_matrix: np.ndarray,
    weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine quantiles and apply transparent monotonic repair (P10 <= P50 <= P90)."""
    q10_c = combine_weighted(q10_matrix, weights)
    q50_c = combine_weighted(q50_matrix, weights)
    q90_c = combine_weighted(q90_matrix, weights)

    # Monotonic repair
    q50_c = np.maximum(q10_c, q50_c)
    q90_c = np.maximum(q50_c, q90_c)

    return q10_c, q50_c, q90_c


def post_process_demand(predictions: np.ndarray) -> np.ndarray:
    """Clip demand predictions to non-negative values."""
    return np.maximum(0.0, np.asarray(predictions, dtype=float))


def post_process_price(predictions: np.ndarray) -> np.ndarray:
    """Enforce positive purchase-price predictions (>= 0.01 MYR)."""
    return np.maximum(0.01, np.asarray(predictions, dtype=float))
