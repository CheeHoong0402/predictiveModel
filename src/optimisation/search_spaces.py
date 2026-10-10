"""Search space definitions and parameter samplers for Optuna trials."""
from typing import Any, Dict, Optional
import optuna


def sample_lightgbm_params(
    trial: optuna.Trial,
    stage: str = "screening",
    best_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Sample hyperparameters for LightGBM Regressor.
    
    Stage A (screening): Compact exploration across viable bounds.
    Stage B (refinement): Focused local search around promising region.
    """
    if stage == "refinement" and best_params:
        # Refinement around previous best
        base_leaves = int(best_params.get("num_leaves", 31))
        base_depth = int(best_params.get("max_depth", 6))
        base_lr = float(best_params.get("learning_rate", 0.05))

        return {
            "num_leaves": trial.suggest_int("num_leaves", max(16, base_leaves - 16), min(96, base_leaves + 16)),
            "max_depth": trial.suggest_int("max_depth", max(4, base_depth - 1), min(9, base_depth + 1)),
            "learning_rate": trial.suggest_float("learning_rate", max(0.01, base_lr * 0.7), min(0.12, base_lr * 1.4), log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 50),
            "subsample": trial.suggest_float("subsample", 0.75, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.75, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 3.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 3.0, log=True),
            "random_state": 42,
            "verbose": -1,
            "n_jobs": 14
        }
    else:
        # Screening search space
        return {
            "num_leaves": trial.suggest_int("num_leaves", 16, 96),
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "subsample": trial.suggest_float("subsample", 0.70, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "random_state": 42,
            "verbose": -1,
            "n_jobs": 14
        }


def sample_catboost_params(
    trial: optuna.Trial,
    stage: str = "screening",
    best_params: Optional[Dict[str, Any]] = None,
    use_gpu: bool = False
) -> Dict[str, Any]:
    """Sample hyperparameters for CatBoost Regressor."""
    if stage == "refinement" and best_params:
        base_depth = int(best_params.get("depth", 6))
        base_lr = float(best_params.get("learning_rate", 0.05))

        params = {
            "depth": trial.suggest_int("depth", max(4, base_depth - 1), min(8, base_depth + 1)),
            "learning_rate": trial.suggest_float("learning_rate", max(0.01, base_lr * 0.7), min(0.12, base_lr * 1.4), log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.5),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_seed": 42,
            "verbose": 0,
            "nan_mode": "Min"
        }
    else:
        params = {
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_seed": 42,
            "verbose": 0,
            "nan_mode": "Min"
        }

    if use_gpu:
        params["task_type"] = "GPU"
        params["devices"] = "0"
    else:
        params["thread_count"] = 14

    return params
