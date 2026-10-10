"""Evaluation engine, objectives, and early stopping for Optuna trials."""
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import optuna

import lightgbm as lgb
import catboost as cb


def evaluate_trial_demand(
    trial: optuna.Trial,
    model_family: str,
    params: Dict[str, Any],
    features: List[str],
    horizon: int,
    folds_df: pd.DataFrame,
    cache_df: pd.DataFrame,
    early_stopping_rounds: int = 30
) -> Tuple[float, Dict[str, float], int, int]:
    """Train and evaluate demand trial across walk-forward folds.
    
    Returns (mean_wape, extra_metrics, actual_iterations, avoided_iterations).
    """
    t_col = f"demand_target_{horizon}d"
    v_col = f"demand_target_{horizon}d_valid"

    fold_wapes = []
    all_actuals = []
    all_preds = []
    trained_iters = []
    max_iters = params.get("n_estimators", params.get("iterations", 300))

    for fold_idx, (_, fold) in enumerate(folds_df.iterrows()):
        origin = pd.Timestamp(fold.forecast_origin)
        t_start = pd.Timestamp(fold.training_start)
        train_cutoff = origin - pd.Timedelta(days=horizon)

        train_mask = (cache_df.date >= t_start) & (cache_df.date <= train_cutoff) & (cache_df[v_col] == 1)
        train_slice = cache_df.loc[train_mask]
        eval_slice = cache_df.loc[cache_df.date == origin]

        if train_slice.empty or eval_slice.empty:
            continue

        X_train = train_slice[features].to_numpy(dtype=np.float32)
        y_train = train_slice[t_col].to_numpy(dtype=np.float32)
        X_eval = eval_slice[features].to_numpy(dtype=np.float32)
        y_eval = eval_slice[t_col].to_numpy(dtype=np.float32)

        # Early stopping validation split: last 15% of training window
        split_idx = int(len(X_train) * 0.85)
        if split_idx > 100:
            X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
            y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
        else:
            X_tr, X_val = X_train, X_train
            y_tr, y_val = y_train, y_train

        if model_family == "lightgbm":
            p = params.copy()
            p["n_estimators"] = max_iters
            model = lgb.LGBMRegressor(**p)
            callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
            best_iter = model.best_iteration_ if hasattr(model, "best_iteration_") and model.best_iteration_ else max_iters
            preds_raw = model.predict(X_eval)
        else:
            p = params.copy()
            p["iterations"] = max_iters
            model = cb.CatBoostRegressor(**p)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=early_stopping_rounds, verbose=0)
            best_iter = model.get_best_iteration() if hasattr(model, "get_best_iteration") and model.get_best_iteration() else max_iters
            preds_raw = model.predict(X_eval)

        trained_iters.append(best_iter)
        preds = np.maximum(0.0, np.asarray(preds_raw, dtype=float))

        all_actuals.extend(y_eval)
        all_preds.extend(preds)

        sum_y = float(np.sum(y_eval))
        f_wape = float(np.sum(np.abs(y_eval - preds)) / sum_y) if sum_y > 0 else 0.0
        fold_wapes.append(f_wape)

        # Optuna pruning checkpoint after initial warmup
        if fold_idx >= 2:
            running_mean = float(np.mean(fold_wapes))
            trial.report(running_mean, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    actual_y = np.array(all_actuals, dtype=float)
    predicted_y = np.array(all_preds, dtype=float)
    err = np.abs(actual_y - predicted_y)
    sum_act = float(np.sum(actual_y))

    wape = float(np.sum(err) / sum_act) if sum_act > 0 else np.nan
    mae = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.sum(predicted_y - actual_y) / sum_act) if sum_act > 0 else 0.0
    fold_std = float(np.std(fold_wapes))
    worst_fold = float(np.max(fold_wapes)) if fold_wapes else np.nan

    avg_best_iter = int(np.mean(trained_iters)) if trained_iters else max_iters
    avoided_iters = max_iters - avg_best_iter

    metrics = {
        "wape": wape,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "fold_std": fold_std,
        "worst_fold": worst_fold
    }
    return wape, metrics, avg_best_iter, avoided_iters


def evaluate_trial_price(
    trial: optuna.Trial,
    model_family: str,
    params: Dict[str, Any],
    features: List[str],
    horizon: int,
    folds_df: pd.DataFrame,
    cache_df: pd.DataFrame,
    early_stopping_rounds: int = 30
) -> Tuple[float, Dict[str, float], int, int]:
    """Train and evaluate price percentage-change trial across walk-forward folds.
    
    Returns (mean_mae, extra_metrics, actual_iterations, avoided_iterations).
    """
    t_col = f"purchase_price_target_{horizon}d"
    v_col = f"purchase_price_target_{horizon}d_valid"
    pct_col = f"purchase_price_pct_change_{horizon}d"

    fold_maes = []
    all_actuals = []
    all_preds = []
    trained_iters = []
    max_iters = params.get("n_estimators", params.get("iterations", 300))

    for fold_idx, (_, fold) in enumerate(folds_df.iterrows()):
        origin = pd.Timestamp(fold.forecast_origin)
        t_start = pd.Timestamp(fold.training_start)
        train_cutoff = origin - pd.Timedelta(days=horizon)

        train_mask = (cache_df.date >= t_start) & (cache_df.date <= train_cutoff) & (cache_df[v_col] == 1)
        train_slice = cache_df.loc[train_mask]
        eval_slice = cache_df.loc[cache_df.date == origin]

        if train_slice.empty or eval_slice.empty:
            continue

        X_train = train_slice[features].to_numpy(dtype=np.float32)
        y_train_pct = train_slice[pct_col].to_numpy(dtype=np.float32)
        X_eval = eval_slice[features].to_numpy(dtype=np.float32)

        eval_curr_price = eval_slice["current_price"].to_numpy(dtype=float)
        eval_actual_price = eval_slice["actual"].to_numpy(dtype=float)

        split_idx = int(len(X_train) * 0.85)
        if split_idx > 100:
            X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
            y_tr, y_val = y_train_pct[:split_idx], y_train_pct[split_idx:]
        else:
            X_tr, X_val = X_train, X_train
            y_tr, y_val = y_train_pct, y_train_pct

        if model_family == "lightgbm":
            p = params.copy()
            p["n_estimators"] = max_iters
            model = lgb.LGBMRegressor(**p)
            callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
            best_iter = model.best_iteration_ if hasattr(model, "best_iteration_") and model.best_iteration_ else max_iters
            pred_pct = np.asarray(model.predict(X_eval), dtype=float)
        else:
            p = params.copy()
            p["iterations"] = max_iters
            model = cb.CatBoostRegressor(**p)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=early_stopping_rounds, verbose=0)
            best_iter = model.get_best_iteration() if hasattr(model, "get_best_iteration") and model.get_best_iteration() else max_iters
            pred_pct = np.asarray(model.predict(X_eval), dtype=float)

        trained_iters.append(best_iter)
        pred_price = np.maximum(0.01, eval_curr_price * (1.0 + pred_pct))

        all_actuals.extend(eval_actual_price)
        all_preds.extend(pred_price)

        f_mae = float(np.mean(np.abs(eval_actual_price - pred_price)))
        fold_maes.append(f_mae)

        if fold_idx >= 2:
            running_mean = float(np.mean(fold_maes))
            trial.report(running_mean, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    actual_p = np.array(all_actuals, dtype=float)
    predicted_p = np.array(all_preds, dtype=float)
    err = np.abs(actual_p - predicted_p)
    sum_act = float(np.sum(actual_p))

    mae = float(np.mean(err))
    wape = float(np.sum(err) / sum_act) if sum_act > 0 else np.nan
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.sum(predicted_p - actual_p) / sum_act) if sum_act > 0 else 0.0
    fold_std = float(np.std(fold_maes))
    worst_fold = float(np.max(fold_maes)) if fold_maes else np.nan

    avg_best_iter = int(np.mean(trained_iters)) if trained_iters else max_iters
    avoided_iters = max_iters - avg_best_iter

    metrics = {
        "mae": mae,
        "wape": wape,
        "rmse": rmse,
        "bias": bias,
        "fold_std": fold_std,
        "worst_fold": worst_fold
    }
    return mae, metrics, avg_best_iter, avoided_iters
