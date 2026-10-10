"""Model diversity and error correlation analysis for Phase 15 Ensemble Modelling."""
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_pairwise_diversity(
    task: str,
    horizon: int,
    model_a: str,
    model_b: str,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    actuals: np.ndarray,
    relative_disagreement_threshold: float = 0.05
) -> Dict[str, Any]:
    """Calculate diversity metrics between two candidate models."""
    res_a = actuals - preds_a
    res_b = actuals - preds_b
    abs_err_a = np.abs(res_a)
    abs_err_b = np.abs(res_b)

    # 1. Prediction correlation
    if np.std(preds_a) > 1e-9 and np.std(preds_b) > 1e-9:
        pred_corr = float(np.corrcoef(preds_a, preds_b)[0, 1])
    else:
        pred_corr = 1.0

    # 2. Residual correlation
    if np.std(res_a) > 1e-9 and np.std(res_b) > 1e-9:
        res_corr = float(np.corrcoef(res_a, res_b)[0, 1])
    else:
        res_corr = 1.0

    # 3. Absolute error correlation
    if np.std(abs_err_a) > 1e-9 and np.std(abs_err_b) > 1e-9:
        abs_err_corr = float(np.corrcoef(abs_err_a, abs_err_b)[0, 1])
    else:
        abs_err_corr = 1.0

    # 4. Disagreement rate
    scale = np.maximum(1.0, (np.abs(preds_a) + np.abs(preds_b)) / 2.0)
    rel_diff = np.abs(preds_a - preds_b) / scale
    disagreement_rate = float(np.mean(rel_diff > relative_disagreement_threshold))

    # A pair is useful if residual correlation is below 0.995 and disagreement is non-trivial
    useful = (res_corr < 0.995) and (disagreement_rate > 0.01)

    return {
        "task": task,
        "horizon": horizon,
        "model_A": model_a,
        "model_B": model_b,
        "prediction_correlation": round(pred_corr, 4),
        "residual_correlation": round(res_corr, 4),
        "absolute_error_correlation": round(abs_err_corr, 4),
        "disagreement_rate": round(disagreement_rate, 4),
        "useful_for_ensembling": bool(useful)
    }


def analyze_model_diversity(
    task: str,
    predictions_df: pd.DataFrame,
    model_col: str = "model",
    pred_col: str = "predicted_demand",
    actual_col: str = "actual_demand"
) -> pd.DataFrame:
    """Compute all pairwise diversities across horizons."""
    records = []
    horizons = sorted(predictions_df["horizon"].unique())

    for h in horizons:
        h_df = predictions_df[predictions_df["horizon"] == h]
        models = sorted(h_df[model_col].unique())
        if len(models) < 2:
            continue

        pvt = h_df.pivot_table(
            index=["validation_fold", "store_id", "product_id"],
            columns=model_col,
            values=pred_col
        )
        act = h_df.groupby(["validation_fold", "store_id", "product_id"])[actual_col].first()
        act_arr = act.to_numpy(dtype=float)

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m_a = models[i]
                m_b = models[j]
                preds_a = pvt[m_a].to_numpy(dtype=float)
                preds_b = pvt[m_b].to_numpy(dtype=float)
                div = compute_pairwise_diversity(
                    task=task,
                    horizon=h,
                    model_a=m_a,
                    model_b=m_b,
                    preds_a=preds_a,
                    preds_b=preds_b,
                    actuals=act_arr
                )
                records.append(div)

    return pd.DataFrame(records)
