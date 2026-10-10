"""Registry manager and decision engine for Phase 15 Ensemble Modelling."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import yaml

from src.utils.paths import PROJECT_ROOT


def decide_architecture(
    task: str,
    horizon: int,
    best_single_name: str,
    best_single_metric: float,
    best_ensemble_method: str,
    best_ensemble_metric: float,
    single_stability: Dict[str, float],
    ensemble_stability: Dict[str, float],
    min_improvement_threshold_pct: float = 0.20
) -> Dict[str, Any]:
    """Select between SINGLE MODEL and ENSEMBLE using the competition decision rule."""
    if best_single_metric > 0:
        rel_imp_pct = ((best_single_metric - best_ensemble_metric) / best_single_metric) * 100.0
    else:
        rel_imp_pct = 0.0

    worst_fold_diff = single_stability["worst_fold_error"] - ensemble_stability["worst_fold_error"]
    std_diff = single_stability["std_fold_error"] - ensemble_stability["std_fold_error"]

    # Select ensemble if it delivers >= min_improvement_threshold_pct improvement
    # OR if it materially reduces worst-fold error / fold variance without metric regression
    if rel_imp_pct >= min_improvement_threshold_pct:
        arch = "ENSEMBLE"
        selected_model = best_ensemble_method
        reason = (
            f"Ensemble ({best_ensemble_method}) achieved {rel_imp_pct:+.2f}% relative improvement "
            f"exceeding practical threshold of {min_improvement_threshold_pct:.2f}%."
        )
    elif rel_imp_pct >= 0.0 and (worst_fold_diff > 0.001 or std_diff > 0.001):
        arch = "ENSEMBLE"
        selected_model = best_ensemble_method
        reason = (
            f"Ensemble ({best_ensemble_method}) improved fold stability "
            f"(std reduced by {std_diff:.4f}, worst-fold reduced by {worst_fold_diff:.4f}) "
            f"with non-negative metric change ({rel_imp_pct:+.2f}%)."
        )
    else:
        arch = "SINGLE_MODEL"
        selected_model = best_single_name
        reason = (
            f"Single model ({best_single_name}) retained: ensemble gain of {rel_imp_pct:+.2f}% "
            f"did not meet practical threshold of {min_improvement_threshold_pct:.2f}%, avoiding unnecessary complexity."
        )

    return {
        "selected_architecture": arch,
        "selected_model": selected_model,
        "relative_improvement_pct": round(rel_imp_pct, 2),
        "selection_reason": reason
    }


def save_ensemble_metadata(
    task: str,
    horizon: int,
    metadata: Dict[str, Any],
    root: Path = PROJECT_ROOT
) -> Path:
    """Save lightweight JSON metadata artifact in models/ensemble/ referencing component models."""
    out_dir = Path(root) / f"models/ensemble/{task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{metadata['selected_architecture'].lower()}_{horizon}d_metadata.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return out_file
