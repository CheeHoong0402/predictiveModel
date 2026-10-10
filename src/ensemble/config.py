"""Configuration and registry loader for Phase 15 Ensemble Modelling."""
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import yaml

from src.utils.paths import PROJECT_ROOT


def load_ensemble_config(root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Load config/final_precalibration_models.yaml if it exists, else return default skeleton."""
    cfg_path = Path(root) / "config/final_precalibration_models.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def save_ensemble_config(config_dict: Dict[str, Any], root: Path = PROJECT_ROOT) -> None:
    """Save machine-readable ensemble configuration to config/final_precalibration_models.yaml."""
    cfg_path = Path(root) / "config/final_precalibration_models.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)


def load_model_registry(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Load final pre-calibration model registry CSV."""
    reg_path = Path(root) / "reports/ensemble/final_precalibration_model_registry.csv"
    if not reg_path.exists():
        raise FileNotFoundError(f"Model registry not found: {reg_path}")
    return pd.read_csv(reg_path)


def validate_holdout_protection(dates: pd.Series, holdout_start: str = "2024-10-01") -> bool:
    """Verify that all dates are strictly prior to the final holdout period."""
    boundary = pd.Timestamp(holdout_start)
    max_d = pd.to_datetime(dates).max()
    if max_d >= boundary:
        raise ValueError(f"Final holdout breached! Maximum date {max_d} is >= {boundary}")
    return True
