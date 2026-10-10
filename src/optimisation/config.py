"""Configuration and manifest manager for Phase 14 Hyperparameter Optimisation."""
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import yaml

from src.utils.paths import PROJECT_ROOT


def load_config(root: Path = PROJECT_ROOT) -> dict:
    """Load config/hyperparameter_optimisation.yaml."""
    cfg_path = Path(root) / "config/hyperparameter_optimisation.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Hyperparameter optimisation config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_optimisation_feature_manifest(task: str, root: Path = PROJECT_ROOT) -> List[str]:
    """Load approved KEEP features from Phase 13 ablation manifests.
    
    Strictly verifies that:
    1. Feature has keep_status == 'KEEP'
    2. Temporal status is 'SAFE'
    3. No target columns enter features
    """
    if task not in ["demand", "price"]:
        raise ValueError(f"Task must be 'demand' or 'price', got '{task}'")
        
    path = Path(root) / f"reports/ablation/{task}_features_for_optimisation.csv"
    if not path.exists():
        raise FileNotFoundError(f"Optimisation feature manifest not found: {path}")
        
    df = pd.read_csv(path)
    # Filter KEEP status
    keep_df = df[df["keep_status"] == "KEEP"].copy()
    
    # Verify temporal safety
    non_safe = keep_df[~keep_df["temporal_status"].isin(["SAFE", "SAFE_WITH_ASSUMPTION"])]
    if not non_safe.empty:
        raise ValueError(f"Non-safe features found in KEEP manifest: {non_safe['feature_name'].tolist()}")
        
    features = keep_df["feature_name"].tolist()
    
    # Strict anti-target leakage check
    for f in features:
        if "target" in f.lower():
            raise ValueError(f"Target column detected in feature manifest: {f}")
            
    return sorted(features)


def validate_holdout_protection(dates: pd.Series, holdout_start: str = "2024-10-01") -> bool:
    """Verify that all dates are strictly prior to the final holdout period."""
    boundary = pd.Timestamp(holdout_start)
    max_d = pd.to_datetime(dates).max()
    if max_d >= boundary:
        raise ValueError(f"Final holdout breached! Maximum date {max_d} is >= {boundary}")
    return True
