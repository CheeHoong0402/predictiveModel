"""Configuration and feature manifest loader for Phase 11 Demand Models."""
from pathlib import Path
import pandas as pd
import yaml

from src.utils.paths import PROJECT_ROOT

BANNED_INPUT_PREFIXES = (
    'demand_target_', 'demand_day_', 'purchase_price_target_',
    'price_direction_', 'y__'
)


def load_config(root=PROJECT_ROOT):
    root = Path(root)
    config_file = root / 'config/demand_models.yaml'
    with config_file.open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['root'] = root
    return cfg


def load_demand_feature_manifest(root=PROJECT_ROOT):
    root = Path(root)
    manifest_path = root / 'reports/leakage/demand_features_temporally_safe.csv'
    if not manifest_path.exists():
        raise FileNotFoundError(f"Demand feature manifest not found at: {manifest_path}")
    df = pd.read_csv(manifest_path)
    return df['feature_name'].tolist()


def validate_feature_input(columns, approved_features):
    """Enforce that only approved features are present and no targets are included."""
    col_set = set(columns)
    banned_found = [c for c in columns if any(c.startswith(p) for p in BANNED_INPUT_PREFIXES)]
    if banned_found:
        raise ValueError(f"Prohibited target columns found in model inputs: {banned_found}")
    
    unapproved = col_set - set(approved_features) - {'date', 'store_id', 'product_id'}
    if unapproved:
        raise ValueError(f"Unapproved feature columns found in model inputs: {list(unapproved)[:10]}")
