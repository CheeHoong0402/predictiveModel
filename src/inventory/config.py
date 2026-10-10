"""Configuration and Cost Assumption Registry loader for Inventory Optimisation."""
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import pandas as pd

from src.utils.paths import PROJECT_ROOT


def load_inventory_config(root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Load config/inventory_optimisation.yaml."""
    cfg_path = Path(root) / "config/inventory_optimisation.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Inventory optimisation config not found at {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_cost_assumptions(root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Load config/cost_assumptions.yaml."""
    cfg_path = Path(root) / "config/cost_assumptions.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Cost assumptions config not found at {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def export_cost_assumptions_csv(root: Path = PROJECT_ROOT) -> Path:
    """Export structured cost assumption registry to reports/inventory/cost_assumptions.csv."""
    cost_data = load_cost_assumptions(root=root)
    params = cost_data.get("parameters", {})
    records = []
    for k, v in params.items():
        records.append({
            "parameter_name": k,
            "value": v.get("value", "DYNAMIC"),
            "unit": v.get("unit", ""),
            "provenance": v.get("source", ""),
            "status": v.get("status", ""),
            "description": v.get("description", "")
        })
    df = pd.DataFrame(records)
    out_dir = Path(root) / "reports/inventory"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cost_assumptions.csv"
    df.to_csv(out_path, index=False)
    return out_path
