"""Configuration loader for Phase 19 Monte Carlo Risk Simulation."""
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from src.utils.paths import PROJECT_ROOT


def load_simulation_config(root: Optional[Path] = None) -> Dict[str, Any]:
    """Load config/monte_carlo.yaml."""
    base = Path(root) if root is not None else PROJECT_ROOT
    cfg_path = base / "config" / "monte_carlo.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Monte Carlo configuration not found at {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
