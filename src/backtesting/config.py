"""Configuration loader for Phase 20 Historical Business Backtesting."""
from typing import Dict, Any
from pathlib import Path
import yaml

from src.utils.paths import PROJECT_ROOT


def load_backtest_config(config_path: Path = None) -> Dict[str, Any]:
    """Load configuration from config/backtesting.yaml."""
    path = config_path or (PROJECT_ROOT / "config" / "backtesting.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Backtesting configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
