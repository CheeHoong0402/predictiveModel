"""Configuration loader for Recommendation Engine and Alerts."""
from pathlib import Path
from typing import Any, Dict
import yaml

from src.utils.paths import PROJECT_ROOT

_REC_CONFIG_CACHE: Dict[str, Any] = {}
_ALERTS_CONFIG_CACHE: Dict[str, Any] = {}
_PRICE_DECISION_CONFIG_CACHE: Dict[str, Any] = {}


def load_recommendation_config(root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Load config/recommendation_engine.yaml with caching."""
    global _REC_CONFIG_CACHE
    if not _REC_CONFIG_CACHE:
        config_path = Path(root) / "config" / "recommendation_engine.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration not found at {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            _REC_CONFIG_CACHE = yaml.safe_load(f)
    return _REC_CONFIG_CACHE


def load_price_decision_config(root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Load config/price_decision.yaml with caching."""
    global _PRICE_DECISION_CONFIG_CACHE
    if not _PRICE_DECISION_CONFIG_CACHE:
        config_path = Path(root) / "config" / "price_decision.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                _PRICE_DECISION_CONFIG_CACHE = yaml.safe_load(f)
        else:
            _PRICE_DECISION_CONFIG_CACHE = {"price_decision": {}}
    return _PRICE_DECISION_CONFIG_CACHE


def load_alerts_config(root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Load config/alerts.yaml with caching."""
    global _ALERTS_CONFIG_CACHE
    if not _ALERTS_CONFIG_CACHE:
        config_path = Path(root) / "config" / "alerts.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Alerts configuration not found at {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            _ALERTS_CONFIG_CACHE = yaml.safe_load(f)
    return _ALERTS_CONFIG_CACHE


def clear_config_caches() -> None:
    """Clear cached configurations."""
    _REC_CONFIG_CACHE.clear()
    _ALERTS_CONFIG_CACHE.clear()
    _PRICE_DECISION_CONFIG_CACHE.clear()
