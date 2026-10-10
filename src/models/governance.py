"""
Model Governance and Validation Gate Module.
============================================
Enforces scientific safety, target leakage prevention, and model approval gates
across forecasting, inventory optimisation, Monte Carlo simulation, and recommendations.

Strict Fail-Safe Principle:
If governance config is missing, malformed, or unreadable, price models
ALWAYS default to safe_for_decision = False.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import yaml

from src.utils.paths import PROJECT_ROOT


class TargetLeakageError(ValueError):
    """Raised when target-derived or future-looking columns contaminate a feature matrix."""
    pass


BANNED_TARGET_SUBSTRINGS = (
    "__targ",
    "_target_",
    "target_",
    "_actual",
    "actual_",
    "future_price",
    "future_purchase_price",
    "future_demand",
    "forward_price",
)

EXPLICIT_FORWARD_TARGET_PATTERNS = (
    "purchase_price_target_",
    "demand_target_",
    "price_direction_",
    "purchase_price_pct_change_",
    "demand_pct_change_",
    "y__",
)


def assert_no_target_leakage(
    feature_columns: List[str],
    target_columns: Optional[List[str]] = None,
    allow_backward_features: bool = True
) -> None:
    """Enforce strict target leakage prohibition on any candidate feature column list.
    
    Args:
        feature_columns: Columns intended for feature input.
        target_columns: Optional list of actual target columns to explicitly prohibit.
        allow_backward_features: Whether to allow backward-looking momentum/change features
                                (e.g. historical purchase_price_pct_change_1d if known at origin).
    
    Raises:
        TargetLeakageError: If any target or future-derived column is found in feature_columns.
    """
    target_set = set(target_columns) if target_columns else set()
    violations = []

    for col in feature_columns:
        col_lower = col.lower()

        # 1. Direct __targ suffix or substring
        if "__targ" in col_lower:
            violations.append(f"{col} (contains '__targ')")
            continue

        # 2. Direct membership in target_columns
        if col in target_set:
            violations.append(f"{col} (is an explicit target column)")
            continue

        # 3. Future target prefix patterns
        if col_lower.startswith(("purchase_price_target_", "demand_target_", "price_direction_")):
            violations.append(f"{col} (matches future target prefix)")
            continue

        # 4. Check for forward percentage change targets (e.g., if tagged with _target, _future, etc.)
        if any(sub in col_lower for sub in ["__future", "_future_", "future_purchase_price"]):
            violations.append(f"{col} (matches forward-looking pattern)")
            continue

    if violations:
        raise TargetLeakageError(
            f"CRITICAL TARGET LEAKAGE DETECTED in feature columns:\n"
            + "\n".join(f" - {v}" for v in violations)
        )


def validate_against_approved_manifest(
    columns: List[str],
    task: str = "price",
    root: Path = PROJECT_ROOT
) -> None:
    """Validate feature columns against the approved Phase 8 feature manifest.
    
    Raises:
        TargetLeakageError: If unapproved or target-derived columns are present.
    """
    root = Path(root)
    manifest_path = root / f"reports/leakage/{task}_features_temporally_safe.csv"
    if not manifest_path.exists():
        manifest_path = root / f"reports/features/{task}_feature_manifest.csv"
        
    if not manifest_path.exists():
        raise FileNotFoundError(f"Safe feature manifest not found for task '{task}' at {manifest_path}")
        
    import pandas as pd
    m_df = pd.read_csv(manifest_path)
    name_col = "feature_name" if "feature_name" in m_df.columns else "name"
    approved_set = set(m_df[name_col].tolist())
    
    # Standard permissible index keys
    index_keys = {"date", "store_id", "product_id", "supplier_id"}
    
    # Check for leakage first
    assert_no_target_leakage(columns)
    
    # Check for unapproved features
    unapproved = [c for c in columns if c not in approved_set and c not in index_keys]
    if unapproved:
        raise TargetLeakageError(
            f"Feature matrix contains {len(unapproved)} unapproved features not in Phase 8 safe manifest: "
            f"{unapproved[:10]}"
        )


def load_model_governance_config(root: Optional[Path] = None) -> Dict[str, Any]:
    """Load model governance YAML configuration with fail-safe fallback."""
    root = Path(root) if root is not None else PROJECT_ROOT
    cfg_path = root / "config/model_governance.yaml"
    if not cfg_path.exists():
        # FAIL-SAFE DEFAULT: Not validated, not safe
        return {
            "price_model": {
                "validation_required": True,
                "current_status": "NOT_VALIDATED",
                "safe_for_decision": False,
                "safe_for_quantity_adjustment": False,
                "safe_for_coverage_adjustment": False,
                "safe_for_buy_wait_signal": False,
                "safe_for_display": False,
                "reasons": ["Governance configuration file missing; fail-safe active."],
            },
            "demand_model": {
                "validation_required": True,
                "current_status": "VALIDATED",
                "safe_for_decision": True,
                "safe_for_quantity_adjustment": True,
                "safe_for_coverage_adjustment": True,
                "safe_for_display": True,
                "reasons": [],
            }
        }
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                raise ValueError("Malformed YAML")
            return cfg
    except Exception as exc:
        # FAIL-SAFE ON ERROR
        return {
            "price_model": {
                "validation_required": True,
                "current_status": "NOT_VALIDATED",
                "safe_for_decision": False,
                "safe_for_quantity_adjustment": False,
                "safe_for_coverage_adjustment": False,
                "safe_for_buy_wait_signal": False,
                "safe_for_display": False,
                "reasons": [f"Governance config unreadable ({exc}); fail-safe active."],
            },
            "demand_model": {
                "validation_required": True,
                "current_status": "VALIDATED",
                "safe_for_decision": True,
                "safe_for_quantity_adjustment": True,
                "safe_for_coverage_adjustment": True,
                "safe_for_display": True,
                "reasons": [],
            }
        }


def get_price_model_validation_status(root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Return formal validation status dictionary for the purchase-price models."""
    cfg = load_model_governance_config(root=root)
    p_cfg = cfg.get("price_model", {})
    return {
        "status": p_cfg.get("current_status", "NOT_VALIDATED"),
        "safe_for_decision": bool(p_cfg.get("safe_for_decision", False)),
        "safe_for_quantity_adjustment": bool(p_cfg.get("safe_for_quantity_adjustment", False)),
        "safe_for_coverage_adjustment": bool(p_cfg.get("safe_for_coverage_adjustment", False)),
        "safe_for_buy_wait_signal": bool(p_cfg.get("safe_for_buy_wait_signal", False)),
        "safe_for_display": bool(p_cfg.get("safe_for_display", False)),
        "reasons": p_cfg.get("reasons", [
            "Critical target leakage found in optimised price models",
            "Clean models do not materially outperform strong mean baselines",
            "Price target provenance is synthetic / unverified"
        ])
    }


def price_model_usable_for_decision(root: Path = PROJECT_ROOT) -> bool:
    """Master single source of truth: is the price model authorized for business decisions?
    
    Returns:
        False if unvalidated, quarantined, or failing scientific standards.
    """
    status = get_price_model_validation_status(root=root)
    return bool(status.get("safe_for_decision", False))


def get_price_model_validation_reason(root: Path = PROJECT_ROOT) -> str:
    """Return consolidated explanation string for why the price model is or is not approved."""
    status = get_price_model_validation_status(root=root)
    reasons = status.get("reasons", [])
    if not reasons:
        return "Model passed validation gates." if status.get("safe_for_decision") else "Model not validated."
    return "; ".join(reasons)


def get_demand_model_validation_status(root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Return validation status dictionary for the demand forecasting models."""
    cfg = load_model_governance_config(root=root)
    d_cfg = cfg.get("demand_model", {})
    return {
        "status": d_cfg.get("current_status", "VALIDATED"),
        "safe_for_decision": bool(d_cfg.get("safe_for_decision", True)),
        "safe_for_quantity_adjustment": bool(d_cfg.get("safe_for_quantity_adjustment", True)),
        "safe_for_coverage_adjustment": bool(d_cfg.get("safe_for_coverage_adjustment", True)),
        "safe_for_display": bool(d_cfg.get("safe_for_display", True)),
        "reasons": d_cfg.get("reasons", [])
    }


def demand_model_usable_for_decision(root: Path = PROJECT_ROOT) -> bool:
    """Master gate for demand forecasting models."""
    status = get_demand_model_validation_status(root=root)
    return bool(status.get("safe_for_decision", True))
