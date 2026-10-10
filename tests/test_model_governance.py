"""
test_model_governance.py
========================
Tests for the model governance and price model validation gate.
Covers:
1. __targ feature rejected.
2. Direct future target rejected.
3. Contaminated model cannot be loaded for decision use.
4. Missing validation registry defaults to disabled.
5. Invalid model cannot influence recommendation.
6. Demand model remains usable.
7. Model registry records invalid reason.
8. Ensemble inference skips invalid price models.
9. Feature manifest rejects unsafe target-derived columns.
10. Protected final holdout remains untouched.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.models.governance import (
    TargetLeakageError,
    assert_no_target_leakage,
    validate_against_approved_manifest,
    get_price_model_validation_status,
    price_model_usable_for_decision,
    get_demand_model_validation_status,
    demand_model_usable_for_decision,
    load_model_governance_config,
)
from src.ensemble.inference import _resolve_model_path
from src.recommendations.engine import generate_final_recommendation


def test_targ_feature_rejected():
    """1. Columns containing '__targ' must be rejected with TargetLeakageError."""
    unsafe_features = ["purchase_price_lag_1", "purchase_price_pct_change_1d__targ"]
    with pytest.raises(TargetLeakageError) as exc_info:
        assert_no_target_leakage(unsafe_features)
    assert "__targ" in str(exc_info.value)


def test_direct_future_target_rejected():
    """2. Direct future target columns must be rejected."""
    unsafe_features = ["purchase_price_target_7d", "current_stock"]
    with pytest.raises(TargetLeakageError) as exc_info:
        assert_no_target_leakage(unsafe_features, target_columns=["purchase_price_target_7d"])
    assert "purchase_price_target_7d" in str(exc_info.value)


def test_contaminated_model_cannot_be_loaded_for_decision_use():
    """3. Attempting to load an unvalidated price model for decisions must raise PermissionError."""
    root = Path(".")
    with pytest.raises(PermissionError) as exc_info:
        _resolve_model_path("price", "lightgbm_pct", 1, root=root, allow_quarantined_price=False)
    assert "BLOCKED by validation gate" in str(exc_info.value)


def test_missing_validation_registry_defaults_to_disabled(tmp_path):
    """4. If governance config is missing, price model must default to disabled (False)."""
    cfg = load_model_governance_config(root=tmp_path)
    assert cfg["price_model"]["safe_for_decision"] is False
    assert cfg["price_model"]["current_status"] == "NOT_VALIDATED"


def test_invalid_model_cannot_influence_recommendation():
    """5. An invalid price model cannot trigger speculative timing (e.g. wait or prebuy)."""
    rec = generate_final_recommendation(
        store_id="STR_KUL_001",
        product_id="PRD_CHK_01",
        root=Path(".")
    )
    # Price model is unvalidated, so price decision effect must reflect unvalidated/disabled state
    pr_dec = rec.get("price_decision", {})
    assert pr_dec.get("predicted_change_pct") in (0.0, None)
    assert pr_dec.get("price_signal") in ("NEUTRAL_PRICE_SIGNAL", None, "PRICE_MODEL_NOT_VALIDATED")
    assert rec.get("price_order_adjustment_units", 0.0) == 0.0


def test_demand_model_remains_usable():
    """6. Demand model must remain fully validated and usable for decisions."""
    assert demand_model_usable_for_decision() is True
    status = get_demand_model_validation_status()
    assert status["safe_for_decision"] is True
    assert status["status"] == "VALIDATED"


def test_model_registry_records_invalid_reason():
    """7. Price model registry must record explicit invalidation reasons."""
    reg_path = Path("reports/model_governance/price_model_registry.csv")
    assert reg_path.exists()
    df = pd.read_csv(reg_path)
    assert "validation_status" in df.columns
    assert "validation_reason" in df.columns
    
    # Check that optimised models are marked with INVALID_TARGET_LEAKAGE
    opt = df[df["source_phase"].str.contains("Optimised")]
    assert len(opt) > 0
    assert (opt["validation_status"] == "INVALID_TARGET_LEAKAGE").all()
    assert (opt["safe_for_decision"] == False).all()


def test_ensemble_inference_skips_invalid_price_models():
    """8. Ensemble inference resolution blocks unvalidated price models by default."""
    root = Path(".")
    with pytest.raises(PermissionError):
        _resolve_model_path("price", "catboost_pct", 7, root=root, allow_quarantined_price=False)


def test_feature_manifest_rejects_unsafe_target_derived_columns():
    """9. Manifest validation rejects unsafe target-derived columns."""
    unsafe = ["demand_target_1d", "inventory_current"]
    with pytest.raises(TargetLeakageError):
        validate_against_approved_manifest(unsafe, task="price")


def test_protected_final_holdout_remains_untouched():
    """10. Ensure no models or caches contain holdout period rows."""
    from src.validation.price_data_sufficiency import HOLDOUT_START, assert_no_holdout
    
    # Verify cached model-ready parquet files don't cross holdout boundary
    for h in [1, 3, 7, 14, 21, 30]:
        cp = Path(f"cache/model_ready/price_{h}d.parquet")
        if cp.exists():
            df = pd.read_parquet(cp, columns=["date"])
            assert_no_holdout(df, date_col="date")
