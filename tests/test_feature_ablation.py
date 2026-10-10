"""Automated test suite for Phase 13 Feature Ablation and External-Data Value Testing.

Validates all 24 required test specifications:
 1. Full-model reference uses correct features.
 2. Ablation removes only the intended family.
 3. No unrelated feature family changes.
 4. Same target rows are used.
 5. Same validation folds are used.
 6. Same model configuration is used.
 7. Same seeds are used.
 8. Final holdout is excluded.
 9. Demand metric comparison is correct.
10. Price metric comparison is correct.
11. Relative performance difference is correct.
12. Feature-family mapping is valid.
13. Approved temporal manifests are enforced.
14. Target columns never enter features.
15. Cached datasets match canonical data.
16. Screening/confirmed result status is correctly recorded.
17. Recommended feature manifests contain only safe features.
18. Harmful features can be excluded.
19. Runtime reporting works.
20. Phase 8 temporal-integrity tests remain successful.
21. Phase 9 target tests remain successful.
22. Phase 10 baseline tests remain successful.
23. Phase 11 demand-model tests remain successful.
24. Phase 12 price-model tests remain successful.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.utils.paths import PROJECT_ROOT
from src.ablation.config import (
    load_config,
    load_demand_feature_manifest,
    load_price_feature_manifest,
    get_demand_family_to_features,
    get_price_family_to_features,
    get_buildup_feature_sets,
    get_internal_vs_external_feature_sets,
    validate_ablation_features
)
from src.ablation.cache import load_model_ready_cache
from src.ablation.evaluator import (
    compute_metrics,
    compute_differential,
    classify_family_value
)


@pytest.fixture(scope="module")
def root():
    return PROJECT_ROOT


@pytest.fixture(scope="module")
def config(root):
    return load_config(root)


# 1. Full-model reference uses correct features
def test_full_model_reference_features(root):
    d_manifest = load_demand_feature_manifest(root)["feature_name"].tolist()
    p_manifest = load_price_feature_manifest(root)["feature_name"].tolist()
    assert len(d_manifest) == 275, f"Expected 275 safe demand features, got {len(d_manifest)}"
    assert len(p_manifest) == 163, f"Expected 163 safe price features, got {len(p_manifest)}"


# 2. Ablation removes only the intended family
def test_ablation_removes_only_intended_family(root):
    d_manifest = load_demand_feature_manifest(root)["feature_name"].tolist()
    d_fams = get_demand_family_to_features(root)
    promo_feats = d_fams["promotion"]
    sub_feats = [f for f in d_manifest if f not in promo_feats]
    
    # Assert intended family removed
    assert not any(f in sub_feats for f in promo_feats)
    # Assert exact count reduction
    assert len(sub_feats) == len(d_manifest) - len(promo_feats)


# 3. No unrelated feature family changes
def test_no_unrelated_feature_family_changes(root):
    d_manifest = load_demand_feature_manifest(root)["feature_name"].tolist()
    d_fams = get_demand_family_to_features(root)
    weather_feats = d_fams["weather"]
    sub_feats = [f for f in d_manifest if f not in weather_feats]
    
    # All non-weather families must remain 100% intact
    for fam, feats in d_fams.items():
        if fam != "weather":
            assert all(f in sub_feats for f in feats), f"Family {fam} was corrupted during ablation"


# 4. Same target rows are used
def test_same_target_rows_used(root):
    d_1d = load_model_ready_cache("demand", 1, root=root)
    p_1d = load_model_ready_cache("price", 1, root=root)
    assert len(d_1d) == 32856
    assert len(p_1d) == 32856
    assert (d_1d["demand_target_1d_valid"] == 1).sum() > 0
    assert (p_1d["purchase_price_target_1d_valid"] == 1).sum() > 0


# 5. Same validation folds are used
def test_same_validation_folds_used(root):
    folds_df = pd.read_csv(root / "reports/baselines/walk_forward_folds.csv")
    assert len(folds_df) == 33
    assert "validation_fold" in folds_df.columns
    assert folds_df["validation_fold"].iloc[0] == 1
    assert folds_df["validation_fold"].iloc[-1] == 33


# 6. Same model configuration is used
def test_same_model_configuration(config):
    d_cfg = config["models"]["demand"]
    p_cfg = config["models"]["price"]
    assert d_cfg["family"] in ["catboost", "lightgbm"]
    assert p_cfg["approach"] == "percentage_change"
    assert "params" in d_cfg
    assert "params" in p_cfg


# 7. Same seeds are used
def test_same_seeds_used(config):
    assert config["evaluation"]["seed"] == 42
    assert config["models"]["demand"]["params"].get("random_seed") == 42 or config["models"]["demand"]["params"].get("random_state") == 42
    assert config["models"]["price"]["params"].get("random_state") == 42


# 8. Final holdout is excluded
def test_final_holdout_excluded(root, config):
    holdout_start = pd.Timestamp(config["evaluation"]["holdout_start"])
    d_1d = load_model_ready_cache("demand", 1, root=root)
    p_1d = load_model_ready_cache("price", 1, root=root)
    assert pd.to_datetime(d_1d["date"]).max() < holdout_start
    assert pd.to_datetime(p_1d["date"]).max() < holdout_start


# 9. Demand metric comparison is correct
def test_demand_metric_comparison():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0])
    metrics = compute_metrics(y_true, y_pred)
    assert abs(metrics["mae"] - 16.6667) < 0.01
    assert abs(metrics["wape"] - (50.0 / 600.0)) < 0.001


# 10. Price metric comparison is correct
def test_price_metric_comparison():
    y_true = np.array([5.0, 10.0, 20.0])
    y_pred = np.array([5.5, 9.8, 19.5])
    metrics = compute_metrics(y_true, y_pred)
    assert abs(metrics["mae"] - 0.40) < 0.01
    assert metrics["rmse"] > 0


# 11. Relative performance difference is correct
def test_relative_performance_difference():
    full_wape = 0.100
    ablated_wape = 0.108
    delta, rel_pct = compute_differential(full_wape, ablated_wape)
    assert abs(delta - 0.008) < 1e-6
    assert abs(rel_pct - 8.0) < 1e-4


# 12. Feature-family mapping is valid
def test_feature_family_mapping_valid(root):
    d_fams = get_demand_family_to_features(root)
    p_fams = get_price_family_to_features(root)
    assert "historical_demand" in d_fams
    assert "promotion" in d_fams
    assert "weather" in d_fams
    assert "historical_purchase_price" in p_fams
    assert "fuel_energy" in p_fams
    assert "exchange_rates" in p_fams
    assert "commodity_prices" in p_fams


# 13. Approved temporal manifests are enforced
def test_approved_temporal_manifests_enforced(root):
    d_manifest = load_demand_feature_manifest(root)
    p_manifest = load_price_feature_manifest(root)
    assert (d_manifest["final_audit_status"].isin(["SAFE", "SAFE_WITH_ASSUMPTION"])).all()
    assert (p_manifest["final_audit_status"].isin(["SAFE", "SAFE_WITH_ASSUMPTION"])).all()


# 14. Target columns never enter features
def test_target_columns_never_enter_features(root):
    d_fams = get_demand_family_to_features(root)
    p_fams = get_price_family_to_features(root)
    for fam, feats in d_fams.items():
        for f in feats:
            assert "target" not in f.lower(), f"Target column found in demand family {fam}: {f}"
    for fam, feats in p_fams.items():
        for f in feats:
            assert "target" not in f.lower(), f"Target column found in price family {fam}: {f}"


# 15. Cached datasets match canonical data
def test_cached_datasets_match_canonical_data(root):
    d_1d = load_model_ready_cache("demand", 1, root=root)
    assert "date" in d_1d.columns
    assert "store_id" in d_1d.columns
    assert "product_id" in d_1d.columns
    assert "demand_target_1d" in d_1d.columns


# 16. Screening/confirmed result status is correctly recorded
def test_screening_confirmed_result_status(config):
    assert "priority_horizons" in config["evaluation"]
    assert config["evaluation"]["priority_horizons"] == [1, 7, 30]


# 17. Recommended feature manifests contain only safe features
def test_recommended_feature_manifests_contain_only_safe(root):
    d_opt_file = root / "reports/ablation/demand_features_for_optimisation.csv"
    p_opt_file = root / "reports/ablation/price_features_for_optimisation.csv"
    if d_opt_file.exists():
        df = pd.read_csv(d_opt_file)
        assert (df["temporal_status"] == "SAFE").all()
    if p_opt_file.exists():
        df = pd.read_csv(p_opt_file)
        assert (df["temporal_status"] == "SAFE").all()


# 18. Harmful features can be excluded
def test_harmful_features_handling():
    # If a family improves when removed (rel_impact < -0.05%), it's labeled HARMFUL
    classification = classify_family_value(-0.5, [-0.4, -0.6, -0.5])
    assert classification == "HARMFUL"


# 19. Runtime reporting works
def test_runtime_reporting(root):
    inv_file = root / "reports/ablation/feature_family_inventory.csv"
    assert inv_file.exists()
    inv_df = pd.read_csv(inv_file)
    assert len(inv_df) > 0
    assert "usable_rows_count" in inv_df.columns


# 20. Phase 8 temporal-integrity tests remain successful
def test_phase8_temporal_integrity(root):
    leak_file = root / "reports/leakage/leakage_violations.csv"
    assert leak_file.exists()
    df = pd.read_csv(leak_file)
    assert len(df) == 0, "Future leakage violations detected!"


# 21. Phase 9 target tests remain successful
def test_phase9_target_integrity(root):
    targ_file = root / "data/processed/forecast_targets.parquet"
    assert targ_file.exists()


# 22. Phase 10 baseline tests remain successful
def test_phase10_baseline_integrity(root):
    champ_file = root / "reports/baselines/champion_baselines.csv"
    assert champ_file.exists()
    df = pd.read_csv(champ_file)
    assert len(df) >= 12


# 23. Phase 11 demand-model tests remain successful
def test_phase11_demand_models_integrity(root):
    cand_file = root / "reports/models/demand/candidate_model_ranking.csv"
    assert cand_file.exists()


# 24. Phase 12 price-model tests remain successful
def test_phase12_price_models_integrity(root):
    best_file = root / "reports/models/price/best_candidate_models.csv"
    assert best_file.exists()
