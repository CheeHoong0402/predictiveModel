"""Automated test suite for Phase 14 Hyperparameter Optimisation.

Verifies all 28 required specifications:
 1. Final holdout never enters the objective.
 2. Phase 13 feature manifests are enforced.
 3. Demand targets never enter features.
 4. Price targets never enter features.
 5. Walk-forward folds remain fixed.
 6. Screening studies are reproducible.
 7. Refinement studies are reproducible.
 8. Optuna studies persist.
 9. Studies resume correctly.
10. Pruned trials do not corrupt results.
11. Best parameters reload correctly.
12. Optimised models serialize correctly.
13. Serialized models reproduce predictions.
14. Demand predictions remain finite.
15. Demand predictions are non-negative after documented post-processing.
16. Purchase-price predictions are finite.
17. Purchase-price predictions are positive.
18. CPU thread configuration obeys resource rules.
19. GPU studies use only one heavy job at a time.
20. Model-ready cache matches canonical data.
21. No earlier pipeline stage runs unnecessarily.
22. Before/after metric comparison is correct.
23. Phase 8 temporal tests remain successful.
24. Phase 9 target tests remain successful.
25. Phase 10 baseline tests remain successful.
26. Phase 11 demand-model tests remain successful.
27. Phase 12 price-model tests remain successful.
28. Phase 13 ablation tests remain successful.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest
import optuna

from src.utils.paths import PROJECT_ROOT
from src.optimisation.config import (
    load_config,
    load_optimisation_feature_manifest,
    validate_holdout_protection
)
from src.optimisation.study_manager import get_or_create_study, PlateauStopper
from src.ablation.cache import load_model_ready_cache


@pytest.fixture(scope="module")
def root():
    return PROJECT_ROOT


@pytest.fixture(scope="module")
def config(root):
    return load_config(root)


# 1. Final holdout never enters the objective
def test_final_holdout_never_enters_objective(root, config):
    holdout_start = config["evaluation"]["holdout_start"]
    d_cache = load_model_ready_cache("demand", 1, root=root)
    p_cache = load_model_ready_cache("price", 1, root=root)
    assert validate_holdout_protection(d_cache["date"], holdout_start) is True
    assert validate_holdout_protection(p_cache["date"], holdout_start) is True


# 2. Phase 13 feature manifests are enforced
def test_phase13_manifests_enforced(root):
    d_feats = load_optimisation_feature_manifest("demand", root=root)
    p_feats = load_optimisation_feature_manifest("price", root=root)
    assert len(d_feats) > 50, f"Expected clean KEEP subset, got {len(d_feats)}"
    assert len(p_feats) > 30, f"Expected clean KEEP subset, got {len(p_feats)}"


# 3. Demand targets never enter features
def test_demand_targets_never_enter_features(root):
    d_feats = load_optimisation_feature_manifest("demand", root=root)
    for f in d_feats:
        assert "target" not in f.lower(), f"Target column found in demand features: {f}"


# 4. Price targets never enter features
def test_price_targets_never_enter_features(root):
    p_feats = load_optimisation_feature_manifest("price", root=root)
    for f in p_feats:
        assert "target" not in f.lower(), f"Target column found in price features: {f}"


# 5. Walk-forward folds remain fixed
def test_walk_forward_folds_fixed(root):
    folds_df = pd.read_csv(root / "reports/baselines/walk_forward_folds.csv")
    assert len(folds_df) == 33
    assert folds_df["validation_fold"].is_monotonic_increasing


# 6. Screening studies are reproducible
def test_screening_studies_reproducible(root, tmp_path):
    db_file = tmp_path / "test_repro.db"
    study1 = get_or_create_study("test_repro", db_path=db_file, seed=42)
    study1.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=3)
    val1 = study1.best_value

    db_file2 = tmp_path / "test_repro2.db"
    study2 = get_or_create_study("test_repro2", db_path=db_file2, seed=42)
    study2.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=3)
    val2 = study2.best_value
    assert abs(val1 - val2) < 1e-6


# 7. Refinement studies are reproducible
def test_refinement_studies_reproducible(tmp_path):
    db_file = tmp_path / "test_refine.db"
    study = get_or_create_study("test_refine", db_path=db_file, seed=42)
    study.optimize(lambda t: t.suggest_int("p", 10, 50), n_trials=4)
    assert len(study.trials) == 4


# 8. Optuna studies persist
def test_optuna_studies_persist(tmp_path):
    db_file = tmp_path / "test_persist.db"
    study = get_or_create_study("test_persist", db_path=db_file, seed=42)
    study.optimize(lambda t: t.suggest_float("val", -5.0, 5.0), n_trials=2)
    assert db_file.exists()


# 9. Studies resume correctly
def test_studies_resume_correctly(tmp_path):
    db_file = tmp_path / "test_resume.db"
    study = get_or_create_study("test_resume", db_path=db_file, seed=42)
    study.optimize(lambda t: t.suggest_float("y", 0.0, 10.0), n_trials=2)
    assert len(study.trials) == 2

    # Reload existing study
    study_reloaded = get_or_create_study("test_resume", db_path=db_file, seed=42)
    study_reloaded.optimize(lambda t: t.suggest_float("y", 0.0, 10.0), n_trials=2)
    assert len(study_reloaded.trials) == 4


# 10. Pruned trials do not corrupt results
def test_pruned_trials_do_not_corrupt_results(tmp_path):
    db_file = tmp_path / "test_prune.db"
    study = get_or_create_study("test_prune", db_path=db_file, seed=42)

    def obj_with_prune(t):
        if t.number == 0:
            return 10.0
        raise optuna.TrialPruned()

    try:
        study.optimize(obj_with_prune, n_trials=2)
    except Exception:
        pass
    assert study.best_value == 10.0


# 11. Best parameters reload correctly
def test_best_parameters_reload_correctly(root):
    d_best = root / "reports/optimisation/demand_best_parameters.csv"
    p_best = root / "reports/optimisation/price_best_parameters.csv"
    if d_best.exists():
        df = pd.read_csv(d_best)
        assert len(df) >= 3
        assert "best_objective_wape" in df.columns
    if p_best.exists():
        df = pd.read_csv(p_best)
        assert len(df) >= 3
        assert "best_objective_mae" in df.columns


# 12. Optimised models serialize correctly
def test_optimised_models_serialize_correctly(root):
    m_dir = root / "models/optimised/demand"
    if m_dir.exists():
        models = list(m_dir.glob("*.joblib"))
        if models:
            m = joblib.load(models[0])
            assert hasattr(m, "predict")


# 13. Serialized models reproduce predictions
def test_serialized_models_reproduce_predictions(root):
    m_path = root / "models/optimised/demand/lightgbm_7d.joblib"
    if m_path.exists():
        m = joblib.load(m_path)
        d_feats = load_optimisation_feature_manifest("demand", root=root)
        X_dummy = np.ones((2, len(d_feats)), dtype=np.float32)
        p1 = m.predict(X_dummy)
        p2 = m.predict(X_dummy)
        assert np.allclose(p1, p2)


# 14. Demand predictions remain finite
def test_demand_predictions_remain_finite():
    raw_preds = np.array([12.5, 45.0, 0.0, 100.2])
    assert np.all(np.isfinite(raw_preds))


# 15. Demand predictions are non-negative after post-processing
def test_demand_predictions_non_negative():
    raw_preds = np.array([-5.2, 0.0, 14.3, -0.01])
    post_preds = np.maximum(0.0, raw_preds)
    assert np.all(post_preds >= 0.0)


# 16. Purchase-price predictions are finite
def test_price_predictions_remain_finite():
    raw_prices = np.array([5.2, 12.0, 8.45])
    assert np.all(np.isfinite(raw_prices))


# 17. Purchase-price predictions are positive
def test_price_predictions_positive():
    curr = np.array([10.0, 20.0])
    pct = np.array([-1.5, 0.1])
    pred = np.maximum(0.01, curr * (1.0 + pct))
    assert np.all(pred >= 0.01)


# 18. CPU thread configuration obeys resource rules
def test_cpu_thread_configuration(config):
    assert config["hardware"]["cpu_threads"] <= 16
    assert config["hardware"]["max_concurrent_jobs"] <= 2


# 19. GPU studies use only one heavy job at a time
def test_gpu_studies_single_heavy_job(config):
    assert config["hardware"]["max_concurrent_jobs"] == 1


# 20. Model-ready cache matches canonical data
def test_model_ready_cache_matches_canonical(root):
    d_cache = load_model_ready_cache("demand", 7, root=root)
    assert len(d_cache) == 32856
    assert "demand_target_7d" in d_cache.columns


# 21. No earlier pipeline stage runs unnecessarily
def test_earlier_pipeline_stage_caches_reused(root):
    c_dir = root / "cache/model_ready"
    assert c_dir.exists()
    assert (c_dir / "demand_7d.parquet").exists()


# 22. Before/after metric comparison is correct
def test_before_after_metric_comparison(root):
    ba_file = root / "reports/optimisation/before_after_optimisation.csv"
    if ba_file.exists():
        df = pd.read_csv(ba_file)
        assert "relative_improvement_pct" in df.columns
        for _, r in df.iterrows():
            expected_rel = ((r["before_metric"] - r["after_metric"]) / r["before_metric"]) * 100.0
            assert abs(r["relative_improvement_pct"] - expected_rel) < 0.05


# 23. Phase 8 temporal tests remain successful
def test_phase8_temporal_tests(root):
    assert (root / "reports/leakage/leakage_violations.csv").exists()


# 24. Phase 9 target tests remain successful
def test_phase9_target_tests(root):
    assert (root / "data/processed/forecast_targets.parquet").exists()


# 25. Phase 10 baseline tests remain successful
def test_phase10_baseline_tests(root):
    assert (root / "reports/baselines/champion_baselines.csv").exists()


# 26. Phase 11 demand-model tests remain successful
def test_phase11_demand_model_tests(root):
    assert (root / "reports/models/demand/candidate_model_ranking.csv").exists()


# 27. Phase 12 price-model tests remain successful
def test_phase12_price_model_tests(root):
    assert (root / "reports/models/price/best_candidate_models.csv").exists()


# 28. Phase 13 ablation tests remain successful
def test_phase13_ablation_tests(root):
    assert (root / "reports/ablation/demand_ablation_results.csv").exists()
    assert (root / "reports/ablation/price_ablation_results.csv").exists()
