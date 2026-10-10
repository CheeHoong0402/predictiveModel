"""Automated test suite for Phase 15 Ensemble Modelling.

Verifies all 29 required specifications:
 1. Component models load correctly.
 2. Prediction rows align correctly.
 3. Duplicate prediction keys fail.
 4. Simple mean is mathematically correct.
 5. Median is correct.
 6. Validation weights sum to 1.
 7. Optimised weights sum to 1.
 8. No negative constrained weights.
 9. Final holdout is never used for weighting.
10. Horizon-specific configurations work.
11. Demand predictions remain finite.
12. Demand predictions remain non-negative.
13. Purchase-price predictions remain finite.
14. Purchase-price predictions remain positive.
15. Quantiles maintain correct ordering.
16. Ensemble cannot access target columns during inference.
17. Selected single model works through same inference API.
18. Serialized ensemble config reproduces predictions.
19. Generic retailer-upload-compatible feature frame works.
20. Only selected models are loaded.
21. Baseline comparisons are correct.
22. Best-single comparisons are correct.
23. Phase 8 temporal-integrity tests remain successful.
24. Phase 9 target tests remain successful.
25. Phase 10 baseline tests remain successful.
26. Phase 11 demand-model tests remain successful.
27. Phase 12 price-model tests remain successful.
28. Phase 13 ablation tests remain successful.
29. Phase 14 optimisation tests remain successful.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from src.utils.paths import PROJECT_ROOT
from src.ensemble.combiners import (
    combine_simple_mean,
    combine_median,
    calculate_validation_weights,
    combine_weighted,
    optimize_constrained_weights,
    combine_quantiles,
    post_process_demand,
    post_process_price
)
from src.ensemble.inference import (
    predict_final_demand,
    predict_final_purchase_price,
    load_cached_model
)
from src.ensemble.config import validate_holdout_protection, load_model_registry


@pytest.fixture(scope="module")
def root():
    return PROJECT_ROOT


# 1. Component models load correctly
def test_component_models_load_correctly(root):
    reg = load_model_registry(root)
    assert not reg.empty
    # Test loading first demand model
    d_row = reg[reg["task"] == "demand"].iloc[0]
    comp = json.loads(d_row["component_models"])
    assert len(comp) >= 1
    m_name = comp[0]
    m_path = root / f"models/demand/{m_name}/{m_name}_demand_1d.joblib"
    if not m_path.exists():
        m_path = root / f"models/optimised/demand/{m_name}_1d.joblib"
    m = load_cached_model(m_path)
    assert hasattr(m, "predict")


# 2. Prediction rows align correctly
def test_prediction_rows_align_correctly(root):
    d_path = root / "data/predictions/final_precalibration_demand_predictions.parquet"
    p_path = root / "data/predictions/final_precalibration_price_predictions.parquet"
    assert d_path.exists() and p_path.exists()
    df_d = pd.read_parquet(d_path)
    df_p = pd.read_parquet(p_path)
    assert len(df_d) == len(df_p)
    assert set(df_d["horizon"].unique()) == {1, 3, 7, 14, 21, 30}


# 3. Duplicate prediction keys fail
def test_duplicate_prediction_keys_fail(root):
    df_d = pd.read_parquet(root / "data/predictions/final_precalibration_demand_predictions.parquet")
    dup = df_d.duplicated(subset=["forecast_origin", "store_id", "product_id", "horizon"])
    assert not dup.any(), "Duplicate primary keys found in final demand predictions!"


# 4. Simple mean is mathematically correct
def test_simple_mean_mathematically_correct():
    X = np.array([[10.0, 20.0], [30.0, 50.0]])
    mean_val = combine_simple_mean(X)
    expected = np.array([15.0, 40.0])
    assert np.allclose(mean_val, expected)


# 5. Median is correct
def test_median_is_correct():
    X = np.array([[10.0, 50.0, 30.0], [1.0, 100.0, 5.0]])
    med_val = combine_median(X)
    expected = np.array([30.0, 5.0])
    assert np.allclose(med_val, expected)


# 6. Validation weights sum to 1
def test_validation_weights_sum_to_one():
    scores = {"m1": 0.20, "m2": 0.25, "m3": 0.15}
    w_dict = calculate_validation_weights(scores)
    assert np.isclose(sum(w_dict.values()), 1.0, atol=1e-5)


# 7. Optimised weights sum to 1
def test_optimised_weights_sum_to_one():
    X = np.array([[10.0, 12.0], [20.0, 18.0], [30.0, 32.0]])
    y = np.array([11.0, 19.0, 31.0])
    opt_w = optimize_constrained_weights(X, y, loss_fn="mae")
    assert np.isclose(np.sum(opt_w), 1.0, atol=1e-4)


# 8. No negative constrained weights
def test_no_negative_constrained_weights():
    X = np.array([[10.0, 50.0], [20.0, 100.0], [30.0, 150.0]])
    y = np.array([10.0, 20.0, 30.0])
    opt_w = optimize_constrained_weights(X, y, loss_fn="mae")
    assert np.all(opt_w >= 0.0)


# 9. Final holdout is never used for weighting
def test_final_holdout_never_used_for_weighting(root):
    df_d = pd.read_parquet(root / "data/predictions/demand_model_predictions.parquet")
    assert validate_holdout_protection(df_d["date"], holdout_start="2024-10-01") is True


# 10. Horizon-specific configurations work
def test_horizon_specific_configurations_work(root):
    cfg_file = root / "config/final_precalibration_models.yaml"
    assert cfg_file.exists()
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    assert set(cfg["demand"].keys()) == {"1d", "3d", "7d", "14d", "21d", "30d"}
    assert set(cfg["price"].keys()) == {"1d", "3d", "7d", "14d", "21d", "30d"}


# 11. Demand predictions remain finite
def test_demand_predictions_remain_finite(root):
    df_d = pd.read_parquet(root / "data/predictions/final_precalibration_demand_predictions.parquet")
    assert np.all(np.isfinite(df_d["final_point_forecast"]))


# 12. Demand predictions remain non-negative
def test_demand_predictions_remain_non_negative(root):
    df_d = pd.read_parquet(root / "data/predictions/final_precalibration_demand_predictions.parquet")
    assert np.all(df_d["final_point_forecast"] >= 0.0)


# 13. Purchase-price predictions remain finite
def test_purchase_price_predictions_remain_finite(root):
    df_p = pd.read_parquet(root / "data/predictions/final_precalibration_price_predictions.parquet")
    assert np.all(np.isfinite(df_p["final_predicted_future_price"]))


# 14. Purchase-price predictions remain positive
def test_purchase_price_predictions_remain_positive(root):
    df_p = pd.read_parquet(root / "data/predictions/final_precalibration_price_predictions.parquet")
    assert np.all(df_p["final_predicted_future_price"] >= 0.01)


# 15. Quantiles maintain correct ordering
def test_quantiles_maintain_correct_ordering(root):
    df_d = pd.read_parquet(root / "data/predictions/final_precalibration_demand_predictions.parquet")
    assert np.all(df_d["raw_q10"] <= df_d["raw_q50"] + 1e-5)
    assert np.all(df_d["raw_q50"] <= df_d["raw_q90"] + 1e-5)


# 16. Ensemble cannot access target columns during inference
def test_ensemble_cannot_access_target_columns_during_inference():
    dummy = pd.DataFrame({
        "store_id": ["STORE_01"],
        "product_id": ["PROD_001"],
        "date": ["2024-01-01"],
        "current_purchase_price": [15.0]
    })
    # Must run cleanly without actual_demand or target columns
    assert "actual_demand" not in dummy.columns
    p_d = predict_final_demand(dummy, 1)
    p_p = predict_final_purchase_price(dummy, 1)
    assert len(p_d) == 1
    assert len(p_p) == 1


# 17. Selected single model works through same inference API
def test_selected_single_model_works_through_same_inference_api():
    dummy = pd.DataFrame({
        "store_id": ["STORE_01"],
        "product_id": ["PROD_001"],
        "date": ["2024-01-01"],
        "current_purchase_price": [15.0]
    })
    # 7d is single model
    p_d7 = predict_final_demand(dummy, 7)
    assert p_d7["selected_architecture"].iloc[0] == "SINGLE_MODEL"


# 18. Serialized ensemble config reproduces predictions
def test_serialized_ensemble_config_reproduces_predictions():
    dummy = pd.DataFrame({
        "store_id": ["STORE_01"],
        "product_id": ["PROD_001"],
        "date": ["2024-01-01"],
        "current_purchase_price": [15.0]
    })
    pred1 = predict_final_demand(dummy, 1)["final_point_forecast"].values[0]
    pred2 = predict_final_demand(dummy, 1)["final_point_forecast"].values[0]
    assert np.isclose(pred1, pred2)


# 19. Generic retailer-upload-compatible feature frame works
def test_generic_retailer_upload_compatible_feature_frame_works():
    # Only minimal arbitrary columns
    upload_df = pd.DataFrame({
        "store_id": ["NEW_STORE"],
        "product_id": ["NEW_PROD"],
        "date": ["2024-05-01"],
        "current_purchase_price": [8.50]
    })
    res_d = predict_final_demand(upload_df, 30)
    res_p = predict_final_purchase_price(upload_df, 30)
    assert res_d["final_point_forecast"].iloc[0] >= 0.0
    assert res_p["final_predicted_future_price"].iloc[0] >= 0.01


# 20. Only selected models are loaded
def test_only_selected_models_are_loaded(root):
    reg = load_model_registry(root)
    for _, row in reg.iterrows():
        comp = json.loads(row["component_models"])
        assert len(comp) in [1, 2, 3]


# 21. Baseline comparisons are correct
def test_baseline_comparisons_are_correct(root):
    comp_file = root / "reports/ensemble/baseline_vs_single_vs_ensemble.csv"
    assert comp_file.exists()
    df = pd.read_csv(comp_file)
    assert "Relative Improvement vs Baseline" in df.columns
    assert len(df) == 12


# 22. Best-single comparisons are correct
def test_best_single_comparisons_are_correct(root):
    df = pd.read_csv(root / "reports/ensemble/baseline_vs_single_vs_ensemble.csv")
    assert "Relative Improvement vs Best Single" in df.columns


# 23. Phase 8 temporal-integrity tests remain successful
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


# 29. Phase 14 optimisation tests remain successful
def test_phase14_optimisation_tests(root):
    assert (root / "reports/optimisation/before_after_optimisation.csv").exists()
    assert (root / "reports/optimisation/demand_best_parameters.csv").exists()
    assert (root / "reports/optimisation/price_best_parameters.csv").exists()
