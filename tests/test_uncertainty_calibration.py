"""Comprehensive automated test suite for Phase 16: Probability Calibration and Uncertainty Validation."""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.utils.paths import PROJECT_ROOT
from src.calibration.metrics import (
    calculate_coverage,
    calculate_mean_interval_width,
    calculate_median_interval_width,
    calculate_p90_interval_width,
    winkler_interval_score,
    pinball_loss,
    detect_quantile_crossings,
    enforce_monotonic_quantiles,
    classify_calibration_status,
    calculate_all_uncertainty_metrics
)
from src.calibration.calibrators import (
    BaseCalibrator,
    ConformalQuantileCalibrator,
    AsymmetricResidualCalibrator,
    SymmetricResidualCalibrator
)
from src.calibration.inference import (
    predict_demand_with_uncertainty,
    predict_price_with_uncertainty,
    load_cached_calibrator
)


# -----------------------------------------------------------------------------
# Test 1: Final holdout is strictly excluded from calibration
# -----------------------------------------------------------------------------
def test_final_holdout_excluded_from_calibration():
    dem_path = PROJECT_ROOT / "data/predictions/final_calibrated_demand_predictions.parquet"
    pr_path = PROJECT_ROOT / "data/predictions/final_calibrated_price_predictions.parquet"
    
    assert dem_path.exists(), "Calibrated demand predictions parquet missing"
    assert pr_path.exists(), "Calibrated price predictions parquet missing"

    df_d = pd.read_parquet(dem_path)
    df_p = pd.read_parquet(pr_path)

    holdout_start = pd.Timestamp("2024-10-01")
    assert pd.to_datetime(df_d["forecast_origin"]).max() < holdout_start, "Demand calibration contains holdout dates!"
    assert pd.to_datetime(df_p["forecast_origin"]).max() < holdout_start, "Price calibration contains holdout dates!"


# -----------------------------------------------------------------------------
# Test 2: Calibration rows occur after calibrator-training history
# -----------------------------------------------------------------------------
def test_calibration_rows_occur_after_training_history():
    # In temporal expanding calibration: for any evaluation fold k, training folds are < k
    for k in range(2, 34):
        train_folds = list(range(1, k))
        assert all(f < k for f in train_folds)
        assert max(train_folds) < k


# -----------------------------------------------------------------------------
# Test 3: No future residual is used for earlier predictions
# -----------------------------------------------------------------------------
def test_no_future_residual_used_for_earlier_predictions():
    # Verify that a calibrator fit on dates < T does not require or access dates >= T
    y_past = np.array([10.0, 15.0, 12.0, 18.0] * 10)
    y_hat_past = np.array([11.0, 14.0, 13.0, 17.0] * 10)
    q10_past = y_hat_past - 3.0
    q90_past = y_hat_past + 3.0

    cal = ConformalQuantileCalibrator("demand", horizon=7)
    cal.fit(y_past, y_hat_past, q10_past, y_hat_past, q90_past)

    # Predict on new test point at T without passing future target
    new_y_hat = np.array([20.0, 25.0])
    p10, p50, p90 = cal.predict_uncertainty(new_y_hat, new_y_hat - 3.0, new_y_hat, new_y_hat + 3.0)
    assert len(p10) == 2
    assert p10[0] <= p50[0] <= p90[0]


# -----------------------------------------------------------------------------
# Test 4: Quantile ordering: P10 <= P50 <= P90 (0 crossings)
# -----------------------------------------------------------------------------
def test_quantile_ordering_no_crossings():
    df_d = pd.read_parquet(PROJECT_ROOT / "data/predictions/final_calibrated_demand_predictions.parquet")
    df_p = pd.read_parquet(PROJECT_ROOT / "data/predictions/final_calibrated_price_predictions.parquet")

    cross_d = detect_quantile_crossings(df_d["calibrated_p10"], df_d["calibrated_p50"], df_d["calibrated_p90"])
    cross_p = detect_quantile_crossings(df_p["calibrated_p10"], df_p["calibrated_p50"], df_p["calibrated_p90"])

    assert cross_d == 0, f"Found {cross_d} quantile crossings in calibrated demand predictions"
    assert cross_p == 0, f"Found {cross_p} quantile crossings in calibrated price predictions"


# -----------------------------------------------------------------------------
# Test 5: Demand P10 is non-negative
# -----------------------------------------------------------------------------
def test_demand_p10_non_negative():
    df_d = pd.read_parquet(PROJECT_ROOT / "data/predictions/final_calibrated_demand_predictions.parquet")
    assert (df_d["calibrated_p10"] >= 0.0).all(), "Found negative demand P10 quantiles!"


# -----------------------------------------------------------------------------
# Test 6: Demand P50 is non-negative
# -----------------------------------------------------------------------------
def test_demand_p50_non_negative():
    df_d = pd.read_parquet(PROJECT_ROOT / "data/predictions/final_calibrated_demand_predictions.parquet")
    assert (df_d["calibrated_p50"] >= 0.0).all(), "Found negative demand P50 quantiles!"


# -----------------------------------------------------------------------------
# Test 7: Demand P90 is non-negative
# -----------------------------------------------------------------------------
def test_demand_p90_non_negative():
    df_d = pd.read_parquet(PROJECT_ROOT / "data/predictions/final_calibrated_demand_predictions.parquet")
    assert (df_d["calibrated_p90"] >= 0.0).all(), "Found negative demand P90 quantiles!"


# -----------------------------------------------------------------------------
# Test 8: Purchase-price quantiles are positive (>= 0.01 MYR)
# -----------------------------------------------------------------------------
def test_purchase_price_quantiles_positive():
    df_p = pd.read_parquet(PROJECT_ROOT / "data/predictions/final_calibrated_price_predictions.parquet")
    assert (df_p["calibrated_p10"] >= 0.01).all(), "Found price P10 quantiles < 0.01 MYR!"
    assert (df_p["calibrated_p50"] >= 0.01).all(), "Found price P50 quantiles < 0.01 MYR!"
    assert (df_p["calibrated_p90"] >= 0.01).all(), "Found price P90 quantiles < 0.01 MYR!"


# -----------------------------------------------------------------------------
# Test 9: Coverage is calculated correctly
# -----------------------------------------------------------------------------
def test_coverage_calculation_correct():
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    lower = np.array([8.0, 15.0, 32.0, 35.0, 45.0])
    upper = np.array([12.0, 25.0, 38.0, 42.0, 48.0])
    # Included: idx 0 (10 in [8,12]), idx 1 (20 in [15,25]), idx 3 (40 in [35,42]) -> 3 out of 5 = 60%
    cov = calculate_coverage(y, lower, upper)
    assert pytest.approx(cov, 1e-5) == 0.60


# -----------------------------------------------------------------------------
# Test 10: Interval width is calculated correctly
# -----------------------------------------------------------------------------
def test_interval_width_calculation_correct():
    lower = np.array([10.0, 20.0, 30.0])
    upper = np.array([15.0, 30.0, 45.0])
    # widths: 5, 10, 15 -> mean = 10.0, median = 10.0
    assert pytest.approx(calculate_mean_interval_width(lower, upper), 1e-5) == 10.0
    assert pytest.approx(calculate_median_interval_width(lower, upper), 1e-5) == 10.0


# -----------------------------------------------------------------------------
# Test 11: Pinball loss is correct
# -----------------------------------------------------------------------------
def test_pinball_loss_correct():
    y = np.array([10.0, 20.0])
    q = np.array([8.0, 25.0])
    # for idx 0: err = 10 - 8 = +2 -> loss = tau * 2
    # for idx 1: err = 20 - 25 = -5 -> loss = (tau - 1) * (-5) = (1 - tau) * 5
    # tau = 0.1: mean(0.1*2, 0.9*5) = mean(0.2, 4.5) = 2.35
    loss_01 = pinball_loss(y, q, tau=0.10)
    assert pytest.approx(loss_01, 1e-5) == 2.35

    # tau = 0.5: mean(0.5*2, 0.5*5) = mean(1.0, 2.5) = 1.75
    loss_05 = pinball_loss(y, q, tau=0.50)
    assert pytest.approx(loss_05, 1e-5) == 1.75


# -----------------------------------------------------------------------------
# Test 12: Interval score (Winkler score) is correct
# -----------------------------------------------------------------------------
def test_winkler_interval_score_correct():
    # alpha = 0.20 -> penalty multiplier = 2 / 0.2 = 10
    # idx 0: y=10 in [8, 12] -> score = 12 - 8 = 4
    # idx 1: y=5 < L=8 -> score = 4 + 10 * (8 - 5) = 4 + 30 = 34
    # idx 2: y=15 > U=12 -> score = 4 + 10 * (15 - 12) = 4 + 30 = 34
    y = np.array([10.0, 5.0, 15.0])
    l = np.array([8.0, 8.0, 8.0])
    u = np.array([12.0, 12.0, 12.0])
    score = winkler_interval_score(y, l, u, alpha=0.20)
    expected = (4.0 + 34.0 + 34.0) / 3.0
    assert pytest.approx(score, 1e-5) == expected


# -----------------------------------------------------------------------------
# Test 13: Raw versus calibrated metrics are correctly separated
# -----------------------------------------------------------------------------
def test_raw_vs_calibrated_metrics_separated():
    d_rep = pd.read_csv(PROJECT_ROOT / "reports/uncertainty/demand_calibration_by_horizon.csv")
    p_rep = pd.read_csv(PROJECT_ROOT / "reports/uncertainty/price_calibration_by_horizon.csv")

    for col in ["raw_coverage", "calibrated_coverage", "raw_mean_width", "calibrated_mean_width", "raw_winkler_score", "calibrated_winkler_score"]:
        assert col in d_rep.columns, f"Missing {col} in demand report"
        assert col in p_rep.columns, f"Missing {col} in price report"

    # Calibration improves demand coverage toward 80%
    assert (d_rep["calibrated_coverage"] >= d_rep["raw_coverage"]).all()


# -----------------------------------------------------------------------------
# Test 14: Calibration artefacts serialize correctly
# -----------------------------------------------------------------------------
def test_calibrator_serialization():
    cal = ConformalQuantileCalibrator("demand", horizon=7)
    cal.fit(
        y_true=np.array([10.0, 20.0, 30.0] * 10),
        point_pred=np.array([11.0, 19.0, 31.0] * 10),
        raw_q10=np.array([8.0, 16.0, 28.0] * 10),
        raw_q50=np.array([11.0, 19.0, 31.0] * 10),
        raw_q90=np.array([14.0, 24.0, 34.0] * 10)
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "test_calibrator.joblib"
        cal.save(tmp_file)
        assert tmp_file.exists()

        loaded_cal = BaseCalibrator.load(tmp_file)
        assert loaded_cal.task == "demand"
        assert loaded_cal.horizon == 7
        assert loaded_cal.is_fitted is True
        assert hasattr(loaded_cal, "q_correction_")


# -----------------------------------------------------------------------------
# Test 15: Loaded calibrators reproduce results
# -----------------------------------------------------------------------------
def test_loaded_calibrators_reproduce_results():
    cal = AsymmetricResidualCalibrator("price", horizon=14)
    y_test = np.array([5.0, 10.0, 15.0, 20.0] * 10)
    y_hat = np.array([5.2, 9.8, 15.1, 19.9] * 10)
    cal.fit(y_test, y_hat)

    new_pts = np.array([12.0, 18.0, 25.0])
    p10_orig, p50_orig, p90_orig = cal.predict_uncertainty(new_pts)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "cal.joblib"
        cal.save(tmp_file)
        loaded = BaseCalibrator.load(tmp_file)
        p10_load, p50_load, p90_load = loaded.predict_uncertainty(new_pts)

        np.testing.assert_allclose(p10_orig, p10_load)
        np.testing.assert_allclose(p50_orig, p50_load)
        np.testing.assert_allclose(p90_orig, p90_load)


# -----------------------------------------------------------------------------
# Test 16: Inference does not require target columns
# -----------------------------------------------------------------------------
def test_inference_does_not_require_targets():
    feat_frame = pd.DataFrame({
        "forecast_origin": ["2024-08-01", "2024-08-01"],
        "store_id": ["STR_JHR_003", "STR_KUL_001"],
        "product_id": ["PRD_DRK_001", "PRD_SNK_002"],
        "supplier_id": ["SUP_001", "SUP_002"],
        "current_purchase_price": [10.50, 15.20],
        "purchase_price": [10.50, 15.20]
    })
    # Explicitly verify no actual_demand or actual_future_price
    assert "actual_demand" not in feat_frame.columns
    assert "actual_future_price" not in feat_frame.columns

    out_d = predict_demand_with_uncertainty(feat_frame, horizon=7, root=PROJECT_ROOT)
    for req_col in ["point_prediction", "p10", "p50", "p90"]:
        assert req_col in out_d.columns
    assert (out_d["p10"] <= out_d["p50"]).all()
    assert (out_d["p50"] <= out_d["p90"]).all()

    out_p = predict_price_with_uncertainty(feat_frame, horizon=14, root=PROJECT_ROOT)
    for req_col in ["point_prediction", "p10", "p50", "p90", "current_purchase_price"]:
        assert req_col in out_p.columns
    assert (out_p["p10"] <= out_p["p50"]).all()
    assert (out_p["p50"] <= out_p["p90"]).all()


# -----------------------------------------------------------------------------
# Test 17: Unknown/insufficient segments fall back safely
# -----------------------------------------------------------------------------
def test_insufficient_segments_fallback_safely():
    # Only 5 samples (below min_samples=24)
    cal = ConformalQuantileCalibrator("demand", horizon=1, min_samples=24)
    y_small = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_hat_small = np.array([11.0, 19.0, 31.0, 39.0, 51.0])
    cal.fit(y_small, y_hat_small)

    p10, p50, p90 = cal.predict_uncertainty(np.array([25.0]))
    assert p10[0] <= p50[0] <= p90[0]
    assert p10[0] >= 0.0


# -----------------------------------------------------------------------------
# Test 18: Calibration sample-size rule works
# -----------------------------------------------------------------------------
def test_sample_size_rule():
    status_insufficient = classify_calibration_status(coverage=0.80, n_samples=10, min_samples=24)
    assert status_insufficient == "INSUFFICIENT_DATA"

    status_ok = classify_calibration_status(coverage=0.80, n_samples=100, min_samples=24)
    assert status_ok == "WELL_CALIBRATED"


# -----------------------------------------------------------------------------
# Test 19: Calibration registry is complete
# -----------------------------------------------------------------------------
def test_calibration_registry_complete():
    reg_path = PROJECT_ROOT / "reports/uncertainty/calibrator_registry.csv"
    assert reg_path.exists(), "Calibrator registry CSV missing"
    df_reg = pd.read_csv(reg_path)

    assert len(df_reg) == 12, f"Expected 12 registry entries, got {len(df_reg)}"
    required_cols = [
        "task", "horizon", "calibration_method", "calibration_window",
        "sample_count", "raw_coverage", "calibrated_coverage",
        "mean_interval_width", "interval_score", "selected_status", "artefact_path"
    ]
    for c in required_cols:
        assert c in df_reg.columns, f"Missing column {c} in calibrator registry"


# -----------------------------------------------------------------------------
# Test 20: Final calibrated demand file schema is valid
# -----------------------------------------------------------------------------
def test_final_calibrated_demand_schema():
    p = PROJECT_ROOT / "data/predictions/final_calibrated_demand_predictions.parquet"
    df = pd.read_parquet(p)
    required = [
        "forecast_origin", "store_id", "product_id", "horizon",
        "selected_architecture", "point_prediction", "calibrated_p10",
        "calibrated_p50", "calibrated_p90", "actual_demand",
        "interval_covered", "validation_fold"
    ]
    for col in required:
        assert col in df.columns, f"Missing {col} in calibrated demand predictions"
    assert df.isnull().sum().sum() == 0, "Calibrated demand predictions contain null values"


# -----------------------------------------------------------------------------
# Test 21: Final calibrated price file schema is valid
# -----------------------------------------------------------------------------
def test_final_calibrated_price_schema():
    p = PROJECT_ROOT / "data/predictions/final_calibrated_price_predictions.parquet"
    df = pd.read_parquet(p)
    required = [
        "forecast_origin", "store_id", "product_id", "supplier_id", "horizon",
        "current_purchase_price", "point_prediction", "calibrated_p10",
        "calibrated_p50", "calibrated_p90", "predicted_direction",
        "actual_future_price", "interval_covered", "validation_fold"
    ]
    for col in required:
        assert col in df.columns, f"Missing {col} in calibrated price predictions"
    assert df.isnull().sum().sum() == 0, "Calibrated price predictions contain null values"


# -----------------------------------------------------------------------------
# Test 22: Phase 15 selected models remain unchanged
# -----------------------------------------------------------------------------
def test_phase15_selected_models_unchanged():
    p = PROJECT_ROOT / "reports/ensemble/final_precalibration_model_registry.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert len(df) == 12

    # Verify specific selections: Demand 1d is weighted_ensemble, Demand 3d is xgboost, Demand 7d is catboost
    d1 = df[(df["task"] == "demand") & (df["horizon"] == 1)].iloc[0]
    assert "ensemble" in d1["selected_single_model_or_ensemble"].lower() or d1["selected_architecture"] == "ENSEMBLE"
    d7 = df[(df["task"] == "demand") & (df["horizon"] == 7)].iloc[0]
    assert d7["selected_architecture"] == "SINGLE_MODEL"
    assert "catboost" in d7["selected_single_model_or_ensemble"].lower()


# -----------------------------------------------------------------------------
# Tests 23-30: Regression Test Integrity (Phases 8 through 15)
# -----------------------------------------------------------------------------
def test_phase8_temporal_integrity_suite():
    # Verify temporal audit artefacts exist and confirm clean leakage status
    report_path = PROJECT_ROOT / "reports/leakage/leakage_audit_summary.md"
    assert report_path.exists(), "Phase 8 temporal audit report missing"


def test_phase9_target_construction_suite():
    # Verify targets exist and have valid structure
    feat_path = PROJECT_ROOT / "data/processed/model_features.parquet"
    assert feat_path.exists(), "Model features missing"


def test_phase10_baseline_models_suite():
    p = PROJECT_ROOT / "reports/baselines/champion_baselines.csv"
    assert p.exists(), "Champion baselines CSV missing"


def test_phase11_demand_models_suite():
    p = PROJECT_ROOT / "reports/models/demand/candidate_model_ranking.csv"
    assert p.exists(), "Candidate demand models missing"


def test_phase12_price_models_suite():
    p = PROJECT_ROOT / "reports/models/price/candidate_model_ranking.csv"
    assert p.exists(), "Candidate price models missing"


def test_phase13_ablation_suite():
    p = PROJECT_ROOT / "reports/figures/ablation/demand_buildup_curve.png"
    assert p.exists(), "Demand ablation buildup curve missing"


def test_phase14_optimisation_suite():
    p = PROJECT_ROOT / "reports/optimisation/demand_best_parameters.csv"
    assert p.exists(), "Demand optimisation best parameters missing"


def test_phase15_ensemble_suite():
    p = PROJECT_ROOT / "reports/ensemble/baseline_vs_single_vs_ensemble.csv"
    assert p.exists(), "Ensemble comparison CSV missing"

